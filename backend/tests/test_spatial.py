"""Spatial filtering: validation, parsing, place lookup, SQL, and the full
chat path. Test points are derived from the sample data (never hardcoded),
and expected counts come from an independent haversine calculation."""
from __future__ import annotations

import re

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scipy import stats
from sqlalchemy.dialects import postgresql

from app.agent.location_parser import extract_location
from app.agent.models import SpatialFilter, validate_coordinates
from app.api.v1 import chat, spatial as spatial_api
from app.agent import planner as planner_module
from app.agent import response_generator as response_module
from app.core.config import get_settings
from app.db.session import get_db
from app.parsers.csv_parser import DSSATParser
from app.services.geocoding_service import GeocodingService
from app.services.metadata_service import MetadataService
from app.services.spatial_sql import within_radius
from app.services.statistics_service import StatisticsService
from conftest import ORIGINAL_GEOCODER_REQUEST, FakeLLM, distances_km, haversine_km


# -----------------------------------------------------------------------------
# Helpers derived from the data
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def locations(sample_frame):
    return sample_frame[["LATITUDE", "LONGITUDE"]].drop_duplicates().reset_index(drop=True)


@pytest.fixture(scope="module")
def center(locations):
    """The data location with the smallest total distance to all others."""
    lat = locations["LATITUDE"].to_numpy(float)
    lon = locations["LONGITUDE"].to_numpy(float)
    total = [haversine_km(lat, lon, a, b).sum() for a, b in zip(lat, lon)]
    best = int(np.argmin(total))
    return float(lat[best]), float(lon[best])


@pytest.fixture(scope="module")
def far_point(locations):
    """A valid point well outside the data (5 degrees south of its extent)."""
    return float(locations["LATITUDE"].min()) - 5.0, float(locations["LONGITUDE"].mean())


def rows_within(frame, lat, lon, radius_km):
    return frame[distances_km(frame, lat, lon) <= radius_km]


@pytest.fixture
def client(fake_db):
    app = FastAPI()
    app.include_router(chat.router, prefix="/api/v1/chat")
    app.include_router(spatial_api.router, prefix="/api/v1/spatial")

    async def fake_get_db():
        yield object()

    app.dependency_overrides[get_db] = fake_get_db
    return TestClient(app)


def ask(client, message, **extra):
    response = client.post("/api/v1/chat/", json={"message": message, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def nominatim(name, lat, lon, importance=0.5):
    return {"display_name": name, "lat": str(lat), "lon": str(lon), "importance": importance, "type": "town"}


# -----------------------------------------------------------------------------
# 1. Coordinate and radius validation
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"latitude": 90.5, "longitude": 0, "radius_km": 5}, "Latitude 90.5 is out of range"),
        ({"latitude": -91, "longitude": 0, "radius_km": 5}, "Latitude -91.0 is out of range"),
        ({"latitude": 0, "longitude": 181, "radius_km": 5}, "Longitude 181.0 is out of range"),
        ({"latitude": float("nan"), "longitude": 0, "radius_km": 5}, "finite"),
        ({"latitude": 0, "longitude": 0, "radius_km": 0}, "greater than 0"),
        ({"latitude": 0, "longitude": 0, "radius_km": -3}, "greater than 0"),
    ],
)
def test_spatial_filter_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SpatialFilter(**kwargs)


def test_radius_above_configured_maximum_is_rejected():
    too_big = get_settings().SPATIAL_MAX_RADIUS_KM + 1
    with pytest.raises(ValueError, match="exceeds the maximum"):
        SpatialFilter(latitude=0, longitude=0, radius_km=too_big)


def test_boundary_coordinates_are_valid():
    assert validate_coordinates(-90, 180) == (-90.0, 180.0)
    assert validate_coordinates("0.5", "35") == (0.5, 35.0)


@pytest.mark.parametrize(
    "payload",
    [
        {"latitude": 1.0},                                  # longitude missing
        {"longitude": 35.0},                                # latitude missing
        {"latitude": 95.0, "longitude": 35.0},              # out of range
        {"latitude": 1.0, "longitude": 35.0, "radius_km": 0},
        {"latitude": 1.0, "longitude": 35.0, "radius_km": 100000},
    ],
)
def test_chat_request_rejects_invalid_map_input(client, payload):
    response = client.post("/api/v1/chat/", json={"message": "average yield", **payload})
    assert response.status_code == 422


# -----------------------------------------------------------------------------
# 2. Finding locations and radii in questions
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question, place, radius, cleaned",
    [
        ("What is the average yield near Kitale?", "Kitale", None, "What is the average yield?"),
        ("What is the average yield near kitale for BASE in 2020?", "kitale", None,
         "What is the average yield for BASE in 2020?"),
        ("Correlation between rainfall and yield within 50 km of Eldoret, Kenya", "Eldoret, Kenya", 50.0,
         "Correlation between rainfall and yield"),
        ("Correlation between rainfall and yield within 50 km of Eldoret, for base", "Eldoret", 50.0,
         "Correlation between rainfall and yield, for base"),
        ("What is the average rainfall in 2020 within a 30 mile radius around Bungoma?", "Bungoma", 48.28,
         "What is the average rainfall in 2020?"),
        ("Correlation between rainfall and yield near the town of Webuye", "Webuye", None,
         "Correlation between rainfall and yield"),
        ("Which cultivar has the highest average yield close to Mount Elgon?", "Mount Elgon", None,
         "Which cultivar has the highest average yield?"),
    ],
)
def test_place_and_radius_extraction(question, place, radius, cleaned):
    mention = extract_location(question)
    assert mention.place == place
    assert mention.radius_km == radius
    assert mention.cleaned_query == cleaned


@pytest.mark.parametrize(
    "question, lat, lon, radius",
    [
        ("average yield near 0.79, 35.04", 0.79, 35.04, None),
        ("average yield at lat 0.7917 lon 35.0417 within 10 km", 0.7917, 35.0417, 10.0),
        ("average yield around (-1.2921, 36.8219)", -1.2921, 36.8219, None),
    ],
)
def test_coordinates_in_question(question, lat, lon, radius):
    mention = extract_location(question)
    assert (mention.latitude, mention.longitude, mention.radius_km) == (lat, lon, radius)
    assert mention.place is None
    assert mention.cleaned_query == "average yield"


@pytest.mark.parametrize(
    "question",
    [
        "What is the average rainfall in 2020?",
        "Is yield near zero for VSHT?",
        "yield around the normal planting date",
        "Is rainfall around 800 mm optimal?",
        "correlation between rainfall and yield near base",
        "What does SHT mean?",
    ],
)
def test_non_locations_are_not_extracted(question):
    mention = extract_location(question)
    assert not mention.has_location
    assert mention.cleaned_query == question


# -----------------------------------------------------------------------------
# 3. Place lookup (separate from distance filtering)
# -----------------------------------------------------------------------------

@pytest.fixture
def real_request(monkeypatch):
    """Use the real HTTP code path against an httpx MockTransport."""
    monkeypatch.setattr(GeocodingService, "_request", ORIGINAL_GEOCODER_REQUEST)

    def make(handler):
        return GeocodingService(transport=httpx.MockTransport(handler))

    return make


@pytest.mark.asyncio
async def test_geocode_single_match_and_polite_request(real_request):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=[nominatim("Kitale, Trans Nzoia, Kenya", 1.0191, 35.0023)])

    result = await real_request(handler).geocode("Kitale")
    assert result.status == "ok"
    assert (result.match.latitude, result.match.longitude) == (1.0191, 35.0023)
    request = seen[0]
    assert request.url.params["q"] == "Kitale"
    assert request.url.params["format"] == "jsonv2"
    assert "DSSAT-RAG" in request.headers["User-Agent"]


@pytest.mark.asyncio
async def test_geocode_no_results(real_request):
    result = await real_request(lambda r: httpx.Response(200, json=[])).geocode("Nowhereville")
    assert result.status == "not_found"
    assert "No place named 'Nowhereville'" in result.message


@pytest.mark.asyncio
async def test_geocode_rejects_unrelated_result(real_request):
    handler = lambda r: httpx.Response(200, json=[nominatim("Kitui, Kenya", -1.37, 38.01)])
    result = await real_request(handler).geocode("Kitale")
    assert result.status == "not_found"
    assert result.match is None
    assert "does not match, so it was not used" in result.message


@pytest.mark.asyncio
async def test_geocode_ambiguous_places(real_request):
    handler = lambda r: httpx.Response(200, json=[
        nominatim("Springfield, Illinois, United States", 39.80, -89.64, 0.62),
        nominatim("Springfield, Massachusetts, United States", 42.10, -72.59, 0.60),
    ])
    result = await real_request(handler).geocode("Springfield")
    assert result.status == "ambiguous"
    assert result.match is None
    assert "matches several different places" in result.message
    assert "Illinois" in result.message and "Massachusetts" in result.message


@pytest.mark.asyncio
async def test_geocode_clear_winner_is_not_ambiguous(real_request):
    handler = lambda r: httpx.Response(200, json=[
        nominatim("Paris, France", 48.85, 2.35, 0.95),
        nominatim("Paris, Texas, United States", 33.66, -95.55, 0.45),
    ])
    result = await real_request(handler).geocode("Paris")
    assert result.status == "ok" and "France" in result.match.name


@pytest.mark.asyncio
async def test_geocode_ambiguity_resolved_by_data_coverage(real_request):
    handler = lambda r: httpx.Response(200, json=[
        nominatim("Springfield, Illinois, United States", 39.80, -89.64, 0.62),
        nominatim("Springfield, Massachusetts, United States", 42.10, -72.59, 0.60),
    ])
    bounds = {"min_lat": 41.0, "max_lat": 43.0, "min_lon": -74.0, "max_lon": -71.0}
    result = await real_request(handler).geocode("Springfield", prefer_bounds=bounds)
    assert result.status == "ok"
    assert "Massachusetts" in result.match.name
    assert "only one inside the data coverage area" in result.message


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["http_500", "connect", "bad_json"])
async def test_geocode_service_failure(real_request, failure):
    def handler(request):
        if failure == "http_500":
            return httpx.Response(500)
        if failure == "bad_json":
            return httpx.Response(200, json={"error": "unexpected"})
        raise httpx.ConnectError("offline")

    result = await real_request(handler).geocode("Kitale")
    assert result.status == "error"
    assert result.match is None
    assert "could not be reached" in result.message


@pytest.mark.asyncio
async def test_geocode_results_are_cached(real_request):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=[nominatim("Kitale, Kenya", 1.0191, 35.0023)])

    service = real_request(handler)
    await service.geocode("Kitale")
    await service.geocode("kitale")
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_geocoder_can_be_disabled(monkeypatch):
    monkeypatch.setattr(get_settings(), "GEOCODER_ENABLED", False)
    result = await GeocodingService().geocode("Kitale")
    assert result.status == "disabled"


# -----------------------------------------------------------------------------
# 4. SQL: the radius is a geodesic condition applied by PostgreSQL
# -----------------------------------------------------------------------------

class CapturingSession:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.statements = []

    async def execute(self, stmt):
        self.statements.append(stmt)
        rows = self.rows

        class Result:
            def all(self_inner):
                return rows

            def one(self_inner):
                return rows[0]

            def first(self_inner):
                return rows[0] if rows else None

            def fetchone(self_inner):
                return rows[0] if rows else None

            def scalar_one(self_inner):
                return 0

        return Result()


def compiled(stmt) -> str:
    text = str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
    return re.sub(r"\s+", " ", text)


RADIUS_SQL = (
    "ST_DWithin(CAST(simulations.location AS geography), "
    "CAST(ST_SetSRID(ST_MakePoint(35.04, 0.79), 4326) AS geography), 25000.0)"
)
FILTER = SpatialFilter(latitude=0.79, longitude=35.04, radius_km=25)


def test_within_radius_sql():
    assert compiled(within_radius(FILTER)) == RADIUS_SQL


@pytest.mark.asyncio
async def test_radius_applied_to_aggregation_sql():
    from types import SimpleNamespace

    session = CapturingSession([SimpleNamespace(value=1.0, count=1, stddev=None, min=1.0, max=1.0)])
    await StatisticsService(session).calculate_aggregation("HWAM", "AVG", cultivar="BASE", spatial=FILTER)
    text = compiled(session.statements[0])
    assert RADIUS_SQL in text and "simulations.cultivar = 'BASE'" in text


@pytest.mark.asyncio
async def test_radius_applied_to_analysis_records_sql():
    session = CapturingSession([])
    await StatisticsService(session).get_analysis_records(["PRCP", "HWAM"], spatial=FILTER)
    assert RADIUS_SQL in compiled(session.statements[0])


@pytest.mark.asyncio
async def test_radius_applied_to_trend_and_breakdown_sql():
    session = CapturingSession([])
    service = StatisticsService(session)
    await service.get_yearly_trend("HWAM", spatial=FILTER)
    await service.calculate_breakdown("HWAM", "AVG", "cultivar", spatial=FILTER)
    assert all(RADIUS_SQL in compiled(stmt) for stmt in session.statements)


@pytest.mark.asyncio
async def test_radius_applied_to_output_count_sql():
    session = CapturingSession([])
    await MetadataService(session).get_record_count("simulation_outputs", spatial=FILTER)
    text = compiled(session.statements[0])
    assert "JOIN simulations ON simulation_outputs.simulation_id = simulations.simulation_id" in text
    assert RADIUS_SQL in text


def test_existing_filters_unchanged_without_spatial():
    service = StatisticsService(CapturingSession())
    assert len(service._build_simulation_filters(cultivar="BASE")) == 1


# -----------------------------------------------------------------------------
# 5. Full chat path: map point, place, radius, empty results
# -----------------------------------------------------------------------------

def test_map_point_filters_analysis_before_python_runs(client, fake_db, sample_frame, center):
    lat, lon = center
    body = ask(client, "correlation between rainfall and yield", latitude=lat, longitude=lon, radius_km=25)
    expected = rows_within(sample_frame, lat, lon, 25)

    assert fake_db.calls[-1]["spatial"].radius_km == 25
    analysis = body["analysis"]
    assert analysis["sample_size"] == len(expected)
    assert len(expected) < len(sample_frame)
    assert analysis["correlation"] == pytest.approx(stats.pearsonr(expected["PRCP"], expected["HWAM"]).statistic)
    assert body["spatial"]["simulations"] == len(expected)
    assert body["spatial"]["locations"] == len(expected[["LATITUDE", "LONGITUDE"]].drop_duplicates())
    assert "within 25 km of" in body["answer"]


def test_default_radius_comes_from_configuration(client, fake_db, sample_frame, center, monkeypatch):
    assert get_settings().SPATIAL_DEFAULT_RADIUS_KM == 25.0  # the shipped, testable default
    monkeypatch.setattr(get_settings(), "SPATIAL_DEFAULT_RADIUS_KM", 12.5)
    lat, lon = center
    body = ask(client, "correlation between rainfall and yield", latitude=lat, longitude=lon)
    assert body["spatial"]["filter"]["radius_km"] == 12.5
    assert body["analysis"]["sample_size"] == len(rows_within(sample_frame, lat, lon, 12.5))


def test_radius_in_question_overrides_request_radius(client, sample_frame, center):
    lat, lon = center
    body = ask(client, "correlation between rainfall and yield within 10 km",
               latitude=lat, longitude=lon, radius_km=50)
    assert body["spatial"]["filter"]["radius_km"] == 10
    assert body["analysis"]["sample_size"] == len(rows_within(sample_frame, lat, lon, 10))


def test_place_in_question_is_geocoded_then_filtered(client, fake_db, sample_frame, center, geocoder_returns):
    lat, lon = center
    requests = geocoder_returns({"testtown": [nominatim("Testtown, Test County, Kenya", lat, lon)]})
    body = ask(client, "correlation between rainfall and yield near Testtown for base")

    assert requests == ["Testtown"]
    assert body["spatial"]["filter"]["label"] == "Testtown, Test County, Kenya"
    assert body["spatial"]["filter"]["source"] == "place"
    expected = rows_within(sample_frame[sample_frame["cultivar"] == "BASE"], lat, lon, 25)
    assert fake_db.calls[-1]["filters"] == {"cultivar": "BASE"}
    assert body["analysis"]["sample_size"] == len(expected)


def test_place_wins_over_map_point_and_says_so(client, sample_frame, center, far_point, geocoder_returns):
    lat, lon = center
    geocoder_returns({"testtown": [nominatim("Testtown, Kenya", lat, lon)]})
    body = ask(client, "correlation between rainfall and yield near Testtown",
               latitude=far_point[0], longitude=far_point[1])
    assert body["spatial"]["filter"]["latitude"] == lat
    assert "instead of the selected map point" in body["spatial"]["note"]


def test_ambiguous_place_stops_without_querying(client, fake_db, geocoder_returns):
    geocoder_returns({"springfield": [
        nominatim("Springfield, Illinois, United States", 39.80, -89.64, 0.62),
        nominatim("Springfield, Massachusetts, United States", 42.10, -72.59, 0.60),
    ]})
    body = ask(client, "correlation between rainfall and yield near Springfield")
    assert "matches several different places" in body["answer"]
    assert fake_db.calls == []
    assert body["analysis"] is None
    assert body["confidence"] == "low"


def test_unknown_place_stops_without_querying(client, fake_db, geocoder_returns):
    geocoder_returns({})
    body = ask(client, "What is the average yield near Nowhereville?")
    assert "No place named 'Nowhereville'" in body["answer"]
    assert fake_db.calls == []


def test_geocoder_outage_is_reported(client, fake_db):
    body = ask(client, "correlation between rainfall and yield near Testtown")
    assert "could not be reached" in body["answer"]
    assert fake_db.calls == []


def test_point_outside_coverage_reports_nearest_data(client, fake_db, far_point, locations):
    body = ask(client, "correlation between rainfall and yield",
               latitude=far_point[0], longitude=far_point[1], radius_km=25)
    distances = haversine_km(locations["LATITUDE"].to_numpy(float), locations["LONGITUDE"].to_numpy(float),
                             *far_point)
    assert "No simulation data was found within 25 km" in body["answer"]
    assert f"{distances.min():,.1f} km away" in body["answer"]
    assert body["spatial"]["simulations"] == 0
    assert fake_db.calls == []  # analysis never ran


def test_radius_without_location_asks_for_one(client, fake_db):
    body = ask(client, "What is the average yield within 40 km?")
    assert "no place or point" in body["answer"]
    assert fake_db.calls == []


def test_definition_question_not_blocked_by_distant_map_point(llm_client, far_point):
    # A map point only filters data questions; definitions still answer.
    body = ask(llm_client, "What does SHT mean?", latitude=far_point[0], longitude=far_point[1])
    assert "No simulation data was found" not in body["answer"]
    assert body["semantic_plan"]["operations"][0]["operation"] == "definition"


def test_follow_up_with_map_point_still_selected(client, fake_db, sample_frame, center):
    # The frontend re-sends the selected point with every question.
    lat, lon = center
    point = {"latitude": lat, "longitude": lon, "radius_km": 25, "session_id": "s-map"}
    ask(client, "correlation between rainfall and yield", **point)
    follow = ask(client, "Now fit a quadratic instead", **point)
    assert follow["analysis"]["analysis_type"] == "quadratic_regression"
    assert follow["analysis"]["sample_size"] == len(rows_within(sample_frame, lat, lon, 25))
    assert fake_db.calls[-1]["spatial"].latitude == lat


def test_clearing_map_point_removes_area_from_follow_up(client, fake_db, sample_frame, center):
    """Regression: a cleared map point must not keep filtering follow-ups."""
    lat, lon = center
    ask(client, "correlation between rainfall and yield", latitude=lat, longitude=lon,
        radius_km=25, session_id="s-clear")
    follow = ask(client, "Now fit a quadratic instead", session_id="s-clear")  # no point sent
    assert follow["analysis"]["analysis_type"] == "quadratic_regression"
    assert "spatial" not in fake_db.calls[-1]
    assert follow["analysis"]["sample_size"] == len(sample_frame)
    assert follow["spatial"] is None
    assert "within" not in follow["answer"]


def test_place_area_carries_into_follow_up(client, fake_db, sample_frame, center, geocoder_returns):
    lat, lon = center
    geocoder_returns({"testtown": [nominatim("Testtown, Kenya", lat, lon)]})
    ask(client, "correlation between rainfall and yield near Testtown", session_id="s-place")
    follow = ask(client, "Now fit a quadratic instead", session_id="s-place")
    assert follow["analysis"]["sample_size"] == len(rows_within(sample_frame, lat, lon, 25))
    assert fake_db.calls[-1]["spatial"].label == "Testtown, Kenya"


@pytest.mark.parametrize(
    "reset_question",
    [
        "Now use all data",
        "Use all data instead",
        "Repeat it for the entire dataset",
        "Remove the location filter",
        "Do the same without the location filter",
    ],
)
def test_asking_for_all_data_removes_area(client, fake_db, sample_frame, center, geocoder_returns, reset_question):
    """Regression: 'use all data' drops a place area carried from earlier questions."""
    lat, lon = center
    geocoder_returns({"testtown": [nominatim("Testtown, Kenya", lat, lon)]})
    ask(client, "correlation between rainfall and yield near Testtown", session_id="s-reset")
    follow = ask(client, reset_question, session_id="s-reset")
    assert "spatial" not in fake_db.calls[-1]
    assert follow["analysis"]["sample_size"] == len(sample_frame)
    assert follow["spatial"]["cleared"] is True


def test_all_data_overrides_a_still_selected_map_point(client, fake_db, sample_frame, center):
    """'Use all data' wins even if the frontend still sends a map point."""
    lat, lon = center
    point = {"latitude": lat, "longitude": lon, "radius_km": 25, "session_id": "s-override"}
    ask(client, "correlation between rainfall and yield", **point)
    follow = ask(client, "Now use all data", **point)
    assert "spatial" not in fake_db.calls[-1]
    assert follow["analysis"]["sample_size"] == len(sample_frame)
    assert follow["spatial"] == {"cleared": True, "note": "Location filter removed; using all data.", "notice": None}


def test_llm_outage_falls_back_to_computed_answer(client, sample_frame, center, monkeypatch):
    """If the LLM call fails, the answer still reports the computed result and area."""
    import app.agent.response_generator as rg

    class FailingCompletions:
        async def create(self, **kwargs):
            raise ConnectionError("gateway reset the connection")

    class FailingClient:
        chat = type("Chat", (), {"completions": FailingCompletions()})()

    monkeypatch.setattr(rg.settings, "OPENAI_API_KEY", "test")
    monkeypatch.setattr(rg, "AsyncOpenAI", lambda **kwargs: FailingClient())
    lat, lon = center
    body = ask(client, "correlation between rainfall and yield", latitude=lat, longitude=lon, radius_km=25)
    assert "Pearson correlation" in body["answer"]
    assert "within 25 km of" in body["answer"] and "simulated locations" in body["answer"]
    assert body["confidence"] == "low"


def test_answer_reports_radius_and_location_count(client, sample_frame, center):
    lat, lon = center
    body = ask(client, "correlation between rainfall and yield", latitude=lat, longitude=lon, radius_km=25)
    inside = rows_within(sample_frame, lat, lon, 25)
    locations = len(inside[["LATITUDE", "LONGITUDE"]].drop_duplicates())
    assert "Area: within 25 km of" in body["answer"]
    assert f"{locations:,} simulated locations with {len(inside):,} simulations in total" in body["answer"]


def test_analysis_without_location_is_unchanged(client, fake_db, sample_frame):
    body = ask(client, "correlation between rainfall and yield for base")
    assert "spatial" not in fake_db.calls[-1]
    assert body["spatial"] is None
    assert body["analysis"]["sample_size"] == int((sample_frame["cultivar"] == "BASE").sum())


# Aggregates via the (fake) LLM planner honor the same radius.

@pytest.fixture
def llm_client(client, monkeypatch):
    fake = FakeLLM({
        "What is the average yield?": {
            "goal": "g", "intent": "aggregate",
            "operations": [{"operation": "aggregate", "entity": "simulation_outputs", "metric": "HWAM",
                            "aggregation": "AVG",
                            # An LLM-invented location filter must be replaced by the radius.
                            "filters": [{"field": "state", "operator": "=", "value": "Rift Valley"}],
                            "group_by": [], "independent": True}],
        },
        "How many simulation records are loaded?": {
            "goal": "g", "intent": "metadata",
            "operations": [{"operation": "count", "entity": "simulations", "filters": [],
                            "group_by": [], "independent": True}],
        },
    })
    monkeypatch.setattr(planner_module.settings, "OPENAI_API_KEY", "test")
    monkeypatch.setattr(planner_module, "AsyncOpenAI", lambda **kwargs: fake)
    monkeypatch.setattr(response_module, "AsyncOpenAI", lambda **kwargs: None)
    return client


def test_average_near_map_point(llm_client, sample_frame, center):
    lat, lon = center
    body = ask(llm_client, "What is the average yield near here?", latitude=lat, longitude=lon, radius_km=25)
    expected = rows_within(sample_frame, lat, lon, 25)["HWAM"]
    assert body["statistics"]["value"] == pytest.approx(expected.mean())
    assert body["statistics"]["count"] == len(expected)


def test_count_within_radius(llm_client, sample_frame, center):
    lat, lon = center
    body = ask(llm_client, "How many simulation records are loaded?", latitude=lat, longitude=lon, radius_km=25)
    assert f"{len(rows_within(sample_frame, lat, lon, 25))}" in str(body["tool_outputs"])


# -----------------------------------------------------------------------------
# 6. Coverage endpoints and ingestion schema requirements
# -----------------------------------------------------------------------------

def test_spatial_config_endpoint(client):
    body = client.get("/api/v1/spatial/config").json()
    assert body["default_radius_km"] == get_settings().SPATIAL_DEFAULT_RADIUS_KM
    assert body["max_radius_km"] == get_settings().SPATIAL_MAX_RADIUS_KM


def test_coverage_endpoint_is_derived_from_data(client, locations, sample_frame):
    body = client.get("/api/v1/spatial/coverage").json()
    assert body["location_count"] == len(locations)
    assert body["simulations"] == len(sample_frame)
    assert body["bounds"]["min_lat"] == pytest.approx(locations["LATITUDE"].min())
    assert body["bounds"]["max_lon"] == pytest.approx(locations["LONGITUDE"].max())


def test_parser_skips_rows_with_missing_or_invalid_coordinates(tmp_path):
    csv = tmp_path / "runs.csv"
    csv.write_text(
        "LATITUDE,LONGITUDE,RUN_NAME,CR,WYEAR,HWAM,PRCP\n"
        "1.0,35.0,MZ_RF_HighN_MZ_BASE__pfrst0,MZ,2000,4000,600\n"
        ",35.0,MZ_RF_HighN_MZ_BASE__pfrst0,MZ,2000,4000,600\n"      # missing latitude
        "1.0,,MZ_RF_HighN_MZ_BASE__pfrst0,MZ,2000,4000,600\n"       # missing longitude
        "95.0,35.0,MZ_RF_HighN_MZ_BASE__pfrst0,MZ,2000,4000,600\n"  # out of range
        "-99,-99,MZ_RF_HighN_MZ_BASE__pfrst0,MZ,2000,4000,600\n"    # DSSAT missing code
    )
    parsed = DSSATParser.parse_csv(str(csv))
    assert len(parsed) == 1
    assert (parsed[0].location.latitude, parsed[0].location.longitude) == (1.0, 35.0)


def test_parser_accepts_alternative_coordinate_headers(tmp_path):
    csv = tmp_path / "runs.csv"
    csv.write_text("LAT,LONG,RUN_NAME,CR,WYEAR,HWAM\n1.5,34.5,MZ_RF_HighN_MZ_LNG__pfrst0,MZ,2001,3000\n")
    parsed = DSSATParser.parse_csv(str(csv))
    assert (parsed[0].location.latitude, parsed[0].location.longitude) == (1.5, 34.5)


# Regression (browser): a follow-up that inherits a place must use the radius
# the UI currently shows, unless the earlier question stated its own radius.

@pytest.mark.fallback
def test_inherited_place_uses_current_ui_radius(client, fake_db, sample_frame, center, geocoder_returns):
    lat, lon = center
    geocoder_returns({"testtown": [nominatim("Testtown, Kenya", lat, lon)]})
    ask(client, "correlation between rainfall and yield near Testtown", session_id="s-radius", radius_km=25)
    follow = ask(client, "Now fit a linear regression instead", session_id="s-radius", radius_km=10)
    assert fake_db.calls[-1]["spatial"].radius_km == 10
    assert fake_db.calls[-1]["spatial"].label == "Testtown, Kenya"
    assert follow["analysis"]["sample_size"] == len(rows_within(sample_frame, lat, lon, 10))
    assert "current radius of 10 km (was 25 km)" in follow["spatial"]["note"]


@pytest.mark.fallback
def test_inherited_place_keeps_radius_stated_in_question(client, fake_db, sample_frame, center, geocoder_returns):
    lat, lon = center
    geocoder_returns({"testtown": [nominatim("Testtown, Kenya", lat, lon)]})
    ask(client, "correlation between rainfall and yield within 50 km of Testtown", session_id="s-own", radius_km=25)
    follow = ask(client, "Now fit a linear regression instead", session_id="s-own", radius_km=10)
    assert fake_db.calls[-1]["spatial"].radius_km == 50
    assert follow["analysis"]["sample_size"] == len(rows_within(sample_frame, lat, lon, 50))


@pytest.mark.fallback
def test_radius_flag_is_not_serialized():
    from app.agent.models import SpatialFilter
    area = SpatialFilter(latitude=0.79, longitude=35.04, radius_km=10, source="place", radius_from_question=True)
    assert "radius_from_question" not in area.model_dump()
    assert area.model_copy(deep=True).radius_from_question is True
