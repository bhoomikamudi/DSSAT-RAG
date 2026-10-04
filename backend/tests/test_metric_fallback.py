"""No-LLM metric resolution: the planner's fallback must pick the variable
the question names (yield -> HWAM, rainfall -> PRCP), never a variable that
merely shares a word such as "average", and must ask when it is ambiguous."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent import planner as planner_module
from app.agent import response_generator as response_module
from app.api.v1 import chat
from app.db.session import get_db
from app.services.statistics_service import StatisticsService
from conftest import FakeDataLayer, distances_km, haversine_km

DB_VARIABLES = ["CWAM", "GNAM", "HWAH", "HWAM", "PRCP", "TMAXA", "TMINA"]


# -----------------------------------------------------------------------------
# Resolver unit tests (no database, no LLM)
# -----------------------------------------------------------------------------

@pytest.fixture
def resolver(monkeypatch):
    async def available(self):
        return list(DB_VARIABLES)

    monkeypatch.setattr(StatisticsService, "get_available_variables", available)
    service = StatisticsService(None)
    return lambda text: asyncio.run(service.resolve_metric(text))


@pytest.mark.parametrize(
    "question, code",
    [
        ("What is the average yield?", "HWAM"),
        ("What is the average yield near Kitale?", "HWAM"),
        ("average harvested yield for BASE", "HWAM"),
        ("What is the average rainfall in 2020?", "PRCP"),
        ("What is the mean precipitation?", "PRCP"),
        ("average HWAM", "HWAM"),
        ("What is the average biomass for BASE?", "CWAM"),
        ("What is the mean grain number?", "GNAM"),
        ("average harvested weight", "HWAH"),
        ("What is the average maximum temperature?", "TMAXA"),
        ("average yield for rain-fed runs", "HWAM"),  # "rain-fed" is not rainfall
    ],
)
def test_common_wording_resolves_to_project_variable(resolver, question, code):
    result = resolver(question)
    assert result.code == code
    assert not result.ambiguous


@pytest.mark.parametrize(
    "question, candidates",
    [
        ("What is the average temperature?", ["TMAXA", "TMINA"]),
        ("What is the average yield and rainfall?", ["HWAM", "PRCP"]),
    ],
)
def test_ambiguous_wording_returns_candidates_not_a_guess(resolver, question, candidates):
    result = resolver(question)
    assert result.code is None
    assert result.ambiguous
    assert result.candidates == candidates


@pytest.mark.parametrize(
    "question",
    [
        "What is the average value?",          # no variable named
        "What is the average nitrogen uptake?",  # NUP is not in the data
    ],
)
def test_no_matching_variable_returns_none(resolver, question):
    result = resolver(question)
    assert result.code is None and not result.ambiguous


def test_aggregation_word_alone_never_selects_a_variable(resolver):
    # The original bug: "average" matched "Average Maximum Temperature".
    assert resolver("What is the average?").code is None


# -----------------------------------------------------------------------------
# Full chat path with no LLM (and with a failing LLM)
# -----------------------------------------------------------------------------

@pytest.fixture
def aggregation_calls(fake_db, monkeypatch):
    """Record what reaches StatisticsService.calculate_aggregation."""
    calls = []

    async def recording(self, variable_code, aggregation, **filters):
        calls.append({"variable_code": variable_code, "aggregation": aggregation, **filters})
        return await FakeDataLayer.calculate_aggregation(self, variable_code, aggregation, **filters)

    monkeypatch.setattr(StatisticsService, "calculate_aggregation", recording)
    return calls


@pytest.fixture
def client(aggregation_calls):
    app = FastAPI()
    app.include_router(chat.router, prefix="/api/v1/chat")

    async def fake_get_db():
        yield object()

    app.dependency_overrides[get_db] = fake_get_db
    return TestClient(app)


@pytest.fixture(params=["no_llm", "failing_llm"])
def llm_mode(request, monkeypatch):
    """Run each scenario with no LLM configured and with an LLM that errors."""
    if request.param == "failing_llm":

        class Failing:
            async def create(self, **kwargs):
                raise ConnectionError("gateway reset the connection")

        class FailingClient:
            responses = Failing()
            chat = type("Chat", (), {"completions": Failing()})()

        monkeypatch.setattr(planner_module.settings, "OPENAI_API_KEY", "test")
        monkeypatch.setattr(planner_module, "AsyncOpenAI", lambda **kwargs: FailingClient())
        monkeypatch.setattr(response_module, "AsyncOpenAI", lambda **kwargs: FailingClient())
    return request.param


@pytest.fixture(scope="module")
def kitale_point(sample_frame):
    """A data location near the centre of the sample coverage."""
    locations = sample_frame[["LATITUDE", "LONGITUDE"]].drop_duplicates()
    lat = locations["LATITUDE"].to_numpy(float)
    lon = locations["LONGITUDE"].to_numpy(float)
    best = int(np.argmin([haversine_km(lat, lon, a, b).sum() for a, b in zip(lat, lon)]))
    return float(lat[best]), float(lon[best])


def ask(client, message, **extra):
    response = client.post("/api/v1/chat/", json={"message": message, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def test_average_yield_near_kitale_without_llm(
    client, llm_mode, aggregation_calls, sample_frame, kitale_point, geocoder_returns
):
    lat, lon = kitale_point
    geocoder_returns({"kitale": [{"display_name": "Kitale, Trans-Nzoia County, Kenya",
                                  "lat": str(lat), "lon": str(lon), "importance": 0.6}]})
    body = ask(client, "What is the average yield near Kitale?")

    # Metric and aggregation
    stats = body["statistics"]
    assert stats["metric"] == "HWAM"
    assert stats["aggregation_type"] == "AVG"

    # The radius filter reached the statistic calculation
    call = aggregation_calls[-1]
    assert call["variable_code"] == "HWAM" and call["aggregation"] == "AVG"
    assert call["spatial"] is not None
    assert (call["spatial"].latitude, call["spatial"].longitude, call["spatial"].radius_km) == (lat, lon, 25.0)
    assert body["spatial"]["filter"]["label"] == "Kitale, Trans-Nzoia County, Kenya"

    # Result: the HWAM mean of simulations within 25 km, computed independently
    inside = sample_frame[distances_km(sample_frame, lat, lon) <= 25.0]
    assert stats["value"] == pytest.approx(inside["HWAM"].mean())
    assert stats["count"] == len(inside) < len(sample_frame)
    assert body["spatial"]["simulations"] == len(inside)
    assert "within 25 km of Kitale" in body["answer"]
    assert "TMAXA" not in body["answer"]
    if llm_mode in ("no_llm", "failing_llm"):  # answer built from computed values
        assert f"Based on {len(inside):,} records" in body["answer"]


def test_average_rainfall_in_2020_without_llm(client, llm_mode, aggregation_calls, sample_frame):
    body = ask(client, "What is the average rainfall in 2020?")

    stats = body["statistics"]
    assert stats["metric"] == "PRCP"
    assert stats["aggregation_type"] == "AVG"
    assert body["query_plan"]["filters"] == {"year": 2020}

    call = aggregation_calls[-1]
    assert call["variable_code"] == "PRCP"
    assert call["year"] == 2020
    assert call.get("spatial") is None

    expected = sample_frame.loc[sample_frame["year"] == 2020, "PRCP"]
    assert stats["value"] == pytest.approx(expected.mean())
    assert stats["count"] == len(expected)


def test_ambiguous_metric_asks_for_clarification(client, llm_mode, aggregation_calls):
    body = ask(client, "What is the average temperature?")

    assert "could refer to more than one variable" in body["answer"]
    assert "TMAXA (Average Maximum Temperature)" in body["answer"]
    assert "TMINA (Average Minimum Temperature)" in body["answer"]
    assert body["statistics"] is None
    assert aggregation_calls == []  # nothing was calculated for a guessed metric
    assert body["confidence"] == "low"


def test_ambiguous_metric_near_a_place_still_asks(client, llm_mode, aggregation_calls,
                                                  kitale_point, geocoder_returns):
    lat, lon = kitale_point
    geocoder_returns({"kitale": [{"display_name": "Kitale, Kenya", "lat": str(lat),
                                  "lon": str(lon), "importance": 0.6}]})
    body = ask(client, "What is the average temperature near Kitale?")
    assert "could refer to more than one variable" in body["answer"]
    assert aggregation_calls == []


def test_cultivar_code_is_not_used_as_a_metric(client, llm_mode, aggregation_calls):
    # The old uppercase heuristic would have treated "BASE" as a metric code.
    body = ask(client, "What is the average for BASE?")
    assert all(call["variable_code"] != "BASE" for call in aggregation_calls)
    assert (body["statistics"] or {}).get("metric") != "BASE"
