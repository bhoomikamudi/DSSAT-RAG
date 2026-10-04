"""Shared fixtures: the sample DSSAT CSVs act as a mocked database.

No live PostgreSQL or LLM is needed. StatisticsService's data-access methods
are patched with pandas equivalents over sample_files/vijaya_data, so the
planner -> executor -> analysis -> response path runs on realistic data while
SQL construction is tested separately in test_statistics_sql.py.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
SAMPLE_DIR = REPO_ROOT / "sample_files" / "vijaya_data"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.parsers.csv_parser import DSSATParser  # noqa: E402
from app.services.statistics_service import StatisticsService  # noqa: E402
from app.services.metadata_service import MetadataService  # noqa: E402
from app.services.spatial_service import SpatialService  # noqa: E402
from app.services.geocoding_service import GeocodingService  # noqa: E402

VARIABLES = ["CWAM", "HWAM", "HWAH", "GNAM", "TMAXA", "TMINA", "PRCP"]
FIELD_COLUMNS = {
    "cultivar": "cultivar",
    "year": "year",
    "planting_stage": "planting_stage",
    "crop": "crop",
    "irrigation": "irrigation",
    "nitrogen_level": "nitrogen_level",
    "state": "state",
    "district": "district",
    "country": "country",
}


def _load_sample_frame() -> pd.DataFrame:
    frames = []
    for path in sorted(SAMPLE_DIR.glob("*.csv")):
        frame = pd.read_csv(path)
        parts = DSSATParser.parse_run_name(frame["RUN_NAME"].iloc[0])
        frame["crop"] = parts["crop"]
        frame["irrigation"] = parts["irrigation"]
        frame["nitrogen_level"] = parts["nitrogen"]
        frame["cultivar"] = parts["cultivar"]
        frame["planting_stage"] = parts["planting_stage"]
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["year"] = data["WYEAR"].astype(int)
    data["state"] = None
    data["district"] = None
    data["country"] = "Kenya"
    return data


@pytest.fixture(scope="session")
def sample_frame() -> pd.DataFrame:
    if not SAMPLE_DIR.exists():
        pytest.skip("sample_files/vijaya_data is not available")
    return _load_sample_frame()


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (vectorized); independent of PostGIS."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * np.arcsin(np.sqrt(h))


def distances_km(frame: pd.DataFrame, latitude: float, longitude: float) -> pd.Series:
    return pd.Series(
        haversine_km(frame["LATITUDE"].to_numpy(float), frame["LONGITUDE"].to_numpy(float),
                     latitude, longitude),
        index=frame.index,
    )


def apply_filters(frame: pd.DataFrame, **filters: Any) -> pd.DataFrame:
    """pandas equivalent of StatisticsService._build_simulation_filters."""
    mask = pd.Series(True, index=frame.index)
    for field, value in filters.items():
        if value is None:
            continue
        if field == "spatial":
            mask &= distances_km(frame, value.latitude, value.longitude) <= value.radius_km
            continue
        column = frame[FIELD_COLUMNS[field]]
        if isinstance(value, (list, tuple, set)):
            mask &= column.isin(list(value))
        else:
            mask &= column == value
    return frame[mask]


class FakeDataLayer:
    """Replacement for StatisticsService/MetadataService data access."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls: List[Dict[str, Any]] = []

    async def get_analysis_records(
        self_service,  # the StatisticsService instance (patched method)
        variables: List[str],
        group_by: Optional[str] = None,
        limit: int = 250_000,
        spatial: Any = None,
        **filters: Any,
    ):
        layer: FakeDataLayer = self_service._fake_layer
        call = {"variables": variables, "group_by": group_by, "filters": filters}
        if spatial is not None:
            call["spatial"] = spatial
        layer.calls.append(call)
        subset = apply_filters(layer.frame, spatial=spatial, **filters).head(limit)
        records = subset[variables].to_dict("records")
        if group_by:
            for record, group in zip(records, subset[FIELD_COLUMNS[group_by]]):
                record["group"] = group
        return records

    async def count_simulations(self_service, **filters):
        return int(len(apply_filters(self_service._fake_layer.frame, **filters)))

    async def get_available_variables(self_service):
        return list(VARIABLES)

    async def get_available_crops(self_service):
        return sorted(self_service._fake_layer.frame["crop"].unique())

    async def get_distinct_field_values(self_service, field: str, **filters: Any):
        subset = apply_filters(self_service._fake_layer.frame, **filters)
        return sorted(subset[FIELD_COLUMNS[field]].dropna().unique().tolist())

    async def resolve_crop_code(self_service, crop_text: str):
        return {"maize": "MZ", "corn": "MZ", "mz": "MZ"}.get(crop_text.lower(), crop_text)

    async def calculate_aggregation(
        self_service,
        variable_code: str,
        aggregation: str,
        **filters: Any,
    ):
        subset = apply_filters(self_service._fake_layer.frame, **filters)[variable_code]
        functions = {"avg": subset.mean, "min": subset.min, "max": subset.max,
                     "sum": subset.sum, "count": subset.count}
        value = functions.get(aggregation.lower(), subset.mean)()
        return {
            "aggregation_type": aggregation,
            "metric": variable_code,
            "value": float(value) if len(subset) else None,
            "count": int(len(subset)),
            "stddev": float(subset.std()) if len(subset) > 1 else None,
            "min": float(subset.min()) if len(subset) else None,
            "max": float(subset.max()) if len(subset) else None,
            "unit": None,
        }

    async def calculate_breakdown(
        self_service,
        variable_code: str,
        aggregation: str,
        group_by: str,
        **filters: Any,
    ):
        column = "year" if group_by == "simulation_year" else group_by
        subset = apply_filters(self_service._fake_layer.frame, **filters)
        grouped = subset.groupby(column)[variable_code].agg([aggregation.lower().replace("avg", "mean"), "count"])
        grouped.columns = ["value", "count"]
        grouped = grouped.sort_values("value", ascending=False)
        return [
            {"group_value": index, "value": float(row.value), "count": int(row["count"])}
            for index, row in grouped.iterrows()
        ]

    async def get_extremum_simulation(self_service, *args: Any, **kwargs: Any):
        return None

    async def get_yearly_trend(
        self_service,
        variable_code: str,
        crop=None,
        cultivar=None,
        start_year=None,
        end_year=None,
        spatial=None,
        **filters,
    ):
        subset = apply_filters(self_service._fake_layer.frame, crop=crop, cultivar=cultivar, spatial=spatial, **filters)
        if start_year is not None:
            subset = subset[subset["year"] >= start_year]
        if end_year is not None:
            subset = subset[subset["year"] <= end_year]
        grouped = subset.groupby("year")[variable_code].agg(["mean", "count"])
        return [
            {"year": int(year), "avg_value": float(row["mean"]), "count": int(row["count"])}
            for year, row in grouped.iterrows()
        ]


class FakeSpatialLayer:
    """Replacement for SpatialService queries, over the same sample frame."""

    async def summarize_radius(self_service, spatial: Any):
        frame = StatisticsService._fake_layer.frame
        subset = apply_filters(frame, spatial=spatial)
        return {
            "simulations": int(len(subset)),
            "locations": int(len(subset[["LATITUDE", "LONGITUDE"]].drop_duplicates())),
        }

    async def nearest_location(self_service, latitude: float, longitude: float):
        frame = StatisticsService._fake_layer.frame
        if frame.empty:
            return None
        locations = frame[["LATITUDE", "LONGITUDE"]].drop_duplicates()
        distance = distances_km(locations, latitude, longitude)
        best = distance.idxmin()
        return {
            "latitude": float(locations.loc[best, "LATITUDE"]),
            "longitude": float(locations.loc[best, "LONGITUDE"]),
            "distance_km": float(distance[best]),
        }

    async def get_bounds(self_service, crop=None):
        frame = StatisticsService._fake_layer.frame
        return {
            "min_lat": float(frame["LATITUDE"].min()),
            "max_lat": float(frame["LATITUDE"].max()),
            "min_lon": float(frame["LONGITUDE"].min()),
            "max_lon": float(frame["LONGITUDE"].max()),
        }

    async def get_coverage(self_service, max_locations: int = 5000):
        frame = StatisticsService._fake_layer.frame
        grouped = frame.groupby(["LATITUDE", "LONGITUDE"]).size().reset_index(name="simulations")
        bounds = await FakeSpatialLayer.get_bounds(self_service)
        return {
            "simulations": int(len(frame)),
            "location_count": int(len(grouped)),
            "locations_truncated": False,
            "bounds": bounds,
            "locations": [
                {"latitude": float(r.LATITUDE), "longitude": float(r.LONGITUDE), "simulations": int(r.simulations)}
                for r in grouped.itertuples()
            ],
        }


async def fake_record_count(self, entity: str = "simulations", spatial: Any = None, **filters):
    frame = apply_filters(StatisticsService._fake_layer.frame, spatial=spatial, **filters)
    per_simulation = len(VARIABLES) if entity == "simulation_outputs" else 1
    return int(len(frame) * per_simulation)


@pytest.fixture
def fake_db(monkeypatch, sample_frame) -> FakeDataLayer:
    """Patch the DB access layer with the CSV-backed fake."""
    layer = FakeDataLayer(sample_frame)
    monkeypatch.setattr(StatisticsService, "_fake_layer", layer, raising=False)
    for name in (
        "count_simulations",
        "get_analysis_records",
        "get_available_variables",
        "get_available_crops",
        "get_distinct_field_values",
        "resolve_crop_code",
        "calculate_aggregation",
        "calculate_breakdown",
        "get_extremum_simulation",
        "get_yearly_trend",
    ):
        monkeypatch.setattr(StatisticsService, name, getattr(FakeDataLayer, name))

    for name in ("summarize_radius", "nearest_location", "get_bounds", "get_coverage"):
        monkeypatch.setattr(SpatialService, name, getattr(FakeSpatialLayer, name))

    async def no_simulations(self, *args, **kwargs):
        return []

    monkeypatch.setattr(MetadataService, "get_simulations", no_simulations)
    monkeypatch.setattr(MetadataService, "get_record_count", fake_record_count)
    monkeypatch.setattr(SpatialService, "search_by_radius", no_simulations)

    # Management scope from the CSV rows (one row per simulation).
    from app.services.management_service import MANAGEMENT_FIELDS, ManagementService, sort_values

    def management_values(frame):
        return {
            "simulations": int(len(frame)),
            "values": {
                field: sort_values(field, frame[FIELD_COLUMNS[field]].dropna().unique().tolist())
                for field, _, _ in MANAGEMENT_FIELDS
            },
        }

    async def matched_values(self, filters, spatial=None):
        return management_values(apply_filters(layer.frame, spatial=spatial, **filters))

    async def dataset_values(self):
        return management_values(layer.frame)

    monkeypatch.setattr(ManagementService, "matched_values", matched_values)
    monkeypatch.setattr(ManagementService, "dataset_values", dataset_values)
    return layer


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """Never call a real LLM from tests."""
    from app.agent import planner, response_generator

    monkeypatch.setattr(planner.settings, "OPENAI_API_KEY", None)
    monkeypatch.setattr(response_generator.settings, "OPENAI_API_KEY", None)


ORIGINAL_GEOCODER_REQUEST = GeocodingService._request


@pytest.fixture(autouse=True)
def offline_geocoder(monkeypatch):
    """Tests never reach the real geocoding service.

    Individual tests install canned responses with `geocoder_returns`.
    """
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "GEOCODER_MIN_INTERVAL_SECONDS", 0.0)
    GeocodingService.clear_cache()

    async def unreachable(self, query):
        raise httpx.ConnectError("network disabled in tests")

    monkeypatch.setattr(GeocodingService, "_request", unreachable)
    yield
    GeocodingService.clear_cache()


@pytest.fixture
def geocoder_returns(monkeypatch):
    """Install canned geocoder results: geocoder_returns({"kitale": [...]})."""
    requests: List[str] = []

    def install(responses: Dict[str, List[Dict[str, Any]]]):
        async def canned(self, query):
            requests.append(query)
            return responses.get(query.lower(), [])

        monkeypatch.setattr(GeocodingService, "_request", canned)
        return requests

    return install


@pytest.fixture(autouse=True)
def clear_conversation_context():
    from app.agent.conversation_context import analysis_context_store

    analysis_context_store.clear()
    yield
    analysis_context_store.clear()


class FakeLLM:
    """MOCKED LLM: stands in for AsyncOpenAI's Chat Completions.

    Returns the canned plan scripted for the question (as the gateway would,
    optionally wrapped in a ```json fence). A question with no scripted plan
    raises APIConnectionError-like failure, which exercises the fallback.
    """

    def __init__(self, plans, fenced=False):
        self.plans = plans
        self.fenced = fenced
        self.calls = []
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        self.calls.append(prompt)
        question = prompt.split("User Query:")[-1].strip().splitlines()[0]
        if question not in self.plans:
            raise ConnectionResetError("fake gateway reset (no scripted plan)")
        text = json.dumps(self.plans[question])
        if self.fenced:
            text = f"```json\n{text}\n```"
        message = SimpleNamespace(content=text)
        return SimpleNamespace(model="fake-llm", choices=[SimpleNamespace(message=message)])


# -----------------------------------------------------------------------------
# Test labels: mocked_llm / fallback (real_llm runs only via the opt-in script)
# -----------------------------------------------------------------------------

MOCKED_LLM_FIXTURES = {"llm_client", "use_llm"}
FALLBACK_MODULES = {"test_metric_fallback", "test_statistics_fallback"}


def pytest_collection_modifyitems(config, items):
    """Label planner tests by the path they exercise.

    Explicit markers win. Tests using a mocked-LLM fixture are mocked_llm;
    remaining tests in the fallback modules are fallback.
    """
    for item in items:
        if item.get_closest_marker("mocked_llm") or item.get_closest_marker("fallback"):
            continue
        try:
            source = inspect.getsource(item.function)
        except (OSError, TypeError):
            source = ""
        if MOCKED_LLM_FIXTURES & set(getattr(item, "fixturenames", ())) or "FakeLLM(" in source:
            item.add_marker(pytest.mark.mocked_llm)
        elif item.module.__name__.rsplit(".", 1)[-1] in FALLBACK_MODULES:
            item.add_marker(pytest.mark.fallback)
