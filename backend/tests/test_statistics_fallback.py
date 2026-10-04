"""No-LLM statistics fallback (MIN/MAX/AVG/SUM), and that the LLM stays the
primary planner and response generator whenever it works.

Expected values are recomputed independently from the sample CSVs.
"""
from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent import planner as planner_module
from app.agent import response_generator as response_module
from app.agent.analysis_parser import _find_planting_stage
from app.agent.statistics_fallback import detect_aggregations
from app.api.v1 import chat
from app.db.session import get_db
from app.services.statistics_service import StatisticsService
from conftest import FakeDataLayer, apply_filters, distances_km, haversine_km


# -----------------------------------------------------------------------------
# Wording -> aggregation and planting stage (pure functions)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question, expected",
    [
        ("What is the minimum rainfall?", ["MIN"]),
        ("lowest yield for BASE", ["MIN"]),
        ("smallest harvested weight", ["MIN"]),
        ("Which run had the least rainfall?", ["MIN"]),
        ("What is the maximum yield?", ["MAX"]),
        ("highest rainfall in 2015", ["MAX"]),
        ("largest biomass", ["MAX"]),
        ("peak yield for LNG", ["MAX"]),
        ("What is the average yield?", ["AVG"]),
        ("mean rainfall", ["AVG"]),
        ("typical grain number", ["AVG"]),
        ("total rainfall in 2020", ["SUM"]),
        ("sum of rainfall", ["SUM"]),
        ("cumulative precipitation", ["SUM"]),
        ("minimum and maximum yield", ["MIN", "MAX"]),
        # Words inside variable names are not aggregations:
        ("What is the average maximum temperature?", ["AVG"]),
        ("mean minimum temperature in 2010", ["AVG"]),
        # "at least" is not a minimum request:
        ("simulations with at least 500 mm rainfall", []),
        ("What does SHT stand for?", []),
    ],
)
def test_aggregation_wording(question, expected):
    assert detect_aggregations(question) == expected


@pytest.mark.parametrize(
    "wording, stage",
    [
        ("planted two weeks later", "pfrst15"),
        ("planted 14 days after the normal date", "pfrst15"),
        ("2 weeks late", "pfrst15"),
        ("a fortnight later", "pfrst15"),
        ("15 days after", "pfrst15"),
        ("two weeks earlier", "pfrst-15"),
        ("14 days before", "pfrst-15"),
        ("30 days late", "pfrst30"),
        ("a month later", "pfrst30"),
        ("45 days delayed", "pfrst45"),
        ("normal planting", "pfrst0"),
        ("planted on time", "pfrst0"),
    ],
)
def test_planting_wording_maps_to_supported_stage(wording, stage, sample_frame):
    assert _find_planting_stage(wording) == stage
    assert stage in set(sample_frame["planting_stage"])  # a stage the data has


def test_fourteen_days_is_never_pfrst14():
    for wording in ("two weeks later", "14 days after", "a fortnight after"):
        assert _find_planting_stage(wording) != "pfrst14"


# -----------------------------------------------------------------------------
# Chat path fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def stat_calls(fake_db, monkeypatch):
    """Record every aggregate/breakdown call that reaches StatisticsService."""
    calls = []

    async def aggregation(self, variable_code, aggregation, **filters):
        calls.append({"kind": "aggregate", "variable_code": variable_code, "aggregation": aggregation, **filters})
        return await FakeDataLayer.calculate_aggregation(self, variable_code, aggregation, **filters)

    async def breakdown(self, variable_code, aggregation, group_by, **filters):
        calls.append({"kind": "breakdown", "variable_code": variable_code, "aggregation": aggregation,
                      "group_by": group_by, **filters})
        return await FakeDataLayer.calculate_breakdown(self, variable_code, aggregation, group_by, **filters)

    monkeypatch.setattr(StatisticsService, "calculate_aggregation", aggregation)
    monkeypatch.setattr(StatisticsService, "calculate_breakdown", breakdown)
    return calls


@pytest.fixture
def client(stat_calls):
    app = FastAPI()
    app.include_router(chat.router, prefix="/api/v1/chat")

    async def fake_get_db():
        yield object()

    app.dependency_overrides[get_db] = fake_get_db
    return TestClient(app)


class FailingLLM:
    """An LLM client whose every call fails, counting the attempts."""

    def __init__(self):
        self.attempts = 0
        self.responses = SimpleNamespace(create=self._fail)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._fail))

    async def _fail(self, **kwargs):
        self.attempts += 1
        raise ConnectionError("gateway reset the connection")


@pytest.fixture(params=["no_llm", "failing_llm"])
def fallback_mode(request, monkeypatch):
    """Fallback runs when no LLM is configured and when the LLM fails."""
    state = {"mode": request.param, "planner_llm": None}
    if request.param == "failing_llm":
        planner_llm, responder_llm = FailingLLM(), FailingLLM()
        state["planner_llm"] = planner_llm
        monkeypatch.setattr(planner_module.settings, "OPENAI_API_KEY", "test")
        monkeypatch.setattr(planner_module, "AsyncOpenAI", lambda **kwargs: planner_llm)
        monkeypatch.setattr(response_module, "AsyncOpenAI", lambda **kwargs: responder_llm)
    yield state
    # Statistical questions must try the LLM before falling back. (Known-code
    # definitions are answered by the existing shortcut without the LLM.)
    llm = state["planner_llm"]
    if llm is not None and not state.get("no_llm_expected"):
        assert llm.attempts > 0, "the LLM must be tried before falling back"


@pytest.fixture(scope="module")
def kitale_point(sample_frame):
    locations = sample_frame[["LATITUDE", "LONGITUDE"]].drop_duplicates()
    lat = locations["LATITUDE"].to_numpy(float)
    lon = locations["LONGITUDE"].to_numpy(float)
    best = int(np.argmin([haversine_km(lat, lon, a, b).sum() for a, b in zip(lat, lon)]))
    return float(lat[best]), float(lon[best])


def ask(client, message, **extra):
    response = client.post("/api/v1/chat/", json={"message": message, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def aggregate_calls(calls):
    return [c for c in calls if c["kind"] == "aggregate"]


# -----------------------------------------------------------------------------
# Fallback path: metric + aggregation + filters + radius
# -----------------------------------------------------------------------------

def test_minimum_rainfall_within_25_km_of_kitale(
    client, fallback_mode, stat_calls, sample_frame, kitale_point, geocoder_returns
):
    lat, lon = kitale_point
    geocoder_returns({"kitale": [{"display_name": "Kitale, Trans-Nzoia County, Kenya",
                                  "lat": str(lat), "lon": str(lon), "importance": 0.6}]})
    body = ask(client, "What is the minimum rainfall within 25 km of Kitale?")

    call = aggregate_calls(stat_calls)[-1]
    assert (call["variable_code"], call["aggregation"]) == ("PRCP", "MIN")
    assert (call["spatial"].latitude, call["spatial"].longitude, call["spatial"].radius_km) == (lat, lon, 25.0)

    inside = sample_frame[distances_km(sample_frame, lat, lon) <= 25.0]
    assert body["statistics"]["metric"] == "PRCP"
    assert body["statistics"]["aggregation_type"] == "MIN"
    assert body["statistics"]["value"] == pytest.approx(inside["PRCP"].min())
    assert body["statistics"]["count"] == len(inside)
    assert "within 25 km of Kitale" in body["answer"]


@pytest.mark.parametrize(
    "question, metric, aggregation, filters",
    [
        ("What was the lowest yield for BASE in 2010?", "HWAM", "MIN",
         {"cultivar": "BASE", "year": 2010}),
        ("What was the highest rainfall in 2015?", "PRCP", "MAX", {"year": 2015}),
        ("mean yield for vsht planted two weeks later", "HWAM", "AVG",
         {"cultivar": "VSHT", "planting_stage": "pfrst15"}),
        ("What is the total rainfall in 2020?", "PRCP", "SUM", {"year": 2020}),
        ("peak biomass for LNG from 1990 to 2000", "CWAM", "MAX",
         {"cultivar": "LNG", "year": list(range(1990, 2001))}),
        ("maximum yield for sht planted two weeks earlier", "HWAM", "MAX",
         {"cultivar": "SHT", "planting_stage": "pfrst-15"}),
        ("average rainfall for lng and vlng in rainfed runs", "PRCP", "AVG",
         {"cultivar": ["LNG", "VLNG"], "irrigation": "RF"}),
        ("What is the average maximum temperature in 2000?", "TMAXA", "AVG", {"year": 2000}),
    ],
)
def test_varied_phrasings(client, fallback_mode, stat_calls, sample_frame,
                          question, metric, aggregation, filters):
    body = ask(client, question)

    call = aggregate_calls(stat_calls)[-1]
    assert (call["variable_code"], call["aggregation"]) == (metric, aggregation)
    for field, value in filters.items():
        assert call[field] == value, field

    subset = apply_filters(sample_frame, **filters)[metric]
    expected = {"MIN": subset.min(), "MAX": subset.max(), "AVG": subset.mean(), "SUM": subset.sum()}[aggregation]
    assert body["statistics"]["metric"] == metric
    assert body["statistics"]["aggregation_type"] == aggregation
    assert body["statistics"]["value"] == pytest.approx(expected)
    assert body["statistics"]["count"] == len(subset)


def test_equivalent_planting_wordings_give_identical_statistics(client, fallback_mode, stat_calls):
    wordings = [
        "average yield for BASE planted two weeks later",
        "average yield for BASE planted 14 days after normal",
        "average yield for BASE planted a fortnight later",
        "average yield for BASE planted 15 days after the normal date",
    ]
    results = [ask(client, wording)["statistics"] for wording in wordings]
    stages = [call["planting_stage"] for call in aggregate_calls(stat_calls)]
    assert stages == ["pfrst15"] * 4
    assert all(result == results[0] for result in results)


def test_unsupported_planting_offset_asks_instead_of_inventing(client, fallback_mode, stat_calls, sample_frame):
    body = ask(client, "average yield for BASE planted one week later")
    assert "no planting stage 'pfrst7'" in body["answer"]
    for stage in sorted(set(sample_frame["planting_stage"])):
        assert stage in body["answer"]
    assert stat_calls == []


def test_minimum_and_maximum_together(client, fallback_mode, stat_calls, sample_frame):
    body = ask(client, "What are the minimum and maximum yield in 2005?")
    calls = aggregate_calls(stat_calls)
    assert sorted(c["aggregation"] for c in calls[-2:]) == ["MAX", "MIN"]
    values = {s["aggregation_type"]: s["value"] for s in body["additional_statistics"]}
    year = sample_frame[sample_frame["year"] == 2005]["HWAM"]
    assert values == {"MIN": pytest.approx(year.min()), "MAX": pytest.approx(year.max())}


def test_which_cultivar_has_highest_average_yield(client, fallback_mode, stat_calls, sample_frame):
    body = ask(client, "Which cultivar has the highest average yield?")
    breakdown = [c for c in stat_calls if c["kind"] == "breakdown"][-1]
    assert (breakdown["variable_code"], breakdown["aggregation"], breakdown["group_by"]) == ("HWAM", "AVG", "cultivar")
    means = sample_frame.groupby("cultivar")["HWAM"].mean()
    assert body["statistics"]["breakdown"]["values"][0]["group_value"] == means.idxmax()
    if fallback_mode["mode"] in ("no_llm", "failing_llm"):
        assert f"highest average is {means.idxmax()}" in body["answer"]
        assert f"lowest is {means.idxmin()}" in body["answer"]


def test_ambiguous_metric_still_asks(client, fallback_mode, stat_calls):
    body = ask(client, "What is the maximum temperature variable's lowest value?")
    # "maximum temperature" names TMAXA; "lowest" is the aggregation.
    call = aggregate_calls(stat_calls)[-1]
    assert (call["variable_code"], call["aggregation"]) == ("TMAXA", "MIN")
    body = ask(client, "What is the lowest temperature?")
    assert "could refer to more than one variable" in body["answer"]


def test_statistic_without_a_variable_asks_which(client, fallback_mode, stat_calls):
    body = ask(client, "What is the minimum for BASE in 2010?")
    assert "Which variable should the minimum be calculated for?" in body["answer"]
    assert stat_calls == []


def test_definition_is_not_read_as_an_average(client, fallback_mode, stat_calls):
    fallback_mode["no_llm_expected"] = True  # existing definition shortcut
    body = ask(client, "What does SHT mean?")
    assert body["semantic_plan"]["operations"][0]["operation"] == "definition"
    assert stat_calls == []


# -----------------------------------------------------------------------------
# The LLM stays primary when it works
# -----------------------------------------------------------------------------

def _chat_reply(text):
    message = SimpleNamespace(content=text)
    return SimpleNamespace(model="fake-llm", choices=[SimpleNamespace(message=message)])


class WorkingPlannerLLM:
    """MOCKED LLM planner (Chat Completions)."""

    def __init__(self, plan):
        self.plan = plan
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls += 1
        return _chat_reply(json.dumps(self.plan))


class WorkingResponderLLM:
    """MOCKED LLM response writer (Chat Completions)."""

    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls += 1
        return _chat_reply("LLM-written answer.")


@pytest.mark.mocked_llm
def test_llm_plans_and_answers_when_available(client, stat_calls, monkeypatch):
    # The LLM plan groups by cultivar, which the fallback would not do, so the
    # executed statistics prove which planner was used.
    llm_plan = {
        "goal": "g", "intent": "aggregate",
        "operations": [{"operation": "aggregate", "entity": "simulation_outputs", "metric": "PRCP",
                        "aggregation": "MIN", "filters": [{"field": "year", "operator": "=", "value": 2020}],
                        "group_by": ["cultivar"], "independent": True}],
    }
    planner_llm, responder_llm = WorkingPlannerLLM(llm_plan), WorkingResponderLLM()
    monkeypatch.setattr(planner_module.settings, "OPENAI_API_KEY", "test")
    monkeypatch.setattr(planner_module, "AsyncOpenAI", lambda **kwargs: planner_llm)
    monkeypatch.setattr(response_module, "AsyncOpenAI", lambda **kwargs: responder_llm)

    async def must_not_run(*args, **kwargs):
        raise AssertionError("deterministic fallback used while the LLM works")

    monkeypatch.setattr(planner_module, "build_fallback_statistics_plan", must_not_run)

    body = ask(client, "What is the minimum rainfall in 2020?")
    assert planner_llm.calls == 1
    assert responder_llm.calls == 1
    assert body["answer"] == "LLM-written answer."
    assert body["planner"]["planner"] == "llm" and body["planner"]["answer_by"] == "llm"
    assert any(c["kind"] == "breakdown" and c["group_by"] == "cultivar" for c in stat_calls)
    assert aggregate_calls(stat_calls)[-1]["aggregation"] == "MIN"


def test_fallback_answers_when_llm_fails(client, stat_calls, sample_frame, monkeypatch):
    planner_llm, responder_llm = FailingLLM(), FailingLLM()
    monkeypatch.setattr(planner_module.settings, "OPENAI_API_KEY", "test")
    monkeypatch.setattr(planner_module, "AsyncOpenAI", lambda **kwargs: planner_llm)
    monkeypatch.setattr(response_module, "AsyncOpenAI", lambda **kwargs: responder_llm)

    body = ask(client, "What is the minimum rainfall in 2020?")
    assert planner_llm.attempts >= 1 and responder_llm.attempts >= 1
    call = aggregate_calls(stat_calls)[-1]
    assert (call["variable_code"], call["aggregation"], call["year"]) == ("PRCP", "MIN", 2020)
    expected = sample_frame.loc[sample_frame["year"] == 2020, "PRCP"].min()
    assert body["statistics"]["value"] == pytest.approx(expected)  # full precision kept
    shown = int(Decimal(str(expected)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    assert f"Min PRCP: {shown:,}." in body["answer"]  # displayed as a whole number
    assert body["confidence"] == "low"
    assert body["planner"]["planner"] == "fallback"
    assert "gateway reset" in body["planner"]["fallback_reason"]
    assert body["planner"]["answer_by"] == "fallback_template"
