"""Whole-number display, available management variables, and the management
conditions stated with each result.

Fallback-path tests use the CSV-backed fake database (45 sample files); the
LLM-path test uses a mocked LLM (no real model is called).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent import planner as planner_module
from app.agent import response_generator as response_module
from app.api.v1 import chat, dataset
from app.db.session import get_db
from app.services.display_format import format_stat
from app.services.management_service import ManagementService, sort_values, value_label
from conftest import FakeLLM, apply_filters, distances_km

MANAGEMENT = ("cultivar", "planting_stage", "irrigation", "nitrogen_level")


@pytest.fixture
def client(fake_db):
    app = FastAPI()
    app.include_router(chat.router, prefix="/api/v1/chat")
    app.include_router(dataset.router, prefix="/api/v1/dataset")

    async def session():
        yield object()

    app.dependency_overrides[get_db] = session
    return TestClient(app)


def ask(client, message, **extra):
    response = client.post("/api/v1/chat/", json={"message": message, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def distinct(frame, field):
    return sort_values(field, frame[field].dropna().unique().tolist())


# -----------------------------------------------------------------------------
# 1. Whole-number display
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("value,shown", [
    (4502.36, "4,502"), (4002.3639, "4,002"), (4502.5, "4,503"), (0.4, "0"),
    (-2.5, "-3"), (153180, "153,180"), (None, "n/a"), (float("nan"), "n/a"),
])
def test_format_stat_whole_numbers(value, shown):
    assert format_stat(value) == shown


@pytest.mark.fallback
def test_average_answer_is_whole_number_but_value_keeps_precision(client, sample_frame):
    body = ask(client, "What is the average yield in 2020?")
    expected = sample_frame.loc[sample_frame["year"] == 2020, "HWAM"].mean()
    assert body["statistics"]["value"] == pytest.approx(expected)  # full precision in the data
    assert f"Avg HWAM: {format_stat(expected)}." in body["answer"]
    assert f"{expected:.2f}" not in body["answer"]


# -----------------------------------------------------------------------------
# 2. Available management variables
# -----------------------------------------------------------------------------

class RowsSession:
    """Returns canned rows for the catalogue queries, in call order."""

    def __init__(self, results):
        self.results = list(results)

    async def execute(self, stmt):
        rows = self.results.pop(0)
        return SimpleNamespace(all=lambda: rows, one=lambda: rows)


async def test_available_lists_only_values_present():
    session = RowsSession([
        [("SHT", 2), ("BASE", 3), ("", 9), (None, 1)],          # cultivar (blank/NULL dropped)
        [("pfrst15", 1), ("pfrst-30", 1), ("pfrst0", 1)],       # planting_stage
        [("RF", 5)],                                            # irrigation
        [],                                                     # nitrogen_level: none stored
        (5, 1990, 1995, ["MZ"]),                                # scope
    ])
    result = await ManagementService(session).available()
    fields = {f["field"]: f for f in result["fields"]}
    assert list(fields) == ["cultivar", "planting_stage", "irrigation"]  # empty field omitted
    assert [v["value"] for v in fields["cultivar"]["values"]] == ["BASE", "SHT"]
    assert [v["value"] for v in fields["planting_stage"]["values"]] == ["pfrst-30", "pfrst0", "pfrst15"]
    assert fields["irrigation"]["values"] == [{"value": "RF", "label": "rainfed", "simulations": 5}]
    assert result["years"] == {"min": 1990, "max": 1995} and result["crops"] == ["MZ"]


@pytest.mark.parametrize("field,value,label", [
    ("cultivar", "LNG", "Long-season cultivar"),
    ("planting_stage", "pfrst-15", "15 days before the normal planting date"),
    ("planting_stage", "pfrst0", "normal planting date"),
    ("irrigation", "RF", "rainfed"),
    ("nitrogen_level", "HighN", "high nitrogen"),
    ("cultivar", "UNKNOWN", "UNKNOWN"),  # never invents a meaning
])
def test_value_labels(field, value, label):
    assert value_label(field, value) == label


# -----------------------------------------------------------------------------
# 3. Management conditions stated with each result
# -----------------------------------------------------------------------------

@pytest.mark.fallback
def test_broad_question_states_all_conditions_it_combines(client, sample_frame):
    body = ask(client, "What is the average yield in 2020?")
    subset = sample_frame[sample_frame["year"] == 2020]
    scope = body["management"]
    assert scope["simulations"] == len(subset)
    included = {item["field"]: item["values"] for item in scope["included"]}
    assert included == {field: distinct(subset, field) for field in MANAGEMENT}
    assert not any(item["filtered"] for item in scope["included"])
    assert [r["field"] for r in scope["requested"]] == ["year"]
    sentence = scope["sentence"]
    cultivars = distinct(subset, "cultivar")
    assert f"all {len(cultivars)} cultivars ({', '.join(cultivars)})" in sentence
    assert f"all {len(distinct(subset, 'planting_stage'))} planting dates" in sentence
    assert "(year 2020)" in sentence and "not specific to one cultivar or planting date" in sentence
    assert "Filtered to" not in sentence
    assert sentence in body["answer"]


@pytest.mark.fallback
def test_cultivar_filter_is_applied_and_stated(client, fake_db, sample_frame):
    body = ask(client, "What is the average yield for BASE?")
    subset = sample_frame[sample_frame["cultivar"] == "BASE"]
    assert body["statistics"]["count"] == len(subset)
    scope = body["management"]
    included = {item["field"]: item for item in scope["included"]}
    assert included["cultivar"]["values"] == ["BASE"] and included["cultivar"]["filtered"]
    assert included["planting_stage"]["values"] == distinct(subset, "planting_stage")
    assert scope["sentence"].startswith("Management conditions: Filtered to cultivar BASE (baseline cultivar).")
    assert "not specific to one planting date" in scope["sentence"]
    assert "one cultivar" not in scope["sentence"]


@pytest.mark.fallback
def test_cultivar_and_planting_filters(client, sample_frame):
    body = ask(client, "What is the average yield for LNG planted 15 days late?")
    subset = sample_frame[(sample_frame["cultivar"] == "LNG") & (sample_frame["planting_stage"] == "pfrst15")]
    assert body["statistics"]["count"] == len(subset)
    sentence = body["management"]["sentence"]
    assert "cultivar LNG (long-season cultivar)" in sentence
    assert "planting date pfrst15 (15 days after the normal planting date)" in sentence
    assert "not specific" not in sentence


@pytest.mark.fallback
def test_scope_uses_the_same_area_as_the_result(client, sample_frame):
    lat, lon = sample_frame[["LATITUDE", "LONGITUDE"]].iloc[0]
    body = ask(client, "What is the average yield for BASE here?", latitude=float(lat), longitude=float(lon), radius_km=25)
    area = sample_frame[(distances_km(sample_frame, lat, lon) <= 25) & (sample_frame["cultivar"] == "BASE")]
    assert body["statistics"]["count"] == len(area)
    assert body["management"]["simulations"] == len(area)


@pytest.mark.fallback
def test_analysis_answer_states_conditions(client, sample_frame):
    body = ask(client, "What is the correlation between rainfall and yield for VSHT?")
    assert body["analysis"]["status"] == "ok"
    sentence = body["management"]["sentence"]
    assert sentence.startswith("Management conditions: Filtered to cultivar VSHT")
    assert sentence in body["answer"]


@pytest.mark.fallback
def test_definition_has_no_management_scope(client):
    body = ask(client, "What does HWAM mean?")
    assert body["management"] is None


class CapturingResponder:
    """MOCKED LLM response writer that records the prompt it receives."""

    def __init__(self):
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.prompts.append(kwargs["messages"][-1]["content"])
        message = SimpleNamespace(content="LLM answer.")
        return SimpleNamespace(model="fake", choices=[SimpleNamespace(message=message)])


@pytest.mark.mocked_llm
def test_llm_prompt_receives_conditions_and_rounding_rule(client, monkeypatch, sample_frame):
    question = "What is the average yield in 2020?"
    plan = {"goal": "g", "intent": "aggregate", "operations": [{
        "operation": "aggregate", "entity": "simulation_outputs", "metric": "yield", "aggregation": "AVG",
        "filters": [{"field": "year", "operator": "=", "value": 2020}], "group_by": [], "independent": True}]}
    planner_llm, responder = FakeLLM({question: plan}), CapturingResponder()
    monkeypatch.setattr(planner_module.settings, "OPENAI_API_KEY", "test")
    monkeypatch.setattr(planner_module, "AsyncOpenAI", lambda **kwargs: planner_llm)
    monkeypatch.setattr(response_module, "AsyncOpenAI", lambda **kwargs: responder)

    body = ask(client, question)
    assert body["planner"]["planner"] == "llm" and body["planner"]["answer_by"] == "llm"
    prompt = responder.prompts[-1]
    assert f"Management Conditions:\n{body['management']['sentence']}" in prompt
    assert "Round measured quantities to whole numbers" in prompt
    assert "never describe it as the result for one" in prompt


# -----------------------------------------------------------------------------
# 4. Requested conditions the pipeline cannot apply are reported, not implied
# -----------------------------------------------------------------------------

from app.agent.unsupported_conditions import find_unsupported_conditions  # noqa: E402


@pytest.mark.parametrize("question,key", [
    ("What is the average yield on sandy soil in 2020?", "soil"),
    ("Average yield at 120 kg N/ha", "nitrogen_rate"),
    ("average yield for BASE planted in March", "planting_calendar"),
    ("yield with no-till", "tillage"),
    ("average yield at a planting density of 5 plants per m2", "plant_density"),
    ("average yield in the highland agro-ecological zone", "ecological_zone"),
])
def test_unsupported_conditions_detected(question, key):
    assert [item["key"] for item in find_unsupported_conditions(question)] == [key]


@pytest.mark.parametrize("question", [
    "What is the average yield in 2020?",
    "average yield for BASE planted 15 days late",
    "yield for standard planting",
    "Does planting may matter?",
    "average yield for HighN rainfed simulations",
    "Is there a linear relationship between rainfall and yield?",
])
def test_supported_wording_is_not_flagged(question):
    assert find_unsupported_conditions(question) == []


@pytest.mark.fallback
def test_fallback_reports_condition_it_could_not_apply(client, sample_frame):
    body = ask(client, "What is the average yield on sandy soil in 2020?")
    subset = sample_frame[sample_frame["year"] == 2020]
    assert body["statistics"]["count"] == len(subset)  # soil could not be filtered
    scope = body["management"]
    assert [u["key"] for u in scope["unapplied"]] == ["soil"]
    assert 'Not applied: soil type ("sandy soil": the dataset has no soil information)' in scope["sentence"]
    assert "covers all matching records regardless of this condition" in scope["sentence"]
    assert scope["sentence"] in body["answer"]


@pytest.mark.mocked_llm
def test_llm_dropped_filter_is_reported_once(client, monkeypatch, sample_frame):
    question = "What is the average yield on sandy soil in 2020?"
    plan = {"goal": "g", "intent": "aggregate", "operations": [{
        "operation": "aggregate", "entity": "simulation_outputs", "metric": "HWAM", "aggregation": "AVG",
        "filters": [{"field": "year", "operator": "=", "value": 2020},
                    {"field": "soil_type", "operator": "=", "value": "sandy"}],
        "group_by": [], "independent": True}]}
    monkeypatch.setattr(planner_module.settings, "OPENAI_API_KEY", "test")
    monkeypatch.setattr(planner_module, "AsyncOpenAI", lambda **kwargs: FakeLLM({question: plan}))
    monkeypatch.setattr(response_module, "AsyncOpenAI", lambda **kwargs: None)
    body = ask(client, question)
    assert body["planner"]["planner"] == "llm"
    assert [u["key"] for u in body["planner"]["unapplied_filters"]] == ["soil"]
    assert [u["key"] for u in body["management"]["unapplied"]] == ["soil"]  # text + dropped filter merged
    assert body["statistics"]["count"] == int((sample_frame["year"] == 2020).sum())


@pytest.mark.mocked_llm
def test_llm_dropped_unknown_field_is_reported(client, monkeypatch):
    question = "What is the average yield for large harvest areas?"
    plan = {"goal": "g", "intent": "aggregate", "operations": [{
        "operation": "aggregate", "entity": "simulation_outputs", "metric": "HWAM", "aggregation": "AVG",
        "filters": [{"field": "harvest_area", "operator": "=", "value": "large"}],
        "group_by": [], "independent": True}]}
    monkeypatch.setattr(planner_module.settings, "OPENAI_API_KEY", "test")
    monkeypatch.setattr(planner_module, "AsyncOpenAI", lambda **kwargs: FakeLLM({question: plan}))
    monkeypatch.setattr(response_module, "AsyncOpenAI", lambda **kwargs: None)
    body = ask(client, question)
    unapplied = body["management"]["unapplied"]
    assert [u["condition"] for u in unapplied] == ["harvest_area"]
    assert "it is not a field the query pipeline can filter on" in body["management"]["sentence"]


@pytest.mark.fallback
def test_supported_question_has_nothing_unapplied(client):
    body = ask(client, "What is the average yield for BASE?")
    assert "unapplied" not in body["management"]
    assert "Not applied" not in body["answer"]
