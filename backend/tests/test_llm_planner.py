"""LLM-first planning: the LLM plans, deterministic code validates/corrects.

Test labels
  MOCKED LLM  the planner talks to FakeLLM (a scripted Chat Completions
              stand-in); proves the pipeline wiring, validation and
              normalization, NOT the behavior of a real model.
  FALLBACK    no LLM, or a failing LLM; proves the resilience path.
Real-LLM verification is done separately against the gateway (see the
live end-to-end script), never by these tests.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent import planner as planner_module
from app.agent import response_generator as response_module
from app.agent.models import AnalysisRequest, FilterCondition
from app.agent.planner import QueryPlanner
from app.api.v1 import chat
from app.db.session import get_db
from conftest import FakeLLM, apply_filters, distances_km


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

@pytest.fixture
def client(fake_db):
    app = FastAPI()
    app.include_router(chat.router, prefix="/api/v1/chat")

    async def fake_get_db():
        yield object()

    app.dependency_overrides[get_db] = fake_get_db
    return TestClient(app)


@pytest.fixture
def use_llm(monkeypatch):
    """Install a MOCKED LLM planner; the answer text stays deterministic."""

    def install(plans, fenced=True):
        fake = FakeLLM(plans, fenced=fenced)
        monkeypatch.setattr(planner_module.settings, "OPENAI_API_KEY", "test")
        monkeypatch.setattr(planner_module, "AsyncOpenAI", lambda **kwargs: fake)
        monkeypatch.setattr(response_module, "AsyncOpenAI", lambda **kwargs: None)
        return fake

    return install


def ask(client, message, **extra):
    response = client.post("/api/v1/chat/", json={"message": message, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def aggregate(metric, aggregation="AVG", filters=None, group_by=None):
    return {"goal": "g", "intent": "aggregate", "operations": [{
        "operation": "aggregate", "entity": "simulation_outputs", "metric": metric,
        "aggregation": aggregation, "filters": filters or [], "group_by": group_by or [],
        "independent": True}]}


def analysis(operation, x, y, filters=None, group_by=None):
    return {"goal": "g", "intent": "analysis", "operations": [{
        "operation": "analysis", "analysis": {
            "operation": operation, "x_variable": x, "y_variable": y,
            "filters": filters or [], "group_by": group_by}}]}


def within(frame, lat, lon, km):
    return frame[distances_km(frame, lat, lon) <= km]


@pytest.fixture(scope="module")
def center(sample_frame):
    locations = sample_frame[["LATITUDE", "LONGITUDE"]].drop_duplicates()
    lat, lon = locations["LATITUDE"].to_numpy(float), locations["LONGITUDE"].to_numpy(float)
    from conftest import haversine_km
    best = int(np.argmin([haversine_km(lat, lon, a, b).sum() for a, b in zip(lat, lon)]))
    return round(float(lat[best]), 4), round(float(lon[best]), 4)


def nominatim(name, lat, lon):
    return {"display_name": name, "lat": str(lat), "lon": str(lon), "importance": 0.6, "type": "town"}


# -----------------------------------------------------------------------------
# MOCKED LLM: the required end-to-end questions
# -----------------------------------------------------------------------------

@pytest.mark.mocked_llm
def test_mocked_llm_average_yield_with_map_point(client, use_llm, sample_frame, center):
    """MOCKED LLM: 'yield' alias -> HWAM; the map radius reaches the query."""
    lat, lon = center
    fake = use_llm({"What is the average yield?": aggregate("yield")})
    body = ask(client, "What is the average yield here?", latitude=lat, longitude=lon, radius_km=25)

    assert len(fake.calls) == 1 and "here" not in fake.calls[0].split("User Query:")[-1]
    assert body["planner"]["planner"] == "llm"
    assert "aggregate: metric 'yield' -> 'HWAM'" in body["planner"]["corrections"]
    assert body["semantic_plan"]["operations"][0]["metric"] == "HWAM"
    assert body["semantic_plan"]["spatial"]["radius_km"] == 25
    assert body["semantic_plan"]["spatial"]["source"] == "map"
    expected = within(sample_frame, lat, lon, 25)["HWAM"]
    assert body["statistics"]["metric"] == "HWAM"
    assert body["statistics"]["value"] == pytest.approx(expected.mean())
    assert body["statistics"]["count"] == len(expected)


@pytest.mark.mocked_llm
def test_mocked_llm_coordinate_average_yield(client, use_llm, sample_frame, center):
    """MOCKED LLM: coordinates are parsed before planning, never geocoded."""
    lat, lon = center
    fake = use_llm({"What is the average yield?": aggregate("HWAM")})
    body = ask(client, f"What is the average yield within 10 km of these coordinates: {lat}, {lon}?")

    assert str(lat) not in fake.calls[0].split("User Query:")[-1]
    assert body["planner"]["planner"] == "llm"
    assert body["planner"]["corrections"] == []  # the LLM plan needed no change
    spatial = body["semantic_plan"]["spatial"]
    assert (spatial["latitude"], spatial["longitude"], spatial["radius_km"], spatial["source"]) == (lat, lon, 10, "coordinates")
    expected = within(sample_frame, lat, lon, 10)["HWAM"]
    assert body["statistics"]["value"] == pytest.approx(expected.mean())
    assert body["statistics"]["count"] == len(expected)


@pytest.mark.mocked_llm
def test_mocked_llm_base_correlation_near_place(client, use_llm, fake_db, sample_frame, center, geocoder_returns):
    """MOCKED LLM: aliases in the LLM's analysis plan are canonicalized and
    the geocoded area and cultivar filter reach the analysis query."""
    lat, lon = center
    geocoder_returns({"kitale": [nominatim("Kitale, Trans-Nzoia County, Kenya", lat, lon)]})
    fake = use_llm({
        "What is the correlation between rainfall and yield for BASE?": analysis(
            "correlation", "rainfall", "Yield", [{"field": "cultivar", "operator": "=", "value": "baseline"}]),
    })
    body = ask(client, "What is the correlation between rainfall and yield for BASE near Kitale?")

    assert len(fake.calls) == 1
    assert body["planner"]["planner"] == "llm"
    request = body["semantic_plan"]["operations"][0]["analysis"]
    assert (request["operation"], request["x_variable"], request["y_variable"]) == ("correlation", "PRCP", "HWAM")
    assert request["filters"] == [{"field": "cultivar", "operator": "=", "value": "BASE"}]
    assert request["spatial"]["label"] == "Kitale, Trans-Nzoia County, Kenya"

    call = fake_db.calls[-1]
    assert call["filters"] == {"cultivar": "BASE"} and call["spatial"].radius_km == 25
    subset = within(apply_filters(sample_frame, cultivar="BASE"), lat, lon, 25)
    assert body["analysis"]["sample_size"] == len(subset)
    assert body["analysis"]["correlation"] == pytest.approx(np.corrcoef(subset["PRCP"], subset["HWAM"])[0, 1])


@pytest.mark.mocked_llm
@pytest.mark.parametrize("question,llm_operation,expected_operation,corrected", [
    ("Is there a linear relationship between rainfall and yield?", "linear_regression", "linear_regression", False),
    ("Fit a quadratic regression of yield on precipitation", "quadratic", "quadratic_regression", False),
    # The LLM picked the wrong method; the named method in the question wins.
    ("Is there a quadratic relationship between precipitation and yield?", "linear_regression", "quadratic_regression", True),
])
def test_mocked_llm_regression_plans(client, use_llm, sample_frame, question, llm_operation, expected_operation, corrected):
    """MOCKED LLM: linear and quadratic regression plans are validated."""
    use_llm({question: analysis(llm_operation, "PRCP", "HWAM")})
    body = ask(client, question)
    assert body["planner"]["planner"] == "llm"
    assert body["analysis"]["analysis_type"] == expected_operation
    assert body["analysis"]["sample_size"] == len(sample_frame)
    corrections = body["planner"]["corrections"]
    assert any("analysis: operation" in c for c in corrections) == corrected
    degree = 2 if expected_operation == "quadratic_regression" else 1
    coefficients = np.polyfit(sample_frame["PRCP"], sample_frame["HWAM"], degree)
    key = "quadratic" if degree == 2 else "slope"
    assert body["analysis"]["coefficients"][key] == pytest.approx(coefficients[0], rel=1e-6)


@pytest.mark.mocked_llm
def test_mocked_llm_aggregate_for_analysis_question_is_corrected(client, use_llm, sample_frame):
    """MOCKED LLM: an LLM plan that ignores the stated analysis is corrected,
    and the correction is recorded (the plan is still LLM-sourced)."""
    question = "What is the correlation between rainfall and yield?"
    use_llm({question: aggregate("HWAM")})
    body = ask(client, question)
    assert body["planner"]["planner"] == "llm"
    assert any("replaced with the correlation request" in c for c in body["planner"]["corrections"])
    assert body["analysis"]["analysis_type"] == "correlation"
    assert body["analysis"]["sample_size"] == len(sample_frame)


# -----------------------------------------------------------------------------
# MOCKED LLM: normalization and filter preservation
# -----------------------------------------------------------------------------

async def plan_with(plans, question, previous=None):
    planner = QueryPlanner(api_key=None, db_session=None)
    planner.api_key = "test"
    planner.client = FakeLLM(plans, fenced=True)
    qp = await planner.plan_with_fallback(question, analysis_context=previous)
    return planner, qp


@pytest.mark.mocked_llm
@pytest.mark.parametrize("llm_metric,question,expected", [
    ("yield", "What is the average yield?", "HWAM"),
    ("Yield", "What is the average yield in 2015?", "HWAM"),
    ("rainfall", "What is the average rainfall?", "PRCP"),
    ("precipitation", "What is the total precipitation?", "PRCP"),
    ("TMAXA", "What is the average TMAXA?", "TMAXA"),  # explicit code preserved
    (None, "What is the average HWAM for BASE?", "HWAM"),
])
async def test_mocked_llm_metric_aliases_are_canonical(llm_metric, question, expected):
    """MOCKED LLM: metric aliases become DSSAT codes; explicit codes stay."""
    planner, qp = await plan_with({question: aggregate(llm_metric)}, question)
    assert planner.get_plan_source()["planner"] == "llm"
    assert planner.get_semantic_plan().operations[0].metric == expected
    assert qp.metric == expected


@pytest.mark.mocked_llm
async def test_mocked_llm_analysis_aliases_are_canonical():
    """MOCKED LLM: analysis variable/method aliases become canonical values."""
    question = "How strongly does rain track with maize output across runs?"
    planner, _ = await plan_with({question: analysis("regression", "rainfall", "yield")}, question)
    request = planner.get_semantic_plan().operations[0].analysis
    assert (request.operation, request.x_variable, request.y_variable) == ("linear_regression", "PRCP", "HWAM")


@pytest.mark.mocked_llm
async def test_mocked_llm_filters_stated_in_question_are_preserved():
    """MOCKED LLM: cultivar/year/planting stated in the question are kept even
    if the LLM drops or changes them; unsupported fields are removed."""
    question = "What is the average yield for LNG planted two weeks later between 2010 and 2012?"
    plan = aggregate("HWAM", filters=[
        {"field": "cultivar", "operator": "=", "value": "BASE"},           # wrong cultivar
        {"field": "ecological_zone", "operator": "=", "value": "highland"},  # unsupported
    ])
    planner, _ = await plan_with({question: plan}, question)
    filters = {f.field: (f.operator, f.value) for f in planner.get_semantic_plan().operations[0].filters}
    assert filters == {
        "cultivar": ("=", "LNG"),
        "planting_stage": ("=", "pfrst15"),
        "year": ("BETWEEN", [2010, 2012]),
    }
    corrections = planner.get_plan_source()["corrections"]
    assert any("dropped unsupported filter ecological_zone" in c for c in corrections)
    assert any("replaced cultivar" in c for c in corrections)


@pytest.mark.mocked_llm
async def test_mocked_llm_equivalent_year_list_is_not_rewritten():
    """MOCKED LLM: an IN list equal to the stated range is accepted as is."""
    question = "What is the average yield from 2010 to 2012?"
    plan = aggregate("HWAM", filters=[{"field": "year", "operator": "IN", "value": [2010, 2011, 2012]}])
    planner, _ = await plan_with({question: plan}, question)
    assert planner.get_plan_source()["corrections"] == []
    assert planner.get_semantic_plan().operations[0].filters[0].operator == "IN"


@pytest.mark.mocked_llm
async def test_mocked_llm_analysis_filters_are_grounded():
    """MOCKED LLM: a cultivar the LLM missed is restored from the question."""
    question = "Correlation between rainfall and yield for short-season cultivar in 2015"
    planner, _ = await plan_with({question: analysis("correlation", "PRCP", "HWAM")}, question)
    request = planner.get_semantic_plan().operations[0].analysis
    assert {(f.field, f.value) for f in request.filters} == {("cultivar", "SHT"), ("year", 2015)}


@pytest.mark.mocked_llm
async def test_mocked_llm_follow_up_uses_previous_analysis():
    """MOCKED LLM: the previous analysis is in the prompt; the follow-up keeps
    its variables and filters and applies the new method."""
    previous = AnalysisRequest(operation="correlation", x_variable="PRCP", y_variable="HWAM",
                               filters=[FilterCondition(field="cultivar", operator="=", value="BASE")])
    question = "Now fit a quadratic instead"
    planner, _ = await plan_with({question: analysis("quadratic_regression", "PRCP", "HWAM",
                                                     [{"field": "cultivar", "operator": "=", "value": "BASE"}])},
                                 question, previous)
    assert "PREVIOUS ANALYSIS IN THIS CONVERSATION" in planner.client.calls[0]
    request = planner.get_semantic_plan().operations[0].analysis
    assert request.operation == "quadratic_regression"
    assert [(f.field, f.value) for f in request.filters] == [("cultivar", "BASE")]
    assert planner.get_plan_source()["corrections"] == []


@pytest.mark.mocked_llm
async def test_mocked_llm_uses_chat_completions_only():
    """MOCKED LLM: exactly one Chat Completions call; no Responses API."""
    question = "What is the average yield?"
    planner, _ = await plan_with({question: aggregate("HWAM")}, question)
    assert not hasattr(planner.client, "responses")
    assert len(planner.client.requests) == 1
    assert planner.client.requests[0]["messages"][0]["role"] == "user"


@pytest.mark.mocked_llm
async def test_mocked_llm_definition_is_planned_by_llm():
    """MOCKED LLM: definition questions go to the LLM too."""
    question = "What does SHT mean?"
    plan = {"goal": "g", "intent": "definition",
            "operations": [{"operation": "definition", "variable": "SHT"}]}
    planner, qp = await plan_with({question: plan}, question)
    assert planner.get_plan_source()["planner"] == "llm"
    assert qp.intent == "definition"


# -----------------------------------------------------------------------------
# FALLBACK: LLM unavailable or its plan unusable
# -----------------------------------------------------------------------------

@pytest.mark.fallback
@pytest.mark.parametrize("question,check", [
    ("What is the average yield?", lambda body, frame: body["statistics"]["value"] == pytest.approx(frame["HWAM"].mean())),
    ("What is the correlation between rainfall and yield for BASE?",
     lambda body, frame: body["analysis"]["sample_size"] == int((frame["cultivar"] == "BASE").sum())),
])
def test_fallback_when_llm_unavailable(client, use_llm, sample_frame, question, check):
    """FALLBACK: the gateway fails -> the deterministic planner answers, labeled."""
    fake = use_llm({})  # every call fails like the reset gateway
    body = ask(client, question)
    assert len(fake.calls) == 1
    assert body["planner"]["planner"] == "fallback"
    assert "reset" in body["planner"]["fallback_reason"]
    assert check(body, sample_frame)


@pytest.mark.fallback
def test_fallback_when_no_llm_configured(client, sample_frame):
    """FALLBACK: no key configured."""
    body = ask(client, "What is the average yield?")
    assert body["planner"]["planner"] == "fallback"
    assert body["planner"]["fallback_reason"] == "no LLM configured"
    assert body["statistics"]["value"] == pytest.approx(sample_frame["HWAM"].mean())


@pytest.mark.fallback
async def test_fallback_when_llm_plan_is_invalid():
    """FALLBACK: an unusable LLM plan (no metric, nothing to resolve) is not
    executed; the labeled fallback plans instead."""
    question = "What is the average HWAM?"
    planner = QueryPlanner(api_key=None, db_session=None)
    planner.api_key = "test"
    planner.client = FakeLLM({question: {"goal": "g", "intent": "aggregate", "operations": []}})
    await planner.plan_with_fallback(question)
    source = planner.get_plan_source()
    assert source["planner"] == "fallback"
    assert "no supported operations" in source["fallback_reason"]
    assert planner.get_semantic_plan().operations[0].metric == "HWAM"


# -----------------------------------------------------------------------------
# Early guard: "here" without a selected point
# -----------------------------------------------------------------------------

@pytest.mark.mocked_llm
def test_here_without_point_is_rejected_before_planning(client, use_llm, fake_db):
    """No LLM call and no data query: the user must pick a point first."""
    fake = use_llm({"What is the average yield?": aggregate("HWAM")})
    body = ask(client, "What is the average yield here?")
    assert fake.calls == []
    assert fake_db.calls == []
    assert body["statistics"] is None
    assert "no map point is selected" in body["answer"]
    assert body["planner"]["planner"] == "not_run"
    assert body["planner"]["answer_by"] == "notice"
