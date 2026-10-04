"""End-to-end tests through the FastAPI chat route.

The sample DSSAT CSVs stand in for PostgreSQL (see conftest.fake_db). Analysis
questions need no LLM; the existing aggregate questions use a fake LLM that
returns the semantic plans the planner LLM produces for them. Expected numbers
are recomputed independently from the CSVs with pandas/SciPy.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scipy import stats

from app.agent import planner as planner_module
from app.agent import response_generator as response_module
from app.api.v1 import chat
from app.db.session import get_db
from conftest import FakeLLM, apply_filters


@pytest.fixture
def client(fake_db):
    app = FastAPI()
    app.include_router(chat.router, prefix="/api/v1/chat")

    async def fake_get_db():
        yield object()

    app.dependency_overrides[get_db] = fake_get_db
    return TestClient(app)


def ask(client, message, session_id=None):
    response = client.post("/api/v1/chat/", json={"message": message, "session_id": session_id})
    assert response.status_code == 200, response.text
    return response.json()


def xy(frame, **filters):
    subset = apply_filters(frame, **filters)
    return subset["PRCP"].to_numpy(float), subset["HWAM"].to_numpy(float)


# -----------------------------------------------------------------------------
# 1-5: analysis questions
# -----------------------------------------------------------------------------

def test_correlation_prcp_hwam_for_base(client, fake_db, sample_frame):
    body = ask(client, "What is the correlation between precipitation and yield for the BASE cultivar?")
    analysis = body["analysis"]
    x, y = xy(sample_frame, cultivar="BASE")
    expected = stats.pearsonr(x, y)

    assert fake_db.calls[-1] == {
        "variables": ["PRCP", "HWAM"],
        "group_by": None,
        "filters": {"cultivar": "BASE"},
    }
    assert analysis["status"] == "ok"
    assert analysis["analysis_type"] == "correlation"
    assert analysis["sample_size"] == len(x)
    assert analysis["correlation"] == pytest.approx(expected.statistic)
    assert analysis["chart_type"] == "scatter"
    assert analysis["chart_data"]["plotted_points"] <= 1000
    assert f"r = {expected.statistic:.3f}" in body["answer"]
    assert body["statistics"] is None  # existing field untouched
    assert body["semantic_plan"]["intent"] == "analysis"


def test_linear_regression_prcp_hwam(client, sample_frame):
    analysis = ask(client, "Is there a linear relationship between rainfall and yield?")["analysis"]
    x, y = xy(sample_frame)
    expected = stats.linregress(x, y)

    assert analysis["analysis_type"] == "linear_regression"
    assert analysis["sample_size"] == len(sample_frame)
    assert analysis["coefficients"]["slope"] == pytest.approx(expected.slope)
    assert analysis["coefficients"]["intercept"] == pytest.approx(expected.intercept)
    assert analysis["r_squared"] == pytest.approx(expected.rvalue ** 2)
    assert analysis["chart_type"] == "scatter_with_line"
    assert len(analysis["chart_data"]["series"][0]["fit"]) == 2


def test_quadratic_regression_prcp_hwam(client, sample_frame):
    analysis = ask(client, "Is there a quadratic relationship between precipitation and yield?")["analysis"]
    x, y = xy(sample_frame)
    a, b, c = np.polyfit(x, y, 2)

    assert analysis["analysis_type"] == "quadratic_regression"
    assert analysis["coefficients"]["quadratic"] == pytest.approx(a)
    assert analysis["coefficients"]["linear"] == pytest.approx(b)
    assert analysis["coefficients"]["intercept"] == pytest.approx(c)
    assert analysis["r_squared"] >= analysis["details"]["linear_r_squared"]
    assert analysis["chart_type"] == "scatter_with_curve"


def test_grouped_correlation_by_cultivar(client, fake_db, sample_frame):
    analysis = ask(client, "Show the precipitation-yield relationship separately for each cultivar.")["analysis"]

    assert fake_db.calls[-1]["group_by"] == "cultivar"
    groups = {group["group"]: group for group in analysis["groups"]}
    assert set(groups) == {"BASE", "LNG", "SHT", "VLNG", "VSHT"}
    for cultivar, group in groups.items():
        x, y = xy(sample_frame, cultivar=cultivar)
        assert group["correlation"] == pytest.approx(stats.pearsonr(x, y).statistic)
        assert group["sample_size"] == len(x)
    series = analysis["chart_data"]["series"]
    assert [s["name"] for s in series] == ["BASE", "LNG", "SHT", "VLNG", "VSHT"]
    assert sum(len(s["points"]) for s in series) <= 1000 + len(series)


def test_multiple_filters_base_rainfed(client, fake_db, sample_frame):
    body = ask(client, "Analyze precipitation and yield for BASE rainfed simulations.")
    analysis = body["analysis"]
    x, y = xy(sample_frame, cultivar="BASE", irrigation="RF")

    assert fake_db.calls[-1]["filters"] == {"cultivar": "BASE", "irrigation": "RF"}
    assert [f["field"] for f in analysis["filters"]] == ["cultivar", "irrigation"]
    assert analysis["sample_size"] == len(x)
    assert analysis["correlation"] == pytest.approx(stats.pearsonr(x, y).statistic)
    assert "irrigation regime = RF" in body["answer"]


# -----------------------------------------------------------------------------
# Paraphrases -> identical results
# -----------------------------------------------------------------------------

def test_paraphrases_give_identical_results(client):
    questions = [
        "What is the relationship between rainfall and yield?",
        "Does precipitation affect yield?",
        "Is PRCP correlated with HWAM?",
        "How does rainfall influence harvested yield?",
    ]
    results = [ask(client, q)["analysis"] for q in questions]
    assert all(result == results[0] for result in results)
    assert results[0]["analysis_type"] == "correlation"


# -----------------------------------------------------------------------------
# Unavailable / invalid requests are reported, never invented
# -----------------------------------------------------------------------------

def test_nitrogen_rate_question_is_reported_unavailable(client):
    body = ask(client, "Does nitrogen rate affect yield?")
    assert body["analysis"]["status"] == "unavailable"
    assert "only one nitrogen level value (HighN)" in body["answer"]
    assert body["analysis"]["chart_type"] is None
    assert body["confidence"] == "low"


def test_irrigated_filter_is_reported_unavailable(client):
    body = ask(client, "What is the correlation between rainfall and yield for irrigated simulations?")
    assert body["analysis"]["status"] == "unavailable"
    assert "irrigation regime = IR" in body["answer"]
    assert "Available irrigation regime values: RF" in body["answer"]


def test_unknown_variable_is_reported_unavailable(client):
    body = ask(client, "What is the correlation between soil moisture and yield?")
    assert body["analysis"]["status"] == "unavailable"
    assert "'soil moisture' is not an available variable" in body["answer"]


def test_no_matching_year_is_reported(client):
    body = ask(client, "Correlation between rainfall and yield in 1950")
    assert body["analysis"]["status"] == "unavailable"
    assert "year = 1950" in body["answer"]


def test_descriptive_statistics_by_cultivar(client, sample_frame):
    analysis = ask(client, "Give descriptive statistics for yield by cultivar")["analysis"]
    expected = sample_frame.groupby("cultivar")["HWAM"].mean()
    assert analysis["chart_type"] == "bar"
    for bar in analysis["chart_data"]["bars"]:
        assert bar["mean"] == pytest.approx(expected[bar["group"]])


# -----------------------------------------------------------------------------
# Conversation context
# -----------------------------------------------------------------------------

def test_follow_up_reuses_filters_and_requeries_database(client, fake_db, sample_frame):
    first = ask(client, "Analyze precipitation and yield for BASE rainfed simulations.", session_id="s1")
    follow = ask(client, "Now fit a quadratic instead", session_id="s1")

    assert first["analysis"]["analysis_type"] == "correlation"
    assert follow["analysis"]["analysis_type"] == "quadratic_regression"
    assert follow["analysis"]["filters"] == first["analysis"]["filters"]
    # The follow-up retrieved the records again rather than reusing chart points.
    assert len(fake_db.calls) == 2
    assert fake_db.calls[1]["filters"] == {"cultivar": "BASE", "irrigation": "RF"}
    x, y = xy(sample_frame, cultivar="BASE", irrigation="RF")
    assert follow["analysis"]["coefficients"]["quadratic"] == pytest.approx(np.polyfit(x, y, 2)[0])


def test_follow_up_is_scoped_to_session(client):
    ask(client, "Analyze precipitation and yield for BASE rainfed simulations.", session_id="s1")
    other = ask(client, "Now fit a quadratic instead", session_id="s2")
    assert other["analysis"] is None


# -----------------------------------------------------------------------------
# 6-8: existing aggregate behavior through the same API
# -----------------------------------------------------------------------------

def aggregate_plan(metric, aggregation, filters=None, group_by=None):
    return {
        "goal": "g",
        "intent": "aggregate",
        "operations": [
            {
                "operation": "aggregate",
                "entity": "simulation_outputs",
                "metric": metric,
                "aggregation": aggregation,
                "filters": filters or [],
                "group_by": group_by or [],
                "independent": True,
            }
        ],
    }


YEAR_2020 = [{"field": "year", "operator": "=", "value": 2020}]
LLM_PLANS = {
    "What is the average rainfall in 2020?": aggregate_plan("PRCP", "AVG", YEAR_2020),
    "What is the minimum rainfall in 2020?": aggregate_plan("PRCP", "MIN", YEAR_2020),
    "What is the maximum yield for BASE?": aggregate_plan(
        "HWAM", "MAX", [{"field": "cultivar", "operator": "=", "value": "BASE"}]
    ),
    "Which cultivar has the highest average yield?": aggregate_plan("HWAM", "AVG", group_by=["cultivar"]),
}


@pytest.fixture
def llm_client(client, monkeypatch):
    fake = FakeLLM(LLM_PLANS)
    monkeypatch.setattr(planner_module.settings, "OPENAI_API_KEY", "test")
    monkeypatch.setattr(planner_module, "AsyncOpenAI", lambda **kwargs: fake)
    # Keep the response generator on its deterministic heuristic path.
    monkeypatch.setattr(response_module, "AsyncOpenAI", lambda **kwargs: None)
    return client


def test_existing_average_rainfall_2020(llm_client, sample_frame):
    body = ask(llm_client, "What is the average rainfall in 2020?")
    expected = sample_frame.loc[sample_frame["year"] == 2020, "PRCP"]
    assert body["analysis"] is None
    assert body["statistics"]["aggregation_type"] == "AVG"
    assert body["statistics"]["value"] == pytest.approx(expected.mean())
    assert body["statistics"]["count"] == len(expected)


def test_existing_minimum_rainfall_2020(llm_client, sample_frame):
    body = ask(llm_client, "What is the minimum rainfall in 2020?")
    expected = sample_frame.loc[sample_frame["year"] == 2020, "PRCP"]
    assert body["analysis"] is None
    assert body["statistics"]["aggregation_type"] == "MIN"
    assert body["statistics"]["value"] == pytest.approx(expected.min())


def test_existing_maximum_yield_base(llm_client, sample_frame):
    body = ask(llm_client, "What is the maximum yield for BASE?")
    expected = sample_frame.loc[sample_frame["cultivar"] == "BASE", "HWAM"]
    assert body["statistics"]["aggregation_type"] == "MAX"
    assert body["statistics"]["value"] == pytest.approx(expected.max())


def test_existing_highest_average_yield_cultivar(llm_client, sample_frame):
    body = ask(llm_client, "Which cultivar has the highest average yield?")
    expected = sample_frame.groupby("cultivar")["HWAM"].mean()
    breakdown = body["statistics"]["breakdown"]
    assert body["analysis"] is None
    assert breakdown["group_by"] == "cultivar"
    assert breakdown["values"][0]["group_value"] == expected.idxmax()
    assert breakdown["values"][0]["value"] == pytest.approx(expected.max())
