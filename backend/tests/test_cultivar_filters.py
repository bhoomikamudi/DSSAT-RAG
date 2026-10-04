"""Cultivar filter recognition: case-insensitive codes, verified aliases,
no silent drops, and no confusion with other parts of the question."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent.analysis_parser import find_cultivars, parse_analysis_request
from app.agent.models import AnalysisRequest, FilterCondition
from app.api.v1 import chat
from app.db.session import get_db
from app.services.analysis_vocabulary import CULTIVAR_CODES, normalize_cultivar

CODES = ["BASE", "LNG", "SHT", "VLNG", "VSHT"]
QUESTION = "What is the correlation between rainfall and yield for {}?"


def cultivar_filter(query):
    request = parse_analysis_request(query)
    assert request is not None, query
    return [c for c in request.filters if c.field == "cultivar"]


def expected(code):
    return [FilterCondition(field="cultivar", operator="=", value=code)]


# -----------------------------------------------------------------------------
# 1. Codes in any capitalization
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("spelling", [str.upper, str.lower, str.title])
def test_codes_are_case_insensitive(code, spelling):
    query = QUESTION.format(spelling(code))
    assert cultivar_filter(query) == expected(code), query


@pytest.mark.parametrize("code", CODES)
def test_code_variants_produce_identical_requests(code):
    requests = [
        parse_analysis_request(QUESTION.format(variant))
        for variant in (code, code.lower(), code.title(), f"the {code.lower()} cultivar")
    ]
    assert all(request == requests[0] for request in requests)
    assert (requests[0].x_variable, requests[0].y_variable) == ("PRCP", "HWAM")


# -----------------------------------------------------------------------------
# 2-3. Verified aliases normalize to the canonical database value
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "wording, code",
    [
        ("the standard cultivar", "BASE"),
        ("the Standard Cultivar", "BASE"),
        ("the baseline cultivar", "BASE"),
        ("baseline", "BASE"),
        ("the base cultivar", "BASE"),
        ("the long-season cultivar", "LNG"),
        ("the long season cultivar", "LNG"),
        ("Long-Season", "LNG"),
        ("the short-season cultivar", "SHT"),
        ("short season", "SHT"),
        ("the very-long-season cultivar", "VLNG"),
        ("very long season", "VLNG"),
        ("the very-short-season cultivar", "VSHT"),
        ("Very Short Season", "VSHT"),
    ],
)
def test_verified_aliases(wording, code):
    assert cultivar_filter(QUESTION.format(wording)) == expected(code)


def test_alias_and_code_give_identical_request():
    by_code = parse_analysis_request(QUESTION.format("BASE"))
    by_alias = parse_analysis_request(QUESTION.format("the standard cultivar"))
    assert by_alias == by_code


def test_every_alias_maps_to_an_existing_code():
    for code, spec in CULTIVAR_CODES.items():
        assert code in CODES
        for alias in spec["aliases"]:
            assert normalize_cultivar(alias) == code


@pytest.mark.parametrize(
    "value, code",
    [("base", "BASE"), ("Base", "BASE"), ("lng", "LNG"), ("long-season", "LNG"),
     ("very short season cultivar", "VSHT"), ("standard", "BASE"), ("baseline", "BASE")],
)
def test_llm_supplied_cultivar_values_are_normalized(value, code):
    request = AnalysisRequest(
        operation="correlation", x_variable="PRCP", y_variable="HWAM",
        filters=[{"field": "cultivar", "operator": "=", "value": value}],
    )
    assert request.filters[0].value == code


def test_llm_supplied_cultivar_list_is_normalized():
    request = AnalysisRequest(
        operation="correlation", x_variable="PRCP", y_variable="HWAM",
        filters=[{"field": "cultivar", "operator": "IN", "value": ["lng", "Vlng"]}],
    )
    assert request.filters[0].value == ["LNG", "VLNG"]


def test_unknown_cultivar_value_is_left_for_the_database_to_reject():
    assert normalize_cultivar("hybrid-x") == "hybrid-x"


@pytest.mark.parametrize(
    "wording",
    ["lng and vlng", "VLNG and LNG", "Vlng, lng", "the long-season and very long season cultivars"],
)
def test_multiple_cultivars_equivalent_in_any_order_or_case(wording):
    assert cultivar_filter(QUESTION.format(wording)) == [
        FilterCondition(field="cultivar", operator="IN", value=["LNG", "VLNG"])
    ]


@pytest.mark.parametrize("wording", ["base and lng", "LNG and base", "base, lng"])
def test_lowercase_base_in_a_cultivar_list(wording):
    assert cultivar_filter(QUESTION.format(wording)) == [
        FilterCondition(field="cultivar", operator="IN", value=["BASE", "LNG"])
    ]


# -----------------------------------------------------------------------------
# 4. A cultivar filter is not mistaken for another part of the question,
#    and other words are not mistaken for a cultivar
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("code", ["base", "Base", "lng", "sht", "the long-season cultivar"])
def test_cultivar_does_not_change_variables_operation_or_grouping(code):
    request = parse_analysis_request(QUESTION.format(code))
    assert request.operation == "correlation"
    assert (request.x_variable, request.y_variable) == ("PRCP", "HWAM")
    assert request.group_by is None
    assert request.requires_group_variation is False
    assert [c.field for c in request.filters] == ["cultivar"]


def test_cultivar_filter_with_other_filters_keeps_all_of_them():
    request = parse_analysis_request(
        "Correlation between rainfall and yield for base rainfed runs from 1990 to 2000"
    )
    assert [(c.field, c.value) for c in request.filters] == [
        ("cultivar", "BASE"),
        ("irrigation", "RF"),
        ("year", [1990, 2000]),
    ]


def test_lowercase_code_with_grouping_request():
    request = parse_analysis_request("Regress yield on rainfall for sht per year")
    assert request.filters == expected("SHT")
    assert request.group_by == "year"


@pytest.mark.parametrize(
    "query",
    [
        "Correlation between rainfall and yield using base temperature",
        "Correlation between rainfall and yield for the base case",
        "Correlation between rainfall and yield in the database",
        "Correlation between rainfall and yield based on all runs",
        "Give the standard deviation of yield",
        "Correlation between rainfall and yield for standard planting",
    ],
)
def test_other_words_are_not_read_as_a_cultivar(query):
    assert find_cultivars(query) == []


def test_standard_planting_stays_a_planting_filter():
    request = parse_analysis_request("Correlation between rainfall and yield for standard planting")
    assert [(c.field, c.value) for c in request.filters] == [("planting_stage", "pfrst0")]


def test_follow_up_with_lowercase_code():
    first = parse_analysis_request(
        "Show the relationship between rainfall and yield separately for each cultivar."
    )
    follow = parse_analysis_request("What about base?", previous=first)
    assert follow.filters == expected("BASE")
    assert follow.group_by is None


# -----------------------------------------------------------------------------
# 6. Full analysis path: the filter reaches the database and the sample is
#    BASE-only, not all cultivars
# -----------------------------------------------------------------------------

@pytest.fixture
def client(fake_db):
    app = FastAPI()
    app.include_router(chat.router, prefix="/api/v1/chat")

    async def fake_get_db():
        yield object()

    app.dependency_overrides[get_db] = fake_get_db
    return TestClient(app)


def ask(client, message):
    response = client.post("/api/v1/chat/", json={"message": message})
    assert response.status_code == 200, response.text
    return response.json()


def test_full_path_lowercase_base_applies_filter(client, fake_db, sample_frame):
    body = ask(client, "correlation between rainfall and yield for base")
    analysis = body["analysis"]
    base_rows = int((sample_frame["cultivar"] == "BASE").sum())

    assert fake_db.calls[-1]["filters"] == {"cultivar": "BASE"}
    assert analysis["filters"] == [{"field": "cultivar", "operator": "=", "value": "BASE"}]
    assert analysis["sample_size"] == base_rows
    assert analysis["sample_size"] < len(sample_frame)
    assert "cultivar = BASE" in body["answer"]


def test_full_path_equivalent_wordings_give_identical_results(client):
    wordings = [
        "correlation between rainfall and yield for BASE",
        "correlation between rainfall and yield for base",
        "correlation between rainfall and yield for Base",
        "correlation between rainfall and yield for the standard cultivar",
        "correlation between rainfall and yield for the baseline cultivar",
    ]
    results = [ask(client, wording)["analysis"] for wording in wordings]
    assert all(result == results[0] for result in results)


@pytest.mark.parametrize("code", ["LNG", "SHT", "VLNG", "VSHT"])
def test_full_path_other_codes_lowercase(client, fake_db, sample_frame, code):
    analysis = ask(client, f"correlation between rainfall and yield for {code.lower()}")["analysis"]
    assert fake_db.calls[-1]["filters"] == {"cultivar": code}
    assert analysis["sample_size"] == int((sample_frame["cultivar"] == code).sum())
