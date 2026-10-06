"""Whole-number display, available management variables, and the management
conditions stated with each result.

Fallback-path tests use the CSV-backed fake database (45 sample files); the
LLM-path test uses a mocked LLM (no real model is called).
"""
from __future__ import annotations

import json
import re
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
        [("pfrst15", "03-30", "04-14"), ("pfrst-30", "02-13", "02-28"), ("pfrst0", "03-15", "03-30")],  # MM-DD spans
        [("RF", 5)],                                            # irrigation
        [],                                                     # nitrogen_level: none stored
        (5, 1990, 1995, ["MZ"]),                                # scope
    ])
    result = await ManagementService(session).available()
    fields = {f["field"]: f for f in result["fields"]}
    assert list(fields) == ["cultivar", "planting_stage", "irrigation"]  # empty field omitted
    assert [v["value"] for v in fields["cultivar"]["values"]] == ["BASE", "SHT"]
    assert [v["value"] for v in fields["planting_stage"]["values"]] == ["pfrst-30", "pfrst0", "pfrst15"]
    assert [v["label"] for v in fields["planting_stage"]["values"]] == [
        "30 days before the normal planting date (planted 13–28 Feb)",
        "normal planting date (planted 15–30 Mar)",
        "15 days after the normal planting date (planted 30 Mar–14 Apr)"]
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
    ("What is the average yield with pesticide use?", "pesticide"),
    ("average yield where fungicides were sprayed", "pesticide"),
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


@pytest.mark.fallback
def test_fallback_pesticide_question_states_data_unavailable(client, sample_frame):
    """Regression: the no-LLM path silently averaged all records for a pesticide question."""
    body = ask(client, "What is the average yield with pesticide use?")
    scope = body["management"]
    assert [u["key"] for u in scope["unapplied"]] == ["pesticide"]
    assert "the dataset has no pesticide information" in body["answer"]
    assert "covers all matching records regardless of this condition" in body["answer"]


# -----------------------------------------------------------------------------
# 5. Planting dates: the actual PDAT range of the records behind each answer
# -----------------------------------------------------------------------------

from datetime import date  # noqa: E402

from app.parsers.csv_parser import DSSATParser  # noqa: E402
from app.services.management_service import date_text, day_text, planting_date_note  # noqa: E402
from conftest import pdat_to_date  # noqa: E402


def pdat_range(frame):
    dates = [pdat_to_date(v) for v in frame["PDAT"]]
    days = [f"{d:%m-%d}" for d in dates]
    return min(dates), max(dates), min(days), max(days)


@pytest.mark.parametrize("pdat,expected", [
    (2020060, date(2020, 2, 29)),   # leap year: day 60 is 29 Feb
    (2019060, date(2019, 3, 1)),    # non-leap year: day 60 is 1 Mar
    (2020366, date(2020, 12, 31)),
    (2019365, date(2019, 12, 31)),
])
def test_pdat_conversion_handles_leap_years(pdat, expected):
    assert DSSATParser.parse_dssat_date(pdat) == expected  # what ingestion stores
    assert pdat_to_date(pdat) == expected                  # what the test fake uses
    assert DSSATParser.parse_dssat_date(2019366) is None    # no day 366 outside leap years


def test_date_formatting_keeps_leap_days():
    assert day_text("02-29") == "29 Feb" and date_text(date(2020, 2, 29)) == "29 Feb 2020"


def test_planting_date_note_wording():
    one_year = {"first": date(2020, 2, 29), "last": date(2020, 3, 14), "first_day": "02-29", "last_day": "03-14",
                "dates": 9, "locations": 92, "years": 1, "stages": 1}
    assert planting_date_note(one_year) == (
        "Actual planting dates (PDAT) in the matching records range from 29 Feb to 14 Mar 2020; "
        "the exact date varies across the matching records (by location).")
    many_years = {**one_year, "first": date(1984, 3, 30), "last": date(2020, 4, 4), "first_day": "03-30",
                  "last_day": "04-14", "years": 37}
    assert planting_date_note(many_years) == (
        "Actual planting dates (PDAT) in the matching records range from 30 Mar to 14 Apr each year "
        "(1984–2020); the exact date varies across the matching records (by location and year).")
    single = {**one_year, "last": date(2020, 2, 29), "dates": 1}
    assert planting_date_note(single) == "Actual planting date (PDAT) in the matching records: 29 Feb 2020."
    assert planting_date_note(None) == ""


def planting_part(sentence):
    return sentence[sentence.index("Actual planting"):]


@pytest.mark.fallback
def test_broad_year_answer_uses_that_years_pdat_range(client, sample_frame):
    body = ask(client, "What is the average yield in 2020?")
    subset = sample_frame[sample_frame["year"] == 2020]
    first, last, _, _ = pdat_range(subset)
    note = planting_part(body["management"]["sentence"])
    assert note.startswith(f"Actual planting dates (PDAT) in the matching records range from "
                           f"{first.day} {first:%b} to {date_text(last)};")
    assert "the exact date varies across the matching records" in note
    # "Normal planting date" appears only together with its actual pfrst0 dates.
    mentions = re.findall(r"normal planting date[^.;]*", note)
    assert mentions and all(re.search(r"\(pfrst0\) for .* (?:ranges from|was) \d", m) for m in mentions)
    assert body["management"]["planting_dates"]["first"] == first.isoformat()
    # Not the whole-dataset range: the answer used only 2020 records.
    assert pdat_range(sample_frame)[0] != first
    assert body["management"]["sentence"] in body["answer"]


@pytest.mark.fallback
def test_stage_filtered_answer_uses_that_stages_pdat_range(client, sample_frame):
    body = ask(client, "What is the average yield for LNG planted 15 days late?")
    subset = sample_frame[(sample_frame["cultivar"] == "LNG") & (sample_frame["planting_stage"] == "pfrst15")]
    first, last, first_day, last_day = pdat_range(subset)
    note = planting_part(body["management"]["sentence"])
    assert note.startswith(f"Actual planting dates (PDAT) in the matching records range from "
                           f"{day_text(first_day)} to {day_text(last_day)} each year ({first.year}–{last.year});")
    info = body["management"]["planting_dates"]
    assert (info["first"], info["last"], info["stages"]) == (first.isoformat(), last.isoformat(), 1)


@pytest.mark.fallback
def test_leap_year_stage_range_starts_on_29_february(client, sample_frame):
    body = ask(client, "What is the average yield in 2020 for planting 15 days early?")
    subset = sample_frame[(sample_frame["year"] == 2020) & (sample_frame["planting_stage"] == "pfrst-15")]
    first, last, _, _ = pdat_range(subset)
    assert body["statistics"]["count"] == len(subset)
    assert first == date(2020, 2, 29)  # the data really contains a leap-day planting
    assert planting_part(body["management"]["sentence"]).startswith(
        f"Actual planting dates (PDAT) in the matching records range from 29 Feb to {date_text(last)};")


# -----------------------------------------------------------------------------
# 6. The normal planting (pfrst0) baseline for the answer's own scope
# -----------------------------------------------------------------------------

def baseline_part(sentence):
    return sentence[sentence.index("Actual planting"):]


def expected_baseline(frame, scope):
    """The pfrst0 sentence the answer should contain, from the sample rows."""
    first, last, first_day, last_day = pdat_range(frame)
    if first == last:
        return f"The normal planting date (pfrst0) for {scope} was {date_text(first)}."
    if first.year == last.year:
        return f"The normal planting date (pfrst0) for {scope} ranges from {first.day} {first:%b} to {date_text(last)};"
    return (f"The normal planting date (pfrst0) for {scope} ranges from {day_text(first_day)} to "
            f"{day_text(last_day)} each year ({first.year}–{last.year});")


@pytest.mark.fallback
def test_year_query_shows_all_stage_range_and_pfrst0_baseline(client, sample_frame):
    body = ask(client, "What is the average yield in 2020?")
    year = sample_frame[sample_frame["year"] == 2020]
    note = baseline_part(body["management"]["sentence"])
    first, last, _, _ = pdat_range(year)
    assert note.startswith(f"Actual planting dates (PDAT) in the matching records range from {first.day} {first:%b} "
                           f"to {date_text(last)};")
    assert expected_baseline(year[year["planting_stage"] == "pfrst0"], "the same year") in note
    assert body["management"]["normal_planting_status"] == "ok"
    # The yield still uses every planting stage of 2020.
    assert body["statistics"]["count"] == len(year)


@pytest.mark.fallback
def test_year_and_area_query_scopes_the_baseline_to_that_area(client, sample_frame):
    lat, lon = map(float, sample_frame[["LATITUDE", "LONGITUDE"]].iloc[0])
    body = ask(client, "What is the average yield in 2020 here?", latitude=lat, longitude=lon, radius_km=25)
    area = sample_frame[(distances_km(sample_frame, lat, lon) <= 25) & (sample_frame["year"] == 2020)]
    assert body["statistics"]["count"] == len(area)
    pfrst0 = area[area["planting_stage"] == "pfrst0"]
    note = baseline_part(body["management"]["sentence"])
    assert expected_baseline(pfrst0, "the same year and area") in note
    assert body["management"]["normal_planting_dates"]["first"] == pdat_range(pfrst0)[0].isoformat()
    # Scoped, not dataset-wide: the area's baseline is not the whole 2020 baseline.
    year0 = sample_frame[(sample_frame["year"] == 2020) & (sample_frame["planting_stage"] == "pfrst0")]
    assert pdat_range(pfrst0)[:2] != pdat_range(year0)[:2] or len(pfrst0) == len(year0)


@pytest.mark.fallback
def test_stage_filtered_query_shows_stage_range_and_baseline(client, sample_frame):
    body = ask(client, "What is the average yield for LNG planted 15 days late?")
    lng = sample_frame[sample_frame["cultivar"] == "LNG"]
    stage = lng[lng["planting_stage"] == "pfrst15"]
    note = baseline_part(body["management"]["sentence"])
    _, _, first_day, last_day = pdat_range(stage)
    assert f"range from {day_text(first_day)} to {day_text(last_day)} each year" in note
    assert expected_baseline(lng[lng["planting_stage"] == "pfrst0"], "the same cultivar") in note
    assert body["statistics"]["count"] == len(stage)


@pytest.mark.fallback
def test_missing_pfrst0_baseline_is_reported_not_guessed(client, fake_db, sample_frame):
    """No pfrst0 records in the answer's scope: say so instead of guessing."""
    frame = fake_db.frame
    fake_db.frame = frame[~((frame["year"] == 2020) & (frame["planting_stage"] == "pfrst0"))]
    body = ask(client, "What is the average yield in 2020?")
    note = baseline_part(body["management"]["sentence"])
    assert body["management"]["normal_planting_status"] == "unavailable"
    assert body["management"]["normal_planting_dates"] is None
    assert ("No normal planting (pfrst0) records exist for the same year, so the normal planting date "
            "cannot be shown for this answer.") in note
    assert "The normal planting date (pfrst0) for" not in note


def test_pfrst0_only_answer_is_not_compared_with_itself():
    info = {"first": date(2020, 3, 15), "last": date(2020, 3, 29), "first_day": "03-15", "last_day": "03-29",
            "dates": 15, "locations": 92, "years": 1, "stages": 1}
    note = planting_date_note(info, None, "same", "the same year")
    assert note.endswith("These are the normal planting (pfrst0) records.")
    assert note.count("15 Mar") == 1
