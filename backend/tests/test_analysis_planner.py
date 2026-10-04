"""Planner tests: paraphrase normalization, filters, follow-ups, and that the
existing aggregate/definition planning is unchanged."""
from __future__ import annotations

import pytest

from app.agent.analysis_parser import parse_analysis_request
from app.agent.models import AnalysisRequest, FilterCondition
from app.agent.planner import QueryPlanner
from conftest import FakeLLM


def request_of(query, previous=None):
    request = parse_analysis_request(query, previous=previous)
    assert request is not None, f"not recognized as analysis: {query}"
    return request


# -----------------------------------------------------------------------------
# Paraphrases -> one canonical request
# -----------------------------------------------------------------------------

RELATIONSHIP_PARAPHRASES = [
    "What is the relationship between rainfall and yield?",
    "Does precipitation affect yield?",
    "Is PRCP correlated with HWAM?",
    "How does rainfall influence harvested yield?",
    "Is yield associated with rainfall?",
    "Analyze precipitation versus yield.",
    "What is the correlation between HWAM and PRCP?",
]


def test_paraphrases_produce_identical_requests():
    requests = [request_of(q) for q in RELATIONSHIP_PARAPHRASES]
    expected = AnalysisRequest(operation="correlation", x_variable="PRCP", y_variable="HWAM")
    for query, request in zip(RELATIONSHIP_PARAPHRASES, requests):
        assert request == expected, query


@pytest.mark.parametrize(
    "queries, operation",
    [
        (
            [
                "Is there a linear relationship between rainfall and yield?",
                "Fit a linear regression of yield on precipitation.",
                "Regress HWAM on PRCP",
                "What is the slope of yield against rainfall?",
            ],
            "linear_regression",
        ),
        (
            [
                "Is there a quadratic relationship between precipitation and yield?",
                "Fit a second-order polynomial of yield vs rainfall",
                "Is the rainfall-yield relationship nonlinear?",
            ],
            "quadratic_regression",
        ),
    ],
)
def test_operation_synonyms(queries, operation):
    for query in queries:
        request = request_of(query)
        assert request.operation == operation, query
        assert (request.x_variable, request.y_variable) == ("PRCP", "HWAM"), query


def test_llm_style_values_are_canonicalized():
    request = AnalysisRequest(
        operation="polynomial",
        x_variable="rainfall",
        y_variable="Harvested Yield",
        group_by=["varieties"],
        filters=[{"field": "nitrogen", "operator": "=", "value": "HighN"}],
    )
    assert request.operation == "quadratic_regression"
    assert (request.x_variable, request.y_variable) == ("PRCP", "HWAM")
    assert request.group_by == "cultivar"
    assert request.filters[0].field == "nitrogen_level"


# -----------------------------------------------------------------------------
# Filters and grouping
# -----------------------------------------------------------------------------

def test_cultivar_filter():
    request = request_of("What is the correlation between precipitation and yield for the BASE cultivar?")
    assert request.filters == [FilterCondition(field="cultivar", operator="=", value="BASE")]


def test_multiple_filters_equivalent_wording():
    a = request_of("Analyze precipitation and yield for BASE rainfed simulations.")
    b = request_of("Filter to BASE and rainfed simulations, then analyze precipitation versus yield.")
    c = request_of("For rain-fed runs of the baseline cultivar, how does rainfall affect yield?")
    expected = [
        FilterCondition(field="cultivar", operator="=", value="BASE"),
        FilterCondition(field="irrigation", operator="=", value="RF"),
    ]
    for request in (a, b, c):
        assert request.filters == expected
        assert (request.x_variable, request.y_variable) == ("PRCP", "HWAM")


def test_year_planting_and_nitrogen_filters():
    request = request_of(
        "Correlation between rainfall and yield from 1990 to 2000 for high nitrogen, "
        "30 days late planting"
    )
    assert request.filters == [
        FilterCondition(field="nitrogen_level", operator="=", value="HighN"),
        FilterCondition(field="planting_stage", operator="=", value="pfrst30"),
        FilterCondition(field="year", operator="BETWEEN", value=[1990, 2000]),
    ]


def test_multiple_cultivars_become_in_filter():
    request = request_of("Correlation between rainfall and yield for LNG and VLNG")
    assert request.filters == [FilterCondition(field="cultivar", operator="IN", value=["LNG", "VLNG"])]


@pytest.mark.parametrize(
    "query, group_by",
    [
        ("Show the relationship between rainfall and yield separately for each cultivar.", "cultivar"),
        ("Show the precipitation-yield relationship separately for each cultivar.", "cultivar"),
        ("Correlation of rainfall and yield by cultivar", "cultivar"),
        ("Regress yield on rainfall per year", "year"),
        ("Rainfall vs yield correlation for each planting date", "planting_stage"),
    ],
)
def test_group_by_detection(query, group_by):
    assert request_of(query).group_by == group_by


def test_rainfed_is_not_read_as_rain_variable():
    request = request_of("Is yield correlated with maximum temperature for rain-fed runs?")
    assert (request.x_variable, request.y_variable) == ("TMAXA", "HWAM")
    assert request.filters == [FilterCondition(field="irrigation", operator="=", value="RF")]


def test_ambiguous_single_variable_is_not_guessed():
    # "temperature" alone could be TMAXA or TMINA; the parser does not guess.
    assert parse_analysis_request("Does yield depend on temperature?") is None


# -----------------------------------------------------------------------------
# Unavailable data is represented explicitly
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query, factor",
    [
        ("Does nitrogen rate affect yield?", "nitrogen_level"),
        ("What is the effect of nitrogen on yield?", "nitrogen_level"),
        ("What is the effect of irrigation on yield?", "irrigation"),
    ],
)
def test_factor_questions_require_variation(query, factor):
    request = request_of(query)
    assert request.operation == "descriptive_statistics"
    assert request.group_by == factor
    assert request.requires_group_variation is True


def test_unknown_variable_is_kept_for_reporting():
    request = request_of("What is the correlation between soil moisture and yield?")
    assert request.x_variable == "soil moisture"
    assert request.y_variable == "HWAM"


# -----------------------------------------------------------------------------
# Existing question types are NOT captured by the analysis parser
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "What is the average rainfall in 2020?",
        "What is the minimum rainfall in 2020?",
        "What is the maximum yield for BASE?",
        "Which cultivar has the highest average yield?",
        "What is the average yield by cultivar?",
        "What does PRCP mean?",
        "Define HWAM",
        "Show the yield trend over time",
        "How many simulation records are loaded?",
        "Compare yield between BASE and LNG",
    ],
)
def test_existing_questions_are_not_analysis(query):
    assert parse_analysis_request(query) is None


# -----------------------------------------------------------------------------
# Follow-up questions reuse the previous request
# -----------------------------------------------------------------------------

def test_follow_up_changes_operation_keeps_filters():
    first = request_of("Analyze precipitation and yield for BASE rainfed simulations.")
    follow = request_of("Now fit a quadratic instead", previous=first)
    assert follow.operation == "quadratic_regression"
    assert (follow.x_variable, follow.y_variable) == ("PRCP", "HWAM")
    assert follow.filters == first.filters


def test_follow_up_replaces_filter_and_drops_matching_group():
    first = request_of("Show the relationship between rainfall and yield separately for each cultivar.")
    follow = request_of("What about only LNG?", previous=first)
    assert follow.filters == [FilterCondition(field="cultivar", operator="=", value="LNG")]
    assert follow.group_by is None
    assert follow.operation == "correlation"


def test_follow_up_all_cultivars_resets_cultivar_filter():
    first = request_of("Correlation between rainfall and yield for BASE rainfed")
    follow = request_of("Now do the same for all cultivars, by cultivar", previous=first)
    assert follow.group_by == "cultivar"
    assert [c.field for c in follow.filters] == ["irrigation"]


def test_aggregate_question_after_analysis_is_not_a_follow_up():
    first = request_of("Does precipitation affect yield?")
    assert parse_analysis_request("What is the average rainfall in 2020?", previous=first) is None


# -----------------------------------------------------------------------------
# Planner integration
# -----------------------------------------------------------------------------

def make_planner(plans=None):
    planner = QueryPlanner(api_key=None, db_session=None)
    planner.api_key = "test"
    planner.client = FakeLLM(plans or {})
    return planner


@pytest.mark.fallback
@pytest.mark.asyncio
async def test_fallback_analysis_plan_when_llm_fails():
    """FALLBACK: the LLM is asked first; the parser plans only after it fails."""
    planner = make_planner()  # mocked LLM with no scripted plan -> call fails
    plan = await planner.plan_with_fallback("Is there a linear relationship between rainfall and yield?")
    semantic = planner.get_semantic_plan()

    assert len(planner.client.calls) == 1  # LLM tried first
    assert planner.get_plan_source()["planner"] == "fallback"
    assert plan.intent == "analysis"
    assert semantic.intent == "analysis"
    assert semantic.operations[0].operation == "analysis"
    assert semantic.operations[0].analysis.operation == "linear_regression"


@pytest.mark.fallback
@pytest.mark.asyncio
async def test_paraphrases_produce_equivalent_semantic_plans():
    """FALLBACK: without a working LLM, paraphrases still plan identically."""
    plans = []
    for query in RELATIONSHIP_PARAPHRASES[:4]:
        planner = make_planner()
        await planner.plan_with_fallback(query)
        plans.append(planner.get_semantic_plan().model_dump())
    assert all(plan == plans[0] for plan in plans)


@pytest.mark.fallback
@pytest.mark.asyncio
async def test_planner_uses_conversation_context():
    """FALLBACK: follow-ups resolve from the previous analysis."""
    previous = AnalysisRequest(
        operation="correlation",
        x_variable="PRCP",
        y_variable="HWAM",
        filters=[FilterCondition(field="cultivar", operator="=", value="BASE")],
    )
    planner = make_planner()
    await planner.plan_with_fallback("Now fit a linear regression instead", analysis_context=previous)
    request = planner.get_semantic_plan().operations[0].analysis
    assert request.operation == "linear_regression"
    assert request.filters == previous.filters


@pytest.mark.mocked_llm
@pytest.mark.asyncio
async def test_llm_emitted_analysis_operation_is_normalized():
    """MOCKED LLM: aliases in the LLM's analysis plan become DSSAT codes."""
    question = "How strongly does rain track with maize output across runs?"
    plans = {
        question: {
            "goal": "relationship",
            "intent": "analysis",
            "operations": [
                {
                    "operation": "analysis",
                    "analysis": {
                        "operation": "regression",
                        "x_variable": "rainfall",
                        "y_variable": "yield",
                        "filters": [],
                        "group_by": None,
                    },
                }
            ],
        }
    }
    planner = make_planner(plans)
    plan = await planner.plan_with_fallback(question)
    request = planner.get_semantic_plan().operations[0].analysis
    assert plan.intent == "analysis"
    assert request.operation == "linear_regression"
    assert (request.x_variable, request.y_variable) == ("PRCP", "HWAM")


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


@pytest.mark.mocked_llm
@pytest.mark.asyncio
async def test_existing_average_rainfall_single_year_uses_avg():
    question = "What is the average rainfall in 2020?"
    planner = make_planner({question: aggregate_plan("PRCP", "SUM", [{"field": "year", "operator": "=", "value": 2020}])})
    plan = await planner.plan_with_fallback(question)
    operation = planner.get_semantic_plan().operations[0]
    assert plan.aggregation == "AVG"
    assert operation.aggregation == "AVG"
    assert operation.filters[0].value == 2020


@pytest.mark.mocked_llm
@pytest.mark.asyncio
async def test_existing_minimum_and_maximum_overrides():
    q_min = "What is the minimum rainfall in 2020?"
    q_max = "What is the maximum yield for BASE?"
    planner = make_planner({
        q_min: aggregate_plan("PRCP", "AVG", [{"field": "year", "operator": "=", "value": 2020}]),
        q_max: aggregate_plan("HWAM", "AVG", [{"field": "cultivar", "operator": "=", "value": "BASE"}]),
    })
    plan = await planner.plan_with_fallback(q_min)
    assert plan.aggregation == "MIN"
    assert planner.get_semantic_plan().operations[0].aggregation == "MIN"

    plan = await planner.plan_with_fallback(q_max)
    assert plan.aggregation == "MAX"
    assert planner.get_semantic_plan().operations[0].aggregation == "MAX"


@pytest.mark.mocked_llm
@pytest.mark.asyncio
async def test_existing_highest_average_by_cultivar():
    question = "Which cultivar has the highest average yield?"
    plan_json = aggregate_plan("HWAM", "AVG", group_by=["cultivar"])
    plan_json["operations"].insert(0, aggregate_plan("HWAM", "MAX")["operations"][0])
    planner = make_planner({question: plan_json})
    await planner.plan_with_fallback(question)
    operations = planner.get_semantic_plan().operations
    assert len(operations) == 1
    assert operations[0].aggregation == "AVG"
    assert operations[0].group_by == ["cultivar"]


@pytest.mark.fallback
@pytest.mark.asyncio
async def test_fallback_definition_when_llm_fails():
    """FALLBACK: definitions are planned by the LLM; the shortcut is the fallback."""
    planner = make_planner()
    await planner.plan_with_fallback("What does SHT mean?")
    semantic = planner.get_semantic_plan()
    assert len(planner.client.calls) == 1
    assert planner.get_plan_source()["planner"] == "fallback"
    assert semantic.intent == "definition"
    assert semantic.operations[0].variable == "SHT"
