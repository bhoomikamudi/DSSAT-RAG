"""Deterministic planning of statistical questions when the LLM is unavailable.

Used only by QueryPlanner when no LLM is configured or the LLM call fails;
the LLM remains the primary planner. It reuses the project's existing
pieces rather than a separate vocabulary:
- metric: StatisticsService.resolve_metric (canonical synonyms, DB codes,
  CDE definitions; ambiguous wording -> clarification)
- filters: analysis_parser.find_filters (cultivar, irrigation, nitrogen,
  planting stage, years), with planting stages checked against the data
- output: the same SemanticPlan the LLM path produces, so the orchestrator
  attaches a spatial radius and the executor runs it unchanged
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from app.agent.analysis_parser import find_filters, find_group_by
from app.agent.models import FilterCondition, QueryPlan, SemanticOperation, SemanticPlan
from app.services.analysis_vocabulary import find_variable_spans

# Order matters only for reporting; each aggregation is detected independently.
AGGREGATION_PATTERNS = [
    # "the least", not "at least 5000 kg/ha"
    ("MIN", re.compile(r"\b(minimum|min|lowest|smallest|the\s+least)\b")),
    ("MAX", re.compile(r"\b(maximum|max|highest|largest|greatest|biggest|peak)\b")),
    ("AVG", re.compile(r"\b(average|avg|mean|typical)\b")),
    ("SUM", re.compile(r"\b(total|sum|cumulative|combined)\b")),
]

# "What does SHT mean?" is a definition, not an average.
DEFINITION_PATTERN = re.compile(
    r"\b(what\s+(?:does|do)\b.*\bmean|meaning\s+of|define|definition|stands?\s+for)\b"
)
# "which cultivar has the highest ..." ranks groups.
WHICH_GROUP_PATTERN = re.compile(
    r"\bwhich\s+(cultivars?|variet(?:y|ies)|years?|planting\s+(?:dates?|stages?))\b"
)
AGGREGATION_NAMES = {"MIN": "minimum", "MAX": "maximum", "AVG": "average", "SUM": "total"}


@dataclass
class FallbackStatisticsPlan:
    query_plan: QueryPlan
    semantic_plan: Optional[SemanticPlan] = None
    clarification: Optional[str] = None


def detect_aggregations(query: str) -> List[str]:
    """Aggregations requested, ignoring words inside variable names.

    "average maximum temperature" is AVG of TMAXA, not MAX.
    """
    lowered = query.lower()
    for start, end, _ in find_variable_spans(query):
        # Keep a leading aggregation word visible: in the synonym "average
        # maximum temperature", "average" is the statistic, "maximum
        # temperature" the variable.
        prefix = re.match(r"(?:average|avg|mean)\s+", lowered[start:end])
        if prefix:
            start += prefix.end()
        lowered = lowered[:start] + " " * (end - start) + lowered[end:]
    hits = []
    for name, pattern in AGGREGATION_PATTERNS:
        match = pattern.search(lowered)
        if match:
            hits.append((match.start(), name))
    return [name for _, name in sorted(hits)]


def _group_by(query: str) -> Optional[str]:
    match = WHICH_GROUP_PATTERN.search(query.lower())
    if match:
        word = match.group(1)
        if word.startswith(("cultivar", "variet")):
            return "cultivar"
        if word.startswith("year"):
            return "year"
        return "planting_stage"
    return find_group_by(query)


def _clarify(query_plan_text: str) -> FallbackStatisticsPlan:
    return FallbackStatisticsPlan(
        query_plan=QueryPlan(intent="metadata", filters={}, required_tools=["metadata"],
                             response_type="summary"),
        clarification=query_plan_text,
    )


async def build_fallback_statistics_plan(
    query: str,
    stats,
    metric_clarification,
) -> Optional[FallbackStatisticsPlan]:
    """Plan MIN/MAX/AVG/SUM questions without an LLM.

    Returns None when the question is not a statistical question, so the
    caller's generic fallback handles it.
    """
    lowered = query.lower()
    if DEFINITION_PATTERN.search(lowered):
        return None
    aggregations = detect_aggregations(query)
    if not aggregations:
        return None

    group_by = _group_by(query)
    # "highest average yield" ranks group averages: the average is the
    # statistic and highest/lowest selects among the groups.
    if "AVG" in aggregations and ({"MIN", "MAX"} & set(aggregations)):
        aggregations = ["AVG"]

    resolution = await stats.resolve_metric(query)
    if resolution.ambiguous:
        return _clarify(metric_clarification(resolution.candidates))
    if resolution.code is None:
        available = await stats.get_available_variables()
        wanted = " and ".join(AGGREGATION_NAMES[a] for a in aggregations)
        return _clarify(
            f"Which variable should the {wanted} be calculated for? "
            f"Available variables: {', '.join(sorted(available))}."
        )
    metric = resolution.code

    filters: List[FilterCondition] = find_filters(query)
    stage = next((f for f in filters if f.field == "planting_stage"), None)
    if stage is not None:
        try:
            available_stages = await stats.get_distinct_field_values("planting_stage")
        except Exception:
            available_stages = None
        if available_stages and stage.value not in available_stages:
            return _clarify(
                f"The dataset has no planting stage '{stage.value}' for that wording. "
                f"Available planting stages: {', '.join(map(str, available_stages))} "
                "(e.g. pfrst15 = 15 days after normal planting, pfrst-15 = 15 days before)."
            )

    operations = [
        SemanticOperation(
            operation="aggregate",
            entity="simulation_outputs",
            metric=metric,
            aggregation=aggregation,
            filters=filters,
            group_by=[group_by] if group_by else [],
            independent=True,
        )
        for aggregation in aggregations
    ]
    semantic_plan = SemanticPlan(
        goal=f"{' and '.join(aggregations)} of {metric}",
        intent="aggregate",
        operations=operations,
    )
    query_plan = QueryPlan(
        intent="aggregate",
        metric=metric,
        aggregation=aggregations[0],
        filters={f.field: f.value for f in filters},
        required_tools=["statistics"],
        response_type="summary",
    )
    return FallbackStatisticsPlan(query_plan=query_plan, semantic_plan=semantic_plan)
