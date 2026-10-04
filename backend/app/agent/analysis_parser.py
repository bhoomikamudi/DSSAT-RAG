"""Deterministic natural-language normalizer for analysis questions.

Maps paraphrases ("Does precipitation affect yield?", "Is PRCP correlated with
HWAM?", "How does rainfall influence harvested yield?") onto one canonical
AnalysisRequest, so different wording never creates a separate workflow.

The parser only recognizes a question as an analysis when it contains an
analysis cue (correlation/regression/relationship wording) together with
recognizable variables. Everything else returns None and continues through
the existing planner unchanged.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from app.agent.location_parser import mentions_location_reset
from app.agent.models import AnalysisRequest, FilterCondition
from app.services.analysis_vocabulary import (
    ANALYSIS_VARIABLES,
    CULTIVAR_CODES,
    cultivar_alias_pairs,
    find_variable_mentions,
)


# =============================================================================
# CUE PATTERNS
# =============================================================================

QUADRATIC_PATTERN = re.compile(
    r"\b(quadratic|second[- ]order|polynomial|curvilinear|curved|"
    r"non[- ]?linear|parabol\w*)\b"
)
LINEAR_PATTERN = re.compile(
    r"\b(linear|regression|regress(?:ed|ing)?|slope|line of best fit|"
    r"best[- ]fit line|trend ?line|fit a line)\b"
)
DESCRIPTIVE_PATTERN = re.compile(
    r"\b(descriptive statistics|summary statistics|statistical summary|"
    r"describe|summari[sz]e|distribution|spread|variability)\b"
)
CORRELATION_PATTERN = re.compile(r"\bcorrelat\w*\b")
RELATIONSHIP_PATTERN = re.compile(
    r"\b(correlat\w*|relationship|relation|related|relate[sd]?|association|"
    r"associated|affects?|affected|affecting|effects?|influenc\w*|impacts?|"
    r"impacted|depend\w*|versus|vs|against|analy[sz]e|analysis|linked|"
    r"connection|respond\w*|sensitiv\w*|drives?|driven)\b"
)

# Questions that belong to the existing aggregate/definition workflows.
AGGREGATE_PATTERN = re.compile(
    r"\b(average|avg|mean|minimum|maximum|min|max|highest|lowest|total|sum|"
    r"how many|count|define|meaning|what does)\b"
)
FOLLOW_UP_PATTERN = re.compile(
    r"\b(same|instead|now|also|again|what about|how about|repeat|redo|only|"
    r"just|then|too|as well)\b"
)
RESET_ALL_FILTERS_PATTERN = re.compile(
    r"\b(all data|all simulations|all records|without (?:any )?filters?|"
    r"no filters?|remove (?:the )?filters?)\b"
)

# Factor terms: categorical simulation attributes asked about as a cause.
FACTOR_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("nitrogen_level", re.compile(
        r"\b(nitrogen|n[- ]rates?|n levels?|fertili[sz]\w*)\b")),
    ("irrigation", re.compile(
        r"\b(irrigat\w*|water supply|water management)\b")),
    ("planting_stage", re.compile(
        r"\b(planting (?:dates?|times?|stages?|windows?)|sowing (?:dates?|times?)|planting)\b")),
    ("cultivar", re.compile(
        r"\b(cultivars?|variet(?:y|ies)|hybrids?|maturity (?:groups?|class(?:es)?))\b")),
]

GROUP_BY_PATTERN = re.compile(
    r"\b(?:separately for(?: each)?|for each|for every|within each|broken down by|"
    r"split by|grouped by|group by|by|per|each|across)\s+(?:the\s+)?"
    r"(cultivars?|variet(?:y|ies)|years?|planting (?:dates?|stages?|times?)|"
    r"sowing dates?|crops?)\b"
)
GROUP_BY_WORDS: Dict[str, str] = {
    "cultivar": "cultivar",
    "variet": "cultivar",
    "year": "year",
    "planting": "planting_stage",
    "sowing": "planting_stage",
    "crop": "crop",
}

# Phrases that name filters and must not be read as variables ("rain-fed").
RAINFED_PATTERN = re.compile(
    r"\b(rain[- ]?fed|non[- ]irrigated|dryland|without irrigation)\b"
)
IRRIGATED_PATTERN = re.compile(
    r"(?<!non-)(?<!non )\b(irrigated|under irrigation|with irrigation|full irrigation)\b"
)

# Cultivar codes other than BASE are not English words, so any case is safe.
UNAMBIGUOUS_CULTIVAR_PATTERN = re.compile(r"\b(vlng|vsht|lng|sht)\b")
# "base" is also an ordinary word ("base temperature"), so lowercase or
# title-case "base" only counts where the wording marks it as a cultivar:
# after a filter word, before a cultivar noun, or next to another code.
_NOT_CULTIVAR_AFTER_BASE = (
    r"(?!\s+(?:temperatures?|temp|case|years?|periods?|lines?|scenarios?|"
    r"values?|levels?|rates?))"
)
BASE_CONTEXT_PATTERNS = [
    re.compile(
        r"\b(?:for|of|with|under|using|only|just|about|and|or|vs\.?|versus)\s+"
        r"(?:the\s+)?(base)\b" + _NOT_CULTIVAR_AFTER_BASE
    ),
    re.compile(r"\b(base)\s+(?:cultivars?|variet(?:y|ies)|hybrids?)\b"),
    re.compile(r"\b(base)\s*(?:,|and|or|vs\.?|versus)\s*(?:vlng|vsht|lng|sht)\b"),
]
CULTIVAR_ORDER = list(CULTIVAR_CODES)

YEAR_RANGE_PATTERNS = [
    re.compile(
        r"\b(?:from|between)\s+((?:19|20)\d{2})\s+(?:to|and|-|–|through|until)\s+((?:19|20)\d{2})\b"
    ),
    re.compile(r"\b((?:19|20)\d{2})\s*(?:-|–|to|through)\s*((?:19|20)\d{2})\b"),
]
YEAR_PATTERN = re.compile(r"\b((?:19|20)\d{2})\b")

# Patterns that capture the two terms of a relationship question, used to
# report terms that are not available in the dataset.
TERM_PATTERNS: List[Tuple[re.Pattern, bool]] = [
    # (pattern, first group is the independent variable)
    (re.compile(r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\s+(?:for|in|under|across|among|with|during|by|separately)\b|[?.!,]|$)"), True),
    (re.compile(r"\b(?:effect|impact|influence)\s+of\s+(.+?)\s+on\s+(.+?)(?:\s+(?:for|in|under|across|among|with|during|by)\b|[?.!,]|$)"), True),
    (re.compile(r"\b(?:does|do|did|can|will)\s+(.+?)\s+(?:affect|influence|impact|drive|change)\s+(.+?)(?:\s+(?:for|in|under|across|among|with|during|by)\b|[?.!,]|$)"), True),
    (re.compile(r"\b(?:is|are)\s+(.+?)\s+(?:correlated|related|associated|linked)\s+(?:with|to)\s+(.+?)(?:\s+(?:for|in|under|across|among|during|by)\b|[?.!,]|$)"), True),
]
STOP_WORDS = {"the", "a", "an", "of", "and", "on", "in", "there", "any", "is"}


# =============================================================================
# EXTRACTION HELPERS
# =============================================================================

def _mask(text: str, pattern: re.Pattern) -> str:
    return pattern.sub(lambda m: " " * len(m.group(0)), text)


def find_variables(text: str) -> List[Tuple[int, str]]:
    """Return (position, canonical code) for each variable mention, in order."""
    return find_variable_mentions(text)


def find_factor(text: str) -> Optional[str]:
    lowered = _mask(text.lower(), RAINFED_PATTERN)
    hits = []
    for field, pattern in FACTOR_PATTERNS:
        match = pattern.search(lowered)
        if match:
            hits.append((match.start(), field))
    return min(hits)[1] if hits else None


def find_group_by(text: str) -> Optional[str]:
    match = GROUP_BY_PATTERN.search(text.lower())
    if not match:
        return None
    word = match.group(1)
    for prefix, field in GROUP_BY_WORDS.items():
        if word.startswith(prefix):
            return field
    return None


def find_operation(text: str) -> Tuple[Optional[str], bool]:
    """Return (operation, explicit) where explicit means a named method."""
    lowered = text.lower()
    if QUADRATIC_PATTERN.search(lowered):
        return "quadratic_regression", True
    if LINEAR_PATTERN.search(lowered):
        return "linear_regression", True
    if DESCRIPTIVE_PATTERN.search(lowered):
        return "descriptive_statistics", True
    if CORRELATION_PATTERN.search(lowered):
        return "correlation", True
    if RELATIONSHIP_PATTERN.search(lowered):
        return "correlation", False
    return None, False


def _condition(field: str, values: List) -> FilterCondition:
    if len(values) == 1:
        return FilterCondition(field=field, operator="=", value=values[0])
    return FilterCondition(field=field, operator="IN", value=values)


def _alias_pattern(phrase: str) -> re.Pattern:
    body = r"[- ]".join(re.escape(word) for word in phrase.split())
    return re.compile(r"(?<![\w-])" + body + r"(?![\w-])")


def find_cultivars(text: str) -> List[str]:
    """Return canonical cultivar codes mentioned as filters.

    Codes match in any case (BASE/base/Base), verified aliases map to their
    code ("standard cultivar" -> BASE, "long-season" -> LNG), and the result
    is in canonical order so equivalent wording yields an identical filter.
    """
    remaining = text.lower()
    found = set()

    for phrase, code in cultivar_alias_pairs():
        pattern = _alias_pattern(phrase)
        if pattern.search(remaining):
            found.add(code)
            remaining = _mask(remaining, pattern)

    for match in UNAMBIGUOUS_CULTIVAR_PATTERN.finditer(remaining):
        found.add(match.group(1).upper())

    if re.search(r"\bBASE\b", text) or any(
        pattern.search(remaining) for pattern in BASE_CONTEXT_PATTERNS
    ):
        found.add("BASE")

    return [code for code in CULTIVAR_ORDER if code in found]


def find_filters(text: str) -> List[FilterCondition]:
    """Extract dataset filters in a fixed field order."""
    lowered = text.lower()
    filters: List[FilterCondition] = []

    if re.search(r"\b(maize|corn)\b", lowered):
        filters.append(FilterCondition(field="crop", operator="=", value="MZ"))

    cultivars = find_cultivars(text)
    if cultivars:
        filters.append(_condition("cultivar", cultivars))

    if RAINFED_PATTERN.search(lowered):
        filters.append(FilterCondition(field="irrigation", operator="=", value="RF"))
    elif IRRIGATED_PATTERN.search(lowered):
        filters.append(FilterCondition(field="irrigation", operator="=", value="IR"))

    if re.search(r"\bhigh[- ]?n(?:itrogen)?\b", lowered):
        filters.append(FilterCondition(field="nitrogen_level", operator="=", value="HighN"))
    elif re.search(r"\blow[- ]?n(?:itrogen)?\b", lowered):
        filters.append(FilterCondition(field="nitrogen_level", operator="=", value="LowN"))

    stage = _find_planting_stage(lowered)
    if stage:
        filters.append(FilterCondition(field="planting_stage", operator="=", value=stage))

    year_filter = _find_year_filter(lowered)
    if year_filter:
        filters.append(year_filter)

    return filters


_NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
                 "five": 5, "six": 6, "eight": 8, "ten": 10, "twelve": 12}
_UNIT_DAYS = {"day": 1, "week": 7, "fortnight": 14, "month": 30}
PLANTING_OFFSET_PATTERN = re.compile(
    r"\b(?P<count>\d+|a|an|one|two|three|four|five|six|eight|ten|twelve)[\s-]*"
    r"(?P<unit>days?|weeks?|fortnights?|months?)\s+"
    r"(?P<direction>late|later|after|delayed|early|earlier|before|sooner|ahead)\b"
)
# Equivalences stated by the project's planner rules ("two weeks late" or
# "14 days late" maps to pfrst15). Other offsets are used as-is and checked
# against the dataset's planting stages by the caller.
PLANTING_DAY_EQUIVALENTS = {14: 15}


def planting_offset_days(lowered: str) -> Optional[int]:
    """Signed planting offset in days from wording, or None."""
    match = PLANTING_OFFSET_PATTERN.search(lowered)
    if not match:
        return None
    count = match.group("count")
    number = int(count) if count.isdigit() else _NUMBER_WORDS[count]
    unit = match.group("unit").rstrip("s")
    days = number * _UNIT_DAYS[unit]
    days = PLANTING_DAY_EQUIVALENTS.get(days, days)
    direction = match.group("direction")
    return -days if direction in {"early", "earlier", "before", "sooner", "ahead"} else days


def _find_planting_stage(lowered: str) -> Optional[str]:
    literal = re.search(r"\bpfrst-?\d+\b", lowered)
    if literal:
        return literal.group(0)
    if re.search(
        r"\b(normal|reference|standard|on[- ]time|usual) planting\b|\bplanted on time\b",
        lowered,
    ):
        return "pfrst0"
    days = planting_offset_days(lowered)
    if days is not None:
        return f"pfrst{days}"
    return None


def _find_year_filter(lowered: str) -> Optional[FilterCondition]:
    for pattern in YEAR_RANGE_PATTERNS:
        match = pattern.search(lowered)
        if match:
            start, end = sorted((int(match.group(1)), int(match.group(2))))
            return FilterCondition(field="year", operator="BETWEEN", value=[start, end])
    years: List[int] = []
    for match in YEAR_PATTERN.finditer(lowered):
        year = int(match.group(1))
        if year not in years:
            years.append(year)
    if years:
        return _condition("year", years)
    return None


def _unresolved_terms(text: str) -> List[Tuple[str, bool]]:
    """Return raw terms from relationship phrasing that are not variables.

    Each item is (term, is_independent).
    """
    lowered = text.lower()
    for pattern, first_is_x in TERM_PATTERNS:
        match = pattern.search(lowered)
        if not match:
            continue
        terms = []
        for index, raw in enumerate(match.groups()):
            term = " ".join(w for w in raw.strip().split() if w not in STOP_WORDS)
            if not term:
                continue
            if find_variables(term) or find_factor(term):
                continue
            is_x = (index == 0) == first_is_x
            terms.append((term, is_x))
        return terms
    return []


def _orient(
    text: str,
    mentions: List[Tuple[int, str]],
) -> Tuple[str, str]:
    """Choose (x, y) for two variables.

    Weather drivers are independent and crop responses dependent; otherwise
    phrasing decides ("B vs A" and "regress B on A" put B on y), falling back
    to mention order.
    """
    first, second = mentions[0][1], mentions[1][1]
    roles = {code: ANALYSIS_VARIABLES[code]["role"] for code in (first, second)}
    if roles[first] != roles[second]:
        return (first, second) if roles[first] == "driver" else (second, first)

    lowered = text.lower()
    if re.search(r"\b(vs\.?|versus|against)\b", lowered) or re.search(
        r"\bregress\w*\b.+\bon\b", lowered
    ):
        return second, first
    return first, second


# =============================================================================
# PUBLIC API
# =============================================================================

def parse_analysis_request(
    query: str,
    previous: Optional[AnalysisRequest] = None,
) -> Optional[AnalysisRequest]:
    """Normalize an analysis question into a canonical AnalysisRequest.

    Args:
        query: User question.
        previous: The last analysis request in this conversation, used to
            complete follow-ups such as "now fit a quadratic instead".

    Returns:
        An AnalysisRequest, or None when the question is not an analysis
        question (so the existing planner handles it).
    """
    operation, explicit = find_operation(query)
    mentions = find_variables(query)
    filters = find_filters(query)
    group_by = find_group_by(query)
    filter_fields = {condition.field for condition in filters}

    if len(mentions) >= 2 and operation:
        x_variable, y_variable = _orient(query, mentions)
        return AnalysisRequest(
            operation=operation,
            x_variable=x_variable,
            y_variable=y_variable,
            filters=filters,
            group_by=group_by,
        )

    if len(mentions) == 1 and operation:
        (_, variable), = mentions

        if operation == "descriptive_statistics":
            return AnalysisRequest(
                operation=operation,
                y_variable=variable,
                filters=filters,
                group_by=group_by,
            )

        factor = find_factor(query)
        if factor and factor not in filter_fields:
            # "Does nitrogen rate affect yield?" -> compare yield across the
            # factor's values; the data decides whether that is possible.
            return AnalysisRequest(
                operation="descriptive_statistics",
                y_variable=variable,
                filters=filters,
                group_by=factor,
                requires_group_variation=True,
            )

        unresolved = _unresolved_terms(query)
        if unresolved:
            term, is_x = unresolved[0]
            return AnalysisRequest(
                operation=operation,
                x_variable=term if is_x else variable,
                y_variable=variable if is_x else term,
                filters=filters,
                group_by=group_by,
            )

    if previous is not None:
        return _follow_up(query, previous, operation, explicit, mentions, filters, group_by)

    return None


def _follow_up(
    query: str,
    previous: AnalysisRequest,
    operation: Optional[str],
    explicit: bool,
    mentions: List[Tuple[int, str]],
    filters: List[FilterCondition],
    group_by: Optional[str],
) -> Optional[AnalysisRequest]:
    """Complete a follow-up question from the previous analysis request."""
    lowered = query.lower()

    if AGGREGATE_PATTERN.search(lowered) and not explicit:
        return None
    if len(mentions) >= 2:
        return None

    reset_all = bool(RESET_ALL_FILTERS_PATTERN.search(lowered))
    reset_location = mentions_location_reset(query)
    reset_cultivar = bool(re.search(r"\b(all|every) (cultivars?|variet(?:y|ies))\b", lowered))
    # Asking for all data is itself a follow-up ("use all data").
    has_cue = bool(FOLLOW_UP_PATTERN.search(lowered)) or explicit or reset_all or reset_location
    changes_something = bool(
        explicit or filters or group_by or mentions or reset_all or reset_cultivar
        or reset_location
    )
    if not (has_cue and changes_something):
        return None

    new_operation = operation if explicit else previous.operation

    x_variable, y_variable = previous.x_variable, previous.y_variable
    if len(mentions) == 1:
        code = mentions[0][1]
        if ANALYSIS_VARIABLES[code]["role"] == "driver":
            x_variable = code
        else:
            y_variable = code

    merged: Dict[str, FilterCondition] = {}
    if not reset_all:
        merged = {condition.field: condition for condition in previous.filters}
    if reset_cultivar:
        merged.pop("cultivar", None)
    for condition in filters:
        merged[condition.field] = condition

    new_group_by = group_by
    if new_group_by is None and not reset_all:
        new_group_by = previous.group_by
        new_filter_fields = {condition.field for condition in filters}
        if new_group_by in new_filter_fields:
            new_group_by = None

    return AnalysisRequest(
        operation=new_operation,
        x_variable=x_variable,
        y_variable=y_variable,
        filters=list(merged.values()),
        group_by=new_group_by,
        include_plot=previous.include_plot,
        max_points=previous.max_points,
        requires_group_variation=previous.requires_group_variation,
        spatial=_carried_spatial(previous, reset_all or reset_location),
    )


def _carried_spatial(previous: AnalysisRequest, reset: bool):
    """The area a follow-up inherits from the previous question.

    Only an area named in a question ("near Kitale", coordinates) carries
    over. A map point is re-sent by the frontend while it is selected, so
    inheriting it would keep filtering after the user cleared the map.
    A new location in the current request replaces this in the executor.
    """
    if reset or previous.spatial is None or previous.spatial.source == "map":
        return None
    return previous.spatial
