"""Controlled vocabulary for the Python analysis layer.

Single source of truth for which DSSAT variables, operations, group-by fields
and filter fields an analysis request may use. The planner, the Pydantic
models and the analysis service all normalize through these tables so that
paraphrased questions resolve to the same canonical request.

This module intentionally has no project imports to avoid import cycles.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# VARIABLES
# =============================================================================

# Allowlist of numeric DSSAT output variables that may be analyzed.
# "role" drives the default x/y orientation: drivers (weather) are treated as
# independent variables and responses (crop outputs) as dependent variables.
ANALYSIS_VARIABLES: Dict[str, Dict[str, Any]] = {
    "HWAM": {
        "label": "Yield",
        "unit": "kg/ha",
        "role": "response",
        "synonyms": [
            "harvested yield",
            "harvest yield",
            "grain yield",
            "crop yield",
            "maize yield",
            "yields",
            "yield",
            "hwam",
        ],
    },
    "HWAH": {
        "label": "Harvested weight at harvest",
        "unit": "kg/ha",
        "role": "response",
        "synonyms": ["harvested weight", "harvest weight", "hwah"],
    },
    "CWAM": {
        "label": "Above-ground biomass at maturity",
        "unit": "kg/ha",
        "role": "response",
        "synonyms": ["crop biomass", "above-ground biomass", "biomass", "cwam"],
    },
    "GNAM": {
        "label": "Grain number",
        "unit": None,
        "role": "response",
        "synonyms": ["grain number", "grain count", "number of grains", "gnam"],
    },
    "PRCP": {
        "label": "Precipitation",
        "unit": "mm",
        "role": "driver",
        "synonyms": [
            "seasonal precipitation",
            "seasonal rainfall",
            "precipitation",
            "rainfall",
            "rain",
            "prcp",
        ],
    },
    "TMAXA": {
        "label": "Average maximum temperature",
        "unit": "°C",
        "role": "driver",
        "synonyms": [
            "average maximum temperature",
            "maximum temperature",
            "max temperature",
            "max temp",
            "tmax",
            "tmaxa",
        ],
    },
    "TMINA": {
        "label": "Average minimum temperature",
        "unit": "°C",
        "role": "driver",
        "synonyms": [
            "average minimum temperature",
            "minimum temperature",
            "min temperature",
            "min temp",
            "tmin",
            "tmina",
        ],
    },
}


# =============================================================================
# CULTIVARS
# =============================================================================

# Canonical cultivar codes as stored in simulations.cultivar. Meanings come
# from the reference_codes table / CDEService.PROJECT_MAPPINGS ("project
# run-name convention"); "standard cultivar" for BASE comes from the planner
# prompt's DATASET MAPPINGS ("BASE = baseline or standard cultivar").
# Aliases are lowercase phrases; a space also matches a hyphen when parsing.
# Bare "standard" is deliberately not an alias: it also appears in
# "standard planting" (pfrst0) and "standard deviation".
CULTIVAR_CODES: Dict[str, Dict[str, Any]] = {
    "BASE": {
        "meaning": "Baseline cultivar",
        "aliases": [
            "baseline",
            "standard cultivar",
            "standard variety",
            "standard hybrid",
        ],
    },
    "LNG": {"meaning": "Long-season cultivar", "aliases": ["long season"]},
    "SHT": {"meaning": "Short-season cultivar", "aliases": ["short season"]},
    "VLNG": {"meaning": "Very-long-season cultivar", "aliases": ["very long season"]},
    "VSHT": {"meaning": "Very-short-season cultivar", "aliases": ["very short season"]},
}

CULTIVAR_NOUNS = ("cultivars", "cultivar", "varieties", "variety", "hybrids", "hybrid")


# =============================================================================
# OPERATIONS
# =============================================================================

ANALYSIS_OPERATIONS: Tuple[str, ...] = (
    "correlation",
    "linear_regression",
    "quadratic_regression",
    "descriptive_statistics",
)

OPERATION_ALIASES: Dict[str, str] = {
    "correlation": "correlation",
    "correlate": "correlation",
    "pearson": "correlation",
    "pearson_correlation": "correlation",
    "relationship": "correlation",
    "linear_regression": "linear_regression",
    "linear": "linear_regression",
    "regression": "linear_regression",
    "ols": "linear_regression",
    "quadratic_regression": "quadratic_regression",
    "quadratic": "quadratic_regression",
    "polynomial": "quadratic_regression",
    "polynomial_regression": "quadratic_regression",
    "second_order": "quadratic_regression",
    "descriptive_statistics": "descriptive_statistics",
    "descriptive": "descriptive_statistics",
    "describe": "descriptive_statistics",
    "summary": "descriptive_statistics",
    "summary_statistics": "descriptive_statistics",
}

# Minimum number of valid (x, y) pairs needed per operation.
MIN_SAMPLE_SIZE: Dict[str, int] = {
    "correlation": 3,
    "linear_regression": 3,
    "quadratic_regression": 4,
    "descriptive_statistics": 1,
}


# =============================================================================
# GROUPING AND FILTERS
# =============================================================================

# Canonical group-by field -> Simulation column name.
GROUP_BY_FIELDS: Dict[str, str] = {
    "cultivar": "cultivar",
    "year": "simulation_year",
    "planting_stage": "planting_stage",
    "crop": "crop",
    "irrigation": "irrigation",
    "nitrogen_level": "nitrogen_level",
    "state": "state",
    "district": "district",
    "country": "country",
}

GROUP_BY_ALIASES: Dict[str, str] = {
    "cultivar": "cultivar",
    "cultivars": "cultivar",
    "variety": "cultivar",
    "varieties": "cultivar",
    "year": "year",
    "years": "year",
    "simulation_year": "year",
    "planting_stage": "planting_stage",
    "planting_date": "planting_stage",
    "planting": "planting_stage",
    "sowing_date": "planting_stage",
    "crop": "crop",
    "irrigation": "irrigation",
    "nitrogen": "nitrogen_level",
    "nitrogen_level": "nitrogen_level",
    "nitrogen_rate": "nitrogen_level",
    "state": "state",
    "district": "district",
    "country": "country",
}

# Canonical filter field names accepted by the analysis layer.
FILTER_FIELDS: Tuple[str, ...] = (
    "crop",
    "cultivar",
    "irrigation",
    "nitrogen_level",
    "planting_stage",
    "year",
    "state",
    "district",
    "country",
)

FILTER_FIELD_ALIASES: Dict[str, str] = {
    "nitrogen": "nitrogen_level",
    "simulation_year": "year",
}

# Human-readable names for explanations.
FIELD_LABELS: Dict[str, str] = {
    "cultivar": "cultivar",
    "year": "year",
    "planting_stage": "planting stage",
    "crop": "crop",
    "irrigation": "irrigation regime",
    "nitrogen_level": "nitrogen level",
    "state": "state",
    "district": "district",
    "country": "country",
}


# =============================================================================
# NORMALIZERS
# =============================================================================

def _key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _synonym_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for code, spec in ANALYSIS_VARIABLES.items():
        lookup[code.lower()] = code
        for synonym in spec["synonyms"]:
            lookup[synonym.lower()] = code
    return lookup


_VARIABLE_LOOKUP = _synonym_lookup()


def normalize_variable(value: Optional[str]) -> Optional[str]:
    """Map a variable code or synonym to its canonical code.

    Unknown values are returned unchanged (stripped) so validation can report
    them to the user instead of silently discarding them.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return _VARIABLE_LOOKUP.get(text.lower(), text)


def normalize_operation(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return OPERATION_ALIASES.get(_key(str(value)), str(value).strip())


def normalize_group_by(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return GROUP_BY_ALIASES.get(_key(text), text)


def normalize_filter_field(value: str) -> str:
    key = _key(value)
    return FILTER_FIELD_ALIASES.get(key, key)


def _cultivar_key(value: str) -> str:
    text = " ".join(value.strip().lower().replace("-", " ").replace("_", " ").split())
    for noun in CULTIVAR_NOUNS:
        if text.endswith(" " + noun):
            return text[: -len(noun) - 1]
    return text


def cultivar_alias_pairs() -> List[Tuple[str, str]]:
    """Return (alias phrase, code) pairs sorted longest-phrase first."""
    pairs = [
        (alias, code)
        for code, spec in CULTIVAR_CODES.items()
        for alias in spec["aliases"]
    ]
    return sorted(pairs, key=lambda item: len(item[0]), reverse=True)


def normalize_cultivar(value: Any) -> Any:
    """Map a cultivar code (any case) or verified alias to its canonical code.

    Used for values already known to be cultivars (a "cultivar" filter), so a
    trailing noun is optional here ("standard" == "standard cultivar"); free
    text is matched more strictly by the analysis parser. Lists are
    normalized element-wise. Unknown values are returned unchanged so the
    database can report that no such cultivar exists.
    """
    if isinstance(value, list):
        return [normalize_cultivar(item) for item in value]
    if not isinstance(value, str):
        return value
    key = _cultivar_key(value)
    for code, spec in CULTIVAR_CODES.items():
        if key == code.lower():
            return code
        for alias in spec["aliases"]:
            if key in (alias, _cultivar_key(alias)):
                return code
    return value


def variable_label(code: Optional[str]) -> str:
    if not code:
        return ""
    spec = ANALYSIS_VARIABLES.get(code)
    return spec["label"] if spec else code


def variable_unit(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    spec = ANALYSIS_VARIABLES.get(code)
    return spec["unit"] if spec else None


def variable_synonym_pairs() -> List[Tuple[str, str]]:
    """Return (phrase, code) pairs sorted longest-phrase first for matching."""
    return sorted(
        _VARIABLE_LOOKUP.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )


# Filter phrases that contain variable words but are not variables
# ("rain-fed" is an irrigation regime, not precipitation).
NON_VARIABLE_PHRASES = re.compile(
    r"\b(rain[- ]?fed|non[- ]irrigated|dryland|without irrigation)\b", re.I
)


def find_variable_spans(text: str) -> List[Tuple[int, int, str]]:
    """Return (start, end, canonical code) for every variable mention.

    Longest synonyms match first ("harvested weight" before "harvest") and
    each part of the text is used once.
    """
    lowered = NON_VARIABLE_PHRASES.sub(lambda m: " " * len(m.group(0)), text.lower())
    taken = [False] * len(lowered)
    found: List[Tuple[int, int, str]] = []

    for phrase, code in variable_synonym_pairs():
        pattern = re.compile(r"(?<![\w])" + re.escape(phrase) + r"(?![\w])")
        for match in pattern.finditer(lowered):
            start, end = match.span()
            if any(taken[start:end]):
                continue
            for index in range(start, end):
                taken[index] = True
            found.append((start, end, code))
    return sorted(found)


def find_variable_mentions(text: str) -> List[Tuple[int, str]]:
    """Return (position, canonical code) for each variable, first mention only."""
    ordered: List[Tuple[int, str]] = []
    seen = set()
    for position, _, code in find_variable_spans(text):
        if code not in seen:
            seen.add(code)
            ordered.append((position, code))
    return ordered
