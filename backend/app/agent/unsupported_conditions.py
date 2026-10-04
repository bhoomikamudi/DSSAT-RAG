"""Management conditions a question asks for that the pipeline cannot filter on.

The query pipeline can filter on cultivar, planting stage (days relative to
the normal planting date), irrigation, nitrogen level, crop, year and area.
A question about anything else (soil, nitrogen rate, a calendar planting
date, ...) would otherwise be answered over all records without saying so.
These conditions are reported as "not applied" with the reason, so the
answer never implies a filter that was not used.
"""
from __future__ import annotations

import re
from typing import Dict, List

_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
          r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")

# key -> (label, pattern, reason)
UNSUPPORTED: Dict[str, tuple] = {
    "soil": (
        "soil type",
        re.compile(r"\b(?:(?:sandy|clay|clayey|loam|loamy|silty|silt|volcanic|red|black|acidic|alkaline|shallow|deep)\s+)?soils?\b", re.I),
        "the dataset has no soil information",
    ),
    "nitrogen_rate": (
        "nitrogen rate",
        re.compile(r"\b(?:nitrogen|n|fertili[sz]er)\s+(?:application\s+)?(?:rates?|amounts?|doses?|levels?\s+of\s+\d+)\b"
                   r"|\b\d+(?:\.\d+)?\s*kg\s*(?:of\s+)?(?:n|nitrogen|fertili[sz]er)\b", re.I),
        "the dataset has no nitrogen-rate field; every run is HighN",
    ),
    "plant_density": (
        "plant density",
        re.compile(r"\b(?:plant(?:ing)?\s+(?:density|population)|seeding\s+rate|sowing\s+density|plants?\s+per\s+(?:m2|m²|hectare|ha))\b", re.I),
        "the dataset has no plant-density field",
    ),
    "tillage": (
        "tillage",
        re.compile(r"\b(?:tillage|tilled|no-?till|minimum\s+till(?:age)?|ploughing|plowing)\b", re.I),
        "the dataset has no tillage information",
    ),
    "irrigation_amount": (
        "irrigation amount",
        re.compile(r"\birrigation\s+(?:amounts?|depths?|volumes?|schedules?)\b|\b\d+(?:\.\d+)?\s*mm\s+(?:of\s+)?irrigation\b", re.I),
        "the dataset has no irrigation amounts; every run is rainfed (RF)",
    ),
    "planting_calendar": (
        "calendar planting date",
        re.compile(rf"\b(?:planted|planting|sown|sowing|sow)\s+(?:on|in|during|by|before|after|around)\s+"
                   rf"(?:(?:early|mid|late)[\s-]+)?(?:{_MONTH}\b|\d{{1,2}}[/-]\d{{1,2}}\b)", re.I),
        "planting is recorded only relative to the normal planting date (pfrst-30 to pfrst90), not as calendar dates",
    ),
    "ecological_zone": (
        "ecological zone",
        re.compile(r"\b(?:agro-?)?ecological\s+zones?\b|\bagro-?ecozones?\b", re.I),
        "the ecological zone is empty for every record",
    ),
}

# Planner filter fields that correspond to the conditions above.
_FIELD_HINTS = {
    "soil": ("soil",),
    "nitrogen_rate": ("nitrogen_rate", "n_rate", "fertilizer", "fertiliser", "nitrogen_amount"),
    "plant_density": ("density", "population", "seeding"),
    "tillage": ("till",),
    "irrigation_amount": ("irrigation_amount", "irrigation_depth", "irrigation_mm"),
    "planting_calendar": ("planting_date", "sowing_date", "plant_date", "planting_month"),
    "ecological_zone": ("ecological", "ecozone"),
}


def find_unsupported_conditions(text: str) -> List[Dict[str, str]]:
    """Conditions mentioned in the question that cannot be applied as filters."""
    found = []
    for key, (label, pattern, reason) in UNSUPPORTED.items():
        match = pattern.search(text or "")
        if match:
            found.append({"key": key, "condition": label, "phrase": match.group(0).strip(), "reason": reason})
    return found


def classify_dropped_filter(field: str, value) -> Dict[str, str]:
    """Describe a planner filter that was dropped because no field supports it."""
    lowered = str(field).lower()
    for key, hints in _FIELD_HINTS.items():
        if any(hint in lowered for hint in hints):
            label, _, reason = UNSUPPORTED[key]
            return {"key": key, "condition": label, "phrase": f"{field} = {value}", "reason": reason}
    return {"key": f"field:{lowered}", "condition": str(field), "phrase": f"{field} = {value}",
            "reason": "it is not a field the query pipeline can filter on"}


def merge_unapplied(*groups: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """One entry per condition (text mention and dropped filter may coincide)."""
    merged: Dict[str, Dict[str, str]] = {}
    for group in groups:
        for item in group or []:
            merged.setdefault(item["key"], item)
    return list(merged.values())


def describe_unapplied(items: List[Dict[str, str]]) -> str:
    if not items:
        return ""
    parts = [f"{i['condition']} (\"{i['phrase']}\": {i['reason']})" for i in items]
    return ("Not applied: " + "; ".join(parts)
            + ". The result covers all matching records regardless of "
            + ("this condition." if len(items) == 1 else "these conditions."))
