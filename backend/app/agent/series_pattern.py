"""The actual shape of a grouped result across its natural order.

Grouped statistics arrive sorted by value (highest first), so the narrative
writer cannot see whether yields rise or fall along the planting calendar or
over the years. This module restores the natural order (planting dates by
days relative to the normal date, years ascending), classifies the pattern
(steadily increasing, steadily decreasing, flat, or mixed) and states the
highs and lows. Groups with no natural order (e.g. cultivars) get no trend.

claims_steady_trend() flags narrative that asserts a steady/progressive
change, so it can be rejected when the data do not show one.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.services.display_format import format_stat
from app.services.management_service import planting_offset

ORDERED_GROUPS = {"planting_stage": "planting date", "year": "year", "simulation_year": "year"}


def _order_key(group_by: str, label: Any) -> Optional[float]:
    if group_by == "planting_stage":
        offset = planting_offset(label)
        return float(offset) if offset is not None else None
    try:
        return float(label)
    except (TypeError, ValueError):
        return None


def ordered_series(breakdown: Optional[Dict[str, Any]]) -> Optional[List[Tuple[Any, float]]]:
    """(group, value) in natural order, or None for unordered groupings."""
    if not breakdown:
        return None
    group_by = str(breakdown.get("group_by") or "")
    if group_by not in ORDERED_GROUPS:
        return None
    rows = []
    for item in breakdown.get("values") or []:
        label = item.get("group_value", item.get("year"))
        value = item.get("value", item.get("avg_value"))
        key = _order_key(group_by, label)
        if label is None or value is None or key is None:
            return None  # cannot order reliably; make no trend statement
        rows.append((key, label, float(value)))
    if len(rows) < 3:
        return None
    return [(label, value) for _, label, value in sorted(rows)]


def describe_pattern(breakdown: Optional[Dict[str, Any]], unit: str = "") -> Optional[Dict[str, Any]]:
    """Pattern of a grouped result in natural order, with a plain sentence."""
    series = ordered_series(breakdown)
    if series is None:
        return None
    noun = ORDERED_GROUPS[str(breakdown.get("group_by"))]
    unit = f" {unit}" if unit else ""
    steps = [b[1] - a[1] for a, b in zip(series, series[1:])]
    signs = [1 if s > 0 else -1 if s < 0 else 0 for s in steps]
    nonzero = [s for s in signs if s]
    if not nonzero:
        shape = "flat"
    elif all(s >= 0 for s in signs):
        shape = "increasing"
    elif all(s <= 0 for s in signs):
        shape = "decreasing"
    else:
        shape = "mixed"
    reversals = sum(1 for a, b in zip(nonzero, nonzero[1:]) if a != b)
    high = max(series, key=lambda item: item[1])
    low = min(series, key=lambda item: item[1])
    order = f"{series[0][0]} to {series[-1][0]}"
    fmt = lambda item: f"{item[0]} ({format_stat(item[1])}{unit})"
    if shape == "mixed":
        sentence = (
            f"In {noun} order ({order}) the values are not monotonic: they rise and fall, changing "
            f"direction {reversals} time{'s' if reversals != 1 else ''}. Highest: {fmt(high)}; lowest: {fmt(low)}. "
            f"Do not describe a steady or progressive increase or decrease."
        )
    elif shape == "flat":
        sentence = f"In {noun} order ({order}) the values do not change."
    else:
        verb = "increase" if shape == "increasing" else "decrease"
        sentence = (
            f"In {noun} order ({order}) the values {verb} steadily from {fmt(series[0])} to {fmt(series[-1])}."
        )
    return {
        "group_by": breakdown.get("group_by"),
        "shape": shape,
        "monotonic": shape in {"increasing", "decreasing", "flat"},
        "reversals": reversals,
        "ordered": [{"group": label, "value": value} for label, value in series],
        "highest": high[0],
        "lowest": low[0],
        "sentence": sentence,
    }


# Narrative that asserts a steady, progressive or one-directional change.
_STEADY_TREND = re.compile(
    r"\b(?:progressive(?:ly)?|steadil?y|steady|consistent(?:ly)?\s+(?:de|in)creas\w*|"
    r"continu(?:ous(?:ly)?|e[sd]?)\s+to\s+(?:de|in)creas\w*|monoton\w*|"
    r"(?:de|in)creas\w*\s+(?:with\s+)?(?:each|every)\s+\w+|"
    r"(?:declin|decreas|increas|fall|fell|drop|rise|rose|ris)\w*\s+(?:with|as)\s+(?:later|earlier|each|every)\b|"
    r"the\s+(?:later|earlier)\b.{0,40}?\bthe\s+(?:lower|higher|less|more)\b)",
    re.I,
)


_LOW_WORDS = re.compile(r"\b(?:lowest|minimum|smallest|least)\b", re.I)
_HIGH_WORDS = re.compile(r"\b(?:highest|maximum|largest|greatest|best|top)\b", re.I)
# Clauses: sentences, plus "…, with …", "… while …", "… — …".
_CLAUSE_SPLIT = re.compile(r"[.;!?\n]|\s[—–]\s|,\s*(?:with|while|whereas|but|and)\b")


def extreme_label_errors(breakdown: Optional[Dict[str, Any]], answer: str) -> List[str]:
    """Clauses that call the wrong group the highest or lowest.

    Only clauses that name exactly one group and exactly one of
    highest/lowest are checked, so lists and ranges are left alone.
    """
    rows = []
    for item in (breakdown or {}).get("values") or []:
        label = item.get("group_value", item.get("year"))
        value = item.get("value", item.get("avg_value"))
        if label is not None and value is not None:
            rows.append((str(label), float(value)))
    if len(rows) < 2:
        return []
    highest = max(rows, key=lambda item: item[1])
    lowest = min(rows, key=lambda item: item[1])
    patterns = {label: re.compile(rf"(?<![\w-]){re.escape(label)}(?![\w])") for label, _ in rows}
    errors = []
    for clause in _CLAUSE_SPLIT.split(answer or ""):
        named = [label for label, pattern in patterns.items() if pattern.search(clause)]
        says_low, says_high = bool(_LOW_WORDS.search(clause)), bool(_HIGH_WORDS.search(clause))
        if len(named) != 1 or says_low == says_high:
            continue
        expected = lowest if says_low else highest
        if named[0] != expected[0]:
            errors.append(
                f"it names {named[0]} as the {'lowest' if says_low else 'highest'}, but the "
                f"{'lowest' if says_low else 'highest'} is {expected[0]} ({format_stat(expected[1])})"
            )
    return errors


# "without a steady trend", "no progressive decline", "did not fall steadily"
_NEGATION = re.compile(
    r"\b(?:no|not|without|never|neither|nor|rather\s+than|instead\s+of|isn't|aren't|wasn't|weren't|"
    r"doesn't|don't|didn't|lacks?|lacking)\b[^.;:!?]*$",
    re.I,
)


def claims_steady_trend(text: str) -> Optional[str]:
    """The phrase that claims a steady trend, if any (negated mentions excluded)."""
    text = text or ""
    for match in _STEADY_TREND.finditer(text):
        clause_start = max(text.rfind(mark, 0, match.start()) for mark in ".;:!?\n")
        before = text[clause_start + 1:match.start()]
        if _NEGATION.search(before[-60:]):
            continue
        return match.group(0)
    return None
