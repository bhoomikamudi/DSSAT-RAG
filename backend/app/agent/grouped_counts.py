"""Record counts for grouped comparisons (e.g. average yield by cultivar).

A grouped result carries one count per group (records in that group) and the
records across all groups together. Answers must report both so that a
single group's count ("180 records") is never read as the total ("900").
Works for any number of groups and for unequal group sizes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Singular noun used in "180 records per cultivar".
GROUP_NOUNS = {
    "cultivar": "cultivar",
    "planting_stage": "planting date",
    "year": "year",
    "simulation_year": "year",
    "irrigation": "irrigation condition",
    "nitrogen_level": "nitrogen level",
    "crop": "crop",
}


def _rows(breakdown: Dict[str, Any]) -> List[Tuple[Any, Optional[float], int]]:
    """(group, value, count) from grouped statistics or yearly trend rows."""
    rows = []
    for item in breakdown.get("values") or []:
        group = item.get("group_value", item.get("year"))
        value = item.get("value", item.get("avg_value"))
        count = item.get("count")
        if group is None or count is None:
            continue
        rows.append((group, value, int(count)))
    return rows


def grouped_counts(breakdown: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Total and per-group record counts of a grouped result, or None."""
    if not breakdown:
        return None
    rows = _rows(breakdown)
    if len(rows) < 2:
        return None
    group_by = str(breakdown.get("group_by") or "group")
    noun = GROUP_NOUNS.get(group_by, group_by.replace("_", " "))
    counts = [count for _, _, count in rows]
    total = sum(counts)
    equal = len(set(counts)) == 1
    if equal:
        per_group = f"{counts[0]:,} records per {noun}"
    else:
        per_group = f"records per {noun}: " + ", ".join(f"{group} {count:,}" for group, _, count in rows)
    return {
        "group_by": group_by,
        "noun": noun,
        "groups": len(rows),
        "total": total,
        "equal": equal,
        "counts": {str(group): count for group, _, count in rows},
        "per_group_text": per_group,
        "sentence": (
            f"Grouped comparison by {noun}: {len(rows)} groups, {total:,} output records in total "
            f"({per_group})."
        ),
    }
