"""Management variables: what the dataset contains, and what a result covers.

Management fields are the simulation settings a user can ask about or filter
on: cultivar, planting date (planting_stage), irrigation and nitrogen level.
Every value shown comes from the simulations table; nothing is hard-coded,
so the catalogue and the per-result description always match the data.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.simulation import Simulation
from app.services.analysis_vocabulary import CULTIVAR_CODES
from app.services.statistics_service import StatisticsService

# (field, column, label) in display order.
MANAGEMENT_FIELDS = (
    ("cultivar", Simulation.cultivar, "Cultivar"),
    ("planting_stage", Simulation.planting_stage, "Planting date"),
    ("irrigation", Simulation.irrigation, "Irrigation"),
    ("nitrogen_level", Simulation.nitrogen_level, "Nitrogen level"),
)
FIELD_LABELS = {field: label for field, _, label in MANAGEMENT_FIELDS}
# Plural nouns used in sentences ("all 5 cultivars").
FIELD_NOUNS = {
    "cultivar": ("cultivar", "cultivars"),
    "planting_stage": ("planting date", "planting dates"),
    "irrigation": ("irrigation condition", "irrigation conditions"),
    "nitrogen_level": ("nitrogen level", "nitrogen levels"),
}
OTHER_FILTER_LABELS = {"year": "year", "crop": "crop", "country": "country", "state": "state", "district": "district"}

_PLANTING = re.compile(r"^pfrst(-?\d+)$", re.I)
_IRRIGATION = {"RF": "rainfed", "IR": "irrigated"}
_NITROGEN = {"HIGHN": "high nitrogen", "LOWN": "low nitrogen"}


def planting_offset(value: Any) -> Optional[int]:
    match = _PLANTING.match(str(value))
    return int(match.group(1)) if match else None


def value_label(field: str, value: Any) -> str:
    """Plain-language meaning of a stored code, or the code itself."""
    text = str(value)
    if field == "cultivar" and text.upper() in CULTIVAR_CODES:
        return CULTIVAR_CODES[text.upper()]["meaning"]
    if field == "planting_stage":
        days = planting_offset(text)
        if days is not None:
            if days == 0:
                return "normal planting date"
            return f"{abs(days)} days {'before' if days < 0 else 'after'} the normal planting date"
    if field == "irrigation" and text.upper() in _IRRIGATION:
        return _IRRIGATION[text.upper()]
    if field == "nitrogen_level" and text.upper() in _NITROGEN:
        return _NITROGEN[text.upper()]
    return text


def sort_values(field: str, values: List[Any]) -> List[Any]:
    if field == "planting_stage":
        return sorted(values, key=lambda v: (planting_offset(v) is None, planting_offset(v) or 0, str(v)))
    return sorted(values, key=str)


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def _join(items: List[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


class ManagementService:
    """Reads management fields and values from the simulations table."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.stats = StatisticsService(db)

    async def _distinct(self, conditions: List[Any]) -> Dict[str, Any]:
        columns = [func.array_agg(func.distinct(column)).label(field) for field, column, _ in MANAGEMENT_FIELDS]
        stmt = select(func.count().label("simulations"), *columns).select_from(Simulation)
        if conditions:
            stmt = stmt.where(*conditions)
        row = (await self.db.execute(stmt)).one()
        mapping = row._mapping
        return {
            "simulations": int(mapping["simulations"] or 0),
            "values": {
                field: sort_values(field, [v for v in (mapping[field] or []) if v not in (None, "")])
                for field, _, _ in MANAGEMENT_FIELDS
            },
        }

    async def matched_values(self, filters: Dict[str, Any], spatial: Any = None) -> Dict[str, Any]:
        """Distinct management values of the simulations matching the filters.

        Uses StatisticsService's filter builder, so the predicates (including
        the spatial one) are the same as the statistic being described.
        """
        return await self._distinct(self.stats._build_simulation_filters(spatial=spatial, **filters))

    async def dataset_values(self) -> Dict[str, Any]:
        """Distinct management values in the whole dataset."""
        return await self._distinct([])

    async def available(self) -> Dict[str, Any]:
        """Management fields with their values and simulation counts."""
        fields = []
        for field, column, label in MANAGEMENT_FIELDS:
            rows = (await self.db.execute(
                select(column, func.count()).group_by(column)
            )).all()
            counts = {value: int(n) for value, n in rows if value not in (None, "")}
            fields.append({
                "field": field,
                "label": label,
                "values": [
                    {"value": value, "label": value_label(field, value), "simulations": counts[value]}
                    for value in sort_values(field, list(counts))
                ],
            })
        scope = (await self.db.execute(select(
            func.count(), func.min(Simulation.simulation_year), func.max(Simulation.simulation_year),
            func.array_agg(func.distinct(Simulation.crop)),
        ))).one()
        return {
            "fields": [f for f in fields if f["values"]],
            "simulations": int(scope[0] or 0),
            "years": {"min": scope[1], "max": scope[2]},
            "crops": sorted(c for c in (scope[3] or []) if c),
        }

    async def scope(
        self,
        filters: Dict[str, Any],
        spatial: Any = None,
        group_by: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Which management conditions the records behind a result include.

        filters are the same keyword filters the statistic used (management
        and other fields, e.g. year); spatial is the same area filter.
        """
        filters = {k: v for k, v in filters.items() if v is not None}
        if filters.get("crop") and not isinstance(filters["crop"], list):
            resolved = await self.stats.resolve_crop_code(str(filters["crop"]))
            if resolved:
                filters["crop"] = resolved
        matched = await self.matched_values(filters, spatial)
        if not matched["simulations"]:
            return None
        dataset = await self.dataset_values()

        requested = []
        for field, value in filters.items():
            label = FIELD_LABELS.get(field) or OTHER_FILTER_LABELS.get(field, field)
            values = _as_list(value)
            requested.append({
                "field": field,
                "label": label,
                "values": [str(v) for v in values],
                "management": field in FIELD_LABELS,
            })

        included = []
        for field, _, label in MANAGEMENT_FIELDS:
            values = matched["values"][field]
            available = dataset["values"][field]
            included.append({
                "field": field,
                "label": label,
                "values": values,
                "value_labels": [value_label(field, v) for v in values],
                "available_count": len(available),
                "all_available": bool(values) and set(values) == set(available),
                "filtered": field in filters,
                "grouped": field == group_by,
            })

        return {
            "simulations": matched["simulations"],
            "requested": requested,
            "included": included,
            "sentence": describe_scope(requested, included),
        }


def _multi_phrase(item: Dict[str, Any]) -> str:
    """'all 5 cultivars (BASE, …)' for a field with several values."""
    field, values = item["field"], item["values"]
    plural = FIELD_NOUNS[field][1]
    amount = f"all {len(values)}" if item["all_available"] else f"{len(values)} of {item['available_count']}"
    offsets = [planting_offset(v) for v in values]
    if field == "planting_stage" and None not in offsets:
        first, last = offsets[0], offsets[-1]
        if first != 0 and last != 0:
            def side(days: int) -> str:
                return f"{abs(days)} days {'before' if days < 0 else 'after'}"
            span = f"from {side(first)} to {side(last)} the normal planting date"
        else:
            span = f"{value_label(field, values[0])} to {value_label(field, values[-1])}"
        return f"{amount} {plural} ({values[0]} to {values[-1]}: {span})"
    return f"{amount} {plural} ({', '.join(map(str, values))})"


def _named(field: str, value: Any) -> str:
    """'rainfed (RF)': meaning first, for values described in a sentence."""
    meaning = value_label(field, value)
    return str(value) if meaning == str(value) else f"{meaning} ({value})"


def _coded(field: str, value: Any) -> str:
    """'BASE (baseline cultivar)': code first, for filters the user asked for."""
    meaning = value_label(field, value)
    if meaning == str(value):
        return str(value)
    return f"{value} ({meaning[0].lower() + meaning[1:]})"


def describe_scope(requested: List[Dict[str, Any]], included: List[Dict[str, Any]]) -> str:
    """Plain sentences stating the management conditions behind a result."""
    filtered = [f for f in requested if f["management"]]
    other = [f for f in requested if not f["management"]]
    other_text = _join([f"{f['label']} {', '.join(f['values'])}" for f in other])

    parts = []
    if filtered:
        parts.append("Filtered to " + _join([
            f"{f['label'].lower()} " + (
                _coded(f["field"], f["values"][0]) if len(f["values"]) == 1 else ", ".join(f["values"]))
            for f in filtered
        ]) + (f" ({other_text})" if other_text else "") + ".")

    rest = [item for item in included if not item["filtered"] and item["values"]]
    grouped = [item for item in rest if item["grouped"]]
    multi = [item for item in rest if not item["grouped"] and len(item["values"]) > 1]
    single = [item for item in rest if not item["grouped"] and len(item["values"]) == 1]

    if grouped:
        parts.append("Shown separately for " + "; ".join(_multi_phrase(i) for i in grouped) + ".")
    if multi:
        lead = "The result combines all matching records"
        if other_text and not filtered:
            lead += f" ({other_text})"
        parts.append(f"{lead} across " + "; ".join(_multi_phrase(i) for i in multi) + ".")
    if single:
        sentence = "All matching records are " + _join([_named(i["field"], i["values"][0]) for i in single])
        only = [FIELD_NOUNS[i["field"]][0] for i in single if i["available_count"] == 1]
        if only:
            sentence += f", the only {_join(only)} in the data"
        parts.append(sentence + ".")
    mixed = [FIELD_NOUNS[i["field"]][0] for i in multi if i["field"] in ("cultivar", "planting_stage")]
    if mixed:
        parts.append(f"It is not specific to one {' or '.join(mixed)}.")
    return "Management conditions: " + " ".join(parts)
