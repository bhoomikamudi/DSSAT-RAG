"""Display formatting for calculated statistics.

Measured quantities (yield, biomass, rainfall, temperature, counts, and
averages, totals, minima, maxima, standard deviations and intercepts of them)
are shown as whole numbers with thousands separators: 4,502.36 -> "4,502".
Only the displayed text is rounded; calculations and stored values keep full
precision. Dimensionless statistics and rates (r, R², p-values, regression
slopes and polynomial coefficients) keep their own precision, because rounding
them to whole numbers would erase their meaning (r = 0.246 would become 0).

Rounding is half away from zero, the same as the frontend's toLocaleString.
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional


def round_stat(value: float) -> int:
    """Whole-number display value, rounding halves away from zero."""
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def format_stat(value: Optional[float], missing: str = "n/a") -> str:
    """A measured statistic for display: 4502.36 -> '4,502'."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return missing
    return f"{round_stat(value):,}"
