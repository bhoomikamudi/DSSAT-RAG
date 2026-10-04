"""Analysis Service - controlled statistical analysis in Python.

Receives already-filtered tabular records (retrieved from PostgreSQL by
StatisticsService.get_analysis_records) and runs a fixed set of validated
operations with pandas/NumPy/SciPy. No user- or LLM-provided code or SQL is
ever executed here: operations, variables and group fields are checked
against the allowlists in analysis_vocabulary.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.agent.models import (
    AnalysisGroupResult,
    AnalysisRequest,
    AnalysisResult,
)
from app.services.display_format import format_stat
from app.services.analysis_vocabulary import (
    ANALYSIS_OPERATIONS,
    ANALYSIS_VARIABLES,
    FIELD_LABELS,
    FILTER_FIELDS,
    GROUP_BY_FIELDS,
    MIN_SAMPLE_SIZE,
    variable_label,
    variable_unit,
)

logger = logging.getLogger(__name__)

# DSSAT writes -99 (and occasionally -99.9 / -999 / -9999) for missing values.
DSSAT_MISSING_SENTINELS = (-99.0, -99.9, -999.0, -9999.0)

GROUP_KEY = "group"

CHART_TYPES = {
    "correlation": "scatter",
    "linear_regression": "scatter_with_line",
    "quadratic_regression": "scatter_with_curve",
}

SUPPORTED_OPERATORS = {"=", "IN", "BETWEEN"}


class AnalysisValidationError(ValueError):
    """Raised when an analysis request cannot be run.

    status is "invalid" for malformed requests (unsupported operation, field
    or operator) and "unavailable" when the dataset lacks the requested data.
    """

    def __init__(self, message: str, status: str = "invalid"):
        super().__init__(message)
        self.status = status


def _clean_float(value: Any) -> Optional[float]:
    """Convert NumPy/None/NaN values to a JSON-safe float or None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _group_label(value: Any) -> str:
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _sort_key(label: str):
    try:
        return (0, float(label), label)
    except ValueError:
        return (1, 0.0, label)


def correlation_strength(r: float) -> str:
    magnitude = abs(r)
    if magnitude < 0.1:
        return "negligible"
    if magnitude < 0.3:
        return "weak"
    if magnitude < 0.5:
        return "moderate"
    if magnitude < 0.7:
        return "strong"
    return "very strong"


def _format_p(p_value: Optional[float]) -> str:
    if p_value is None:
        return "p not available"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def _format_number(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}"


class AnalysisService:
    """Runs validated correlation, regression and descriptive analyses."""

    def __init__(self, random_seed: int = 0):
        self.random_seed = random_seed

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def required_variables(request: AnalysisRequest) -> List[str]:
        """Variables that must be retrieved for this request, in x, y order."""
        variables: List[str] = []
        for code in (request.x_variable, request.y_variable):
            if code and code not in variables:
                variables.append(code)
        return variables

    def validate(
        self,
        request: AnalysisRequest,
        available_variables: Optional[Iterable[str]] = None,
    ) -> None:
        """Raise AnalysisValidationError with a user-facing message if invalid."""

        if request.operation not in ANALYSIS_OPERATIONS:
            raise AnalysisValidationError(
                f"Unsupported analysis operation '{request.operation}'. "
                f"Supported operations: {', '.join(ANALYSIS_OPERATIONS)}."
            )

        supported = ", ".join(
            f"{code} ({spec['label']})"
            for code, spec in ANALYSIS_VARIABLES.items()
        )

        for role, code in (("x", request.x_variable), ("y", request.y_variable)):
            if code is not None and code not in ANALYSIS_VARIABLES:
                raise AnalysisValidationError(
                    f"'{code}' is not an available variable in this dataset, "
                    f"so the requested analysis cannot be performed. "
                    f"Available variables: {supported}.",
                    status="unavailable",
                )

        if request.operation == "descriptive_statistics":
            if not request.x_variable and not request.y_variable:
                raise AnalysisValidationError(
                    "Descriptive statistics need at least one variable. "
                    f"Available variables: {supported}."
                )
        else:
            if not request.x_variable or not request.y_variable:
                raise AnalysisValidationError(
                    f"A {request.operation.replace('_', ' ')} needs two "
                    "variables (an independent and a dependent variable). "
                    f"Available variables: {supported}."
                )
            if request.x_variable == request.y_variable:
                raise AnalysisValidationError(
                    "The independent and dependent variables must be different."
                )

        if request.group_by is not None and request.group_by not in GROUP_BY_FIELDS:
            raise AnalysisValidationError(
                f"Cannot group by '{request.group_by}'. Supported grouping "
                f"fields: {', '.join(GROUP_BY_FIELDS)}."
            )

        for condition in request.filters:
            if condition.field not in FILTER_FIELDS:
                raise AnalysisValidationError(
                    f"Cannot filter by '{condition.field}'. Supported filter "
                    f"fields: {', '.join(FILTER_FIELDS)}."
                )
            if condition.operator not in SUPPORTED_OPERATORS:
                raise AnalysisValidationError(
                    f"Filter operator '{condition.operator}' is not supported "
                    f"for analysis. Use one of: {', '.join(sorted(SUPPORTED_OPERATORS))}."
                )

        if available_variables is not None:
            available = set(available_variables)
            missing = [
                code for code in self.required_variables(request)
                if code not in available
            ]
            if missing:
                raise AnalysisValidationError(
                    f"The loaded dataset does not contain {', '.join(missing)}, "
                    "so this analysis cannot be performed.",
                    status="unavailable",
                )

    def invalid_result(
        self,
        request: AnalysisRequest,
        message: str,
        status: str = "invalid",
    ) -> AnalysisResult:
        return AnalysisResult(
            status=status,  # type: ignore[arg-type]
            analysis_type=request.operation,
            x_variable=request.x_variable,
            y_variable=request.y_variable,
            x_unit=variable_unit(request.x_variable),
            y_unit=variable_unit(request.y_variable),
            filters=request.filters,
            group_by=request.group_by,
            explanation=message,
        )

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        request: AnalysisRequest,
        records: Sequence[Mapping[str, Any]],
    ) -> AnalysisResult:
        """Run the requested analysis on filtered records.

        Args:
            request: Validated analysis request.
            records: Rows with one key per variable code and an optional
                "group" key holding the group_by value.
        """
        try:
            self.validate(request)
        except AnalysisValidationError as exc:
            return self.invalid_result(request, str(exc), status=exc.status)

        variables = self.required_variables(request)
        frame = self._prepare_frame(records, variables, request.group_by)
        rows_retrieved = len(records)
        rows_dropped = rows_retrieved - len(frame)

        warnings: List[str] = []
        if rows_dropped:
            warnings.append(
                f"{rows_dropped:,} of {rows_retrieved:,} records were excluded "
                "because of missing or non-numeric values."
            )

        overall = self._analyze_subset(request, frame, group=None)

        groups: List[AnalysisGroupResult] = []
        if request.group_by:
            labels = sorted(
                frame[GROUP_KEY].dropna().map(_group_label).unique(),
                key=_sort_key,
            )
            labelled = frame.assign(_label=frame[GROUP_KEY].map(_group_label))
            for label in labels:
                subset = labelled[labelled["_label"] == label]
                groups.append(self._analyze_subset(request, subset, group=label))

            if request.requires_group_variation and len(groups) < 2:
                field_label = FIELD_LABELS.get(request.group_by, request.group_by)
                values = ", ".join(g.group or "" for g in groups) or "none"
                return self.invalid_result(
                    request,
                    (
                        "The current dataset cannot support this analysis: the "
                        f"matching simulations contain only one {field_label} "
                        f"value ({values}), so there is no {field_label} "
                        "variation to analyze."
                    ),
                    status="unavailable",
                )

        status = "ok" if overall.status == "ok" else "insufficient_data"

        chart_type, chart_data = (None, None)
        if request.include_plot and status == "ok":
            chart_type, chart_data = self._build_chart(request, frame, overall, groups)

        result = AnalysisResult(
            status=status,
            analysis_type=request.operation,
            x_variable=request.x_variable,
            y_variable=request.y_variable,
            x_unit=variable_unit(request.x_variable),
            y_unit=variable_unit(request.y_variable),
            filters=request.filters,
            group_by=request.group_by,
            sample_size=overall.sample_size,
            rows_retrieved=rows_retrieved,
            rows_dropped=rows_dropped,
            correlation=overall.correlation,
            p_value=overall.p_value,
            coefficients=overall.coefficients,
            r_squared=overall.r_squared,
            descriptive=overall.descriptive,
            details=overall.details,
            groups=groups,
            chart_type=chart_type,
            chart_data=chart_data,
            warnings=warnings,
        )
        result.explanation = self._explain(request, result, overall)
        return result

    # ------------------------------------------------------------------ #
    # Data preparation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _prepare_frame(
        records: Sequence[Mapping[str, Any]],
        variables: List[str],
        group_by: Optional[str],
    ) -> pd.DataFrame:
        columns = list(variables) + ([GROUP_KEY] if group_by else [])
        frame = pd.DataFrame.from_records(list(records), columns=columns)

        cleaned = {}
        for code in variables:
            numeric = pd.to_numeric(frame[code], errors="coerce").astype(float)
            numeric = numeric.replace([np.inf, -np.inf], np.nan)
            cleaned[code] = numeric.mask(numeric.isin(DSSAT_MISSING_SENTINELS))
        frame = frame.assign(**cleaned)

        required = list(variables) + ([GROUP_KEY] if group_by else [])
        return frame.dropna(subset=required).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Calculations
    # ------------------------------------------------------------------ #

    def _analyze_subset(
        self,
        request: AnalysisRequest,
        frame: pd.DataFrame,
        group: Optional[str],
    ) -> AnalysisGroupResult:
        n = int(len(frame))
        minimum = MIN_SAMPLE_SIZE[request.operation]
        result = AnalysisGroupResult(group=group, sample_size=n)

        if n < minimum:
            result.status = "insufficient_data"
            result.message = (
                f"Only {n} valid record(s) available; at least {minimum} are "
                f"required for {request.operation.replace('_', ' ')}."
            )
            return result

        if request.operation == "descriptive_statistics":
            result.descriptive = {
                code: self._describe(frame[code])
                for code in self.required_variables(request)
            }
            return result

        x = frame[request.x_variable].to_numpy(dtype=float)
        y = frame[request.y_variable].to_numpy(dtype=float)

        if np.ptp(x) == 0 or np.ptp(y) == 0:
            constant = request.x_variable if np.ptp(x) == 0 else request.y_variable
            result.status = "insufficient_data"
            result.message = (
                f"{constant} has the same value in every record, so a "
                "relationship cannot be estimated."
            )
            return result

        if request.operation == "correlation":
            pearson = scipy_stats.pearsonr(x, y)
            spearman = scipy_stats.spearmanr(x, y)
            result.correlation = _clean_float(pearson.statistic)
            result.p_value = _clean_float(pearson.pvalue)
            result.details = {
                "method": "pearson",
                "spearman_correlation": _clean_float(spearman.statistic),
                "spearman_p_value": _clean_float(spearman.pvalue),
            }

        elif request.operation == "linear_regression":
            fit = scipy_stats.linregress(x, y)
            result.correlation = _clean_float(fit.rvalue)
            result.r_squared = _clean_float(fit.rvalue ** 2)
            result.p_value = _clean_float(fit.pvalue)
            result.coefficients = {
                "slope": _clean_float(fit.slope),
                "intercept": _clean_float(fit.intercept),
            }
            result.details = {
                "slope_std_error": _clean_float(fit.stderr),
                "intercept_std_error": _clean_float(fit.intercept_stderr),
            }

        elif request.operation == "quadratic_regression":
            a, b, c = np.polyfit(x, y, deg=2)
            predicted = np.polyval([a, b, c], x)
            ss_res = float(np.sum((y - predicted) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

            p_value = None
            if r_squared is not None and n > 3 and r_squared < 1.0:
                # Overall F-test of the 2-predictor model against the mean.
                f_stat = (r_squared / 2.0) / ((1.0 - r_squared) / (n - 3))
                p_value = float(scipy_stats.f.sf(f_stat, 2, n - 3))
            elif r_squared == 1.0:
                p_value = 0.0

            linear_fit = scipy_stats.linregress(x, y)
            details: Dict[str, Any] = {
                "p_value_test": "overall F-test",
                "linear_r_squared": _clean_float(linear_fit.rvalue ** 2),
            }
            if a != 0:
                vertex_x = -b / (2.0 * a)
                if x.min() <= vertex_x <= x.max():
                    details["vertex_x"] = _clean_float(vertex_x)
                    details["vertex_y"] = _clean_float(np.polyval([a, b, c], vertex_x))
                    details["vertex_type"] = "maximum" if a < 0 else "minimum"

            result.r_squared = _clean_float(r_squared)
            result.p_value = _clean_float(p_value)
            result.coefficients = {
                "quadratic": _clean_float(a),
                "linear": _clean_float(b),
                "intercept": _clean_float(c),
            }
            result.details = details

        return result

    @staticmethod
    def _describe(series: pd.Series) -> Dict[str, Optional[float]]:
        values = series.to_numpy(dtype=float)
        return {
            "count": float(len(values)),
            "mean": _clean_float(np.mean(values)),
            "std": _clean_float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "min": _clean_float(np.min(values)),
            "q1": _clean_float(np.percentile(values, 25)),
            "median": _clean_float(np.median(values)),
            "q3": _clean_float(np.percentile(values, 75)),
            "max": _clean_float(np.max(values)),
        }

    # ------------------------------------------------------------------ #
    # Chart specification
    # ------------------------------------------------------------------ #

    def _build_chart(
        self,
        request: AnalysisRequest,
        frame: pd.DataFrame,
        overall: AnalysisGroupResult,
        groups: List[AnalysisGroupResult],
    ):
        x_code, y_code = request.x_variable, request.y_variable
        axis = lambda code: (  # noqa: E731
            {"code": code, "label": variable_label(code), "unit": variable_unit(code)}
            if code
            else None
        )

        if request.operation == "descriptive_statistics":
            if request.group_by and groups:
                target = y_code or x_code
                bars = [
                    {
                        "group": g.group,
                        "mean": (g.descriptive or {}).get(target, {}).get("mean"),
                        "std": (g.descriptive or {}).get(target, {}).get("std"),
                        "count": g.sample_size,
                    }
                    for g in groups
                    if g.status == "ok"
                ]
                return "bar", {
                    "y": axis(target),
                    "group_by": request.group_by,
                    "bars": bars,
                }
            if not (x_code and y_code):
                return None, None

        chart_type = CHART_TYPES.get(request.operation, "scatter")

        if request.group_by and groups:
            labelled = frame.assign(_label=frame[GROUP_KEY].map(_group_label))
            subsets = [
                (g, labelled[labelled["_label"] == g.group])
                for g in groups
            ]
        else:
            subsets = [(overall, frame)]

        total = int(len(frame))
        budget = request.max_points
        rng = np.random.default_rng(self.random_seed)
        series = []
        plotted = 0

        for group_result, subset in subsets:
            if subset.empty:
                continue
            share = max(1, round(budget * len(subset) / max(total, 1)))
            sample = subset
            if len(subset) > share:
                index = rng.choice(len(subset), size=share, replace=False)
                sample = subset.iloc[np.sort(index)]
            sample = sample.sort_values(x_code)
            points = [
                {"x": _clean_float(px), "y": _clean_float(py)}
                for px, py in zip(sample[x_code], sample[y_code])
            ]
            plotted += len(points)
            series.append(
                {
                    "name": group_result.group or "All records",
                    "n": group_result.sample_size,
                    "points": points,
                    "fit": self._fit_curve(request, subset[x_code], group_result),
                    "correlation": group_result.correlation,
                    "r_squared": group_result.r_squared,
                }
            )

        return chart_type, {
            "x": axis(x_code),
            "y": axis(y_code),
            "group_by": request.group_by,
            "series": series,
            "total_points": total,
            "plotted_points": plotted,
            "sampled": plotted < total,
        }

    @staticmethod
    def _fit_curve(
        request: AnalysisRequest,
        x_values: pd.Series,
        group_result: AnalysisGroupResult,
    ) -> Optional[List[Dict[str, Optional[float]]]]:
        coefficients = group_result.coefficients
        if group_result.status != "ok" or not coefficients:
            return None
        x_min, x_max = float(x_values.min()), float(x_values.max())

        if request.operation == "linear_regression":
            slope, intercept = coefficients["slope"], coefficients["intercept"]
            if slope is None or intercept is None:
                return None
            return [
                {"x": x_min, "y": _clean_float(intercept + slope * x_min)},
                {"x": x_max, "y": _clean_float(intercept + slope * x_max)},
            ]

        if request.operation == "quadratic_regression":
            a = coefficients["quadratic"]
            b = coefficients["linear"]
            c = coefficients["intercept"]
            if a is None or b is None or c is None:
                return None
            xs = np.linspace(x_min, x_max, 50)
            ys = np.polyval([a, b, c], xs)
            return [
                {"x": _clean_float(px), "y": _clean_float(py)}
                for px, py in zip(xs, ys)
            ]

        return None

    # ------------------------------------------------------------------ #
    # Explanation
    # ------------------------------------------------------------------ #

    def _explain(
        self,
        request: AnalysisRequest,
        result: AnalysisResult,
        overall: AnalysisGroupResult,
    ) -> str:
        x_name = f"{result.x_variable} ({variable_label(result.x_variable).lower()})"
        y_name = f"{result.y_variable} ({variable_label(result.y_variable).lower()})"
        scope = self._describe_scope(request)
        lines: List[str] = []

        if overall.status != "ok":
            lines.append(
                f"Not enough data for this analysis{scope}. {overall.message}"
            )
            return " ".join(lines)

        lines.append(self._explain_subset(request, overall, x_name, y_name, scope))

        if result.groups:
            field_label = FIELD_LABELS.get(request.group_by or "", request.group_by)
            items = []
            for group in result.groups:
                if group.status != "ok":
                    items.append(f"- {group.group}: {group.message}")
                else:
                    items.append(
                        f"- {group.group}: "
                        + self._explain_subset(request, group, x_name, y_name, "", short=True)
                    )
            lines.append(f"By {field_label}:\n\n" + "\n".join(items))

        if request.operation in {"correlation", "linear_regression", "quadratic_regression"}:
            lines.append(
                "These are statistical associations in the simulation outputs "
                "and do not by themselves establish causation."
            )
        # Blank lines between paragraphs keep the markdown list intact.
        return "\n\n".join(lines)

    @staticmethod
    def _describe_scope(request: AnalysisRequest) -> str:
        area = f" {request.spatial.describe()}" if request.spatial else ""
        if not request.filters:
            return f" across all matching simulations{area}"
        parts = []
        for condition in request.filters:
            label = FIELD_LABELS.get(condition.field, condition.field)
            if condition.operator == "BETWEEN" and isinstance(condition.value, list):
                parts.append(f"{label} {condition.value[0]}–{condition.value[-1]}")
            elif isinstance(condition.value, list):
                parts.append(f"{label} in {', '.join(map(str, condition.value))}")
            else:
                parts.append(f"{label} = {condition.value}")
        return " for " + ", ".join(parts) + area

    @staticmethod
    def _explain_subset(
        request: AnalysisRequest,
        subset: AnalysisGroupResult,
        x_name: str,
        y_name: str,
        scope: str,
        short: bool = False,
    ) -> str:
        n = f"n = {subset.sample_size:,}"
        p_text = _format_p(subset.p_value)
        significant = (
            subset.p_value is not None and subset.p_value < 0.05
        )

        if request.operation == "descriptive_statistics":
            pieces = []
            for code, summary in (subset.descriptive or {}).items():
                unit = variable_unit(code) or ""
                pieces.append(
                    f"{code}: mean {format_stat(summary.get('mean'))} {unit}, "
                    f"median {format_stat(summary.get('median'))}, "
                    f"SD {format_stat(summary.get('std'))}, "
                    f"range {format_stat(summary.get('min'))}–{format_stat(summary.get('max'))}"
                    .replace("  ", " ")
                )
            prefix = "" if short else f"Descriptive statistics{scope} ({n}): "
            return prefix + "; ".join(pieces) + (f" ({n})" if short else "")

        r = subset.correlation
        if request.operation == "correlation":
            direction = "positive" if (r or 0) >= 0 else "negative"
            body = (
                f"r = {r:.3f} ({correlation_strength(r)} {direction}), {p_text}, {n}"
            )
            if short:
                return body
            return (
                f"Pearson correlation between {x_name} and {y_name}{scope}: {body}. "
                + (
                    "The correlation is statistically significant at the 0.05 level."
                    if significant
                    else "The correlation is not statistically significant at the 0.05 level."
                )
            )

        if request.operation == "linear_regression":
            coefficients = subset.coefficients or {}
            slope = coefficients.get("slope")
            intercept = coefficients.get("intercept")
            x_unit = variable_unit(request.x_variable) or "unit"
            y_unit = variable_unit(request.y_variable) or ""
            direction = "positive" if (r or 0) >= 0 else "negative"
            body = (
                f"slope = {_format_number(slope, 3)} {y_unit} per {x_unit}, "
                f"intercept = {format_stat(intercept)}, "
                f"r = {r:.3f} ({correlation_strength(r)} {direction}), "
                f"R² = {_format_number(subset.r_squared, 3)}, {p_text}, {n}"
            ).replace("  ", " ")
            if short:
                return body
            return (
                f"Linear regression of {y_name} on {x_name}{scope}: {body}. "
                f"{request.x_variable} explains about "
                f"{(subset.r_squared or 0) * 100:.1f}% of the variation in "
                f"{request.y_variable}."
            )

        coefficients = subset.coefficients or {}
        body = (
            f"{request.y_variable} = {_format_number(coefficients.get('quadratic'), 6)}·x² "
            f"+ {_format_number(coefficients.get('linear'), 4)}·x "
            f"+ {format_stat(coefficients.get('intercept'))}, "
            f"R² = {_format_number(subset.r_squared, 3)}, {p_text} (overall F-test), {n}"
        )
        if short:
            return body
        text = f"Quadratic regression of {y_name} on {x_name}{scope}: {body}."
        details = subset.details or {}
        linear_r2 = details.get("linear_r_squared")
        if linear_r2 is not None and subset.r_squared is not None:
            text += (
                f" A straight line gives R² = {linear_r2:.3f}, so the quadratic "
                f"term adds {max(subset.r_squared - linear_r2, 0):.3f} explained variance."
            )
        if "vertex_x" in details:
            x_unit = variable_unit(request.x_variable) or ""
            text += (
                f" The fitted curve reaches a {details['vertex_type']} at "
                f"{request.x_variable} ≈ {format_stat(details['vertex_x'])} {x_unit}."
            ).replace("  ", " ")
        return text
