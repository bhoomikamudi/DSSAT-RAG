"""Unit tests for the Python analysis layer (no database)."""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import stats

from app.agent.models import AnalysisRequest, FilterCondition
from app.services.analysis_service import AnalysisService, AnalysisValidationError


def records(xs, ys, groups=None, x="PRCP", y="HWAM"):
    rows = []
    for index, (xv, yv) in enumerate(zip(xs, ys)):
        row = {x: xv, y: yv}
        if groups is not None:
            row["group"] = groups[index]
        rows.append(row)
    return rows


@pytest.fixture
def service():
    return AnalysisService()


@pytest.fixture
def noisy_linear():
    rng = np.random.default_rng(7)
    x = rng.uniform(200, 900, 300)
    y = 1500 + 4.0 * x + rng.normal(0, 400, 300)
    return x, y


def test_correlation_matches_scipy(service, noisy_linear):
    x, y = noisy_linear
    result = service.analyze(
        AnalysisRequest(operation="correlation", x_variable="PRCP", y_variable="HWAM"),
        records(x, y),
    )
    expected = stats.pearsonr(x, y)

    assert result.status == "ok"
    assert result.sample_size == 300
    assert result.correlation == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue, abs=1e-12)
    assert result.chart_type == "scatter"
    assert result.chart_data["series"][0]["fit"] is None
    assert "Pearson correlation" in result.explanation


def test_linear_regression_recovers_exact_line(service):
    x = np.arange(10, 60, dtype=float)
    y = 2.0 * x + 1.0
    result = service.analyze(
        AnalysisRequest(operation="linear_regression", x_variable="PRCP", y_variable="HWAM"),
        records(x, y),
    )

    assert result.status == "ok"
    assert result.coefficients["slope"] == pytest.approx(2.0)
    assert result.coefficients["intercept"] == pytest.approx(1.0)
    assert result.r_squared == pytest.approx(1.0)
    assert result.sample_size == 50
    assert result.chart_type == "scatter_with_line"
    fit = result.chart_data["series"][0]["fit"]
    assert fit == [{"x": 10.0, "y": 21.0}, {"x": 59.0, "y": 119.0}]


def test_linear_regression_p_value_and_r2(service, noisy_linear):
    x, y = noisy_linear
    result = service.analyze(
        AnalysisRequest(operation="linear_regression", x_variable="PRCP", y_variable="HWAM"),
        records(x, y),
    )
    expected = stats.linregress(x, y)
    assert result.coefficients["slope"] == pytest.approx(expected.slope)
    assert result.r_squared == pytest.approx(expected.rvalue ** 2)
    assert result.p_value == pytest.approx(expected.pvalue, abs=1e-12)


def test_quadratic_regression_recovers_curve_and_vertex(service):
    x = np.linspace(100, 900, 81)
    y = -0.02 * x ** 2 + 20 * x + 1000  # peak at x = 500
    result = service.analyze(
        AnalysisRequest(operation="quadratic_regression", x_variable="PRCP", y_variable="HWAM"),
        records(x, y),
    )

    assert result.status == "ok"
    assert result.coefficients["quadratic"] == pytest.approx(-0.02)
    assert result.coefficients["linear"] == pytest.approx(20.0)
    assert result.coefficients["intercept"] == pytest.approx(1000.0, abs=1e-6)
    assert result.r_squared == pytest.approx(1.0)
    assert result.details["vertex_x"] == pytest.approx(500.0)
    assert result.details["vertex_type"] == "maximum"
    assert result.details["linear_r_squared"] < 0.1
    assert result.chart_type == "scatter_with_curve"
    assert len(result.chart_data["series"][0]["fit"]) == 50


def test_quadratic_p_value_is_overall_f_test(service, noisy_linear):
    x, y = noisy_linear
    result = service.analyze(
        AnalysisRequest(operation="quadratic_regression", x_variable="PRCP", y_variable="HWAM"),
        records(x, y),
    )
    n, r2 = 300, result.r_squared
    f_stat = (r2 / 2) / ((1 - r2) / (n - 3))
    assert result.p_value == pytest.approx(stats.f.sf(f_stat, 2, n - 3), abs=1e-12)


def test_grouped_correlation_by_cultivar(service):
    rng = np.random.default_rng(1)
    x = rng.uniform(200, 900, 200)
    groups = ["BASE"] * 100 + ["LNG"] * 100
    y = np.where(np.array(groups) == "BASE", 3 * x, -3 * x) + rng.normal(0, 50, 200)
    result = service.analyze(
        AnalysisRequest(
            operation="correlation", x_variable="PRCP", y_variable="HWAM", group_by="cultivar"
        ),
        records(x, y, groups),
    )

    by_group = {group.group: group for group in result.groups}
    assert set(by_group) == {"BASE", "LNG"}
    assert by_group["BASE"].correlation > 0.9
    assert by_group["LNG"].correlation < -0.9
    assert by_group["BASE"].sample_size == 100
    assert result.sample_size == 200  # overall pooled result is also reported
    assert [s["name"] for s in result.chart_data["series"]] == ["BASE", "LNG"]
    assert "By cultivar:" in result.explanation


def test_grouped_regression_by_year_sorts_numerically(service):
    x = np.tile(np.arange(10, dtype=float), 3)
    y = x * 2
    groups = [2010] * 10 + [2009] * 10 + [2011] * 10
    result = service.analyze(
        AnalysisRequest(
            operation="linear_regression", x_variable="PRCP", y_variable="HWAM", group_by="year"
        ),
        records(x, y, groups),
    )
    assert [group.group for group in result.groups] == ["2009", "2010", "2011"]
    assert all(group.coefficients["slope"] == pytest.approx(2.0) for group in result.groups)


def test_missing_and_invalid_values_are_removed(service):
    rows = [
        {"PRCP": 100, "HWAM": 1000},
        {"PRCP": 200, "HWAM": 2100},
        {"PRCP": None, "HWAM": 1500},
        {"PRCP": 300, "HWAM": float("nan")},
        {"PRCP": -99, "HWAM": 1800},  # DSSAT missing sentinel
        {"PRCP": "n/a", "HWAM": 1200},
        {"PRCP": 400, "HWAM": float("inf")},
        {"PRCP": 500, "HWAM": 4900},
        {"PRCP": "600", "HWAM": "6050"},  # numeric strings are accepted
    ]
    result = service.analyze(
        AnalysisRequest(operation="linear_regression", x_variable="PRCP", y_variable="HWAM"),
        rows,
    )
    assert result.status == "ok"
    assert result.rows_retrieved == 9
    assert result.sample_size == 4
    assert result.rows_dropped == 5
    assert any("excluded" in warning for warning in result.warnings)


def test_insufficient_data(service):
    result = service.analyze(
        AnalysisRequest(operation="quadratic_regression", x_variable="PRCP", y_variable="HWAM"),
        records([1, 2, 3], [4, 5, 7]),
    )
    assert result.status == "insufficient_data"
    assert result.chart_type is None
    assert "at least 4" in result.explanation


def test_constant_variable_is_insufficient(service):
    result = service.analyze(
        AnalysisRequest(operation="correlation", x_variable="PRCP", y_variable="HWAM"),
        records([5, 5, 5, 5], [1, 2, 3, 4]),
    )
    assert result.status == "insufficient_data"
    assert "same value" in result.explanation


def test_empty_group_reports_insufficient_but_others_succeed(service):
    x = list(range(20)) + [1, 2]
    y = [v * 3 for v in x]
    groups = ["BASE"] * 20 + ["SHT"] * 2
    result = service.analyze(
        AnalysisRequest(
            operation="correlation", x_variable="PRCP", y_variable="HWAM", group_by="cultivar"
        ),
        records(x, y, groups),
    )
    by_group = {group.group: group for group in result.groups}
    assert by_group["BASE"].status == "ok"
    assert by_group["SHT"].status == "insufficient_data"
    assert result.status == "ok"


@pytest.mark.parametrize(
    "request_kwargs, expected",
    [
        ({"operation": "anova", "x_variable": "PRCP", "y_variable": "HWAM"}, "Unsupported analysis operation"),
        ({"operation": "correlation", "x_variable": "soil moisture", "y_variable": "HWAM"}, "not an available variable"),
        ({"operation": "correlation", "x_variable": "PRCP", "y_variable": None}, "needs two"),
        ({"operation": "correlation", "x_variable": "PRCP", "y_variable": "PRCP"}, "must be different"),
        ({"operation": "correlation", "x_variable": "PRCP", "y_variable": "HWAM", "group_by": "soil_type"}, "Cannot group by"),
    ],
)
def test_invalid_requests_are_rejected(service, request_kwargs, expected):
    request = AnalysisRequest(**request_kwargs)
    with pytest.raises(AnalysisValidationError, match=expected):
        service.validate(request)
    result = service.analyze(request, records([1, 2, 3], [1, 2, 3]))
    assert result.status in {"invalid", "unavailable"}
    assert expected in result.explanation


def test_unknown_variable_is_unavailable_not_invalid(service):
    request = AnalysisRequest(operation="correlation", x_variable="soil moisture", y_variable="HWAM")
    result = service.analyze(request, [])
    assert result.status == "unavailable"


def test_unsupported_filter_operator_rejected(service):
    request = AnalysisRequest(
        operation="correlation",
        x_variable="PRCP",
        y_variable="HWAM",
        filters=[FilterCondition(field="cultivar", operator="LIKE", value="%BA%")],
    )
    with pytest.raises(AnalysisValidationError, match="not supported"):
        service.validate(request)


def test_variable_missing_from_loaded_dataset(service):
    request = AnalysisRequest(operation="correlation", x_variable="TMAXA", y_variable="HWAM")
    with pytest.raises(AnalysisValidationError, match="does not contain TMAXA"):
        service.validate(request, available_variables=["HWAM", "PRCP"])


def test_single_group_for_required_factor_is_unavailable(service):
    request = AnalysisRequest(
        operation="descriptive_statistics",
        y_variable="HWAM",
        group_by="nitrogen_level",
        requires_group_variation=True,
    )
    rows = [{"HWAM": value, "group": "HighN"} for value in (4000, 4500, 5000)]
    result = service.analyze(request, rows)
    assert result.status == "unavailable"
    assert "only one nitrogen level value (HighN)" in result.explanation


def test_descriptive_statistics_grouped_bar_chart(service):
    rows = [{"HWAM": v, "group": g} for v, g in [(1, "A"), (3, "A"), (10, "B"), (20, "B"), (30, "B")]]
    result = service.analyze(
        AnalysisRequest(operation="descriptive_statistics", y_variable="HWAM", group_by="cultivar"),
        rows,
    )
    assert result.status == "ok"
    assert result.descriptive["HWAM"]["mean"] == pytest.approx(64 / 5)
    assert result.chart_type == "bar"
    assert result.chart_data["bars"] == [
        {"group": "A", "mean": 2.0, "std": pytest.approx(np.std([1, 3], ddof=1)), "count": 2},
        {"group": "B", "mean": 20.0, "std": 10.0, "count": 3},
    ]


def test_plot_sampling_caps_points_but_stats_use_all_rows(service, noisy_linear):
    x, y = noisy_linear
    request = AnalysisRequest(
        operation="linear_regression", x_variable="PRCP", y_variable="HWAM", max_points=50
    )
    result = service.analyze(request, records(x, y))
    assert result.sample_size == 300
    assert result.chart_data["plotted_points"] == 50
    assert result.chart_data["sampled"] is True
    # Deterministic sampling: same input -> same plotted points.
    again = service.analyze(request, records(x, y))
    assert again.chart_data == result.chart_data


def test_include_plot_false_returns_no_chart(service, noisy_linear):
    x, y = noisy_linear
    result = service.analyze(
        AnalysisRequest(
            operation="correlation", x_variable="PRCP", y_variable="HWAM", include_plot=False
        ),
        records(x, y),
    )
    assert result.chart_type is None and result.chart_data is None


@pytest.mark.parametrize(
    "operation", ["correlation", "linear_regression", "quadratic_regression", "descriptive_statistics"]
)
def test_results_are_json_serializable(service, noisy_linear, operation):
    x, y = noisy_linear
    groups = ["BASE", "LNG"] * 150
    result = service.analyze(
        AnalysisRequest(operation=operation, x_variable="PRCP", y_variable="HWAM", group_by="cultivar"),
        records(x, y, groups),
    )
    payload = result.model_dump(mode="json")
    json.dumps(payload, allow_nan=False)  # raises on NaN/inf or NumPy types
