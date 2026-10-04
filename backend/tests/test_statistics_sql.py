"""SQL construction tests for StatisticsService (no live database).

Statements are captured from a fake session and compiled with the
PostgreSQL dialect, which verifies that filtering and pairing are pushed down
to the database and that existing aggregation queries are unchanged.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql

from app.services.statistics_service import StatisticsService


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeSession:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.statements = []

    async def execute(self, stmt):
        self.statements.append(stmt)
        return FakeResult(self.rows)


def sql(stmt) -> str:
    compiled = stmt.compile(
        dialect=postgresql.dialect(),
        compile_kwargs={"literal_binds": True},
    )
    return re.sub(r"\s+", " ", str(compiled))


@pytest.mark.asyncio
async def test_analysis_records_pairs_variables_and_filters_in_sql():
    rows = [SimpleNamespace(value_0=512.0, value_1=4873.0, group_value="BASE")]
    session = FakeSession(rows)
    service = StatisticsService(session)

    records = await service.get_analysis_records(
        variables=["PRCP", "HWAM"],
        group_by="cultivar",
        cultivar="BASE",
        irrigation="RF",
        year=[1990, 1991],
    )

    assert records == [{"PRCP": 512.0, "HWAM": 4873.0, "group": "BASE"}]
    text = sql(session.statements[0])

    # Only the requested columns are selected.
    select_list = text.split(" FROM ")[0]
    assert select_list == (
        "SELECT output_0.value AS value_0, output_1.value AS value_1, "
        "simulations.cultivar AS group_value"
    )
    # Each variable is its own join, keyed by simulation and variable code.
    assert "JOIN simulation_outputs AS output_0 ON output_0.simulation_id = simulations.simulation_id AND output_0.variable_code = 'PRCP'" in text
    assert "JOIN simulation_outputs AS output_1 ON output_1.simulation_id = simulations.simulation_id AND output_1.variable_code = 'HWAM'" in text
    # Filters are applied by PostgreSQL.
    assert "simulations.cultivar = 'BASE'" in text
    assert "simulations.irrigation = 'RF'" in text
    assert "simulations.simulation_year IN (1990, 1991)" in text
    assert "LIMIT 250000" in text


@pytest.mark.asyncio
async def test_analysis_records_single_variable_without_group():
    session = FakeSession([SimpleNamespace(value_0=4000.0)])
    records = await StatisticsService(session).get_analysis_records(["HWAM"])
    assert records == [{"HWAM": 4000.0}]
    text = sql(session.statements[0])
    assert "group_value" not in text
    assert "WHERE" not in text


@pytest.mark.asyncio
async def test_analysis_records_list_filter_uses_in():
    session = FakeSession()
    await StatisticsService(session).get_analysis_records(
        ["PRCP", "HWAM"], cultivar=["LNG", "VLNG"], country="Kenya"
    )
    text = sql(session.statements[0])
    assert "simulations.cultivar IN ('LNG', 'VLNG')" in text
    assert "simulations.country = 'Kenya'" in text


@pytest.mark.asyncio
async def test_analysis_records_rejects_unknown_group():
    with pytest.raises(ValueError):
        await StatisticsService(FakeSession()).get_analysis_records(["HWAM"], group_by="soil")


@pytest.mark.asyncio
async def test_distinct_field_values():
    session = FakeSession([("HighN",), (None,)])
    values = await StatisticsService(session).get_distinct_field_values("nitrogen_level")
    assert values == ["HighN"]
    assert "SELECT DISTINCT simulations.nitrogen_level" in sql(session.statements[0])


# -----------------------------------------------------------------------------
# Existing behavior: AVG / MIN / MAX / grouped average
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("aggregation, sql_function", [("AVG", "avg"), ("MIN", "min"), ("MAX", "max")])
async def test_calculate_aggregation_existing_behavior(aggregation, sql_function):
    row = SimpleNamespace(value=812.5, count=3404, stddev=120.0, min=300.0, max=1400.0)
    session = FakeSession([row])

    result = await StatisticsService(session).calculate_aggregation(
        variable_code="PRCP",
        aggregation=aggregation,
        year=2020,
        cultivar="BASE",
        irrigation="RF",
        nitrogen_level="HighN",
        planting_stage="pfrst0",
    )

    assert result == {
        "aggregation_type": aggregation,
        "metric": "PRCP",
        "value": 812.5,
        "count": 3404,
        "stddev": 120.0,
        "min": 300.0,
        "max": 1400.0,
        "unit": None,
    }
    text = sql(session.statements[0])
    assert text.startswith(f"SELECT {sql_function}(simulation_outputs.value) AS value")
    for clause in (
        "simulation_outputs.variable_code = 'PRCP'",
        "simulations.simulation_year = 2020",
        "simulations.cultivar = 'BASE'",
        "simulations.irrigation = 'RF'",
        "simulations.nitrogen_level = 'HighN'",
        "simulations.planting_stage = 'pfrst0'",
    ):
        assert clause in text


@pytest.mark.asyncio
async def test_calculate_aggregation_year_range_uses_in():
    session = FakeSession([SimpleNamespace(value=1.0, count=1, stddev=None, min=1.0, max=1.0)])
    await StatisticsService(session).calculate_aggregation("HWAM", "AVG", year=[2018, 2019, 2020])
    assert "simulations.simulation_year IN (2018, 2019, 2020)" in sql(session.statements[0])


@pytest.mark.asyncio
async def test_calculate_breakdown_grouped_average_by_cultivar():
    rows = [
        SimpleNamespace(group_value="VLNG", value=5200.0, count=30636),
        SimpleNamespace(group_value="BASE", value=4800.0, count=30636),
    ]
    session = FakeSession(rows)
    result = await StatisticsService(session).calculate_breakdown(
        variable_code="HWAM", aggregation="AVG", group_by="cultivar"
    )
    assert result == [
        {"group_value": "VLNG", "value": 5200.0, "count": 30636},
        {"group_value": "BASE", "value": 4800.0, "count": 30636},
    ]
    text = sql(session.statements[0])
    assert "GROUP BY simulations.cultivar" in text
    assert "ORDER BY avg(simulation_outputs.value) DESC" in text


@pytest.mark.asyncio
async def test_calculate_breakdown_by_year():
    session = FakeSession([])
    await StatisticsService(session).calculate_breakdown(
        variable_code="PRCP", aggregation="AVG", group_by="simulation_year", cultivar="BASE"
    )
    text = sql(session.statements[0])
    assert "GROUP BY simulations.simulation_year" in text
    assert "simulations.cultivar = 'BASE'" in text
