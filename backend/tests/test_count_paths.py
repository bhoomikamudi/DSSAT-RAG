"""Every list path reports the true count, the returned sample and the limit.

Covers the older country/polygon/region branches: the SimulationTool
(semantic metadata path) and the legacy Executor._execute_spatial tool.
The total must come from a COUNT query with the same predicates as the
sample, never from len(sample).
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent.executor import SPATIAL_SAMPLE_LIMIT, Executor
from app.agent.models import (
    LocationFilter,
    QueryPlan,
    SimulationSpatialInput,
    SimulationToolInput,
)
from app.agent.tools import SIMULATION_SAMPLE_LIMIT, SimulationTool
from app.services.metadata_service import MetadataService
from app.services.spatial_service import SpatialService
from app.services.statistics_service import StatisticsService
from test_statistics_sql import sql

POLYGON = "POLYGON((34.5 0.5, 35.5 0.5, 35.5 1.5, 34.5 1.5, 34.5 0.5))"


def fake_simulation(index):
    return SimpleNamespace(
        simulation_id=index, experiment_name="E", run_name="R", country="Kenya",
        state="S", district="D", latitude=0.8 + index / 1000, longitude=35.0,
        crop="MZ", cultivar="BASE", simulation_year=2010,
    )


class CountingSession:
    """Returns `returned` simulations for SELECTs and `total` for COUNTs."""

    def __init__(self, total, returned):
        self.total = total
        self.returned = returned
        self.statements = []

    async def execute(self, stmt):
        self.statements.append(stmt)
        rows = [fake_simulation(i) for i in range(self.returned)]
        return SimpleNamespace(
            scalar_one=lambda: self.total,
            scalars=lambda: SimpleNamespace(all=lambda: rows),
        )


def sample_and_count_sql(session):
    sample, count = (sql(stmt) for stmt in session.statements)
    assert "count(" in count.lower() and "LIMIT" not in count
    assert f"LIMIT {SPATIAL_SAMPLE_LIMIT}" in sample
    # Both queries use the same WHERE clause.
    where = lambda text: text.split(" WHERE ", 1)[1].split(" LIMIT ")[0]
    assert where(sample) == where(count)
    return where(count)


LEGACY_LOCATIONS = [
    (LocationFilter(type="radius", latitude=0.79, longitude=35.04, radius_meters=25000), ["ST_DWithin"]),
    (LocationFilter(type="polygon", polygon_wkt=POLYGON), ["ST_Within", "POLYGON((34.5 0.5"]),
    (LocationFilter(type="country", country="Kenya", state="Rift"), ["simulations.country = 'Kenya'", "simulations.state = 'Rift'"]),
    (LocationFilter(type="state", state="Rift"), ["simulations.state = 'Rift'", "simulations.crop = 'MZ'"]),
    (LocationFilter(type="district", district="Kitale"), ["simulations.district = 'Kitale'"]),
    (LocationFilter(type="ecological_zone", ecological_zone="Highland"), ["simulations.ecological_zone = 'Highland'"]),
]


@pytest.mark.parametrize("location,predicates", LEGACY_LOCATIONS, ids=lambda v: getattr(v, "type", ""))
@pytest.mark.parametrize("total,returned", [(33300, 100), (42, 42), (0, 0)])
async def test_legacy_spatial_reports_true_count(location, predicates, total, returned):
    session = CountingSession(total, returned)
    plan = QueryPlan(intent="spatial_search", filters={"crop": "MZ"}, location=location,
                     required_tools=["spatial"], response_type="summary")
    result = await Executor(None)._execute_spatial(SpatialService(session), plan, {}, [])

    assert result.total_count == total
    assert result.returned_count == returned
    assert result.sample_limit == SPATIAL_SAMPLE_LIMIT
    where = sample_and_count_sql(session)
    for predicate in predicates:
        assert predicate in where


async def test_legacy_spatial_incomplete_location_is_zero_without_query():
    session = CountingSession(total=999, returned=5)
    plan = QueryPlan(intent="spatial_search", filters={}, location=LocationFilter(type="polygon"),
                     required_tools=["spatial"], response_type="summary")
    result = await Executor(None)._execute_spatial(SpatialService(session), plan, {}, [])
    assert (result.total_count, result.returned_count) == (0, 0)
    assert session.statements == []


def test_legacy_spatial_context_carries_counts():
    from app.agent.context_builder import ContextBuilder
    from app.agent.models import SpatialResult
    processed = ContextBuilder()._process_spatial(SpatialResult(
        simulations=[{}] * 100, total_count=33300, returned_count=100, sample_limit=100,
        bounds={}))
    assert (processed["total_count"], processed["returned_count"], processed["sample_limit"]) == (33300, 100, 100)


@pytest.mark.parametrize("spatial,expected_filters,predicate", [
    (SimulationSpatialInput(type="polygon", polygon_wkt=POLYGON), {"cultivar": "BASE"}, "ST_Within"),
    (SimulationSpatialInput(type="country", country="Kenya"), {"cultivar": "BASE", "country": "Kenya"},
     "simulations.country = 'Kenya'"),
    (SimulationSpatialInput(type="state", state="Rift"), {"cultivar": "BASE", "state": "Rift"},
     "simulations.state = 'Rift'"),
], ids=["polygon", "country", "state"])
@pytest.mark.parametrize("total", [6660, 0])
async def test_simulation_tool_country_and_polygon_counts(spatial, expected_filters, predicate, total):
    tool = SimulationTool(None)
    returned = min(total, SIMULATION_SAMPLE_LIMIT)
    tool.meta.get_simulations = AsyncMock(return_value=[{"simulation_id": str(i)} for i in range(returned)])
    tool.stats.count_simulations = AsyncMock(return_value=total)

    output = await tool.run(SimulationToolInput(filters={"cultivar": "BASE"}, spatial=spatial))

    assert output.metadata == {"count": returned, "total_count": total, "sample_limit": SIMULATION_SAMPLE_LIMIT}
    sample_kwargs = tool.meta.get_simulations.call_args.kwargs
    count_kwargs = tool.stats.count_simulations.call_args.kwargs
    # Same predicates for the sample and the count; only the limit differs.
    assert sample_kwargs.pop("limit") == SIMULATION_SAMPLE_LIMIT
    assert sample_kwargs == count_kwargs
    assert {k: v for k, v in count_kwargs.items() if k != "spatial"} == expected_filters

    # The count query itself carries the location predicate and no LIMIT.
    captured = []
    async def execute(stmt):
        captured.append(stmt)
        return SimpleNamespace(scalar_one=lambda: total)
    assert await StatisticsService(SimpleNamespace(execute=execute)).count_simulations(**count_kwargs) == total
    count_sql = sql(captured[0])
    assert predicate in count_sql and "LIMIT" not in count_sql


async def test_simulation_tool_conflicting_country_filter_matches_nothing():
    tool = SimulationTool(None)
    tool.meta.get_simulations = AsyncMock(return_value=[])
    tool.stats.count_simulations = AsyncMock(return_value=0)
    output = await tool.run(SimulationToolInput(
        filters={"country": "Uganda"}, spatial=SimulationSpatialInput(type="country", country="Kenya")))
    assert tool.stats.count_simulations.call_args.kwargs["country"] == []
    assert output.metadata["total_count"] == 0 and output.metadata["count"] == 0


async def test_polygon_sample_sql_matches_count_sql():
    """The repository sample and the statistics count share one polygon predicate."""
    captured = []
    async def execute(stmt):
        captured.append(stmt)
        return SimpleNamespace(
            scalar_one=lambda: 0,
            scalars=lambda: SimpleNamespace(all=lambda: []),
        )
    session = SimpleNamespace(execute=execute)
    spatial = SimulationSpatialInput(type="polygon", polygon_wkt=POLYGON)
    assert await MetadataService(session).get_simulations(country="Kenya", spatial=spatial, limit=100) == []
    assert await StatisticsService(session).count_simulations(country="Kenya", spatial=spatial) == 0
    sample, count = (sql(stmt) for stmt in captured)
    for text in (sample, count):
        assert "ST_Within" in text and "simulations.country = 'Kenya'" in text
    assert "LIMIT 100" in sample and "LIMIT" not in count


@pytest.mark.parametrize("field", ["country", "state", "district"])
async def test_simulation_tool_blank_region_is_rejected(field):
    tool = SimulationTool(None)
    tool.meta.get_simulations = AsyncMock()
    tool.stats.count_simulations = AsyncMock()
    with pytest.raises(ValueError, match=f"A {field} is required"):
        await tool.run(SimulationToolInput(spatial=SimulationSpatialInput(type=field, **{field: ""})))
    tool.stats.count_simulations.assert_not_called()
