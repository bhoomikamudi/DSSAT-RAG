"""Regression coverage for map scope, coordinate questions and honest counts."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent.location_parser import extract_location
from app.agent.location_resolver import resolve_spatial
from app.agent.models import SpatialFilter, SimulationToolInput, SimulationSpatialInput
from app.agent.planner import QueryPlanner
from app.agent.orchestrator import AgentOrchestrator
from app.agent.models import QueryPlan, SemanticPlan, SemanticOperation, FilterCondition
from app.agent.location_resolver import SpatialResolution
from app.agent.tools import SimulationTool
from app.agent import planner as planner_module
from app.agent import response_generator as response_module
from app.api.v1 import chat
from app.db.session import get_db
from app.services.metadata_service import MetadataService
from app.services.statistics_service import StatisticsService
from conftest import FakeLLM, apply_filters
from test_statistics_sql import FakeSession, sql


@pytest.fixture
def client(fake_db):
    app = FastAPI()
    app.include_router(chat.router)
    async def session():
        yield object()
    app.dependency_overrides[get_db] = session
    return TestClient(app)


def ask(client, message, **kwargs):
    result = client.post('/', json={'message': message, **kwargs})
    assert result.status_code == 200, result.text
    return result.json()


@pytest.mark.parametrize('question,lat,lon', [
    ('What is the average yield within 10 km of these coordinates: 0.79, 35.04?', .79, 35.04),
    ('average yield within 10 km of coordinates: 0, 35?', 0, 35),
    ('average yield within 10 km of lat -0.79 lon +35.04', -.79, 35.04),
])
async def test_coordinates_cleaned_and_never_geocoded(question, lat, lon):
    geocoder = SimpleNamespace(geocode=AsyncMock(side_effect=AssertionError('must not geocode')))
    result = await resolve_spatial(question, geocoder=geocoder, latitude=1, longitude=2, radius_km=25)
    assert result.notice is None
    assert result.spatial.model_dump() == dict(latitude=lat, longitude=lon, radius_km=10, source='coordinates', label=None)
    assert result.query.strip('?') in ('What is the average yield', 'average yield')
    geocoder.geocode.assert_not_called()


@pytest.mark.parametrize('pair', ['100.79, 35.04', '-100.79, 35.04', '0.79, 350.04', '0.79, -350.04'])
async def test_invalid_coordinates_do_not_fall_back_to_map(pair):
    result = await resolve_spatial(f'average yield at coordinates: {pair}', latitude=0, longitude=35)
    assert result.spatial is None
    assert 'out of range' in result.notice


@pytest.mark.parametrize('question', ['compare 2010 and 2015', 'compare 2010, 2015', 'compare years 2010, 2015 near Kitale'])
def test_year_pairs_are_not_coordinates(question):
    assert extract_location(question).latitude is None


@pytest.mark.parametrize('question', [
    'What is the average yield here?',
    'What is the average yield at the selected map point?',
    'What is the average yield around this location?',
    'What is the average yield at selected location?',
])
def test_exact_here_without_selection_does_not_query(client, fake_db, question):
    result = ask(client, question)
    assert 'Select a point on the map' in result['answer']
    assert result['statistics'] is None and result['query_plan'] is None
    assert fake_db.calls == []


def test_exact_here_with_selection(client, sample_frame):
    result = ask(client, 'What is the average yield here?', latitude=.625, longitude=35.0417, radius_km=25)
    subset = apply_filters(sample_frame, spatial=SpatialFilter(latitude=.625, longitude=35.0417, radius_km=25))
    assert result['statistics']['metric'] == 'HWAM'
    assert result['statistics']['count'] == len(subset) == 13320
    assert result['statistics']['value'] == pytest.approx(subset.HWAM.mean())
    assert round(result['statistics']['value'], 2) == 4207.96


def test_current_selection_overrides_carried_place_and_here_requires_map(client, geocoder_returns):
    geocoder_returns({'kitale': [{'display_name': 'Kitale, Kenya', 'lat': '1.015', 'lon': '35.006', 'importance': .8}]})
    first = ask(client, 'correlation between rainfall and yield for BASE near Kitale', session_id='scope', radius_km=25)
    assert first['analysis']['sample_size'] == 6660
    missing = ask(client, 'What is the average yield here?', session_id='scope')
    assert 'Select a point' in missing['answer']
    changed = ask(client, 'Now fit a linear regression here', session_id='scope', latitude=.625, longitude=35.0417, radius_km=25)
    assert changed['spatial']['filter']['source'] == 'map'
    assert changed['analysis']['sample_size'] == 2664
    assert changed['semantic_plan']['operations'][0]['analysis']['spatial']['latitude'] == .625


def test_base_kitale_area_and_analysis_counts(client, sample_frame):
    body = ask(client, 'correlation between rainfall and yield for BASE within 25 km of coordinates: 1.015, 35.006')
    area = apply_filters(sample_frame, spatial=SpatialFilter(latitude=1.015, longitude=35.006, radius_km=25))
    assert len(area) == body['spatial']['simulations'] == 33300
    assert len(area[area.cultivar == 'BASE']) == body['analysis']['sample_size'] == 6660
    assert body['analysis']['filters'] == [{'field': 'cultivar', 'operator': '=', 'value': 'BASE'}]
    assert '33,300 simulations in total' in body['answer']
    assert '6,660 paired records were retrieved after cultivar = BASE' in body['answer']
    assert '6,660 had valid values for the analysis' in body['answer']


@pytest.mark.parametrize('metric,question,expected', [
    ('yield', 'What is the average yield here?', 'HWAM'),
    ('TMAXA', 'What is the average yield here?', 'HWAM'),
    ('rainfall', 'What is the average rainfall here?', 'PRCP'),
    ('HWAH', 'What is the average HWAH here?', 'HWAH'),
    ('yield', 'What is the average HWAH yield here?', 'HWAH'),
])
async def test_mock_llm_plan_normalizes_metrics(fake_db, metric, question, expected):
    planner = QueryPlanner(db_session=object())
    planner.client = FakeLLM({question: {'goal': 'average', 'intent': 'aggregate', 'operations': [
        {'operation': 'aggregate', 'metric': metric, 'aggregation': 'AVG'}]}})
    plan = await planner.plan_with_fallback(question)
    assert planner.client.calls
    assert plan.metric == expected
    assert planner.get_semantic_plan().operations[0].metric == expected


@pytest.mark.parametrize('total', [6660, 0, 100])
def test_metadata_true_count_and_capped_sample(client, monkeypatch, total):
    async def sample(self, **filters):
        assert filters['cultivar'] == 'BASE'
        assert filters['spatial'].radius_km == 25
        return [{'simulation_id': str(i), 'crop': 'MZ', 'cultivar': 'BASE'} for i in range(min(total, 100))]
    async def count(self, **filters):
        assert filters['cultivar'] == 'BASE'
        assert filters['spatial'].latitude == .79
        return total
    monkeypatch.setattr(MetadataService, 'get_simulations', sample)
    monkeypatch.setattr(StatisticsService, 'count_simulations', count)
    body = ask(client, 'Show simulations for BASE here', latitude=.79, longitude=35.04, radius_km=25)
    assert body['simulation_matches'] == {'total_count': total, 'returned_count': min(total, 100), 'sample_limit': 100}
    assert f'Found {total:,} matching simulations' in body['answer']
    assert ('capped sample of 100' in body['answer']) == (total > 100)


async def test_count_failure_is_not_reported_as_sample_total():
    tool = SimulationTool(None)
    tool.meta.get_simulations = AsyncMock(return_value=[{}] * 100)
    tool.stats.count_simulations = AsyncMock(side_effect=RuntimeError('count unavailable'))
    with pytest.raises(RuntimeError, match='count unavailable'):
        await tool.run(SimulationToolInput(filters={}))


def test_spatial_scope_preserves_explicit_domain_location_filters():
    orchestrator = AgentOrchestrator()
    plan = SemanticPlan(goal='average', intent='aggregate', operations=[SemanticOperation(
        operation='aggregate', metric='HWAM', aggregation='AVG', filters=[
            FilterCondition(field='country', operator='=', value='Kenya'),
            FilterCondition(field='district', operator='=', value='Kitale'),
            FilterCondition(field='cultivar', operator='=', value='BASE'),
        ])])
    result = orchestrator._apply_spatial(plan, QueryPlan(intent='aggregate', filters={},
        required_tools=['statistics'], response_type='summary'), 'average yield for BASE in Kenya',
        SpatialResolution(query='', spatial=SpatialFilter(latitude=1.015, longitude=35.006,
                                                         radius_km=25, source='place')))
    assert [(f.field, f.value) for f in result.operations[0].filters] == [('country', 'Kenya'), ('cultivar', 'BASE')]


def test_count_and_trend_keep_domain_filters(client, sample_frame, monkeypatch):
    question = 'trend of yield for BASE in 2010 and 2015'
    fake = FakeLLM({question: {'goal': 'trend', 'intent': 'trend', 'operations': [
        {'operation': 'trend', 'metric': 'yield', 'filters': [
            {'field': 'cultivar', 'operator': '=', 'value': 'BASE'},
            {'field': 'year', 'operator': 'IN', 'value': [2010, 2015]},
            {'field': 'planting_stage', 'operator': '=', 'value': 'pfrst0'}]}]}})
    monkeypatch.setattr(planner_module.settings, 'OPENAI_API_KEY', 'test')
    monkeypatch.setattr(planner_module, 'AsyncOpenAI', lambda **kwargs: fake)
    monkeypatch.setattr(response_module, 'AsyncOpenAI', lambda **kwargs: None)
    result = ask(client, question, latitude=.79, longitude=35.04, radius_km=25)
    rows = result['statistics']['breakdown']['values']
    assert [row['year'] for row in rows] == [2010, 2015]
    expected = apply_filters(sample_frame, cultivar='BASE', planting_stage='pfrst0', year=[2010, 2015],
                             spatial=SpatialFilter(latitude=.79, longitude=35.04, radius_km=25))
    assert result['statistics']['count'] == len(expected) == 30
    assert result['statistics']['value'] == pytest.approx(expected.HWAM.mean())


@pytest.mark.parametrize('entity', ['simulations', 'simulation_outputs'])
async def test_count_sql_has_radius_and_domain_filters(entity):
    session = FakeSession()
    async def execute(stmt):
        session.statements.append(stmt)
        return SimpleNamespace(scalar_one=lambda: 0)
    session.execute = execute
    assert await MetadataService(session).get_record_count(entity, cultivar='BASE', planting_stage='pfrst0',
        year=[2010, 2015], spatial=SpatialFilter(latitude=.79, longitude=35.04, radius_km=25)) == 0
    statement = sql(session.statements[0])
    for text in ["simulations.cultivar = 'BASE'", "simulations.planting_stage = 'pfrst0'", 'simulations.simulation_year IN (2010, 2015)', 'ST_DWithin']:
        assert text in statement
    assert 'LIMIT' not in statement


@pytest.mark.parametrize('method', ['calculate_aggregation', 'calculate_breakdown', 'get_extremum_simulation', 'get_yearly_trend'])
async def test_statistics_sql_preserves_every_filter(method):
    session = FakeSession()
    kwargs = dict(variable_code='HWAM', crop='MZ', cultivar='BASE', irrigation='RF', nitrogen_level='HighN',
                  planting_stage='pfrst0', country='Kenya', state='test', district='test', year=[2010, 2015],
                  spatial=SpatialFilter(latitude=.79, longitude=35.04, radius_km=25))
    if method != 'get_yearly_trend':
        kwargs['aggregation'] = 'AVG' if method != 'get_extremum_simulation' else 'MAX'
    if method == 'calculate_breakdown':
        kwargs['group_by'] = 'cultivar'
    await getattr(StatisticsService(session), method)(**kwargs)
    statement = sql(session.statements[0])
    for field in ['crop', 'cultivar', 'irrigation', 'nitrogen_level', 'planting_stage', 'country', 'state', 'district']:
        assert f"simulations.{field} = '{kwargs[field]}'" in statement
    assert 'simulations.simulation_year IN (2010, 2015)' in statement
    assert 'ST_DWithin' in statement
