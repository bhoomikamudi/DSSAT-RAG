"""Unique simulation identity (migration 002_unique_simulation_identity).

Unit tests always run. Database tests are opt-in: they apply the real Alembic
migrations to a disposable PostgreSQL/PostGIS database given by
MIGRATION_TEST_DATABASE_URL and refuse to run against the live database
(the database name must contain "test"). Example, from backend/:

    docker run -d --name dssat_migration_test -e POSTGRES_PASSWORD=postgres \
        -e POSTGRES_DB=dssat_migration_test -p 5440:5432 postgis/postgis:15-3.4-alpine
    MIGRATION_TEST_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5440/dssat_migration_test \
        pytest tests/test_unique_identity_migration.py
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.exc import IntegrityError

BACKEND = Path(__file__).resolve().parents[1]


def _import_alembic_library():
    """Import the installed Alembic, not backend/alembic (the migrations
    folder has an __init__.py, so with backend/ on sys.path it shadows the
    library)."""
    saved_path = sys.path[:]
    local = sys.modules.get("alembic")
    if local is not None and Path(getattr(local, "__file__", "") or "").resolve().parent == BACKEND / "alembic":
        del sys.modules["alembic"]
    sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != BACKEND]
    try:
        import alembic.command as alembic_command
        import alembic.config as alembic_config_module
    finally:
        sys.path[:] = saved_path
    return alembic_command, alembic_config_module.Config


command, Config = _import_alembic_library()
from sqlalchemy.engine import make_url

from app.core.config import get_settings
from app.models.simulation import Simulation
from app.repositories.simulation import SimulationRepository
from app.services.ingestion import IngestionService

CONSTRAINT = "uq_simulations_run_location_year"
KEY_COLUMNS = ["run_name", "latitude", "longitude", "simulation_year"]
TEST_URL = os.environ.get("MIGRATION_TEST_DATABASE_URL")


def alembic_config() -> Config:
    config = Config(str(BACKEND / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND / "alembic"))
    return config


# -----------------------------------------------------------------------------
# Unit: model, migration SQL, ingestion's handling of a rejected duplicate
# -----------------------------------------------------------------------------

def test_model_declares_the_unique_identity():
    constraint = next(c for c in Simulation.__table__.constraints if c.name == CONSTRAINT)
    assert [col.name for col in constraint.columns] == KEY_COLUMNS
    assert all(not Simulation.__table__.c[name].nullable for name in KEY_COLUMNS)


def test_migration_sql_adds_and_drops_only_the_constraint():
    config = alembic_config()
    config.output_buffer = io.StringIO()
    command.upgrade(config, "initial_migration:unique_simulation_identity", sql=True)
    upgrade_sql = config.output_buffer.getvalue()
    assert (f"ALTER TABLE simulations ADD CONSTRAINT {CONSTRAINT} "
            "UNIQUE (run_name, latitude, longitude, simulation_year)") in upgrade_sql
    assert "DELETE" not in upgrade_sql.upper() and "UPDATE SIMULATIONS" not in upgrade_sql.upper()

    config.output_buffer = io.StringIO()
    command.downgrade(config, "unique_simulation_identity:initial_migration", sql=True)
    assert f"ALTER TABLE simulations DROP CONSTRAINT {CONSTRAINT}" in config.output_buffer.getvalue()


def test_migration_chain():
    from alembic.script import ScriptDirectory

    scripts = ScriptDirectory.from_config(alembic_config())
    assert scripts.get_current_head() == "unique_simulation_identity"
    assert scripts.get_revision("unique_simulation_identity").down_revision == "initial_migration"


class ConstraintSession:
    """Session double whose flush raises IntegrityError a set number of times."""

    def __init__(self, rejections: int):
        self.rejections = rejections
        self.pending, self.committed, self.rollbacks = [], [], 0

    def add_all(self, objs):
        self.pending.extend(objs)

    async def flush(self):
        for obj in self.pending:
            if type(obj).__name__ == "Simulation" and obj.simulation_id is None:
                obj.simulation_id = uuid.uuid4()
        if self.rejections and any(type(o).__name__ == "Simulation" for o in self.pending):
            self.rejections -= 1
            raise IntegrityError("INSERT INTO simulations ...", {}, Exception(f"violates {CONSTRAINT}"))

    async def commit(self):
        self.committed.extend(self.pending)
        self.pending = []

    async def rollback(self):
        self.pending = []
        self.rollbacks += 1


def write_csv(path: Path, run_name: str = "MZ_RF_HighN_MZ_BASE__pfrst0") -> int:
    rows = [{"LATITUDE": 0.5 + i * 0.1, "LONGITUDE": 35.0, "RUN_NAME": run_name, "CR": "MZ",
             "WYEAR": 2000 + y, "PDAT": (2000 + y) * 1000 + 62, "HWAM": 4000.0, "PRCP": 700.0}
            for i in range(2) for y in range(3)]
    pd.DataFrame(rows).to_csv(path, index=False)
    return len(rows)


@pytest.fixture
def fake_queries(monkeypatch):
    async def lock_runs(self, run_names):
        return None

    async def get_existing_keys(self, run_names):
        return set()

    monkeypatch.setattr(SimulationRepository, "lock_runs", lock_runs)
    monkeypatch.setattr(SimulationRepository, "get_existing_keys", get_existing_keys)


def test_ingestion_retries_once_after_a_constraint_rejection(tmp_path, fake_queries):
    n = write_csv(tmp_path / "run.csv")
    session = ConstraintSession(rejections=1)
    result = asyncio.run(IngestionService(session, log_path=str(tmp_path / "log.jsonl"))
                         .ingest_file(str(tmp_path / "run.csv"), "pp_run.csv"))
    assert result.status == "completed" and result.records_inserted == n
    assert session.rollbacks == 1
    assert len([o for o in session.committed if type(o).__name__ == "Simulation"]) == n


def test_ingestion_fails_cleanly_if_the_constraint_keeps_rejecting(tmp_path, fake_queries):
    write_csv(tmp_path / "run.csv")
    session = ConstraintSession(rejections=2)
    result = asyncio.run(IngestionService(session, log_path=str(tmp_path / "log.jsonl"))
                         .ingest_file(str(tmp_path / "run.csv"), "pp_run.csv"))
    assert result.status == "failed"
    assert "rejected a duplicate simulation" in result.errors[0]
    assert session.committed == [] and session.rollbacks == 2


# -----------------------------------------------------------------------------
# Database: the real migration on a disposable PostgreSQL (opt-in)
# -----------------------------------------------------------------------------

requires_test_db = pytest.mark.skipif(
    not TEST_URL, reason="set MIGRATION_TEST_DATABASE_URL to a disposable PostGIS database"
)


def _guard_not_live(url) -> None:
    live = get_settings()
    assert "test" in (url.database or ""), "refusing: test database name must contain 'test'"
    assert not (url.database == live.POSTGRES_DB and str(url.port or 5432) == str(live.POSTGRES_PORT)
                and url.host in {live.POSTGRES_HOST, "localhost", "127.0.0.1"}), "refusing: live database"


@pytest.fixture(scope="module")
def test_db():
    """Point Alembic at the disposable database, migrate base -> head."""
    url = make_url(TEST_URL)
    _guard_not_live(url)
    settings = get_settings()
    saved = {k: getattr(settings, k) for k in
             ("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")}
    settings.POSTGRES_HOST, settings.POSTGRES_PORT = url.host, int(url.port or 5432)
    settings.POSTGRES_USER, settings.POSTGRES_PASSWORD, settings.POSTGRES_DB = (
        url.username, url.password, url.database)
    config = alembic_config()
    try:
        command.downgrade(config, "base")
        command.upgrade(config, "head")
        engine = __import__("sqlalchemy").create_engine(url)
        yield engine, config
        engine.dispose()
        command.downgrade(config, "base")
    finally:
        for key, value in saved.items():
            setattr(settings, key, value)


def insert_simulation(conn, run_name="RUN_A", lat=0.5, lon=35.0, year=2000):
    from sqlalchemy import text

    conn.execute(text("""
        INSERT INTO simulations (simulation_id, experiment_name, run_name, country, latitude, longitude,
                                 location, crop, simulation_year)
        VALUES (:id, 'e2e', :run, '', :lat, :lon, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), 'MZ', :year)
    """), {"id": uuid.uuid4(), "run": run_name, "lat": lat, "lon": lon, "year": year})


def count(conn):
    from sqlalchemy import text

    return conn.execute(text("SELECT count(*) FROM simulations")).scalar_one()


def clear(engine):
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM simulations"))


@requires_test_db
def test_valid_simulation_is_inserted(test_db):
    engine, _ = test_db
    clear(engine)
    with engine.begin() as conn:
        insert_simulation(conn)
        assert count(conn) == 1


@requires_test_db
def test_same_key_is_rejected_by_the_database(test_db):
    engine, _ = test_db
    clear(engine)
    with engine.begin() as conn:
        insert_simulation(conn)
    with pytest.raises(IntegrityError, match=CONSTRAINT):
        with engine.begin() as conn:
            insert_simulation(conn)
    with engine.connect() as conn:
        assert count(conn) == 1


@requires_test_db
@pytest.mark.parametrize("changed", [{"run_name": "RUN_B"}, {"lat": 0.6}, {"lon": 35.1}, {"year": 2001}])
def test_different_run_location_or_year_is_allowed(test_db, changed):
    engine, _ = test_db
    clear(engine)
    with engine.begin() as conn:
        insert_simulation(conn)
        insert_simulation(conn, **changed)
        assert count(conn) == 2


@requires_test_db
def test_upgrade_refuses_when_duplicates_exist_and_changes_nothing(test_db):
    from sqlalchemy import text

    engine, config = test_db
    clear(engine)
    command.downgrade(config, "initial_migration")
    try:
        with engine.begin() as conn:
            insert_simulation(conn)
            insert_simulation(conn)  # allowed: no constraint at this revision
        with pytest.raises(RuntimeError, match="No changes were made"):
            command.upgrade(config, "head")
        with engine.connect() as conn:
            assert count(conn) == 2
            assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == "initial_migration"
            assert not conn.execute(text(
                "SELECT count(*) FROM pg_constraint WHERE conname = :c"), {"c": CONSTRAINT}).scalar_one()
    finally:
        clear(engine)
        command.upgrade(config, "head")


@requires_test_db
def test_downgrade_removes_only_the_constraint(test_db):
    from sqlalchemy import text

    engine, config = test_db
    clear(engine)
    with engine.begin() as conn:
        insert_simulation(conn)
    command.downgrade(config, "initial_migration")
    try:
        with engine.connect() as conn:
            assert count(conn) == 1
            assert not conn.execute(text(
                "SELECT count(*) FROM pg_constraint WHERE conname = :c"), {"c": CONSTRAINT}).scalar_one()
    finally:
        command.upgrade(config, "head")


def run_async(coro):
    # Async psycopg needs a selector event loop on Windows.
    factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None
    with asyncio.Runner(loop_factory=factory) as runner:
        return runner.run(coro)


async def ingest_into_test_db(path, name, log):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(TEST_URL)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            return await IngestionService(session, log_path=str(log)).ingest_file(str(path), name)
    finally:
        await engine.dispose()


@requires_test_db
def test_ingestion_inserts_then_skips_a_repeat(test_db, tmp_path):
    engine, _ = test_db
    clear(engine)
    n = write_csv(tmp_path / "run.csv")
    first = run_async(ingest_into_test_db(tmp_path / "run.csv", "pp_run.csv", tmp_path / "log.jsonl"))
    second = run_async(ingest_into_test_db(tmp_path / "run.csv", "pp_run.csv", tmp_path / "log.jsonl"))
    assert (first.status, first.records_inserted) == ("completed", n)
    assert (second.status, second.records_inserted) == ("already_present", 0)
    with engine.connect() as conn:
        assert count(conn) == n


@requires_test_db
def test_database_blocks_duplicates_when_the_application_check_fails(test_db, tmp_path, monkeypatch):
    engine, _ = test_db
    clear(engine)
    n = write_csv(tmp_path / "run.csv")
    run_async(ingest_into_test_db(tmp_path / "run.csv", "pp_run.csv", tmp_path / "log.jsonl"))

    async def broken_check(self, run_names):  # the application check misses every row
        return set()

    monkeypatch.setattr(SimulationRepository, "get_existing_keys", broken_check)
    again = run_async(ingest_into_test_db(tmp_path / "run.csv", "pp_run.csv", tmp_path / "log.jsonl"))

    assert again.status == "failed"
    assert "rejected a duplicate simulation" in again.errors[0]
    with engine.connect() as conn:
        assert count(conn) == n          # nothing duplicated, nothing lost


@requires_test_db
def test_retry_recovers_when_only_the_first_check_is_stale(test_db, tmp_path, monkeypatch):
    """Simulates a concurrent upload: the first read misses rows committed meanwhile."""
    engine, _ = test_db
    clear(engine)
    n = write_csv(tmp_path / "run.csv")
    run_async(ingest_into_test_db(tmp_path / "run.csv", "pp_run.csv", tmp_path / "log.jsonl"))

    real = SimulationRepository.get_existing_keys
    calls = {"n": 0}

    async def stale_once(self, run_names):
        calls["n"] += 1
        return set() if calls["n"] == 1 else await real(self, run_names)

    monkeypatch.setattr(SimulationRepository, "get_existing_keys", stale_once)
    again = run_async(ingest_into_test_db(tmp_path / "run.csv", "pp_run.csv", tmp_path / "log.jsonl"))
    assert again.status == "already_present" and again.records_inserted == 0
    with engine.connect() as conn:
        assert count(conn) == n
