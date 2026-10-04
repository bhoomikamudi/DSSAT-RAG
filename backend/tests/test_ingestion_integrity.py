"""Ingestion reliability: identity, idempotency, atomicity, provenance.

Covers the incident in the live database (two files uploaded as random
overlapping halves: 1,702 duplicated + 1,702 missing simulations) and the
pandas-on-Python-3.14 date crash. No live database is used: a recording
session stands in for AsyncSession, and the repository's two queries are
replaced by equivalents over the recorded rows (their SQL is checked
separately below).
"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.dialects import postgresql

from app.parsers.csv_parser import DSSATParser
from app.repositories.simulation import SimulationRepository, simulation_key
from app.services.ingestion import IngestionService

SAMPLE_SUMMARY = Path(__file__).resolve().parents[2] / "sample_files" / "summary.csv"


# -----------------------------------------------------------------------------
# Test doubles
# -----------------------------------------------------------------------------

class RecordingSession:
    """AsyncSession stand-in with transaction semantics (flush/commit/rollback)."""

    def __init__(self, fail_outputs: bool = False):
        self.pending = []
        self.committed = []
        self.fail_outputs = fail_outputs
        self.locked = []
        self.commits = 0

    def add_all(self, objs):
        self.pending.extend(objs)

    async def flush(self):
        for obj in self.pending:
            if type(obj).__name__ == "Simulation" and obj.simulation_id is None:
                obj.simulation_id = uuid.uuid4()
        if self.fail_outputs and any(type(o).__name__ == "SimulationOutput" for o in self.pending):
            raise RuntimeError("simulated failure inserting simulation_outputs")

    async def commit(self):
        self.committed.extend(self.pending)
        self.pending = []
        self.commits += 1

    async def rollback(self):
        self.pending = []

    def simulations(self):
        return [o for o in self.committed if type(o).__name__ == "Simulation"]

    def outputs(self):
        return [o for o in self.committed if type(o).__name__ == "SimulationOutput"]

    def keys(self):
        return [simulation_key(s.run_name, s.latitude, s.longitude, s.simulation_year)
                for s in self.simulations()]


@pytest.fixture(autouse=True)
def fake_repository_queries(monkeypatch):
    async def lock_runs(self, run_names):
        self.db.locked.append(sorted(set(run_names)))

    async def get_existing_keys(self, run_names):
        assert self.db.locked, "runs must be locked before reading existing keys"
        return {k for k in self.db.keys() if k[0] in set(run_names)}

    monkeypatch.setattr(SimulationRepository, "lock_runs", lock_runs)
    monkeypatch.setattr(SimulationRepository, "get_existing_keys", get_existing_keys)


def write_run_csv(path: Path, n_locations: int = 20, years=range(2000, 2010),
                  run_name: str = "MZ_RF_HighN_MZ_VSHT__pfrst75") -> pd.DataFrame:
    """A CSV in the project's summary format: one run, every year at every location."""
    rows = []
    for i in range(n_locations):
        for year in years:
            rows.append({
                "LATITUDE": round(0.0417 + 0.0833 * (i % 5), 4),
                "LONGITUDE": round(34.375 + 0.0833 * (i // 5), 4),
                "RUN_NAME": run_name,
                "HARVEST_AREA": 100.0 + i,
                "CR": "MZ",
                "WYEAR": year,
                "PDAT": year * 1000 + 62,     # real DSSAT YYYYDDD dates
                "MDAT": year * 1000 + 200,
                "HDAT": year * 1000 + 205,
                "HWAM": 3000 + i + year % 7,
                "PRCP": 600.5 + i,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def ingest(session, path, file_name=None, log=None):
    service = IngestionService(session, log_path=str(log) if log else None)
    return asyncio.run(service.ingest_file(str(path), file_name))


def source_keys(frame: pd.DataFrame) -> set:
    return {simulation_key(r.RUN_NAME, r.LATITUDE, r.LONGITUDE, r.WYEAR) for r in frame.itertuples()}


# -----------------------------------------------------------------------------
# Identity
# -----------------------------------------------------------------------------

def test_identity_key_is_unique_in_the_real_source_csvs(sample_frame):
    keys = [simulation_key(r.RUN_NAME, r.LATITUDE, r.LONGITUDE, r.WYEAR) for r in sample_frame.itertuples()]
    assert len(keys) == len(set(keys)) == len(sample_frame)
    # Dropping any part of the key breaks uniqueness, so all four are needed.
    for dropped in ("RUN_NAME", "LATITUDE", "WYEAR"):
        columns = [c for c in ("RUN_NAME", "LATITUDE", "LONGITUDE", "WYEAR") if c != dropped]
        assert sample_frame.duplicated(columns).any(), dropped


def test_key_absorbs_float_noise():
    assert simulation_key("R", 0.0417, 35.1, 2000) == simulation_key("R", 0.04170000001, 35.1, 2000.0)


# -----------------------------------------------------------------------------
# Repeat uploads
# -----------------------------------------------------------------------------

def test_repeat_upload_of_complete_file_adds_nothing(tmp_path):
    csv = write_run_csv(tmp_path / "run.csv")
    session, log = RecordingSession(), tmp_path / "log.jsonl"

    first = ingest(session, tmp_path / "run.csv", "pp_run.csv", log)
    second = ingest(session, tmp_path / "run.csv", "pp_run.csv", log)

    assert first.status == "completed" and first.records_inserted == len(csv)
    assert second.status == "already_present"
    assert second.records_inserted == 0 and second.records_already_present == len(csv)
    assert len(session.simulations()) == len(csv)
    assert len(session.outputs()) == 2 * len(csv)          # HWAM + PRCP, not doubled
    assert any("repeat upload" in w for w in second.warnings)
    assert any("identical file (same SHA-256)" in w for w in second.warnings)


def test_repeat_upload_under_a_different_name_is_still_recognized(tmp_path):
    csv = write_run_csv(tmp_path / "tmpabc.csv")
    session = RecordingSession()
    ingest(session, tmp_path / "tmpabc.csv", "pp_run.csv", tmp_path / "log.jsonl")
    again = ingest(session, tmp_path / "tmpabc.csv", "renamed_copy.csv", tmp_path / "log.jsonl")
    assert again.records_inserted == 0
    assert len(session.keys()) == len(set(session.keys())) == len(csv)


# -----------------------------------------------------------------------------
# Overlapping partial uploads (the incident)
# -----------------------------------------------------------------------------

def test_overlapping_partial_uploads_store_each_simulation_once(tmp_path):
    source = write_run_csv(tmp_path / "full.csv", n_locations=40)
    n = len(source)
    # Part 1 of one random split, part 2 of a different random split.
    source.sample(frac=1, random_state=1).iloc[: n // 2].to_csv(tmp_path / "a.csv", index=False)
    source.sample(frac=1, random_state=2).iloc[n // 2:].to_csv(tmp_path / "b.csv", index=False)
    session, log = RecordingSession(), tmp_path / "log.jsonl"

    first = ingest(session, tmp_path / "a.csv", "pp_run.csv", log)
    second = ingest(session, tmp_path / "b.csv", "pp_run.csv", log)

    # Each half is flagged as incomplete (uneven years per location).
    assert any("looks incomplete" in w for w in first.warnings)
    assert any("looks incomplete" in w for w in second.warnings)
    # The overlap is detected and not duplicated.
    assert second.status == "partial_overlap"
    assert second.records_already_present > 0
    assert second.records_inserted == (n - n // 2) - second.records_already_present
    assert any("overlaps an earlier upload" in w for w in second.warnings)
    keys = session.keys()
    assert len(keys) == len(set(keys))
    missing = source_keys(source) - set(keys)
    assert missing  # the halves alone cannot recover everything...

    complete = ingest(session, tmp_path / "full.csv", "pp_run.csv", log)
    # ...and re-uploading the full file fills exactly the gap, still without duplicates.
    assert complete.records_inserted == len(missing)
    assert set(session.keys()) == source_keys(source)
    assert len(session.keys()) == n
    assert not any("looks incomplete" in w for w in complete.warnings)


def test_partial_file_after_complete_upload_is_flagged(tmp_path):
    source = write_run_csv(tmp_path / "full.csv")
    source.iloc[: len(source) // 3].to_csv(tmp_path / "part.csv", index=False)
    session = RecordingSession()
    ingest(session, tmp_path / "full.csv", "pp_run.csv", tmp_path / "log.jsonl")
    part = ingest(session, tmp_path / "part.csv", "pp_run.csv", tmp_path / "log.jsonl")
    assert part.records_inserted == 0
    assert any("this file may be partial" in w for w in part.warnings)


# -----------------------------------------------------------------------------
# Atomicity
# -----------------------------------------------------------------------------

def test_failed_output_insert_stores_nothing_and_can_be_retried(tmp_path):
    csv = write_run_csv(tmp_path / "run.csv")
    failing = RecordingSession(fail_outputs=True)
    result = ingest(failing, tmp_path / "run.csv", "pp_run.csv", tmp_path / "log.jsonl")

    assert result.status == "failed"
    assert "rolled back; nothing was stored" in result.errors[0]
    assert failing.simulations() == [] and failing.outputs() == []
    assert failing.commits == 0

    # Retrying after the failure stores everything exactly once.
    failing.fail_outputs = False
    retry = ingest(failing, tmp_path / "run.csv", "pp_run.csv", tmp_path / "log.jsonl")
    assert retry.records_inserted == len(csv)
    assert len(failing.simulations()) == len(csv)
    assert failing.commits == 1


def test_simulations_and_outputs_commit_together(tmp_path):
    write_run_csv(tmp_path / "run.csv")
    session = RecordingSession()
    ingest(session, tmp_path / "run.csv", "pp_run.csv", tmp_path / "log.jsonl")
    assert session.commits == 1
    stored_ids = {s.simulation_id for s in session.simulations()}
    assert {str(o.simulation_id) for o in session.outputs()} == {str(i) for i in stored_ids}


# -----------------------------------------------------------------------------
# Provenance
# -----------------------------------------------------------------------------

def test_original_file_name_is_preserved(tmp_path):
    write_run_csv(tmp_path / "tmpabc123.csv")
    session = RecordingSession()
    ingest(session, tmp_path / "tmpabc123.csv", "pp_KenMZ_MZ_RF_HighN_MZ_VSHT__pfrst75.csv",
           tmp_path / "log.jsonl")
    assert {s.experiment_name for s in session.simulations()} == {"pp_KenMZ_MZ_RF_HighN_MZ_VSHT__pfrst75"}


def test_provenance_log_records_each_upload(tmp_path):
    csv = write_run_csv(tmp_path / "run.csv")
    log = tmp_path / "log.jsonl"
    session = RecordingSession()
    first = ingest(session, tmp_path / "run.csv", "pp_run.csv", log)
    ingest(session, tmp_path / "run.csv", "pp_run.csv", log)

    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [e["status"] for e in entries] == ["completed", "already_present"]
    assert entries[0]["file_name"] == "pp_run.csv"
    assert entries[0]["sha256"] == first.source_sha256 and len(first.source_sha256) == 64
    assert entries[0]["rows_in_file"] == len(csv)
    assert (entries[0]["inserted"], entries[1]["already_present"]) == (len(csv), len(csv))
    assert entries[0]["run_names"] == ["MZ_RF_HighN_MZ_VSHT__pfrst75"]


def test_batch_ingestion_keeps_original_names(tmp_path):
    write_run_csv(tmp_path / "tmp1.csv", run_name="MZ_RF_HighN_MZ_BASE__pfrst0")
    write_run_csv(tmp_path / "tmp2.csv", run_name="MZ_RF_HighN_MZ_LNG__pfrst0")
    session = RecordingSession()
    service = IngestionService(session, log_path=str(tmp_path / "log.jsonl"))
    batch = asyncio.run(service.ingest_files([
        (str(tmp_path / "tmp1.csv"), "pp_BASE.csv"),
        (str(tmp_path / "tmp2.csv"), "pp_LNG.csv"),
    ]))
    assert batch["successful"] == 2
    assert {s.experiment_name for s in session.simulations()} == {"pp_BASE", "pp_LNG"}


# -----------------------------------------------------------------------------
# Rejection of files whose rows cannot be identified
# -----------------------------------------------------------------------------

def test_file_with_duplicate_identities_is_rejected(tmp_path):
    frame = write_run_csv(tmp_path / "run.csv")
    pd.concat([frame, frame.head(3)]).to_csv(tmp_path / "dup.csv", index=False)
    session = RecordingSession()
    result = ingest(session, tmp_path / "dup.csv", "dup.csv", tmp_path / "log.jsonl")
    assert result.status == "rejected"
    assert "share the same simulation identity" in result.errors[0]
    assert session.simulations() == []


@pytest.mark.skipif(not SAMPLE_SUMMARY.exists(), reason="sample summary.csv not present")
def test_summary_export_with_unique_rows_is_accepted(tmp_path):
    """The sample Summary.OUT-style export has no RUN_NAME; the parser falls
    back to TNAM ('CerealY'). One treatment x 37 years gives unique keys."""
    session = RecordingSession()
    result = ingest(session, SAMPLE_SUMMARY, "summary.csv", tmp_path / "log.jsonl")
    assert result.status == "completed"
    assert result.records_inserted == result.rows_in_file == 37
    assert {s.run_name for s in session.simulations()} == {"CerealY"}


def test_summary_export_with_colliding_treatments_is_rejected(tmp_path):
    """Several treatments sharing a TNAM at one location/year cannot be told
    apart without RUN_NAME, so the file is rejected rather than merged."""
    rows = [
        {"LAT": 1.875, "LONG": 35.458, "TNAM": "CerealY", "CR": "MZ", "WYEAR": 1984, "TRNO": trno,
         "HWAM": 5000 + trno}
        for trno in (1, 2)
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "multi.csv", index=False)
    session = RecordingSession()
    result = ingest(session, tmp_path / "multi.csv", "multi.csv", tmp_path / "log.jsonl")
    assert result.status == "rejected"
    assert session.simulations() == []


# -----------------------------------------------------------------------------
# Date parsing (pandas 2.2.2 crashed on Python 3.14)
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        (1984062, date(1984, 3, 2)),
        ("1984062", date(1984, 3, 2)),
        (1984001, date(1984, 1, 1)),
        (2020366, date(2020, 12, 31)),   # leap year
        (2021366, None),                 # not a leap year
        (2021000, None),
        (1984, None),
        (-99, None),
        (None, None),
        (".", None),
    ],
)
def test_dssat_dates_parse_without_pandas(value, expected):
    assert DSSATParser.parse_dssat_date(value) == expected


def test_real_dates_are_stored(tmp_path):
    write_run_csv(tmp_path / "run.csv", n_locations=1, years=[1984])
    session = RecordingSession()
    ingest(session, tmp_path / "run.csv", "pp_run.csv", tmp_path / "log.jsonl")
    simulation = session.simulations()[0]
    assert simulation.planting_date == date(1984, 3, 2)
    assert simulation.harvest_date == date(1984, 7, 23)


def test_parser_does_not_call_pandas_to_datetime():
    source = Path(DSSATParser.__module__.replace(".", "/") + ".py")
    text = (Path(__file__).resolve().parents[1] / source).read_text(encoding="utf-8")
    assert not re.search(r"\bpd\.to_datetime\(", text)


# -----------------------------------------------------------------------------
# SQL used for locking and the existing-key lookup
# -----------------------------------------------------------------------------

class CapturingSession:
    def __init__(self):
        self.statements = []

    async def execute(self, stmt):
        self.statements.append(stmt)

        class Result:
            def all(self_inner):
                return [("R", 0.0417, 35.125, 1984)]

        return Result()


def compiled(stmt) -> str:
    text = str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
    return re.sub(r"\s+", " ", text)


def test_repository_sql_for_locking_and_existing_keys(monkeypatch):
    monkeypatch.undo()  # use the real repository methods here
    session = CapturingSession()
    repo = SimulationRepository(session)
    asyncio.run(repo.lock_runs(["B", "A", "B"]))
    keys = asyncio.run(repo.get_existing_keys(["A"]))

    locks = [compiled(s) for s in session.statements[:2]]
    assert locks == ["SELECT pg_advisory_xact_lock(hashtext('A')) AS pg_advisory_xact_lock_1",
                     "SELECT pg_advisory_xact_lock(hashtext('B')) AS pg_advisory_xact_lock_1"]
    lookup = compiled(session.statements[2])
    assert lookup.startswith("SELECT simulations.run_name, simulations.latitude, simulations.longitude, "
                             "simulations.simulation_year FROM simulations")
    assert "simulations.run_name IN ('A')" in lookup
    assert keys == {("R", 0.0417, 35.125, 1984)}
