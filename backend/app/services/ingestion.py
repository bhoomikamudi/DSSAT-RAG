"""Ingestion service for DSSAT data with multi-format support."""
import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.parsers.file_detector import detect_file_type
from app.parsers.csv_parser import DSSATParser as CSVParser
from app.parsers.cultivar_parser import parse_cul_file
from app.parsers.species_parser import parse_spe_file
from app.parsers.ecotype_parser import parse_eco_file
from app.parsers.cde_parser import parse_cde_file
from app.parsers.document_parser import parse_document_file

from app.models.canonical import (
    CanonicalSimulation,
    CanonicalCDE,
    CanonicalDocument,
    IngestionResult,
)
from app.mappers.canonical import (
    map_simulation_to_orm,
    map_outputs_to_orm,
)
from app.repositories.simulation import SimulationRepository, simulation_key

logger = logging.getLogger(__name__)


def _canonical_key(canonical: CanonicalSimulation) -> tuple:
    return simulation_key(
        canonical.simulation.run_name,
        canonical.location.latitude,
        canonical.location.longitude,
        canonical.simulation.year,
    )


class IngestionService:
    """Service for ingesting DSSAT data from multiple file types."""

    def __init__(self, db: AsyncSession, log_path: Optional[str] = None):
        """
        Initialize ingestion service.

        Args:
            db: Database session
            log_path: Provenance log (JSON lines); defaults to INGESTION_LOG_PATH
        """
        self.db = db
        self.log_path = Path(log_path or get_settings().INGESTION_LOG_PATH)

    async def ingest_file(
        self,
        file_path: str,
        file_name: str = None,
    ) -> IngestionResult:
        """
        Ingest a single file, automatically detecting type and routing to parser.

        Args:
            file_path: Path to the file
            file_name: Original file name (uploads are stored under temp names)

        Returns:
            IngestionResult with processing statistics
        """
        start_time = time.time()

        # Detect file type
        file_type = detect_file_type(file_path)

        if not file_type:
            return IngestionResult(
                file_name=file_name or file_path,
                file_type="unknown",
                records_failed=1,
                errors=["Unsupported file type"],
            )

        result = IngestionResult(
            file_name=file_name or file_path,
            file_type=file_type,
        )

        if file_type == "summary_csv":
            await self._ingest_summary_csv(file_path, file_name, result)
            result.execution_time_ms = (time.time() - start_time) * 1000
            return result

        try:
            # Route to appropriate parser based on file type
            if file_type == "cultivar":
                canonical_list = parse_cul_file(file_path)
            elif file_type == "species":
                canonical_list = parse_spe_file(file_path)
            elif file_type == "ecotype":
                canonical_list = parse_eco_file(file_path)
            elif file_type == "cde":
                cde_entities: List[CanonicalCDE] = parse_cde_file(file_path)
                result.cde_entities = [
                    {"entity_type": e.entity_type, "code": e.code}
                    for e in cde_entities
                ]
                canonical_list = []
            elif file_type == "document":
                document_list: List[CanonicalDocument] = parse_document_file(file_path)
                result.document_ids = [f"doc_{i}" for i in range(len(document_list))]
                canonical_list = []
            else:
                result.errors.append(f"Unknown file type: {file_type}")
                return result

            # Process simulations
            if canonical_list:
                simulation_orms = [
                    map_simulation_to_orm(c) for c in canonical_list
                ]

                # Create simulations in database
                sim_repo = SimulationRepository(self.db)
                created_sims = await sim_repo.create_bulk(simulation_orms)

                result.simulation_ids = [str(s.simulation_id) for s in created_sims]

                # Process outputs
                all_outputs = []
                for canonical, simulation in zip(canonical_list, created_sims):
                    outputs = map_outputs_to_orm(canonical, str(simulation.simulation_id))
                    all_outputs.extend(outputs)

                if all_outputs:
                    from app.repositories.simulation import SimulationOutputRepository
                    output_repo = SimulationOutputRepository(self.db)
                    await output_repo.create_bulk(all_outputs)

            result.records_processed = len(canonical_list) + len(result.cde_entities) + len(result.document_ids)

        except Exception as e:
            # Rollback on failure
            await self.db.rollback()
            result.errors.append(str(e))
            result.records_failed = 1

        end_time = time.time()
        result.execution_time_ms = (end_time - start_time) * 1000

        return result

    # ------------------------------------------------------------------ #
    # Summary CSVs: idempotent, atomic, with provenance
    # ------------------------------------------------------------------ #

    async def _ingest_summary_csv(
        self,
        file_path: str,
        file_name: Optional[str],
        result: IngestionResult,
    ) -> None:
        """Insert only simulations not already stored, in one transaction.

        - Identity: (run_name, latitude, longitude, simulation_year).
        - Re-ingesting a complete file inserts nothing (status already_present).
        - Overlapping partial uploads insert only the missing simulations and
          are flagged (status partial_overlap).
        - Simulations and their outputs commit together or not at all.
        """
        source_name = file_name or os.path.basename(file_path)
        result.source_sha256 = self._sha256(file_path)
        previous = self._previous_uploads(result.source_sha256)
        if previous:
            result.warnings.append(
                f"An identical file (same SHA-256) was uploaded before as "
                f"'{previous[-1].get('file_name')}' at {previous[-1].get('timestamp')}."
            )

        try:
            canonical_list, rows_in_file = CSVParser.parse_csv_detailed(file_path, source_name)
        except Exception as exc:
            self._fail(result, f"Could not parse CSV: {exc}", status="rejected")
            self._record(result, source_name, run_names=[])
            return

        result.rows_in_file = rows_in_file
        result.records_invalid = rows_in_file - len(canonical_list)
        if result.records_invalid:
            result.warnings.append(
                f"{result.records_invalid} row(s) were skipped because they have no valid coordinates."
            )

        keys = [_canonical_key(c) for c in canonical_list]
        run_names = sorted({key[0] for key in keys})

        duplicate_in_file = len(keys) - len(set(keys))
        if duplicate_in_file:
            self._fail(
                result,
                f"{duplicate_in_file} row(s) in this file share the same simulation identity "
                "(RUN_NAME, latitude, longitude, year), so simulations cannot be told apart. "
                "Nothing was inserted. Files need a RUN_NAME column that distinguishes treatments.",
                status="rejected",
            )
            self._record(result, source_name, run_names)
            return

        result.warnings.extend(self._completeness_warnings(keys))

        repo = SimulationRepository(self.db)
        # The database's unique constraint (uq_simulations_run_location_year)
        # is the backstop: if it rejects a row that the key check missed
        # (e.g. a concurrent upload), roll back, re-read and retry once.
        for attempt in range(2):
            try:
                await repo.lock_runs(run_names)
                existing = await repo.get_existing_keys(run_names)

                new = [(c, k) for c, k in zip(canonical_list, keys) if k not in existing]
                result.records_already_present = len(keys) - len(new)

                simulations = [map_simulation_to_orm(c) for c, _ in new]
                self.db.add_all(simulations)
                await self.db.flush()  # assigns simulation ids

                outputs = []
                for (canonical, _), simulation in zip(new, simulations):
                    outputs.extend(map_outputs_to_orm(canonical, str(simulation.simulation_id)))
                self.db.add_all(outputs)
                await self.db.flush()

                await self.db.commit()
                break
            except IntegrityError as exc:
                await self.db.rollback()
                if attempt == 0:
                    logger.warning("Unique constraint rejected a simulation; retrying once: %s", exc.orig)
                    continue
                self._fail(
                    result,
                    "The database rejected a duplicate simulation (unique run/location/year) and the "
                    "upload was rolled back; nothing was stored. Retry the upload.",
                    status="failed",
                )
                result.records_already_present = 0
                self._record(result, source_name, run_names)
                return
            except Exception as exc:
                await self.db.rollback()
                self._fail(result, f"Ingestion failed and was rolled back; nothing was stored: {exc}",
                           status="failed")
                result.records_already_present = 0
                self._record(result, source_name, run_names)
                return

        result.records_inserted = len(simulations)
        result.records_processed = len(canonical_list)
        result.simulation_ids = [str(s.simulation_id) for s in simulations]

        stored_per_run = defaultdict(int)
        for key in existing:
            stored_per_run[key[0]] += 1
        file_per_run = defaultdict(int)
        for key in keys:
            file_per_run[key[0]] += 1

        if keys and result.records_inserted == 0:
            result.status = "already_present"
            result.warnings.append(
                f"All {len(keys):,} simulations in this file were already stored; nothing was added "
                "(repeat upload)."
            )
        elif result.records_already_present:
            result.status = "partial_overlap"
            result.warnings.append(
                f"{result.records_already_present:,} of {len(keys):,} simulations were already stored; "
                f"only {result.records_inserted:,} new ones were added. This file overlaps an earlier "
                "upload of the same run; check that the earlier upload was complete."
            )
        for run_name, stored in stored_per_run.items():
            if stored > file_per_run[run_name]:
                result.warnings.append(
                    f"The database already had {stored:,} simulations for {run_name}, more than the "
                    f"{file_per_run[run_name]:,} in this file; this file may be partial."
                )

        self._record(result, source_name, run_names)

    @staticmethod
    def _completeness_warnings(keys: Sequence[tuple]) -> List[str]:
        """Flag files where locations of a run have different sets of years.

        A DSSAT run simulates every year at every location, so uneven year
        coverage indicates a truncated or sampled file (as in the incident
        where random halves of two files were uploaded).
        """
        years = defaultdict(set)
        for run_name, lat, lon, year in keys:
            years[(run_name, lat, lon)].add(year)
        warnings = []
        by_run = defaultdict(list)
        for (run_name, _, _), run_years in years.items():
            by_run[run_name].append(run_years)
        for run_name, year_sets in by_run.items():
            all_years = set().union(*year_sets)
            incomplete = sum(1 for s in year_sets if s != all_years)
            if incomplete:
                warnings.append(
                    f"{run_name}: {incomplete} of {len(year_sets)} locations are missing some of the "
                    f"{len(all_years)} years present elsewhere in the file; the file looks incomplete."
                )
        return warnings

    # ------------------------------------------------------------------ #
    # Provenance
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sha256(file_path: str) -> str:
        digest = hashlib.sha256()
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _fail(result: IngestionResult, message: str, status: str) -> None:
        result.status = status
        result.errors.append(message)
        result.records_failed = 1
        result.records_inserted = 0

    def _previous_uploads(self, sha256: str) -> List[Dict[str, Any]]:
        if not self.log_path.exists():
            return []
        matches = []
        try:
            with self.log_path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("sha256") == sha256 and entry.get("status") != "failed":
                        matches.append(entry)
        except OSError as exc:
            logger.warning("Could not read ingestion log %s: %s", self.log_path, exc)
        return matches

    def _record(self, result: IngestionResult, source_name: str, run_names: List[str]) -> None:
        """Append one provenance line; a logging failure never fails ingestion."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "file_name": source_name,
            "sha256": result.source_sha256,
            "rows_in_file": result.rows_in_file,
            "rows_invalid": result.records_invalid,
            "inserted": result.records_inserted,
            "already_present": result.records_already_present,
            "run_names": run_names,
            "status": result.status,
            "warnings": result.warnings,
            "errors": result.errors,
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        except OSError as exc:
            logger.warning("Could not write ingestion log %s: %s", self.log_path, exc)
        logger.info("Ingestion provenance: %s", entry)

    def _parse_summary_csv(self, file_path: str) -> List[CanonicalSimulation]:
        """
        Parse a summary CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            List of CanonicalSimulation instances
        """
        return CSVParser.parse_csv(file_path)

    async def ingest_files(
        self,
        file_paths: Sequence[Union[str, Tuple[str, str]]],
    ) -> Dict[str, Any]:
        """
        Ingest multiple files.

        Args:
            file_paths: Paths, or (path, original file name) pairs

        Returns:
            Dictionary with batch results
        """
        results = []
        successful = 0
        failed = 0

        for item in file_paths:
            file_path, file_name = item if isinstance(item, tuple) else (item, None)
            try:
                result = await self.ingest_file(file_path, file_name)
                if not result.errors and (result.records_processed > 0 or result.status == "already_present"):
                    successful += 1
                else:
                    failed += 1
                results.append(result)
            except Exception as e:
                results.append(
                    IngestionResult(
                        file_name=file_name or file_path,
                        file_type="unknown",
                        records_failed=1,
                        errors=[str(e)],
                    )
                )
                failed += 1

        return {
            "total_files": len(file_paths),
            "successful": successful,
            "failed": failed,
            "results": results,
        }
