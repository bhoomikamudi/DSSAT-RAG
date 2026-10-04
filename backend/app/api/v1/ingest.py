"""Ingestion endpoint for multi-format file uploads."""
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.services.ingestion import IngestionService
from app.parsers.file_detector import FileTypeDetector

router = APIRouter()


@router.post("/")
async def ingest_file(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Ingest a single DSSAT data file.

    Supports: summary.csv, .CUL, .ECO, .SPE, .CDE, PDF, Markdown, TXT

    Args:
        file: Uploaded file
        db: Database session

    Returns:
        Ingestion result with statistics
    """
    # Check if file type is supported
    file_type = FileTypeDetector.detect_type_from_filename(file.filename)

    if not file_type:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.filename}",
        )

    # Write uploaded file to temporary location
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=os.path.splitext(file.filename)[1],
        delete=False,
    ) as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        temp_path = tmp_file.name

    try:
        # Create ingestion service and process
        service = IngestionService(db)

        result = await service.ingest_file(temp_path, file.filename)

        return {
            **_result_payload(result),
            "execution_time_ms": round(result.execution_time_ms, 2),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest file: {str(e)}",
        )

    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@router.post("/batch")
async def ingest_files_batch(
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Ingest multiple DSSAT data files.

    Args:
        files: Uploaded files
        db: Database session

    Returns:
        Batch ingestion result with statistics for each file
    """
    import tempfile
    import os

    temp_paths = []

    try:
        # Write uploaded files to temporary locations
        for file in files:
            ext = os.path.splitext(file.filename)[1]
            tmp_file = tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=ext,
                delete=False,
            )
            content = await file.read()
            tmp_file.write(content)
            temp_paths.append((tmp_file.name, file.filename))
            tmp_file.close()

        # Create ingestion service and process
        service = IngestionService(db)

        # Pass the original names so provenance is not the temp file name.
        result = await service.ingest_files(temp_paths)

        return {
            "total_files": len(files),
            "successful": result.get("successful", 0),
            "failed": result.get("failed", 0),
            "results": [_result_payload(r) for r in result.get("results", [])],
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest files: {str(e)}",
        )

    finally:
        # Clean up temporary files
        for temp_path, _ in temp_paths:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def _result_payload(result) -> dict:
    """Ingestion result, including provenance and idempotency outcome."""
    return {
        "file_name": result.file_name,
        "file_type": result.file_type,
        "status": result.status,
        "source_sha256": result.source_sha256,
        "rows_in_file": result.rows_in_file,
        "records_processed": result.records_processed,
        "records_inserted": result.records_inserted,
        "records_already_present": result.records_already_present,
        "records_invalid": result.records_invalid,
        "records_failed": result.records_failed,
        "simulation_ids": result.simulation_ids,
        "cde_entities": result.cde_entities,
        "document_ids": result.document_ids,
        "warnings": result.warnings,
        "errors": result.errors,
    }
