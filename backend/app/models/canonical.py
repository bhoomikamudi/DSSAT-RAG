"""Canonical models for all DSSAT data types."""
from typing import List, Dict, Any, Optional
from datetime import date
from pydantic import BaseModel


class LocationModel(BaseModel):
    """Location model with coordinates."""

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    country: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    ecological_zone: Optional[str] = None


class SimulationModel(BaseModel):
    """Simulation canonical model."""

    run_name: str = ""
    experiment_name: str = ""
    crop: str = ""
    cultivar: str = ""
    irrigation: str = ""
    nitrogen_level: str = ""
    planting_stage: str = ""
    harvest_area: Optional[float] = None
    year: int = 2024
    planting_date: Optional[date] = None
    maturity_date: Optional[date] = None
    harvest_date: Optional[date] = None


class SimulationOutputModel(BaseModel):
    """Simulation output canonical model."""

    variable_code: str
    value: float
    unit: Optional[str] = None


class CanonicalSimulation(BaseModel):
    """Complete simulation canonical model."""

    simulation: SimulationModel
    location: LocationModel
    outputs: Dict[str, float]


class CDEntityType:
    """CDE entity types."""
    VARIABLE = "variable"
    RELATIONSHIP = "relationship"
    SYNONYM = "synonym"
    CULTIVAR_DEFINITION = "cultivar_definition"
    SPECIES = "species"
    ECOTYPE = "ecotype"


class CanonicalCDE(BaseModel):
    """CDE canonical model."""

    entity_type: str
    code: str
    attributes: Dict[str, Any]


class DocumentMetadata(BaseModel):
    """Document metadata."""

    source_file: Optional[str] = None
    created_at: Optional[str] = None
    author: Optional[str] = None
    version: Optional[str] = None


class CanonicalDocument(BaseModel):
    """Document canonical model."""

    title: str = ""
    text: str = ""
    metadata: DocumentMetadata = DocumentMetadata()
    document_type: str = "txt"


class IngestionResult(BaseModel):
    """Result of an ingestion operation."""

    file_name: str
    file_type: str
    records_processed: int = 0
    records_failed: int = 0
    simulation_ids: List[str] = []
    cde_entities: List[Dict[str, Any]] = []
    document_ids: List[str] = []
    errors: List[str] = []
    execution_time_ms: float = 0.0

    # Provenance and idempotency (summary CSVs)
    status: str = "completed"
    """completed | already_present | partial_overlap | rejected | failed"""
    source_sha256: Optional[str] = None
    rows_in_file: int = 0
    records_inserted: int = 0
    records_already_present: int = 0
    records_invalid: int = 0
    warnings: List[str] = []


class IngestionBatchResult(BaseModel):
    """Result of a batch ingestion operation."""

    total_files: int = 0
    successful: int = 0
    failed: int = 0
    results: List[IngestionResult] = []
