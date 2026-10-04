"""Agent orchestrator Pydantic models."""
import math
from datetime import date
from typing import Any, Dict, List, Optional, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.config import get_settings
from app.services.analysis_vocabulary import (
    normalize_cultivar,
    normalize_filter_field,
    normalize_group_by,
    normalize_operation,
    normalize_variable,
)


# =============================================================================
# QUERY PLANNER MODELS
# =============================================================================

class LocationFilter(BaseModel):
    """Location-based filtering criteria."""

    type: Literal["radius", "polygon", "country", "state", "district", "ecological_zone"]
    """Type of location filter."""

    # For radius search
    latitude: float | None = None
    longitude: float | None = None
    radius_meters: int | None = None

    # For polygon search
    polygon_wkt: str | None = None

    # For region-based search
    country: str | None = None
    state: str | None = None
    district: str | None = None
    ecological_zone: str | None = None


class QueryPlan(BaseModel):
    intent: Literal[
        "metadata",
        "aggregate",
        "spatial_search",
        "comparison",
        "trend",
        "explanation",
        "definition",
        "hybrid",
        "analysis",
    ]
    """The user's primary intent."""

    metric: Optional[str] = None
    """Target metric (e.g., HWAM, YIELD)."""

    aggregation: Optional[Literal["AVG", "MIN", "MAX", "COUNT", "SUM"]] = None
    """Aggregation function for aggregate queries."""

    filters: Dict[str, Any] = Field(default_factory=dict)
    """Filter criteria (crop, cultivar, year, etc.)."""

    location: Optional[LocationFilter] = None
    """Location-based filtering."""

    comparison: Optional[Dict[str, List[str]]] = None
    """For comparison queries - groups to compare."""

    time_range: Optional[Dict[str, int]] = None
    """Time range for trend analysis."""

    required_tools: List[Literal[
        "metadata",
        "spatial",
        "statistics",
        "cde",
        "embedding"
    ]]
    """Tools needed to fulfill this query."""

    response_type: Literal["summary", "detailed", "comparison", "trend"] = "summary"
    """Preferred response format."""


# =============================================================================
# EXECUTOR MODELS
# =============================================================================

class MetadataResult(BaseModel):
    """Metadata tool result."""

    simulations: List[Dict[str, Any]]
    """Matching simulation records."""

    total_count: int
    """Total number of matching simulations (not the size of `simulations`)."""

    returned_count: Optional[int] = None
    """How many records `simulations` holds (a capped sample for display)."""

    sample_limit: Optional[int] = None
    """Maximum size of the returned sample."""

    crops: List[str]
    """Available crop types in results."""

    cultivars: List[str]
    """Available cultivars in results."""

    years: List[int] = []
    """Years covered by results."""


class SpatialResult(BaseModel):
    """Spatial tool result."""

    simulations: List[Dict[str, Any]]
    """Simulations within spatial filter."""

    total_count: int
    """All matching simulations, from a count query (never the sample size)."""

    returned_count: int = 0
    """Simulations included in the capped sample."""

    sample_limit: Optional[int] = None
    """Maximum sample size requested."""

    bounds: Dict[str, float]
    """Bounding box of the returned sample."""

    distance_stats: Optional[Dict[str, float]] = None
    """Distance statistics for radius searches."""


class StatisticsResult(BaseModel):
    """Statistics tool result."""

    aggregation_type: str
    """Type of aggregation performed."""

    metric: str
    """Target metric."""

    value: Optional[float]
    """Aggregated value."""

    count: int
    """Number of records aggregated."""

    stddev: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None

    breakdown: Optional[Dict[str, Any]] = None
    """Breakdown by groups (e.g., cultivar, year)."""

    unit: Optional[str] = None
    """Unit of measurement."""


class MultiStatisticsResult(BaseModel):
    """Support multiple statistics (e.g., MAX and MIN) in one response."""
    results: List[StatisticsResult]


class CDEResult(BaseModel):
    """CDE tool result."""

    variable_definitions: List[Dict[str, str]]
    """Variable code to definition mappings."""

    relationships: List[Dict[str, Any]]
    """Variable relationships."""

    cultivar_info: Optional[Dict[str, Any]] = None
    """Cultivar-specific information."""

    species_info: Optional[Dict[str, Any]] = None
    """Species-specific information."""


class EmbeddingResult(BaseModel):
    """Embedding tool result."""

    documents: List[Dict[str, Any]]
    """Relevant documents and summaries."""

    scores: List[float]
    """Similarity scores."""

    sources: List[str]
    """Document sources (manuals, papers, summaries)."""


# =============================================================================
# CONTEXT BUILDER MODELS
# =============================================================================

class LLMContext(BaseModel):
    """Context prepared for response generation."""

    metadata: Optional[MetadataResult] = None
    statistics: Optional[StatisticsResult] = None
    additional_statistics: Optional[List[StatisticsResult]] = None
    spatial: Optional[SpatialResult] = None
    cde: Optional[CDEResult] = None
    embeddings: Optional[List[EmbeddingResult]] = None
    tool_outputs: Optional[List[Dict[str, Any]]] = None
    analysis: Optional["AnalysisResult"] = None
    spatial_scope: Optional[Dict[str, Any]] = None
    management_scope: Optional[Dict[str, Any]] = None
    """Management conditions (cultivar, planting date, irrigation, nitrogen)
    of the records behind the result; see ManagementService.scope."""
    """Applied radius filter and how much data it matched."""
    spatial_notice: Optional[str] = None
    """Set when the location could not be used; answered verbatim."""
    clarification: Optional[str] = None
    """A question back to the user (e.g. ambiguous metric); answered verbatim."""

    query_summary: str
    """Natural language summary of what was found."""

    data_quality: Literal["high", "medium", "low"] = "high"
    """Confidence in the data quality."""


# =============================================================================
# RESPONSE GENERATOR MODELS
# =============================================================================

class SourceReference(BaseModel):
    """Reference to data source."""

    type: Literal["metadata", "cde", "qdrant", "statistics"]
    """Source type."""

    id: Optional[str] = None
    """Source identifier."""

    description: str
    """Human-readable description."""


class ResponseGeneration(BaseModel):
    """Final response to user."""

    answer: str
    """Natural language answer."""

    sources: List[SourceReference]
    """Sources used for the answer."""

    simulations: Optional[List[Dict[str, Any]]] = None
    """Detailed simulation data if requested."""

    confidence: Literal["high", "medium", "low"]
    """Confidence in the answer."""

    limitations: Optional[List[str]] = None
    """Known limitations or caveats."""


# =============================================================================
# AGENTIC PLANNER V2 (Structured Outputs)
# =============================================================================

ToolName = Literal["query_simulation_data", "query_cde", "semantic_search"]


class SimulationSpatialInput(BaseModel):
    type: Optional[Literal["radius", "polygon", "country", "state", "district", "bounding_box", "nearest"]] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    radius_meters: Optional[int] = None
    polygon_wkt: Optional[str] = None
    country: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    bounding_box: Optional[Dict[str, float]] = None


class SimulationToolInput(BaseModel):
    filters: Dict[str, Any] = Field(default_factory=dict)
    metrics: List[str] = Field(default_factory=list)
    aggregation: Optional[Literal["AVG", "MAX", "MIN", "COUNT", "SUM"]] = None
    group_by: List[str] = Field(default_factory=list)
    spatial: Optional[SimulationSpatialInput] = None


class SimulationStatistics(BaseModel):
    aggregation_type: Optional[str] = None
    metric: Optional[str] = None
    value: Optional[float] = None
    count: int = 0
    stddev: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    breakdown: Optional[Dict[str, Any]] = None
    unit: Optional[str] = None


class SimulationToolOutput(BaseModel):
    simulations: List[Dict[str, Any]] = Field(default_factory=list)
    statistics: Optional[SimulationStatistics] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CDEToolInput(BaseModel):
    variables: List[str] = Field(default_factory=list)
    cultivars: List[str] = Field(default_factory=list)


class CDEToolOutput(BaseModel):
    definitions: Dict[str, Any] = Field(default_factory=dict)
    relationships: Dict[str, Any] = Field(default_factory=dict)


class SemanticToolInput(BaseModel):
    query: str
    top_k: int = 5


class SemanticToolOutput(BaseModel):
    documents: List[Dict[str, Any]] = Field(default_factory=list)


class PlannerToolCall(BaseModel):
    tool: ToolName
    parameters: Union[SimulationToolInput, CDEToolInput, SemanticToolInput]


class PlannerOutput(BaseModel):
    goal: str
    tools: List[PlannerToolCall]


# =============================================================================
# SEMANTIC PLANNER (Production-grade)
# =============================================================================

Operator = Literal["=", "!=", ">", ">=", "<", "<=", "BETWEEN", "IN", "LIKE", "CONTAINS"]


class FilterCondition(BaseModel):
    field: str
    operator: Operator
    value: Union[str, int, float, List[Union[str, int, float]]]


# =============================================================================
# SPATIAL FILTER
# =============================================================================

def validate_coordinates(latitude: Any, longitude: Any) -> tuple:
    """Return (lat, lon) as floats or raise ValueError with a clear message."""
    try:
        lat, lon = float(latitude), float(longitude)
    except (TypeError, ValueError):
        raise ValueError("Latitude and longitude must be numbers.")
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError("Latitude and longitude must be finite numbers.")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"Latitude {lat} is out of range; it must be between -90 and 90.")
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f"Longitude {lon} is out of range; it must be between -180 and 180.")
    return lat, lon


def validate_radius_km(radius_km: Any, max_radius_km: float) -> float:
    try:
        radius = float(radius_km)
    except (TypeError, ValueError):
        raise ValueError("Radius must be a number of kilometres.")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Radius must be greater than 0 km.")
    if radius > max_radius_km:
        raise ValueError(f"Radius {radius:g} km exceeds the maximum of {max_radius_km:g} km.")
    return radius


class SpatialFilter(BaseModel):
    """Restrict data to simulations within radius_km of a point.

    Applied in SQL before any statistics or analysis runs.
    """

    latitude: float
    longitude: float
    radius_km: float
    source: Literal["map", "place", "coordinates"] = "map"
    label: Optional[str] = None
    """Human-readable description, e.g. the geocoded place name."""
    radius_from_question: bool = Field(default=False, exclude=True)
    """True when the question itself stated the radius ("within 50 km of").
    An area inherited by a follow-up keeps such a radius; otherwise it takes
    the radius currently selected in the UI."""

    @model_validator(mode="after")
    def _validate(self) -> "SpatialFilter":
        self.latitude, self.longitude = validate_coordinates(self.latitude, self.longitude)
        self.radius_km = validate_radius_km(
            self.radius_km, get_settings().SPATIAL_MAX_RADIUS_KM
        )
        return self

    @property
    def radius_meters(self) -> float:
        return self.radius_km * 1000.0

    def describe(self) -> str:
        where = self.label or f"({self.latitude:.4f}, {self.longitude:.4f})"
        return f"within {self.radius_km:g} km of {where}"


# =============================================================================
# ANALYSIS (controlled Python analysis layer)
# =============================================================================

class AnalysisRequest(BaseModel):
    """Structured, validated description of a statistical analysis.

    Free-text values from the LLM (e.g. "rainfall", "regression") are
    canonicalized on construction; values that remain unknown are rejected
    later by AnalysisService with a user-facing explanation.
    """

    operation: str = "correlation"
    """correlation | linear_regression | quadratic_regression | descriptive_statistics"""

    x_variable: Optional[str] = None
    """Independent variable (DSSAT code, e.g. PRCP)."""

    y_variable: Optional[str] = None
    """Dependent variable (DSSAT code, e.g. HWAM)."""

    filters: List[FilterCondition] = Field(default_factory=list)
    group_by: Optional[str] = None
    """Optional grouping field: cultivar, year, planting_stage, ..."""

    include_plot: bool = True
    max_points: int = Field(default=1000, ge=10, le=5000)
    """Maximum scatter points returned for plotting (statistics use all rows)."""

    spatial: Optional[SpatialFilter] = None
    """Optional radius filter; applied in SQL before the analysis runs."""

    requires_group_variation: bool = False
    """True when the grouping field is itself the question's explanatory
    factor (e.g. "does nitrogen rate affect yield?"), so a single group
    means the dataset cannot answer the question."""

    @field_validator("operation", mode="before")
    @classmethod
    def _canonical_operation(cls, value: Any) -> Any:
        return normalize_operation(value) or "correlation"

    @field_validator("x_variable", "y_variable", mode="before")
    @classmethod
    def _canonical_variable(cls, value: Any) -> Any:
        return normalize_variable(value)

    @field_validator("group_by", mode="before")
    @classmethod
    def _canonical_group_by(cls, value: Any) -> Any:
        if isinstance(value, list):
            value = value[0] if value else None
        return normalize_group_by(value)

    @field_validator("filters", mode="after")
    @classmethod
    def _canonical_filters(cls, value: List[FilterCondition]) -> List[FilterCondition]:
        for condition in value:
            condition.field = normalize_filter_field(condition.field)
            if condition.field == "cultivar":
                condition.value = normalize_cultivar(condition.value)
        return value


class AnalysisGroupResult(BaseModel):
    """Result of one analysis on one subset (a group, or all rows)."""

    group: Optional[str] = None
    status: Literal["ok", "insufficient_data"] = "ok"
    sample_size: int = 0
    correlation: Optional[float] = None
    p_value: Optional[float] = None
    coefficients: Optional[Dict[str, Optional[float]]] = None
    r_squared: Optional[float] = None
    descriptive: Optional[Dict[str, Dict[str, Optional[float]]]] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None


class AnalysisResult(BaseModel):
    """JSON-serializable output of the analysis layer."""

    status: Literal["ok", "insufficient_data", "unavailable", "invalid"] = "ok"
    analysis_type: str
    x_variable: Optional[str] = None
    y_variable: Optional[str] = None
    x_unit: Optional[str] = None
    y_unit: Optional[str] = None
    filters: List[FilterCondition] = Field(default_factory=list)
    group_by: Optional[str] = None

    sample_size: int = 0
    """Valid rows used in the (overall) calculation."""
    rows_retrieved: int = 0
    rows_dropped: int = 0

    correlation: Optional[float] = None
    p_value: Optional[float] = None
    coefficients: Optional[Dict[str, Optional[float]]] = None
    r_squared: Optional[float] = None
    descriptive: Optional[Dict[str, Dict[str, Optional[float]]]] = None
    details: Dict[str, Any] = Field(default_factory=dict)

    groups: List[AnalysisGroupResult] = Field(default_factory=list)

    chart_type: Optional[
        Literal["scatter", "scatter_with_line", "scatter_with_curve", "bar"]
    ] = None
    chart_data: Optional[Dict[str, Any]] = None

    explanation: str = ""
    warnings: List[str] = Field(default_factory=list)


class SemanticOperation(BaseModel):
    operation: Literal[
        "aggregate",
        "definition",
        "semantic_search",
        "metadata",
        "count",
        "trend",
        "explanation",
        "spatial",
        "analysis",
    ]
    entity: Optional[Literal[
        "simulations",
        "simulation_outputs"
    ]] = None
    metric: Optional[str] = None
    aggregation: Optional[Literal["AVG", "MIN", "MAX", "COUNT", "SUM"]] = None
    filters: List[FilterCondition] = Field(default_factory=list)
    group_by: List[str] = Field(default_factory=list)
    independent: bool = True
    variable: Optional[str] = None  # for definition ops
    query: Optional[str] = None     # for semantic_search ops
    analysis: Optional[AnalysisRequest] = None  # for analysis ops


class SemanticPlan(BaseModel):
    goal: str
    intent: Literal[
    "metadata",
    "aggregate",
    "spatial_search",
    "comparison",
    "trend",
    "explanation",
    "definition",
    "hybrid",
    "analysis",
    ]
    operations: List[SemanticOperation]
    comparison_axis: Optional[str] = None
    comparison_mode: Optional[Literal["independent", "combined"]] = None
    comparison_values: Optional[List[Union[str, int, float]]] = None
    spatial: Optional[SpatialFilter] = None
    """Request-level radius filter applied to every data operation."""


# =============================================================================
# ERROR MODELS
# =============================================================================

class ToolError(BaseModel):
    """Tool execution error."""

    tool_name: str
    """Name of the tool that failed."""

    error_type: str
    """Type of error."""

    message: str
    """Error message."""

    details: Optional[Dict[str, Any]] = None
    """Additional error details."""


class OrchestratorResponse(BaseModel):
    """Orchestrator response wrapper."""

    success: bool
    """Whether the orchestration succeeded."""

    query_plan: Optional[QueryPlan] = None
    """Generated query plan."""

    context: Optional[LLMContext] = None
    """Built context."""

    response: Optional[ResponseGeneration] = None
    """Final response."""

    errors: List[ToolError] = []
    """Any tool execution errors."""

    timing: Dict[str, float]
    """Timing information for each step."""


# LLMContext references AnalysisResult, which is defined later in this module.
LLMContext.model_rebuild()
OrchestratorResponse.model_rebuild()
