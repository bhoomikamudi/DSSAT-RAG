"""Pydantic schemas for query operations."""
from typing import List, Optional, Dict, Any
import math

from pydantic import BaseModel, Field, model_validator

from app.core.config import get_settings


class QueryFilters(BaseModel):
    """Query filters for simulation retrieval."""

    crop: Optional[str] = Field(None, description="Crop type")
    cultivar: Optional[str] = Field(None, description="Cultivar name")
    year: Optional[int] = Field(None, ge=1900, le=2100, description="Simulation year")
    state: Optional[str] = Field(None, description="State or region")
    district: Optional[str] = Field(None, description="District or county")
    ecological_zone: Optional[str] = Field(None, description="Ecological zone")
    country: Optional[str] = Field(None, description="Country")


class QueryPlan(BaseModel):
    """Query plan for chatbot queries."""

    intent: str = Field(
        ...,
        description="Type of query (aggregate, filter, search)",
    )
    metric: Optional[str] = Field(
        None,
        description="Metric to aggregate (e.g., HWAM, CWAM, PRCP)",
    )
    aggregation: Optional[str] = Field(
        None,
        description="Aggregation function (avg, max, min, sum)",
    )
    filters: QueryFilters = Field(default_factory=QueryFilters, description="Filter criteria")
    radius: Optional[float] = Field(
        None,
        ge=0,
        description="Radius in kilometers for spatial search",
    )
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Latitude for radius search")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude for radius search")


class QueryResult(BaseModel):
    """Query result structure."""

    metric: str
    aggregation: str
    value: float
    count: int
    filters_applied: Dict[str, Any]
    execution_time_ms: float


class ChatResponse(BaseModel):
    """Chat response with query results."""

    message: str = Field(..., description="Response message")
    result: Optional[QueryResult] = Field(None, description="Query result")
    raw_data: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Raw simulation data",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ChatRequest(BaseModel):
    """Chat request from user."""

    message: str = Field(..., description="User query message")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    latitude: Optional[float] = Field(
        None, ge=-90, le=90, description="Latitude of a point selected on the map"
    )
    longitude: Optional[float] = Field(
        None, ge=-180, le=180, description="Longitude of a point selected on the map"
    )
    radius_km: Optional[float] = Field(
        None, gt=0, description="Radius in km; the server default is used when omitted"
    )

    @model_validator(mode="after")
    def _check_point(self) -> "ChatRequest":
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("Provide both latitude and longitude for a map point, or neither.")
        for name in ("latitude", "longitude", "radius_km"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
        max_radius = get_settings().SPATIAL_MAX_RADIUS_KM
        if self.radius_km is not None and self.radius_km > max_radius:
            raise ValueError(f"radius_km must be at most {max_radius:g}.")
        return self
