"""Spatial endpoints used by the map in the frontend."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db
from app.services.spatial_service import SpatialService

router = APIRouter()


@router.get("/config")
async def spatial_config() -> dict:
    """Default and maximum radius, and whether place lookup is available."""
    settings = get_settings()
    return {
        "default_radius_km": settings.SPATIAL_DEFAULT_RADIUS_KM,
        "max_radius_km": settings.SPATIAL_MAX_RADIUS_KM,
        "geocoder_enabled": settings.GEOCODER_ENABLED,
    }


@router.get("/coverage")
async def spatial_coverage(db: AsyncSession = Depends(get_db)) -> dict:
    """Where ingested simulation data exists (bounds and distinct locations)."""
    return await SpatialService(db).get_coverage()
