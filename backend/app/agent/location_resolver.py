"""Decide the spatial filter for one chat request.

Inputs, in priority order:
1. Coordinates written in the question ("near 0.79, 35.04").
2. A place named in the question ("near Kitale"), resolved by geocoding.
3. A point selected on the map (request latitude/longitude).
Radius: the question's radius, else the request's radius_km, else the
configured default.

The result is either a validated SpatialFilter, or a user-facing notice
explaining why the location could not be used. A failed or ambiguous place
lookup never falls back to the map point or to "no location", because that
would silently answer a different question.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional

from app.agent.location_parser import (
    extract_location,
    mentions_location_reset,
    mentions_selected_point,
)
from app.agent.models import SpatialFilter, validate_coordinates, validate_radius_km
from app.core.config import get_settings
from app.services.geocoding_service import GeocodingService

BoundsProvider = Callable[[], Awaitable[Optional[Dict[str, float]]]]


@dataclass
class SpatialResolution:
    query: str
    """The question with any location phrase removed (used for planning)."""
    spatial: Optional[SpatialFilter] = None
    notice: Optional[str] = None
    """Set when a requested location cannot be used; answer with it."""
    note: Optional[str] = None
    """Informational note, e.g. which place a name resolved to."""
    cleared: bool = False
    """The user asked to drop any location filter (e.g. "use all data")."""


def _short_name(display_name: str, parts: int = 3) -> str:
    return ", ".join(p.strip() for p in display_name.split(",")[:parts] if p.strip())


def _expand(bounds: Dict[str, float], radius_km: float) -> Dict[str, float]:
    pad = radius_km / 111.0
    return {
        "min_lat": bounds["min_lat"] - pad,
        "max_lat": bounds["max_lat"] + pad,
        "min_lon": bounds["min_lon"] - pad,
        "max_lon": bounds["max_lon"] + pad,
    }


async def resolve_spatial(
    query: str,
    *,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    radius_km: Optional[float] = None,
    geocoder: Optional[GeocodingService] = None,
    coverage_bounds: Optional[BoundsProvider] = None,
) -> SpatialResolution:
    settings = get_settings()
    mention = extract_location(query)
    resolution = SpatialResolution(query=mention.cleaned_query or query)

    radius_value = (
        mention.radius_km
        if mention.radius_km is not None
        else radius_km
        if radius_km is not None
        else settings.SPATIAL_DEFAULT_RADIUS_KM
    )
    has_map_point = latitude is not None or longitude is not None

    # "Use all data" wins over a still-selected map point and over any area
    # carried from earlier questions (a place named in this question wins).
    if not mention.has_location and mentions_location_reset(query):
        resolution.cleared = True
        resolution.note = "Location filter removed; using all data."
        return resolution

    if not mention.has_location and not has_map_point:
        if mention.radius_km is not None:
            resolution.notice = (
                f"A {mention.radius_km:g} km radius was given, but no place or point. "
                "Name a place (e.g. 'near Kitale'), give coordinates, or select a point on the map."
            )
        elif mentions_selected_point(query):
            # "here" without a selected point must not silently become "all data".
            resolution.notice = (
                "The question refers to the selected location (\"here\"), but no map point is "
                "selected. Select a point on the map, or name a place or coordinates "
                "(e.g. 'near Kitale' or '0.79, 35.04')."
            )
        return resolution

    try:
        radius = validate_radius_km(radius_value, settings.SPATIAL_MAX_RADIUS_KM)
    except ValueError as exc:
        resolution.notice = str(exc)
        return resolution

    try:
        if mention.latitude is not None:
            lat, lon = validate_coordinates(mention.latitude, mention.longitude)
            resolution.spatial = SpatialFilter(
                latitude=lat, longitude=lon, radius_km=radius, source="coordinates",
                radius_from_question=mention.radius_km is not None,
            )
            return resolution

        if mention.place is not None:
            geocoder = geocoder or GeocodingService()
            bounds = await coverage_bounds() if coverage_bounds else None
            result = await geocoder.geocode(
                mention.place,
                prefer_bounds=_expand(bounds, radius) if bounds else None,
            )
            if result.status != "ok" or result.match is None:
                resolution.notice = result.message
                return resolution
            resolution.spatial = SpatialFilter(
                latitude=result.match.latitude,
                longitude=result.match.longitude,
                radius_km=radius,
                source="place",
                label=_short_name(result.match.name),
                radius_from_question=mention.radius_km is not None,
            )
            resolution.note = result.message
            if has_map_point:
                resolution.note += " The place named in the question was used instead of the selected map point."
            return resolution

        if latitude is None or longitude is None:
            resolution.notice = "Both latitude and longitude are needed for a map point."
            return resolution
        lat, lon = validate_coordinates(latitude, longitude)
        resolution.spatial = SpatialFilter(
            latitude=lat, longitude=lon, radius_km=radius, source="map",
            radius_from_question=mention.radius_km is not None,
        )
        return resolution

    except ValueError as exc:
        resolution.notice = str(exc)
        return resolution
