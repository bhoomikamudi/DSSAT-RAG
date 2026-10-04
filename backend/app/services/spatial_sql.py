"""SQL expressions for radius filtering (PostGIS geography, true distances).

Kept free of agent imports so the statistics and spatial services can use it
without import cycles. A "spatial filter" here is any object with latitude,
longitude and radius_km attributes (normally app.agent.models.SpatialFilter).
"""
from __future__ import annotations

from typing import Any

from geoalchemy2 import Geography
from geoalchemy2.elements import WKTElement
from geoalchemy2 import functions as geo_func
from sqlalchemy import cast

from app.models.simulation import Simulation

# Plain "geography": the default Geography() type would compile to
# geography(GEOMETRY,-1), whose SRID is not WGS84 (4326).
GEOGRAPHY = Geography(geometry_type=None, srid=-1)


def point_geography(latitude: float, longitude: float) -> Any:
    """A WGS84 point as PostGIS geography (note ST_MakePoint takes lon, lat)."""
    return cast(
        geo_func.ST_SetSRID(geo_func.ST_MakePoint(longitude, latitude), 4326),
        GEOGRAPHY,
    )


def simulation_geography() -> Any:
    return cast(Simulation.location, GEOGRAPHY)


def within_radius(spatial: Any) -> Any:
    """WHERE clause: simulation location within radius_km of the point.

    Uses geodesic distance on the WGS84 spheroid, not a degree box.
    """
    return geo_func.ST_DWithin(
        simulation_geography(),
        point_geography(spatial.latitude, spatial.longitude),
        float(spatial.radius_km) * 1000.0,
    )


def spatial_condition(spatial: Any) -> Any:
    """Shared geometry predicate for samples, counts, and statistics."""
    if getattr(spatial, "type", None) == "polygon":
        if not spatial.polygon_wkt:
            raise ValueError("A polygon is required for polygon filtering.")
        return geo_func.ST_Within(
            Simulation.location, WKTElement(spatial.polygon_wkt, srid=4326)
        )
    return within_radius(spatial)


def distance_km_to(latitude: float, longitude: float) -> Any:
    """Expression: distance in km from the simulation location to a point."""
    return geo_func.ST_Distance(
        simulation_geography(), point_geography(latitude, longitude)
    ) / 1000.0
