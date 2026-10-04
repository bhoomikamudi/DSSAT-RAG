"""Spatial Service - location-based filtering and analysis."""
import logging
from typing import List, Dict, Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from geoalchemy2 import functions as geo_func
from geoalchemy2 import WKTElement
from sqlalchemy import and_, func, text

from app.models.simulation import Simulation
from app.repositories.simulation import SimulationRepository
from app.services.spatial_sql import distance_km_to, within_radius

logger = logging.getLogger(__name__)


class SpatialService:
    """Service for spatial queries."""

    def __init__(self, db: AsyncSession):
        """
        Initialize service.

        Args:
            db: Database session
        """
        self.db = db
        self.sim_repo = SimulationRepository(db)

    # Legacy list searches. Each builds its predicate set once so the capped
    # sample and the true count (count_where) always describe the same rows.

    @staticmethod
    def radius_conditions(
        latitude: float, longitude: float, radius_meters: float, crop: Optional[str] = None
    ) -> List[Any]:
        # True geodesic radius (previously an approximate degree box).
        conditions = [within_radius(_RadiusFilter(latitude, longitude, radius_meters / 1000.0))]
        if crop:
            conditions.append(Simulation.crop == crop)
        return conditions

    @staticmethod
    def polygon_conditions(polygon_wkt: str, crop: Optional[str] = None) -> List[Any]:
        conditions = [
            geo_func.ST_Within(Simulation.location, WKTElement(polygon_wkt, srid=4326))
        ]
        if crop:
            conditions.append(Simulation.crop == crop)
        return conditions

    @staticmethod
    def region_conditions(
        country: Optional[str] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
        ecological_zone: Optional[str] = None,
        crop: Optional[str] = None,
    ) -> List[Any]:
        conditions: List[Any] = []
        if country:
            conditions.append(Simulation.country == country)
        if state:
            conditions.append(Simulation.state == state)
        if district:
            conditions.append(Simulation.district == district)
        if ecological_zone:
            conditions.append(Simulation.ecological_zone == ecological_zone)
        if crop:
            conditions.append(Simulation.crop == crop)
        return conditions

    async def search_where(self, conditions: List[Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Capped sample of simulations matching the conditions."""
        stmt = select(Simulation).where(*conditions).limit(limit)
        result = await self.db.execute(stmt)
        return [self._simulation_to_dict(sim) for sim in result.scalars().all()]

    async def count_where(self, conditions: List[Any]) -> int:
        """True number of simulations matching the conditions (no limit)."""
        stmt = select(func.count()).select_from(Simulation).where(*conditions)
        return int((await self.db.execute(stmt)).scalar_one())

    async def search_by_radius(
        self,
        latitude: float,
        longitude: float,
        radius_meters: int,
        crop: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Capped sample of simulations within the radius."""
        return await self.search_where(
            self.radius_conditions(latitude, longitude, radius_meters, crop), limit
        )

    async def search_by_polygon(
        self,
        polygon_wkt: str,
        crop: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Capped sample of simulations within the polygon."""
        return await self.search_where(self.polygon_conditions(polygon_wkt, crop), limit)

    async def search_by_country(
        self,
        country: str,
        state: Optional[str] = None,
        district: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Capped sample by country (and optionally state/district)."""
        return await self.search_where(
            self.region_conditions(country=country, state=state, district=district), limit
        )

    async def search_by_state(
        self,
        state: str,
        crop: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Capped sample by state."""
        return await self.search_where(self.region_conditions(state=state, crop=crop), limit)

    async def search_by_district(
        self,
        district: str,
        state: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Capped sample by district."""
        return await self.search_where(
            self.region_conditions(district=district, state=state), limit
        )

    async def search_by_ecological_zone(
        self,
        ecological_zone: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Capped sample by ecological zone."""
        return await self.search_where(
            self.region_conditions(ecological_zone=ecological_zone), limit
        )

    async def get_bounds(self, crop: Optional[str] = None) -> Dict[str, float]:
        """
        Get bounding box of all simulations.

        Args:
            crop: Filter by crop (optional)

        Returns:
            Dictionary with min/max lat/lon
        """
        from sqlalchemy import func

        stmt = select(
            func.min(Simulation.latitude).label("min_lat"),
            func.max(Simulation.latitude).label("max_lat"),
            func.min(Simulation.longitude).label("min_lon"),
            func.max(Simulation.longitude).label("max_lon")
        )

        if crop:
            stmt = stmt.where(Simulation.crop == crop)

        result = await self.db.execute(stmt)
        row = result.fetchone()

        return {
            "min_lat": row.min_lat,
            "max_lat": row.max_lat,
            "min_lon": row.min_lon,
            "max_lon": row.max_lon
        }

    async def calculate_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate distance between two points in meters.

        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates

        Returns:
            Distance in meters
        """
        # Haversine formula approximation
        from math import radians, sin, cos, sqrt, atan2

        R = 6371000  # Earth radius in meters

        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)

        a = sin(delta_lat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        return R * c

    async def summarize_radius(self, spatial: Any) -> Dict[str, Any]:
        """Count simulations and distinct locations inside a radius filter."""

        stmt = select(
            func.count().label("simulations"),
            func.count(
                func.distinct(func.concat(Simulation.latitude, ",", Simulation.longitude))
            ).label("locations"),
        ).where(within_radius(spatial))

        row = (await self.db.execute(stmt)).one()
        return {
            "simulations": int(row.simulations or 0),
            "locations": int(row.locations or 0),
        }

    async def nearest_location(
        self,
        latitude: float,
        longitude: float,
    ) -> Optional[Dict[str, Any]]:
        """Return the simulated location closest to a point, with distance."""

        distance = distance_km_to(latitude, longitude)
        stmt = (
            select(
                Simulation.latitude,
                Simulation.longitude,
                distance.label("distance_km"),
            )
            .order_by(distance)
            .limit(1)
        )
        row = (await self.db.execute(stmt)).first()
        if row is None:
            return None
        return {
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
            "distance_km": float(row.distance_km),
        }

    async def get_coverage(self, max_locations: int = 5000) -> Dict[str, Any]:
        """Describe where ingested data exists (for maps and messages).

        Everything is derived from the simulations table, so new CSVs are
        reflected automatically after ingestion.
        """

        bounds_row = (
            await self.db.execute(
                select(
                    func.min(Simulation.latitude).label("min_lat"),
                    func.max(Simulation.latitude).label("max_lat"),
                    func.min(Simulation.longitude).label("min_lon"),
                    func.max(Simulation.longitude).label("max_lon"),
                    func.count().label("simulations"),
                )
            )
        ).one()

        locations_stmt = (
            select(
                Simulation.latitude,
                Simulation.longitude,
                func.count().label("simulations"),
            )
            .group_by(Simulation.latitude, Simulation.longitude)
            .order_by(Simulation.latitude, Simulation.longitude)
            .limit(max_locations + 1)
        )
        rows = (await self.db.execute(locations_stmt)).all()

        locations = [
            {
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "simulations": int(row.simulations),
            }
            for row in rows[:max_locations]
        ]
        has_data = bool(bounds_row.simulations)

        return {
            "simulations": int(bounds_row.simulations or 0),
            "location_count": len(locations),
            "locations_truncated": len(rows) > max_locations,
            "bounds": (
                {
                    "min_lat": float(bounds_row.min_lat),
                    "max_lat": float(bounds_row.max_lat),
                    "min_lon": float(bounds_row.min_lon),
                    "max_lon": float(bounds_row.max_lon),
                }
                if has_data
                else None
            ),
            "locations": locations,
        }

    def _simulation_to_dict(self, sim: Simulation) -> Dict[str, Any]:
        """Convert simulation to dictionary."""
        return {
            "simulation_id": str(sim.simulation_id),
            "experiment_name": sim.experiment_name,
            "run_name": sim.run_name,
            "country": sim.country,
            "state": sim.state,
            "district": sim.district,
            "latitude": sim.latitude,
            "longitude": sim.longitude,
            "crop": sim.crop,
            "cultivar": sim.cultivar,
            "simulation_year": sim.simulation_year
        }


class _RadiusFilter:
    """Minimal radius filter for callers that pass raw arguments."""

    def __init__(self, latitude: float, longitude: float, radius_km: float):
        self.latitude = latitude
        self.longitude = longitude
        self.radius_km = radius_km
