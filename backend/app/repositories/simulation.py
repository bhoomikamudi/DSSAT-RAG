"""Simulation repository."""
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy import func, and_, or_

from app.models.simulation import Simulation, SimulationOutput


def simulation_key(run_name: Any, latitude: Any, longitude: Any, year: Any) -> tuple:
    """Natural identity of a simulation: (run_name, lat, lon, year).

    RUN_NAME encodes the treatment (crop, irrigation, nitrogen, cultivar,
    planting stage); with the location and weather year it is unique in the
    DSSAT summary CSVs. The source file name is deliberately not part of the
    key, so the same data uploaded under another name is still recognized.
    Coordinates are rounded to 6 decimals (~0.1 m) to absorb float noise.
    """
    return (
        str(run_name),
        round(float(latitude), 6),
        round(float(longitude), 6),
        int(year),
    )


class SimulationRepository:
    """Repository for simulation operations."""

    def __init__(self, db: AsyncSession):
        """
        Initialize repository.

        Args:
            db: Database session
        """
        self.db = db

    async def get(self, id: UUID) -> Optional[Simulation]:
        """Get a single record by ID."""
        return await self.db.get(Simulation, id)

    async def get_by_experiment_and_run(
        self,
        experiment_name: str,
        run_name: str,
    ) -> Optional[Simulation]:
        """
        Get simulation by experiment name and run name.

        Args:
            experiment_name: Name of the experiment
            run_name: Name of the run

        Returns:
            Simulation record or None
        """
        stmt = (
            select(Simulation)
            .where(
                Simulation.experiment_name == experiment_name,
                Simulation.run_name == run_name,
            )
            .options(selectinload(Simulation.outputs))
        )

        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_country_state_district(
        self,
        country: str,
        state: Optional[str] = None,
        district: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Simulation]:
        """
        Get simulations by location.

        Args:
            country: Country name
            state: State name (optional)
            district: District name (optional)
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of simulation records
        """
        stmt = select(Simulation).where(Simulation.country == country).offset(skip).limit(limit)

        if state:
            stmt = stmt.where(Simulation.state == state)

        if district:
            stmt = stmt.where(Simulation.district == district)

        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_by_crop(
        self,
        crop: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Simulation]:
        """
        Get simulations by crop type.

        Args:
            crop: Crop name
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of simulation records
        """
        stmt = (
            select(Simulation)
            .where(Simulation.crop == crop)
            .offset(skip)
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_by_year(
        self,
        year: int,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Simulation]:
        """
        Get simulations by year.

        Args:
            year: Simulation year
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of simulation records
        """
        stmt = (
            select(Simulation)
            .where(Simulation.simulation_year == year)
            .offset(skip)
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_by_ecological_zone(
        self,
        ecological_zone: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Simulation]:
        """
        Get simulations by ecological zone.

        Args:
            ecological_zone: Ecological zone name
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of simulation records
        """
        stmt = (
            select(Simulation)
            .where(Simulation.ecological_zone == ecological_zone)
            .offset(skip)
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_by_cultivar(
        self,
        cultivar: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Simulation]:
        """
        Get simulations by cultivar.

        Args:
            cultivar: Cultivar name
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of simulation records
        """
        stmt = (
            select(Simulation)
            .where(Simulation.cultivar == cultivar)
            .offset(skip)
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_by_state(
        self,
        state: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Simulation]:
        """
        Get simulations by state.

        Args:
            state: State name
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of simulation records
        """
        stmt = (
            select(Simulation)
            .where(Simulation.state == state)
            .offset(skip)
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_by_district(
        self,
        district: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Simulation]:
        """
        Get simulations by district.

        Args:
            district: District name
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of simulation records
        """
        stmt = (
            select(Simulation)
            .where(Simulation.district == district)
            .offset(skip)
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_within_radius(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Simulation]:
        """
        Get simulations within a radius.

        Args:
            latitude: Center latitude
            longitude: Center longitude
            radius_km: Radius in kilometers
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of simulation records
        """
        from geoalchemy2 import functions as geo_func
        from sqlalchemy import text

        # Convert km to degrees (approximate)
        radius_deg = radius_km / 111.0

        stmt = (
            select(Simulation)
            .where(
                and_(
                    Simulation.latitude >= latitude - radius_deg,
                    Simulation.latitude <= latitude + radius_deg,
                    Simulation.longitude >= longitude - radius_deg,
                    Simulation.longitude <= longitude + radius_deg,
                )
            )
            .offset(skip)
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_within_polygon(
        self,
        polygon_wkt: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Simulation]:
        """
        Get simulations within a polygon.

        Args:
            polygon_wkt: Polygon in WKT format
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of simulation records
        """
        from geoalchemy2 import functions as geo_func
        from geoalchemy2 import WKTElement

        polygon_geom = WKTElement(polygon_wkt, srid=4326)

        stmt = (
            select(Simulation)
            .where(
                geo_func.ST_Within(Simulation.location, polygon_geom)
            )
            .offset(skip)
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_with_outputs(
        self,
        simulation_id: UUID,
    ) -> Optional[Simulation]:
        """
        Get simulation with its outputs.

        Args:
            simulation_id: Simulation ID

        Returns:
            Simulation record with outputs loaded
        """
        stmt = (
            select(Simulation)
            .where(Simulation.simulation_id == simulation_id)
            .options(selectinload(Simulation.outputs))
        )

        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_with_filters(
        self,
        crop: Optional[str] = None,
        cultivar: Optional[str] = None,
        planting_stage: Optional[str] = None,
        year: Optional[object] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
        irrigation: Optional[str] = None,
        nitrogen_level: Optional[str] = None,
        ecological_zone: Optional[str] = None,
        country: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
        spatial: Optional[Any] = None,
    ) -> List[Simulation]:
        """
        Get simulations with multiple filters.

        Args:
            crop: Crop type (optional)
            cultivar: Cultivar name (optional)
            year: Simulation year (optional)
            state: State name (optional)
            district: District name (optional)
            ecological_zone: Ecological zone (optional)
            country: Country name (optional)
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of simulation records
        """
        def equals_or_in(column, value):
            # A list (e.g. two cultivars) becomes an IN clause.
            if isinstance(value, (list, tuple, set)):
                return column.in_(list(value))
            return column == value

        filters = []
        if crop is not None:
            filters.append(equals_or_in(Simulation.crop, crop))
        if cultivar is not None:
            filters.append(equals_or_in(Simulation.cultivar, cultivar))

        if irrigation is not None:
            filters.append(equals_or_in(Simulation.irrigation, irrigation))

        if nitrogen_level is not None:
            filters.append(
                equals_or_in(Simulation.nitrogen_level, nitrogen_level)
            )

        if planting_stage is not None:
            filters.append(
                equals_or_in(Simulation.planting_stage, planting_stage)
            )
        if year is not None:
            try:
                from collections.abc import Iterable
                # Treat list-like as IN clause
                if isinstance(year, Iterable) and not isinstance(year, (str, bytes)):
                    years = list(year)
                    filters.append(Simulation.simulation_year.in_(years))
                else:
                    filters.append(Simulation.simulation_year == year)  # type: ignore[arg-type]
            except Exception:
                filters.append(Simulation.simulation_year == year)  # fallback
        if state is not None:
            filters.append(equals_or_in(Simulation.state, state))
        if district is not None:
            filters.append(equals_or_in(Simulation.district, district))
        if ecological_zone is not None:
            filters.append(equals_or_in(Simulation.ecological_zone, ecological_zone))
        if country is not None:
            filters.append(equals_or_in(Simulation.country, country))
        if spatial is not None:
            from app.services.spatial_sql import spatial_condition

            filters.append(spatial_condition(spatial))

        stmt = (
            select(Simulation)
            .where(*filters)
            .offset(skip)
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_aggregate(
        self,
        variable_code: str,
        aggregation: str,
        crop: Optional[str] = None,
        cultivar: Optional[str] = None,
        year: Optional[int] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get aggregated values.

        Args:
            variable_code: Variable code to aggregate
            aggregation: Aggregation function (avg, max, min, sum)
            crop: Crop type (optional filter)
            cultivar: Cultivar name (optional filter)
            year: Simulation year (optional filter)
            state: State name (optional filter)
            district: District name (optional filter)

        Returns:
            Dictionary with aggregation result
        """
        from sqlalchemy import func as sql_func

        # Build filters for simulation join
        sim_filters = []
        if crop is not None:
            sim_filters.append(Simulation.crop == crop)
        if cultivar is not None:
            sim_filters.append(Simulation.cultivar == cultivar)
        if year is not None:
            sim_filters.append(Simulation.simulation_year == year)
        if state is not None:
            sim_filters.append(Simulation.state == state)
        if district is not None:
            sim_filters.append(Simulation.district == district)

        # Build aggregation query
        agg_func = {
            "avg": sql_func.avg,
            "max": sql_func.max,
            "min": sql_func.min,
            "sum": sql_func.sum,
        }.get(aggregation.lower(), sql_func.avg)

        stmt = (
            select(
                agg_func(SimulationOutput.value).label("value"),
                sql_func.count().label("count"),
            )
            .join(Simulation, SimulationOutput.simulation_id == Simulation.simulation_id)
            .where(SimulationOutput.variable_code == variable_code)
        )

        if sim_filters:
            stmt = stmt.where(*sim_filters)

        result = await self.db.execute(stmt)
        row = result.fetchone()

        return {
            "aggregation": aggregation,
            "variable_code": variable_code,
            "value": row.value if row else None,
            "count": row.count if row else 0,
        }

    async def create(self, obj: Simulation) -> Simulation:
        """Create a new simulation."""
        self.db.add(obj)
        await self.db.commit()
        await self.db.refresh(obj)
        return obj

    async def lock_runs(self, run_names: List[str]) -> None:
        """Serialize ingestion of the same runs within the current transaction.

        Transaction-scoped advisory locks (released on commit/rollback), taken
        in sorted order to avoid deadlocks. No schema change is required.
        """
        for run_name in sorted(set(run_names)):
            await self.db.execute(
                select(func.pg_advisory_xact_lock(func.hashtext(run_name)))
            )

    async def get_existing_keys(self, run_names: List[str]) -> set:
        """Natural keys already stored for the given runs.

        Key: (run_name, latitude, longitude, simulation_year), the identity
        of a simulation in the DSSAT summary CSVs.
        """
        if not run_names:
            return set()
        result = await self.db.execute(
            select(
                Simulation.run_name,
                Simulation.latitude,
                Simulation.longitude,
                Simulation.simulation_year,
            ).where(Simulation.run_name.in_(sorted(set(run_names))))
        )
        return {simulation_key(*row) for row in result.all()}

    async def create_bulk(self, objs: List[Simulation]) -> List[Simulation]:
        """
        Create multiple simulations in bulk.

        Args:
            objs: List of Simulation ORM instances

        Returns:
            List of created simulation records with IDs populated
        """
        self.db.add_all(objs)
        await self.db.commit()
        for obj in objs:
            await self.db.refresh(obj)
        return objs


class SimulationOutputRepository:
    """Repository for simulation output operations."""

    def __init__(self, db: AsyncSession):
        """
        Initialize repository.

        Args:
            db: Database session
        """
        self.db = db

    async def get_by_simulation(
        self,
        simulation_id: UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> List[SimulationOutput]:
        """
        Get outputs for a simulation.

        Args:
            simulation_id: Simulation ID
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of output records
        """
        stmt = (
            select(SimulationOutput)
            .where(SimulationOutput.simulation_id == simulation_id)
            .offset(skip)
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_by_variable(
        self,
        variable_code: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[SimulationOutput]:
        """
        Get outputs by variable code.

        Args:
            variable_code: Variable code
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of output records
        """
        stmt = (
            select(SimulationOutput)
            .where(SimulationOutput.variable_code == variable_code)
            .offset(skip)
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def get_by_simulation_and_variable(
        self,
        simulation_id: UUID,
        variable_code: str,
    ) -> Optional[SimulationOutput]:
        """
        Get specific output for a simulation.

        Args:
            simulation_id: Simulation ID
            variable_code: Variable code

        Returns:
            Output record or None
        """
        stmt = (
            select(SimulationOutput)
            .where(
                SimulationOutput.simulation_id == simulation_id,
                SimulationOutput.variable_code == variable_code,
            )
        )

        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def create(self, obj: SimulationOutput) -> SimulationOutput:
        """Create a new output."""
        self.db.add(obj)
        await self.db.commit()
        await self.db.refresh(obj)
        return obj

    async def create_bulk(self, objs: List[SimulationOutput]) -> List[SimulationOutput]:
        """
        Create multiple outputs in bulk.

        Args:
            objs: List of SimulationOutput ORM instances

        Returns:
            List of created output records with IDs populated
        """
        self.db.add_all(objs)
        await self.db.commit()
        for obj in objs:
            await self.db.refresh(obj)
        return objs
