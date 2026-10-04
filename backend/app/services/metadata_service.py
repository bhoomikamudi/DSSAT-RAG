"""Metadata Service - retrieves simulation records and basic information."""

import logging
from typing import List, Dict, Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func
from sqlalchemy.future import select

from app.models.simulation import Simulation, SimulationOutput
from app.repositories.simulation import (
    SimulationRepository,
    SimulationOutputRepository,
)

logger = logging.getLogger(__name__)


class MetadataService:
    """Service for metadata retrieval."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.sim_repo = SimulationRepository(db)
        self.output_repo = SimulationOutputRepository(db)

    async def get_record_count(
        self,
        entity: str = "simulations",
        spatial: Optional[Any] = None,
        **filters: Any,
    ) -> int:
        """Return the total number of records without loading them.

        spatial optionally restricts the count to a radius filter.
        """

        if entity == "simulations":
            statement = select(
                func.count()
            ).select_from(Simulation)

        elif entity == "simulation_outputs":
            statement = select(
                func.count()
            ).select_from(SimulationOutput)
            if spatial is not None or filters:
                statement = statement.join(
                    Simulation,
                    SimulationOutput.simulation_id == Simulation.simulation_id,
                )

        else:
            raise ValueError(
                f"Unsupported count entity: {entity}"
            )

        from app.services.statistics_service import StatisticsService

        statement = statement.where(
            *StatisticsService(self.db)._build_simulation_filters(spatial=spatial, **filters)
        )

        result = await self.db.execute(statement)
        total_count = int(result.scalar_one())

        logger.info(
            "MetadataService.get_record_count "
            "entity=%s total_count=%s",
            entity,
            total_count,
        )

        return total_count

    async def get_simulations(
        self,
        crop: Optional[str] = None,
        cultivar: Optional[str] = None,
        irrigation: Optional[str] = None,
        nitrogen_level: Optional[str] = None,
        planting_stage: Optional[str] = None,
        year: Optional[object] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
        country: Optional[str] = None,
        limit: int = 100,
        spatial: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Get up to `limit` simulations with optional filters (a sample).

        This is a capped list for display; use count_simulations for the
        true number of matches.
        """

        if isinstance(year, str) and "," in year:
            try:
                year = [
                    int(value.strip())
                    for value in year.split(",")
                    if value.strip()
                ]
            except Exception:
                pass

        filters = {
            "crop": crop,
            "cultivar": cultivar,
            "irrigation": irrigation,
            "nitrogen_level": nitrogen_level,
            "planting_stage": planting_stage,
            "year": year,
            "state": state,
            "district": district,
            "country": country,
        }

        filters = {
            key: value
            for key, value in filters.items()
            if value is not None
        }

        logger.info(
            "MetadataService.get_simulations filters=%s",
            filters,
        )

        simulations = await self.sim_repo.get_with_filters(
            skip=0,
            limit=limit,
            spatial=spatial,
            **filters,
        )

        records = [
            self._simulation_to_dict(simulation)
            for simulation in simulations
        ]

        logger.info(
            "MetadataService.get_simulations returned %s records",
            len(records),
        )

        return records

    async def get_simulation_by_id(
        self,
        simulation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a single simulation by ID."""

        from uuid import UUID

        simulation = await self.sim_repo.get(
            UUID(simulation_id)
        )

        return (
            self._simulation_to_dict(simulation)
            if simulation
            else None
        )

    async def get_simulation_outputs(
        self,
        simulation_id: str,
        variable_code: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get outputs for a simulation."""

        from uuid import UUID

        simulation = await self.sim_repo.get(
            UUID(simulation_id)
        )

        if not simulation:
            return []

        outputs = await self.output_repo.get_by_simulation(
            simulation_id=simulation.simulation_id,
            limit=1000,
        )

        return [
            {
                "variable_code": output.variable_code,
                "value": output.value,
                "unit": output.unit,
            }
            for output in outputs
            if (
                variable_code is None
                or output.variable_code == variable_code
            )
        ]

    async def get_unique_values(
        self,
        field: str,
        crop: Optional[str] = None,
    ) -> List[str]:
        """Get unique values for a field."""

        statement = select(
            getattr(Simulation, field)
        ).distinct()

        if crop:
            statement = statement.where(
                Simulation.crop == crop
            )

        result = await self.db.execute(statement)

        return [
            row[0]
            for row in result.all()
            if row[0]
        ]

    async def get_crop_summary(self) -> Dict[str, Any]:
        """Get summary statistics by crop."""

        statement = select(
            Simulation.crop,
            func.count().label("count"),
            func.avg(
                Simulation.simulation_year
            ).label("avg_year"),
        ).group_by(Simulation.crop)

        result = await self.db.execute(statement)

        return {
            "crops": [
                {
                    "name": row[0],
                    "simulation_count": row[1],
                    "avg_year": row[2],
                }
                for row in result.all()
            ]
        }

    def _simulation_to_dict(
        self,
        simulation: Simulation,
    ) -> Dict[str, Any]:
        """Convert a simulation model to a dictionary."""

        return {
            "simulation_id": str(
                simulation.simulation_id
            ),
            "experiment_name": simulation.experiment_name,
            "run_name": simulation.run_name,
            "country": simulation.country,
            "state": simulation.state,
            "district": simulation.district,
            "ecological_zone": simulation.ecological_zone,
            "latitude": simulation.latitude,
            "longitude": simulation.longitude,
            "crop": simulation.crop,
            "cultivar": simulation.cultivar,
            "irrigation": simulation.irrigation,
            "nitrogen_level": simulation.nitrogen_level,
            "planting_date": (
                str(simulation.planting_date)
                if simulation.planting_date
                else None
            ),
            "maturity_date": (
                str(simulation.maturity_date)
                if getattr(
                    simulation,
                    "maturity_date",
                    None,
                )
                else None
            ),
            "harvest_date": (
                str(simulation.harvest_date)
                if simulation.harvest_date
                else None
            ),
            "simulation_year": simulation.simulation_year,
        }