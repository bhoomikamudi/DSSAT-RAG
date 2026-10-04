"""Tool wrappers mapping planner tool calls to internal services.

These tools encapsulate business logic and data access; the LLM never sees SQL or schemas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.models import (
    AnalysisRequest,
    AnalysisResult,
    FilterCondition,
    SimulationToolInput,
    SpatialFilter,
    SimulationToolOutput,
    SimulationStatistics,
    CDEToolInput,
    CDEToolOutput,
    SemanticToolInput,
    SemanticToolOutput,
)
from app.services.analysis_service import (
    AnalysisService,
    AnalysisValidationError,
)
from app.services.analysis_vocabulary import FIELD_LABELS
from app.services.metadata_service import MetadataService
from app.services.spatial_service import SpatialService
from app.services.statistics_service import StatisticsService
from app.services.cde_service import CDEService
from app.services.embedding_service import EmbeddingService


# MetadataService.get_simulations returns at most this many records.
SIMULATION_SAMPLE_LIMIT = 100


class SimulationTool:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.meta = MetadataService(db)
        self.stats = StatisticsService(db)
        self.spatial = SpatialService(db)

    async def run(
        self,
        params: SimulationToolInput,
    ) -> SimulationToolOutput:
        filters = dict(params.filters or {})

        # Normalize multi-year strings like "2015,2016"
        y = filters.get("year")
        if isinstance(y, str) and "," in y:
            try:
                filters["year"] = [
                    int(v.strip())
                    for v in y.split(",")
                    if v.strip()
                ]
            except Exception:
                pass

        # Resolve crop text to database code
        if filters.get("crop"):
            try:
                resolved = await self.stats.resolve_crop_code(
                    str(filters["crop"])
                )
                if resolved:
                    filters["crop"] = resolved
            except Exception:
                pass

        spatial_filter = _radius_filter(params.spatial)
        if params.spatial and params.spatial.type:
            location = params.spatial
            if location.type == "radius" and spatial_filter is None:
                raise ValueError("A radius filter requires latitude, longitude, and a positive radius.")
            if location.type == "polygon":
                if not location.polygon_wkt:
                    raise ValueError("A polygon is required for polygon filtering.")
                spatial_filter = location
            elif location.type in {"country", "state", "district"}:
                if not getattr(location, location.type):
                    raise ValueError(f"A {location.type} is required for this location filter.")
                for field in ("country", "state", "district"):
                    value = getattr(location, field)
                    if value is not None:
                        current = filters.get(field)
                        if current is None:
                            filters[field] = value
                        else:
                            values = current if isinstance(current, list) else [current]
                            filters[field] = [item for item in values if item == value]
            elif location.type not in {"radius", "polygon"}:
                raise ValueError(f"Unsupported spatial filter: {location.type}")

        # The same predicate set supplies the capped list, true count, and
        # every statistic. A count failure must never become the sample size.
        simulations = await self.meta.get_simulations(
            **filters, spatial=spatial_filter, limit=SIMULATION_SAMPLE_LIMIT
        )
        total_count = await self.stats.count_simulations(**filters, spatial=spatial_filter)
        statistics = None

        if params.metrics and params.aggregation:
            metric = params.metrics[0]
            aggregation = params.aggregation

            breakdown = None

            if params.group_by:
                group_by = params.group_by[0]

                if group_by == "year":
                    group_by = "simulation_year"

                breakdown = await self.stats.calculate_breakdown(
                    variable_code=metric,
                    aggregation=aggregation,
                    group_by=group_by,
                    spatial=spatial_filter,
                    **filters,
                )

            result = await self.stats.calculate_aggregation(
                variable_code=metric,
                aggregation=aggregation,
                spatial=spatial_filter,
                **filters,
            )

            statistics = SimulationStatistics(**result)

            if breakdown is not None:
                statistics.breakdown = {
                    "group_by": params.group_by[0],
                    "values": breakdown,
                }

            if aggregation in ("MIN", "MAX"):
                try:
                    extremum = await self.stats.get_extremum_simulation(
                        variable_code=metric,
                        aggregation=aggregation,
                        spatial=spatial_filter,
                        **filters,
                    )

                    if extremum:
                        statistics.breakdown = {
                            **(statistics.breakdown or {}),
                            "extremum_location": {
                                "latitude": extremum.get("latitude"),
                                "longitude": extremum.get("longitude"),
                                "country": extremum.get("country"),
                                "state": extremum.get("state"),
                                "district": extremum.get("district"),
                                "year": extremum.get("year"),
                                "simulation_id": extremum.get(
                                    "simulation_id"
                                ),
                            },
                        }

                except Exception:
                    pass

        return SimulationToolOutput(
            simulations=simulations,
            statistics=statistics,
            metadata={
                "count": len(simulations),       # returned sample (capped)
                "total_count": total_count,       # all matching simulations
                "sample_limit": SIMULATION_SAMPLE_LIMIT,
            },
        )


class CDETool:
    def __init__(self, db=None):
        self.cde = CDEService(db)

    async def run(
        self,
        params: CDEToolInput,
    ) -> CDEToolOutput:
        definitions: Dict[str, Any] = {}
        relationships: Dict[str, Any] = {}

        for variable in params.variables or []:
            # First check standard DSSAT variable definitions
            definition = await self.cde.get_variable_definition(variable)

            # If not found, check project-specific reference_codes
            if definition is None:
                definition = await self.cde.get_project_mapping(variable)

            if definition:
                definitions[variable] = definition

            related_variables = (
                await self.cde.get_variable_relationships(variable)
            )

            if related_variables:
                relationships[variable] = related_variables

        return CDEToolOutput(
            definitions=definitions,
            relationships=relationships,
        )


class SemanticTool:
    def __init__(self):
        self.emb = EmbeddingService()

    async def run(
        self,
        params: SemanticToolInput,
    ) -> SemanticToolOutput:
        documents = []

        for collection in ["summaries", "manuals", "papers"]:
            try:
                results = await self.emb.search_similar(
                    query=params.query,
                    collection=collection,
                    top_k=params.top_k,
                )

                for result in results[:2]:
                    documents.append(
                        {
                            "id": result.get("id"),
                            **result.get("payload", {}),
                            "source": collection,
                            "score": result.get("score"),
                        }
                    )

            except Exception:
                continue

        return SemanticToolOutput(
            documents=documents,
        )




class AnalysisTool:
    """Runs a validated AnalysisRequest: PostgreSQL filters, Python analyzes."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.stats = StatisticsService(db)
        self.analysis = AnalysisService()

    @staticmethod
    def filters_to_kwargs(filters: List[FilterCondition]) -> Dict[str, Any]:
        """Convert validated filter conditions to StatisticsService kwargs."""

        kwargs: Dict[str, Any] = {}

        for condition in filters:
            value = condition.value

            if condition.field == "year":
                if condition.operator == "BETWEEN" and isinstance(value, list):
                    start, end = int(value[0]), int(value[-1])
                    kwargs["year"] = list(range(min(start, end), max(start, end) + 1))
                elif isinstance(value, list):
                    kwargs["year"] = [int(item) for item in value]
                else:
                    kwargs["year"] = int(value)
            else:
                kwargs[condition.field] = value

        return kwargs

    async def run(self, request: AnalysisRequest) -> AnalysisResult:
        try:
            available_variables = await self.stats.get_available_variables()
            self.analysis.validate(request, available_variables=available_variables)
        except AnalysisValidationError as exc:
            return self.analysis.invalid_result(request, str(exc), status=exc.status)

        filter_kwargs = self.filters_to_kwargs(request.filters)

        if filter_kwargs.get("crop") and not isinstance(filter_kwargs["crop"], list):
            try:
                resolved = await self.stats.resolve_crop_code(str(filter_kwargs["crop"]))
                if resolved:
                    filter_kwargs["crop"] = resolved
            except Exception:
                pass

        records = await self.stats.get_analysis_records(
            variables=self.analysis.required_variables(request),
            group_by=request.group_by,
            spatial=request.spatial,
            **filter_kwargs,
        )

        if not records:
            message = await self._explain_empty_result(request, filter_kwargs)
            return self.analysis.invalid_result(request, message, status="unavailable")

        return self.analysis.analyze(request, records)

    async def _explain_empty_result(
        self,
        request: AnalysisRequest,
        filter_kwargs: Dict[str, Any],
    ) -> str:
        """Say which requested filter value does not exist in the dataset."""

        area = f" {request.spatial.describe()}" if request.spatial else ""

        for field, requested in filter_kwargs.items():
            try:
                available = await self.stats.get_distinct_field_values(
                    field, spatial=request.spatial
                )
            except Exception:
                continue

            requested_values = requested if isinstance(requested, list) else [requested]
            available_text = {str(value) for value in available}
            missing = [
                value for value in requested_values
                if str(value) not in available_text
            ]
            if missing and len(missing) == len(requested_values):
                label = FIELD_LABELS.get(field, field)
                shown = ", ".join(map(str, available[:20])) or "none"
                return (
                    "The current dataset cannot support this analysis: it has no "
                    f"simulations with {label} = {', '.join(map(str, missing))}{area}. "
                    f"Available {label} values{area}: {shown}."
                )

        return (
            "No simulations in the current dataset match all of the requested "
            f"filters together{area}, so the analysis cannot be performed."
        )


def _radius_filter(spatial_input: Any) -> Optional[SpatialFilter]:
    """Convert a planner radius input into a SpatialFilter, if it is one."""
    if (
        spatial_input is None
        or spatial_input.type != "radius"
        or spatial_input.latitude is None
        or spatial_input.longitude is None
        or not spatial_input.radius_meters
    ):
        return None
    return SpatialFilter(
        latitude=spatial_input.latitude,
        longitude=spatial_input.longitude,
        radius_km=spatial_input.radius_meters / 1000.0,
    )
