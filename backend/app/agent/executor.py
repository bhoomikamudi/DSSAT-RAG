"""Executor - executes tools in parallel based on Planner output.

Refactored for dynamic tool dispatch with asyncio.gather.
Maintains compatibility with legacy QueryPlan.required_tools for now.
"""
import logging
from typing import List, Dict, Any, Optional

import asyncio
from datetime import datetime

from app.agent.models import (
    QueryPlan,
    PlannerOutput,
    PlannerToolCall,
    SemanticPlan,
    SemanticOperation,
    FilterCondition,
    SimulationToolInput,
    CDEToolInput,
    SemanticToolInput,
    MetadataResult,
    SpatialResult,
    StatisticsResult,
    MultiStatisticsResult,
    CDEResult,
    EmbeddingResult,
    ToolError,
    LLMContext,
    AnalysisRequest,
    AnalysisResult,
    SimulationSpatialInput,
    SpatialFilter,
)
from app.services.metadata_service import MetadataService
from app.services.spatial_service import SpatialService
from app.services.management_service import ManagementService
from app.services.statistics_service import StatisticsService
from app.services.cde_service import CDEService
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

# The legacy spatial tool returns at most this many simulations.
SPATIAL_SAMPLE_LIMIT = 100

# Operations that read simulation data and must honor a spatial filter.
DATA_OPERATIONS = {"aggregate", "metadata", "count", "trend", "analysis"}


class Executor:
    """Executes tools based on query plan."""

    def __init__(self, db_session):
        """
        Initialize executor.

        Args:
            db_session: Database session factory/instance
        """
        self.db_session = db_session

    async def execute(self, query_plan: QueryPlan, planner_output: Optional[PlannerOutput] = None, semantic_plan: Optional[SemanticPlan] = None) -> LLMContext:
        """
        Execute tools based on query plan.

        Args:
            query_plan: Generated query plan

        Returns:
            Built context for response generation
        """
        start_time = datetime.now()
        timing = {}
        errors = []

        # Create services
        db = await self._get_db_session()
        metadata_service = MetadataService(db)
        spatial_service = SpatialService(db)
        statistics_service = StatisticsService(db)
        cde_service = CDEService()
        embedding_service = EmbeddingService()

        # Prefer semantic plan → deterministic mapping to tools
        if semantic_plan and semantic_plan.operations:
            return await self._execute_from_semantic_plan(
                db=db,
                semantic_plan=semantic_plan,
                timing=timing,
                errors=errors,
            )

        # If we only have PlannerOutput with explicit tools, use it for dynamic dispatch
        if planner_output and planner_output.tools:
            return await self._execute_from_planner_output(
                db=db,
                planner_output=planner_output,
                timing=timing,
                errors=errors,
            )

        # Dynamic dispatch (legacy): build task list based on inferred required tools
        tasks = []
        task_names = []

        # Legacy mapping: we don't yet store PlannerOutput, so use required_tools
        if "metadata" in query_plan.required_tools:
            tasks.append(self._execute_metadata(metadata_service, query_plan, timing, errors))
            task_names.append("metadata")
        if "spatial" in query_plan.required_tools:
            tasks.append(self._execute_spatial(spatial_service, query_plan, timing, errors))
            task_names.append("spatial")
        if "statistics" in query_plan.required_tools:
            tasks.append(self._execute_statistics(statistics_service, query_plan, timing, errors))
            task_names.append("statistics")
        if "cde" in query_plan.required_tools:
            tasks.append(self._execute_cde(cde_service, query_plan, timing, errors))
            task_names.append("cde")
        if "embedding" in query_plan.required_tools:
            tasks.append(self._execute_embedding(embedding_service, query_plan, timing, errors))
            task_names.append("embedding")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        context_data: Dict[str, Any] = {}
        for i, result in enumerate(results):
            tool_name = task_names[i]
            if isinstance(result, Exception):
                errors.append(ToolError(tool_name=tool_name, error_type=type(result).__name__, message=str(result)))
                logger.error(f"Tool {tool_name} failed: {result}")
            else:
                context_data[tool_name] = result

        # Build final context
        llm_context = LLMContext(
            metadata=context_data.get("metadata"),
            spatial=context_data.get("spatial"),
            statistics=context_data.get("statistics"),
            cde=context_data.get("cde"),
            embeddings=context_data.get("embeddings", []),
            query_summary=self._build_query_summary(query_plan, context_data),
            data_quality=self._assess_data_quality(context_data)
        )

        # Calculate timing
        total_time = (datetime.now() - start_time).total_seconds()
        timing["total"] = total_time

        return llm_context

    async def _execute_from_semantic_plan(
        self,
        db,
        semantic_plan: SemanticPlan,
        timing: Dict[str, float],
        errors: List[ToolError],
    ) -> LLMContext:
        start_time = datetime.now()

        # Map semantic ops → tool calls
        from app.agent.tools import SimulationTool, CDETool, SemanticTool, AnalysisTool

        sim_tool = SimulationTool(db)
        cde_tool = CDETool(db)
        sem_tool = SemanticTool()
        analysis_tool = AnalysisTool(db)
        statistics_service = StatisticsService(db)
        metadata_service = MetadataService(db)

        tasks = []
        names: List[str] = []
        ops_meta: List[SemanticOperation] = []

        # One radius filter applies to every data operation in the plan. A
        # follow-up analysis may carry it over from the previous question.
        spatial = semantic_plan.spatial or next(
            (
                op.analysis.spatial
                for op in semantic_plan.operations
                if op.operation == "analysis" and op.analysis and op.analysis.spatial
            ),
            None,
        )
        spatial_scope: Optional[Dict[str, Any]] = None
        uses_data = any(
            op.operation in DATA_OPERATIONS for op in semantic_plan.operations
        )

        if spatial is not None and uses_data:
            spatial_service = SpatialService(db)
            summary = await spatial_service.summarize_radius(spatial)
            spatial_scope = {
                "filter": spatial.model_dump(),
                "description": spatial.describe(),
                **summary,
            }
            if summary["simulations"] == 0:
                nearest = await spatial_service.nearest_location(
                    spatial.latitude, spatial.longitude
                )
                spatial_scope["nearest"] = nearest
                notice = self._no_data_in_radius_message(spatial, nearest)
                timing["semantic_execute"] = (datetime.now() - start_time).total_seconds()
                return LLMContext(
                    spatial_scope=spatial_scope,
                    spatial_notice=notice,
                    query_summary=notice,
                    data_quality="low",
                )

        radius_input = (
            SimulationSpatialInput(
                type="radius",
                latitude=spatial.latitude,
                longitude=spatial.longitude,
                radius_meters=max(1, round(spatial.radius_meters)),
            )
            if spatial is not None
            else None
        )

        def filters_to_kwargs(filters: List[FilterCondition]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            # Handle year operators; others as equality for now
            for f in filters:
                if f.field == "year":
                    if f.operator == "=":
                        out["year"] = f.value
                    elif f.operator == "IN" and isinstance(f.value, list):
                        out["year"] = f.value
                    elif f.operator == "BETWEEN" and isinstance(f.value, list) and len(f.value) == 2:
                        # Every year in the inclusive range (previously only
                        # the two end years were used).
                        start, end = sorted((int(f.value[0]), int(f.value[1])))
                        out["year"] = list(range(start, end + 1))
                else:
                    if f.operator == "=":
                        out[f.field] = f.value
                    elif f.operator == "IN" and isinstance(f.value, list):
                        out[f.field] = f.value
                    # Future: add full operator support per field
            return out

        # Group independent ops to run in parallel
        for op in semantic_plan.operations:
            if op.operation == "count":
                entity = op.entity or "simulations"

                tasks.append(
                    metadata_service.get_record_count(
                        entity=entity,
                        spatial=spatial,
                        **filters_to_kwargs(op.filters),
                    )
                )
                names.append("count")
                ops_meta.append(op)

            elif op.operation in {"aggregate", "metadata"}:
                kwargs = filters_to_kwargs(op.filters)

                params = SimulationToolInput(
                    filters=kwargs,
                    metrics=[op.metric] if op.metric and op.operation == "aggregate" else [],
                    aggregation=op.aggregation if op.operation == "aggregate" else None,
                    group_by=op.group_by,
                    spatial=radius_input,
                )

                tasks.append(sim_tool.run(params))
                names.append("simulation")
                ops_meta.append(op)

            elif op.operation == "trend":
                kwargs = filters_to_kwargs(op.filters)

                tasks.append(
                    statistics_service.get_yearly_trend(
                        variable_code=op.metric or "HWAM",
                        spatial=spatial,
                        **kwargs,
                    )
                )
                names.append("trend")
                ops_meta.append(op)

            elif op.operation == "definition":
                logger.info(
                    "Definition operation: variable=%r, query=%r",
                    op.variable,
                    op.query,
                )

                params = CDEToolInput(
                    variables=[op.variable] if op.variable else []
                )
                tasks.append(cde_tool.run(params))
                names.append("cde")
                ops_meta.append(op)

            elif op.operation == "semantic_search":
                params = SemanticToolInput(
                    query=op.query or "",
                    top_k=5,
                )
                tasks.append(sem_tool.run(params))
                names.append("semantic")
                ops_meta.append(op)

            elif op.operation == "analysis":
                request = op.analysis or AnalysisRequest(
                    y_variable=op.metric,
                    filters=op.filters,
                    group_by=op.group_by[0] if op.group_by else None,
                )
                if spatial is not None:
                    request.spatial = spatial
                    op.analysis = request
                tasks.append(analysis_tool.run(request))
                names.append("analysis")
                ops_meta.append(op)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filters/grouping of the first successful data operation; the
        # management scope of the answer is described from the same filters.
        scope_source: Optional[Dict[str, Any]] = None

        # Build context similarly to planner_output path (reuse code where practical)
        metadata_res: Optional[MetadataResult] = None
        statistics_res: Optional[StatisticsResult] = None
        multi_stats: List[StatisticsResult] = []
        cde_res: Optional[CDEResult] = None
        embeddings_res: List[EmbeddingResult] = []
        analysis_res: Optional[AnalysisResult] = None
        tool_outputs: List[Dict[str, Any]] = []

        for i, result in enumerate(results):
            name = names[i]
            op = ops_meta[i]

            if isinstance(result, Exception):
                errors.append(
                    ToolError(
                        tool_name=name,
                        error_type=type(result).__name__,
                        message=str(result),
                    )
                )
                logger.error(
                    "Tool %s failed: %s",
                    name,
                    result,
                )
                continue

            if scope_source is None and name in {"count", "simulation", "trend"}:
                group = op.group_by[0] if op.group_by else None
                scope_source = {"filters": filters_to_kwargs(op.filters),
                                "group_by": "planting_stage" if group == "planting_date" else group}
            elif scope_source is None and name == "analysis" and result.status in {"ok", "insufficient_data"}:
                request = op.analysis
                scope_source = {"filters": AnalysisTool.filters_to_kwargs(request.filters) if request else {},
                                "group_by": request.group_by if request else None}

            if name == "count":
                entity = op.entity or "simulations"

                metadata_res = MetadataResult(
                    simulations=[],
                    total_count=int(result),
                    crops=[],
                    cultivars=[],
                    years=[],
                )

                tool_outputs.append({
                    "operation": op.model_dump(),
                    "tool": "count_records",
                    "output": {
                        "entity": entity,
                        "total_count": int(result),
                    },
                })

                continue
            if name == "simulation":
                sims = result.simulations or []
                metadata_res = MetadataResult(
                    simulations=sims,
                    # True number of matches; sims is a capped display sample.
                    total_count=int(result.metadata.get("total_count", len(sims))),
                    returned_count=len(sims),
                    sample_limit=result.metadata.get("sample_limit"),
                    crops=list({s.get("crop") for s in sims if s.get("crop")}),
                    cultivars=list({s.get("cultivar") for s in sims if s.get("cultivar")}),
                )
                if result.statistics and result.statistics.metric:
                    requested_aggregation = (op.aggregation or "").upper()
                    requested_value = (
                        result.statistics.min
                        if requested_aggregation == "MIN"
                        else result.statistics.max
                        if requested_aggregation == "MAX"
                        else result.statistics.value
                    )

                    stat = StatisticsResult(
                        aggregation_type=(
                            requested_aggregation
                            or result.statistics.aggregation_type
                            or ""
                        ),
                        metric=result.statistics.metric or "",
                        value=requested_value,
                        count=result.statistics.count,
                        stddev=getattr(result.statistics, "stddev", None),
                        min=getattr(result.statistics, "min", None),
                        max=getattr(result.statistics, "max", None),
                        breakdown=result.statistics.breakdown,
                        unit=result.statistics.unit,
                    )
                    multi_stats.append(stat)
                    statistics_res = stat
                tool_outputs.append({
                    "operation": op.model_dump(),
                    "tool": "query_simulation_data",
                    "output": {"simulations": len(sims), "statistics": (stat.model_dump() if 'stat' in locals() and stat else None)}
                })
            elif name == "trend":
                trend_count = sum(
                    row.get("count", 0) for row in result
                )
                trend_values = [
                    (row["avg_value"], row["count"])
                    for row in result
                    if row.get("avg_value") is not None
                ]
                trend_average = (
                    sum(value * count for value, count in trend_values) / trend_count
                    if trend_values and trend_count
                    else None
                )

                statistics_res = StatisticsResult(
                    aggregation_type=op.aggregation or "AVG",
                    metric=op.metric,
                    value=trend_average,
                    count=trend_count,
                    breakdown={
                        "group_by": (
                            op.group_by[0]
                            if op.group_by
                            else "year"
                        ),
                        "values": result,
                    },
                )

                tool_outputs.append({
                    "operation": op.model_dump(),
                    "tool": "query_simulation_data",
                    "output": {
                        "trend": result,
                        "record_count": trend_count,
                    },
                })
            elif name == "cde":
                var_defs = []
                for code, d in (result.definitions or {}).items():
                    d = dict(d)
                    d.setdefault("code", code)
                    var_defs.append(d)
                rels = []
                for code, rel in (result.relationships or {}).items():
                    rels.append({"variable": code, "related": rel})
                cde_res = CDEResult(variable_definitions=var_defs, relationships=rels)
                tool_outputs.append({
                    "operation": op.model_dump(),
                    "tool": "query_cde",
                    "output": {"variable_count": len(var_defs), "relationship_count": len(rels)}
                })
            elif name == "semantic":
                docs = result.documents or []
                if docs:
                    embeddings_res.append(EmbeddingResult(
                        documents=[{k: v for k, v in d.items() if k != "score"} for d in docs[:2]],
                        scores=[d.get("score", 0.0) for d in docs[:2]],
                        sources=[d.get("source", "unknown") for d in docs[:2]],
                    ))
                tool_outputs.append({
                    "operation": op.model_dump(),
                    "tool": "semantic_search",
                    "output": {"documents": len(docs)}
                })
            elif name == "analysis":
                analysis_res = result
                if spatial_scope and result.status in {"ok", "insufficient_data", "unavailable"}:
                    spatial_scope["analysis_rows_retrieved"] = result.rows_retrieved
                    spatial_scope["analysis_records"] = result.sample_size
                    result.explanation += "\n\n" + self._area_counts_sentence(
                        spatial_scope, result.rows_retrieved, result.sample_size,
                        self._describe_filters(result.filters),
                    )
                tool_outputs.append({
                    "operation": op.model_dump(),
                    "tool": "analyze_simulation_data",
                    "output": result.model_dump(
                        mode="json",
                        exclude={"chart_data", "groups", "explanation"},
                    ),
                })

        data_quality = self._assess_data_quality({"metadata": metadata_res, "spatial": None})
        if analysis_res is not None:
            data_quality = self._assess_analysis_quality(analysis_res)

        management_scope = None
        if scope_source is not None:
            try:
                management_scope = await ManagementService(db).scope(
                    scope_source["filters"], spatial=spatial, group_by=scope_source["group_by"]
                )
            except Exception as exc:  # the answer still stands without it
                logger.warning("Management scope unavailable: %s", exc)

        llm_context = LLMContext(
            metadata=metadata_res,
            spatial=None,
            statistics=statistics_res,
            additional_statistics=multi_stats if len(multi_stats) > 1 else None,
            cde=cde_res,
            embeddings=embeddings_res,
            tool_outputs=tool_outputs or None,
            analysis=analysis_res,
            spatial_scope=spatial_scope,
            management_scope=management_scope,
            query_summary=self._build_query_summary(QueryPlan(intent="metadata", filters={}, required_tools=["metadata"], response_type="summary"), {
                "metadata": metadata_res,
                "statistics": statistics_res,
                "cde": cde_res,
                "embeddings": embeddings_res,
            }),
            data_quality=data_quality,
        )
        timing["semantic_execute"] = (datetime.now() - start_time).total_seconds()
        return llm_context

    async def _execute_from_planner_output(
        self,
        db,
        planner_output: PlannerOutput,
        timing: Dict[str, float],
        errors: List[ToolError],
    ) -> LLMContext:
        start_time = datetime.now()

        # Instantiate tool wrappers
        from app.agent.tools import SimulationTool, CDETool, SemanticTool

        sim_tool = SimulationTool(db)
        cde_tool = CDETool(db)
        sem_tool = SemanticTool()

        tasks = []
        names: List[str] = []
        calls_meta: List[PlannerToolCall] = []

        for call in planner_output.tools:
            if call.tool == "query_simulation_data":
                tasks.append(sim_tool.run(call.parameters))
                names.append("simulation")
                calls_meta.append(call)
            elif call.tool == "query_cde":
                tasks.append(cde_tool.run(call.parameters))
                names.append("cde")
                calls_meta.append(call)
            elif call.tool == "semantic_search":
                tasks.append(sem_tool.run(call.parameters))
                names.append("semantic")
                calls_meta.append(call)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Map tool outputs to context models
        metadata_res: Optional[MetadataResult] = None
        statistics_res: Optional[StatisticsResult] = None
        multi_stats: List[StatisticsResult] = []
        cde_res: Optional[CDEResult] = None
        embeddings_res: List[EmbeddingResult] = []

        tool_outputs: List[Dict[str, Any]] = []
        for i, result in enumerate(results):
            name = names[i]
            call = calls_meta[i]
            if isinstance(result, Exception):
                errors.append(ToolError(tool_name=name, error_type=type(result).__name__, message=str(result)))
                logger.error(f"Tool {name} failed: {result}")
                continue
            if name == "simulation":
                # Build MetadataResult
                sims = result.simulations or []
                metadata_res = MetadataResult(
                    simulations=sims,
                    # True number of matches; sims is a capped display sample.
                    total_count=int(result.metadata.get("total_count", len(sims))),
                    returned_count=len(sims),
                    sample_limit=result.metadata.get("sample_limit"),
                    crops=list({s.get("crop") for s in sims if s.get("crop")}),
                    cultivars=list({s.get("cultivar") for s in sims if s.get("cultivar")}),
                )
                # Build StatisticsResult if present
                if result.statistics and result.statistics.metric:
                    requested_aggregation = (
                        getattr(call.parameters, "aggregation", None) or ""
                    ).upper()
                    requested_value = (
                        result.statistics.min
                        if requested_aggregation == "MIN"
                        else result.statistics.max
                        if requested_aggregation == "MAX"
                        else result.statistics.value
                    )

                    stat = StatisticsResult(
                        aggregation_type=(
                            requested_aggregation
                            or result.statistics.aggregation_type
                            or ""
                        ),
                        metric=result.statistics.metric or "",
                        value=requested_value,
                        count=result.statistics.count,
                        stddev=getattr(result.statistics, "stddev", None),
                        min=getattr(result.statistics, "min", None),
                        max=getattr(result.statistics, "max", None),
                        breakdown=result.statistics.breakdown,
                        unit=result.statistics.unit,
                    )
                    # If multiple simulation tool calls are present (e.g., MAX and MIN), collect them
                    multi_stats.append(stat)
                    statistics_res = stat
                # Collect tool-tagged output with parameters
                params = call.parameters.model_dump() if hasattr(call.parameters, "model_dump") else {}
                tool_outputs.append({
                    "tool": call.tool,
                    "parameters": params,
                    "output": {
                        "simulations": len(sims),
                        "statistics": (stat.model_dump() if 'stat' in locals() and stat else None)
                    }
                })
                timing["query_simulation_data"] = (datetime.now() - start_time).total_seconds()
            elif name == "cde":
                var_defs = []
                for code, d in (result.definitions or {}).items():
                    d = dict(d)
                    d.setdefault("code", code)
                    var_defs.append(d)
                rels = []
                for code, rel in (result.relationships or {}).items():
                    rels.append({"variable": code, "related": rel})
                cde_res = CDEResult(variable_definitions=var_defs, relationships=rels)
                # tool-tagged output
                params = call.parameters.model_dump() if hasattr(call.parameters, "model_dump") else {}
                tool_outputs.append({
                    "tool": call.tool,
                    "parameters": params,
                    "output": {"variable_count": len(var_defs), "relationship_count": len(rels)}
                })
                timing["query_cde"] = (datetime.now() - start_time).total_seconds()
            elif name == "semantic":
                docs = result.documents or []
                if docs:
                    embeddings_res.append(EmbeddingResult(
                        documents=[{k: v for k, v in d.items() if k != "score"} for d in docs[:2]],
                        scores=[d.get("score", 0.0) for d in docs[:2]],
                        sources=[d.get("source", "unknown") for d in docs[:2]],
                    ))
                params = call.parameters.model_dump() if hasattr(call.parameters, "model_dump") else {}
                tool_outputs.append({
                    "tool": call.tool,
                    "parameters": params,
                    "output": {"documents": len(docs)}
                })
                timing["semantic_search"] = (datetime.now() - start_time).total_seconds()

        # If multiple stats were computed, include all; keep first in statistics for backward compatibility.
        llm_context = LLMContext(
            metadata=metadata_res,
            spatial=None,
            statistics=statistics_res,
            additional_statistics=multi_stats if len(multi_stats) > 1 else None,
            cde=cde_res,
            embeddings=embeddings_res,
            tool_outputs=tool_outputs or None,
            query_summary=self._build_query_summary(QueryPlan(intent="metadata", filters={}, required_tools=["metadata"], response_type="summary"), {
                "metadata": metadata_res,
                "statistics": statistics_res,
                "cde": cde_res,
                "embeddings": embeddings_res,
            }),
            data_quality=self._assess_data_quality({"metadata": metadata_res, "spatial": None}),
        )
        return llm_context

    async def _get_db_session(self):
        """Get database session."""
        # In production, this would get a session from the pool
        # For now, we'll assume it's passed in or create one
        if hasattr(self.db_session, 'session'):
            return await self.db_session.session()
        return self.db_session

    async def _execute_metadata(
        self,
        service: MetadataService,
        query_plan: QueryPlan,
        timing: Dict[str, float],
        errors: List[ToolError]
    ) -> Optional[MetadataResult]:
        """Execute metadata tool."""
        start = datetime.now()

        try:
            filters = query_plan.filters.copy()

            # Remove location from filters (handled by spatial)
            filters.pop("latitude", None)
            filters.pop("longitude", None)
            filters.pop("radius_meters", None)

            simulations = await service.get_simulations(**filters)
            total = await StatisticsService(service.db).count_simulations(**filters)

            result = MetadataResult(
                simulations=simulations,
                total_count=total,
                returned_count=len(simulations),
                sample_limit=100,
                crops=list(set(s.get("crop") for s in simulations if s.get("crop"))),
                cultivars=list(set(s.get("cultivar") for s in simulations if s.get("cultivar")))
            )

            timing["metadata"] = (datetime.now() - start).total_seconds()
            return result

        except Exception as e:
            errors.append(ToolError(
                tool_name="metadata",
                error_type=type(e).__name__,
                message=str(e)
            ))
            raise

    async def _execute_spatial(
        self,
        service: SpatialService,
        query_plan: QueryPlan,
        timing: Dict[str, float],
        errors: List[ToolError]
    ) -> Optional[SpatialResult]:
        """Execute spatial tool."""
        start = datetime.now()

        try:
            if not query_plan.location:
                return None

            location = query_plan.location
            filters = query_plan.filters.copy()

            crop = filters.get("crop")

            # One predicate set feeds both the capped sample and the true
            # count; the sample size is never reported as the total.
            conditions = None
            if location.type == "radius":
                if location.latitude is not None and location.longitude is not None and location.radius_meters:
                    conditions = service.radius_conditions(
                        location.latitude, location.longitude, location.radius_meters, crop
                    )
            elif location.type == "polygon":
                if location.polygon_wkt:
                    conditions = service.polygon_conditions(location.polygon_wkt, crop)
            elif location.type == "country":
                if location.country:
                    conditions = service.region_conditions(
                        country=location.country, state=location.state, district=location.district
                    )
            elif location.type == "state":
                if location.state:
                    conditions = service.region_conditions(state=location.state, crop=crop)
            elif location.type == "district":
                if location.district:
                    conditions = service.region_conditions(
                        district=location.district, state=location.state
                    )
            elif location.type == "ecological_zone":
                if location.ecological_zone:
                    conditions = service.region_conditions(ecological_zone=location.ecological_zone)

            if conditions is None:
                # Incomplete location: nothing was searched, so nothing matched.
                simulations, total_count = [], 0
            else:
                simulations = await service.search_where(conditions, limit=SPATIAL_SAMPLE_LIMIT)
                total_count = await service.count_where(conditions)

            result = SpatialResult(
                simulations=simulations,
                total_count=total_count,
                returned_count=len(simulations),
                sample_limit=SPATIAL_SAMPLE_LIMIT,
                bounds=self._calculate_bounds(simulations)
            )

            timing["spatial"] = (datetime.now() - start).total_seconds()
            return result

        except Exception as e:
            errors.append(ToolError(
                tool_name="spatial",
                error_type=type(e).__name__,
                message=str(e)
            ))
            raise

    async def _execute_statistics(
        self,
        service: StatisticsService,
        query_plan: QueryPlan,
        timing: Dict[str, float],
        errors: List[ToolError]
    ) -> Optional[StatisticsResult]:
        """Execute statistics tool."""
        start = datetime.now()

        try:
            if not query_plan.metric or not query_plan.aggregation:
                return None

            filters = query_plan.filters.copy()

            # Support comma-separated year ranges like "2015,2016" → [2015, 2016]
            year = filters.get("year")
            if isinstance(year, str) and "," in year:
                try:
                    year = [int(y.strip()) for y in year.split(",") if y.strip()]
                except Exception:
                    pass
            result = await service.calculate_aggregation(
                variable_code=query_plan.metric,
                aggregation=query_plan.aggregation,
                crop=filters.get("crop"),
                cultivar=filters.get("cultivar"),
                planting_stage=filters.get("planting_stage"),
                year=year,
            )

            timing["statistics"] = (datetime.now() - start).total_seconds()
            return StatisticsResult(**result)

        except Exception as e:
            errors.append(ToolError(
                tool_name="statistics",
                error_type=type(e).__name__,
                message=str(e)
            ))
            raise

    async def _execute_cde(
        self,
        service: CDEService,
        query_plan: QueryPlan,
        timing: Dict[str, float],
        errors: List[ToolError]
    ) -> Optional[CDEResult]:
        """Execute CDE tool."""
        start = datetime.now()

        try:
            variable_definitions = []

            # Get definitions for relevant variables
            if query_plan.metric:
                definition = await service.get_variable_definition(query_plan.metric)
                if definition:
                    variable_definitions.append(definition)

            # Get relationships
            relationships = []
            if query_plan.metric:
                rels = await service.get_variable_relationships(query_plan.metric)
                for rel in rels:
                    defn = await service.get_variable_definition(rel)
                    if defn:
                        relationships.append({
                            "variable": rel,
                            **defn
                        })

            # Get cultivar info if cultivar filter exists
            cultivar_info = None
            if query_plan.filters.get("cultivar"):
                crop = query_plan.filters.get("crop", "")
                cultivars = await service.get_cultivar_info(crop)
                for c in cultivars:
                    if c.get("name") == query_plan.filters["cultivar"]:
                        cultivar_info = c
                        break

            result = CDEResult(
                variable_definitions=variable_definitions,
                relationships=relationships,
                cultivar_info=cultivar_info
            )

            timing["cde"] = (datetime.now() - start).total_seconds()
            return result

        except Exception as e:
            errors.append(ToolError(
                tool_name="cde",
                error_type=type(e).__name__,
                message=str(e)
            ))
            raise

    async def _execute_embedding(
        self,
        service: EmbeddingService,
        query_plan: QueryPlan,
        timing: Dict[str, float],
        errors: List[ToolError]
    ) -> Optional[List[EmbeddingResult]]:
        """Execute embedding tool."""
        start = datetime.now()

        try:
            # Build search query from filters
            search_parts = []

            if query_plan.filters.get("crop"):
                search_parts.append(query_plan.filters["crop"])
            if query_plan.filters.get("cultivar"):
                search_parts.append(query_plan.filters["cultivar"])
            if query_plan.location and query_plan.location.country:
                search_parts.append(query_plan.location.country)

            search_query = " ".join(search_parts) if search_parts else "DSSAT simulation"

            # Search in all collections
            results = []

            for collection in ["manuals", "papers", "summaries"]:
                try:
                    docs = await service.search_similar(
                        query=search_query,
                        collection=collection,
                        top_k=3
                    )

                    if docs:
                        results.append(EmbeddingResult(
                            documents=[
                                {"id": d["id"], **d.get("payload", {})}
                                for d in docs[:2]
                            ],
                            scores=[d["score"] for d in docs[:2]],
                            sources=[collection] * min(2, len(docs))
                        ))
                except Exception as e:
                    logger.warning(f"Embedding search failed for {collection}: {e}")

            timing["embedding"] = (datetime.now() - start).total_seconds()
            return results

        except Exception as e:
            errors.append(ToolError(
                tool_name="embedding",
                error_type=type(e).__name__,
                message=str(e)
            ))
            raise

    def _calculate_bounds(self, simulations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate bounding box from simulations."""
        if not simulations:
            return {"min_lat": 0, "max_lat": 0, "min_lon": 0, "max_lon": 0}

        lats = [s.get("latitude", 0) for s in simulations]
        lons = [s.get("longitude", 0) for s in simulations]

        return {
            "min_lat": min(lats),
            "max_lat": max(lats),
            "min_lon": min(lons),
            "max_lon": max(lons)
        }

    def _build_query_summary(
        self,
        query_plan: QueryPlan,
        context_data: Dict[str, Any]
    ) -> str:
        """Build natural language summary of query results."""
        parts = []

        # Intent
        intent_map = {
            "metadata": "Found",
            "aggregate": "Calculated",
            "spatial_search": "Located",
            "comparison": "Compared",
            "trend": "Analyzed trend for"
        }

        if query_plan.intent in intent_map:
            parts.append(intent_map[query_plan.intent])

        # Filters
        filters = []
        if query_plan.filters.get("crop"):
            filters.append(f"{query_plan.filters['crop']}")
        if query_plan.filters.get("year"):
            filters.append(f"year {query_plan.filters['year']}")
        if query_plan.filters.get("cultivar"):
            filters.append(query_plan.filters["cultivar"])

        if filters:
            parts.append(f"for {' '.join(filters)}")

        # Location
        if query_plan.location:
            loc_parts = []
            if query_plan.location.country:
                loc_parts.append(query_plan.location.country)
            if query_plan.location.state:
                loc_parts.append(query_plan.location.state)

            if loc_parts:
                parts.append(f"in {' '.join(loc_parts)}")

        # Results
        result_counts = []
        if context_data.get("metadata"):
            count = context_data["metadata"].total_count
            result_counts.append(f"{count} simulations")
        if context_data.get("spatial"):
            count = context_data["spatial"].total_count
            result_counts.append(f"{count} spatial matches")

        if result_counts:
            parts.append(f"({', '.join(result_counts)})")

        return " ".join(parts) if parts else "Query executed successfully"

    @staticmethod
    def _describe_filters(filters: List[FilterCondition]) -> str:
        parts = []
        for condition in filters or []:
            value = condition.value
            if isinstance(value, list):
                value = f"{value[0]}–{value[-1]}" if condition.operator == "BETWEEN" else ", ".join(map(str, value))
            parts.append(f"{condition.field.replace('_', ' ')} = {value}")
        return "; ".join(parts)

    @staticmethod
    def _area_counts_sentence(
        spatial_scope: Dict[str, Any],
        matching: int,
        analysed: int,
        filter_text: str,
    ) -> str:
        """Explain area vs filtered vs analysed counts without ambiguity."""
        text = (
            f"Area: {spatial_scope['description']} there are {spatial_scope['locations']:,} "
            f"simulated locations with {spatial_scope['simulations']:,} simulations in total"
        )
        if filter_text:
            text += (
                f" (all treatments); {matching:,} paired records were retrieved after {filter_text}"
                f" and {analysed:,} had valid values for the analysis."
            )
        else:
            text += f"; {analysed:,} had valid values for the analysis."
        return text

    @staticmethod
    def _no_data_in_radius_message(
        spatial: SpatialFilter,
        nearest: Optional[Dict[str, Any]],
    ) -> str:
        """Explain an empty radius, pointing to the nearest data."""
        message = f"No simulation data was found {spatial.describe()}."
        if nearest is None:
            return message + " The database does not contain any simulations yet."
        message += (
            f" The nearest simulated location is {nearest['distance_km']:,.1f} km away at "
            f"({nearest['latitude']:.4f}, {nearest['longitude']:.4f})."
        )
        if nearest["distance_km"] > spatial.radius_km:
            message += (
                f" Increase the radius to at least {nearest['distance_km']:.0f} km, "
                "or choose a point inside the data coverage area."
            )
        return message

    @staticmethod
    def _assess_analysis_quality(analysis: AnalysisResult) -> str:
        """Assess data quality of an analysis result from its sample size."""
        if analysis.status != "ok":
            return "low"
        if analysis.sample_size >= 30:
            return "high"
        if analysis.sample_size >= 10:
            return "medium"
        return "low"

    def _assess_data_quality(self, context_data: Dict[str, Any]) -> str:
        """Assess data quality based on results."""
        # Check for sufficient data
        metadata_count = 0
        spatial_count = 0

        if context_data.get("metadata"):
            metadata_count = context_data["metadata"].total_count
        if context_data.get("spatial"):
            spatial_count = context_data["spatial"].total_count

        total = max(metadata_count, spatial_count)

        if total >= 10:
            return "high"
        elif total >= 3:
            return "medium"
        else:
            return "low"
