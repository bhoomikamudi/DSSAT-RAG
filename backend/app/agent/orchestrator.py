"""Orchestrator - main agent coordinator.

Flow:
User → Planner (Structured Outputs) → Executor (parallel tools) → Context → Response Generator
"""
import logging
import re
from typing import Optional

from datetime import datetime

from app.agent.models import (
    QueryPlan,
    LLMContext,
    SpatialFilter,
    ResponseGeneration,
    OrchestratorResponse
)
from app.agent.planner import QueryPlanner
from app.agent.executor import Executor
from app.agent.context_builder import ContextBuilder
from app.agent.response_generator import ResponseGenerator
from app.agent.models import PlannerOutput, SemanticPlan
from app.agent.conversation_context import analysis_context_store
from app.agent.location_resolver import SpatialResolution, resolve_spatial
from app.agent.unsupported_conditions import (
    describe_unapplied,
    find_unsupported_conditions,
    merge_unapplied,
)
from app.services.geocoding_service import GeocodingService
from app.services.spatial_service import SpatialService

# Filters the LLM may invent for a place; a radius filter replaces them.
LOCATION_FIELDS = {"country", "state", "district", "ecological_zone", "location"}

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Main orchestrator for the agent system."""

    def __init__(
        self,
        db_session=None,
        planner_api_key: Optional[str] = None,
        response_api_key: Optional[str] = None,
        geocoder: Optional[GeocodingService] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            db_session: Database session
            planner_api_key: OpenAI API key for planning (optional)
            response_api_key: OpenAI API key for response generation (optional)
        """
        self.db_session = db_session
        self.planner = QueryPlanner(api_key=planner_api_key, db_session=db_session)
        self.executor = Executor(db_session=db_session)
        self.context_builder = ContextBuilder()
        self.response_generator = ResponseGenerator(api_key=response_api_key)
        self.geocoder = geocoder or GeocodingService()

    async def orchestrate(
        self,
        user_query: str
    ) -> OrchestratorResponse:
        """
        Orchestrate the full agent workflow.

        Args:
            user_query: Natural language user query

        Returns:
            Orchestrator response with all components
        """
        start_time = datetime.now()
        timing = {}
        errors = []

        try:
            # Step 1: Plan the query
            logger.info("Step 1: Planning query")
            plan_start = datetime.now()
            query_plan = await self.planner.plan_with_fallback(user_query)
            timing["planning"] = (datetime.now() - plan_start).total_seconds()
            planner_output: PlannerOutput | None = self.planner.get_last_planner_output()
            semantic_plan: SemanticPlan | None = self.planner.get_semantic_plan()

            # Step 2: Execute tools
            logger.info(f"Step 2: Executing tools - {query_plan.required_tools}")
            exec_start = datetime.now()
            context = await self.executor.execute(query_plan, planner_output=planner_output, semantic_plan=semantic_plan)
            timing["execution"] = (datetime.now() - exec_start).total_seconds()

            # Step 3: Generate response
            logger.info("Step 3: Generating response")
            gen_start = datetime.now()
            response = await self.response_generator.generate(
                user_question=user_query,
                context=context,
                query_plan=query_plan
            )
            timing["generation"] = (datetime.now() - gen_start).total_seconds()

            # Calculate total time
            timing["total"] = (datetime.now() - start_time).total_seconds()

            return OrchestratorResponse(
                success=True,
                query_plan=query_plan,
                context=context,
                response=response,
                errors=errors,
                timing=timing
            )

        except Exception as e:
            logger.error(f"Orchestration failed: {e}")

            # Calculate total time even on failure
            timing["total"] = (datetime.now() - start_time).total_seconds()

            return OrchestratorResponse(
                success=False,
                errors=[{
                    "tool_name": "orchestrator",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }],
                timing=timing
            )

    async def orchestrate_with_error_handling(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        radius_km: Optional[float] = None,
    ) -> OrchestratorResponse:
        """
        Orchestrate with comprehensive error handling.

        Args:
            user_query: Natural language user query
            session_id: Optional conversation id; enables follow-up analysis
                questions to reuse the previous analysis variables and filters
            latitude, longitude: Optional point selected on the map
            radius_km: Optional radius; the configured default is used if absent

        Returns:
            Orchestrator response (always succeeds, even on partial failure)
        """
        start_time = datetime.now()
        timing = {}
        errors = []

        try:
            # Step 0: Resolve the location (place lookup is separate from
            # distance filtering, which happens in SQL during execution).
            spatial_start = datetime.now()
            resolution = await resolve_spatial(
                user_query,
                latitude=latitude,
                longitude=longitude,
                radius_km=radius_km,
                geocoder=self.geocoder,
                coverage_bounds=self._coverage_bounds,
            )
            timing["location"] = (datetime.now() - spatial_start).total_seconds()
            if resolution.notice:
                return await self._location_notice_response(
                    user_query, resolution, timing, start_time
                )
            planning_query = resolution.query if resolution.spatial else user_query

            # Step 1: Plan
            logger.info("Step 1: Planning")
            plan_start = datetime.now()
            try:
                query_plan = await self.planner.plan_with_fallback(
                    planning_query,
                    analysis_context=analysis_context_store.get(session_id),
                )
            except Exception as e:
                logger.error(f"Planning failed: {e}")
                errors.append({
                    "tool_name": "planner",
                    "error_type": type(e).__name__,
                    "message": str(e)
                })
                # Create fallback plan
                query_plan = QueryPlan(
                    intent="metadata",
                    filters={},
                    required_tools=["metadata"],
                    response_type="summary"
                )
            timing["planning"] = (datetime.now() - plan_start).total_seconds()
            planner_output: PlannerOutput | None = self.planner.get_last_planner_output()

            # An ambiguous question gets a question back instead of a guess.
            clarification = self.planner.get_clarification()
            if clarification:
                context = LLMContext(
                    clarification=clarification,
                    query_summary=clarification,
                    data_quality="low",
                )
                response = await self.response_generator.generate(
                    user_question=user_query, context=context
                )
                timing["total"] = (datetime.now() - start_time).total_seconds()
                return OrchestratorResponse(
                    success=True, query_plan=query_plan, context=context,
                    response=response, errors=errors, timing=timing,
                )

            # Step 2: Execute
            logger.info("Step 2: Executing")
            exec_start = datetime.now()
            try:
                # Use semantic plan or dynamic PlannerOutput when available
                planner_output: PlannerOutput | None = self.planner.get_last_planner_output()
                semantic_plan: SemanticPlan | None = self.planner.get_semantic_plan()
                if resolution.spatial is not None:
                    semantic_plan = self._apply_spatial(
                        semantic_plan, query_plan, planning_query, resolution
                    )
                elif resolution.cleared and semantic_plan is not None:
                    # Drop any area inherited from earlier questions.
                    semantic_plan.spatial = None
                    for operation in semantic_plan.operations:
                        if operation.analysis is not None:
                            operation.analysis.spatial = None
                elif semantic_plan is not None and radius_km is not None:
                    inherited_note = self._apply_current_radius(semantic_plan, radius_km)
                    if inherited_note:
                        resolution.note = inherited_note
                context = await self.executor.execute(query_plan, planner_output=planner_output, semantic_plan=semantic_plan)
                if context.spatial_scope is not None and resolution.note:
                    context.spatial_scope["note"] = resolution.note
                self._report_unapplied_conditions(user_query, context)
                if resolution.cleared:
                    context.spatial_scope = {"cleared": True, "note": resolution.note}
                self._remember_analysis(session_id, semantic_plan, context)
            except Exception as e:
                logger.error(f"Execution failed: {e}")
                errors.append({
                    "tool_name": "executor",
                    "error_type": type(e).__name__,
                    "message": str(e)
                })
                # Create empty context
                context = LLMContext(
                    query_summary="Partial execution completed",
                    data_quality="low"
                )
            timing["execution"] = (datetime.now() - exec_start).total_seconds()

            # Step 3: Generate response
            logger.info("Step 3: Generating response")
            gen_start = datetime.now()
            try:
                response = await self.response_generator.generate(
                    user_question=user_query,
                    context=context,
                    query_plan=query_plan
                )
            except Exception as e:
                logger.error(f"Response generation failed: {e}")
                errors.append({
                    "tool_name": "response_generator",
                    "error_type": type(e).__name__,
                    "message": str(e)
                })
                response = ResponseGeneration(
                    answer="I encountered an error processing your request. Please try again.",
                    sources=[],
                    confidence="low",
                    limitations=["Service unavailable"]
                )
            timing["generation"] = (datetime.now() - gen_start).total_seconds()

            # Total time
            timing["total"] = (datetime.now() - start_time).total_seconds()

            return OrchestratorResponse(
                success=True,
                query_plan=query_plan,
                context=context,
                response=response,
                errors=errors,
                timing=timing
            )

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            timing["total"] = (datetime.now() - start_time).total_seconds()

            return OrchestratorResponse(
                success=False,
                errors=[{
                    "tool_name": "orchestrator",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }],
                timing=timing
            )

    @staticmethod
    def _remember_analysis(
        session_id: Optional[str],
        semantic_plan: Optional[SemanticPlan],
        context: LLMContext,
    ) -> None:
        """Keep the last successful analysis request for follow-up questions."""
        if not session_id or not semantic_plan or not context.analysis:
            return
        if context.analysis.status not in {"ok", "insufficient_data"}:
            return
        for operation in semantic_plan.operations:
            if operation.operation == "analysis" and operation.analysis:
                analysis_context_store.set(session_id, operation.analysis)
                return

    async def _coverage_bounds(self):
        """Bounding box of ingested data, used to disambiguate place names."""
        try:
            db = await self.executor._get_db_session()
            bounds = await SpatialService(db).get_bounds()
            if bounds and bounds.get("min_lat") is not None:
                return bounds
        except Exception as exc:
            logger.warning(f"Could not read data coverage bounds: {exc}")
        return None

    def _apply_spatial(
        self,
        semantic_plan: Optional[SemanticPlan],
        query_plan: QueryPlan,
        planning_query: str,
        resolution: SpatialResolution,
    ) -> SemanticPlan:
        """Attach the radius filter to the plan every data operation uses."""
        executable = {"aggregate", "metadata", "count", "trend", "analysis", "definition", "semantic_search"}
        if semantic_plan is None or not any(op.operation in executable for op in semantic_plan.operations):
            # The executor only honors spatial filters on semantic plans. A
            # missing plan, or one with no executable step (e.g. only a
            # "spatial" step), would otherwise fall back to the legacy path
            # and silently use all data.
            semantic_plan = self.planner._fallback_semantic(planning_query, query_plan)
            if self.planner.get_plan_source().get("planner") == "llm":
                self.planner._note(
                    "LLM plan had no executable data operation for the location; "
                    f"used a {semantic_plan.operations[0].operation} step built from the plan"
                )
        def keep_filter(condition):
            if condition.field not in LOCATION_FIELDS:
                return True
            # Remove place-derived guesses, but retain an explicit domain
            # restriction still present after the location phrase was removed.
            values = condition.value if isinstance(condition.value, list) else [condition.value]
            return all(
                re.search(r"(?<!\w)" + re.escape(str(value)) + r"(?!\w)", planning_query, re.I)
                for value in values
            )

        for operation in semantic_plan.operations:
            operation.filters = [
                f for f in operation.filters if keep_filter(f)
            ]
            if operation.analysis is not None:
                operation.analysis.spatial = resolution.spatial
                operation.analysis.filters = [
                    f for f in operation.analysis.filters if keep_filter(f)
                ]
        semantic_plan.spatial = resolution.spatial
        self.planner._semantic_plan = semantic_plan
        return semantic_plan

    def _report_unapplied_conditions(self, user_query: str, context: LLMContext) -> None:
        """Say which requested conditions could not be applied as filters.

        Sources: conditions in the question the dataset cannot filter on
        (soil, nitrogen rate, calendar planting dates, ...) and filters the
        planner dropped because no field supports them. Added to the
        management scope so the answer never implies they were applied.
        """
        unapplied = merge_unapplied(
            find_unsupported_conditions(user_query),
            self.planner.get_plan_source().get("unapplied_filters"),
        )
        scope = context.management_scope
        if not unapplied or not scope:
            return
        scope["unapplied"] = unapplied
        scope["sentence"] = f"{scope['sentence']} {describe_unapplied(unapplied)}"

    @staticmethod
    def _apply_current_radius(semantic_plan: SemanticPlan, radius_km: float) -> Optional[str]:
        """Keep an inherited area in step with the radius shown in the UI.

        A follow-up ("now fit a linear regression instead") inherits the place
        or coordinates of the previous question from the session. Unless that
        question stated its own radius, the area uses the radius the user has
        selected now, so the chip, the request and the filter agree.
        """
        note = None
        for operation in semantic_plan.operations:
            area = operation.analysis.spatial if operation.analysis is not None else None
            if area is None or area.radius_from_question or area.radius_km == radius_km:
                continue
            # Rebuilt (not copied) so the new radius is validated.
            operation.analysis.spatial = SpatialFilter(
                **{**area.model_dump(), "radius_km": float(radius_km)}
            )
            note = (
                f"Area carried over from the previous question, using the current "
                f"radius of {float(radius_km):g} km (was {area.radius_km:g} km)."
            )
        return note

    async def _location_notice_response(
        self,
        user_query: str,
        resolution: SpatialResolution,
        timing: dict,
        start_time: datetime,
    ) -> OrchestratorResponse:
        """Answer with the location problem instead of guessing a place."""
        context = LLMContext(
            spatial_notice=resolution.notice,
            query_summary=resolution.notice,
            data_quality="low",
        )
        response = await self.response_generator.generate(
            user_question=user_query, context=context
        )
        timing["total"] = (datetime.now() - start_time).total_seconds()
        return OrchestratorResponse(
            success=True, context=context, response=response, errors=[], timing=timing
        )
