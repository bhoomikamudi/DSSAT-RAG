"""Query Planner - converts natural language to structured execution plan.

Refactored to support production-grade Agentic Retrieval planning:
- Uses OpenAI Responses API (Structured Outputs) when available
- Injects dynamic Tool specifications (not DB schemas/SQL)
- Returns a structured PlannerOutput with explicit tool calls
- Never answers; only plans tools and parameters
"""
import json
import logging
import re
from typing import Optional, Any, Dict, List

from openai import AsyncOpenAI
from pydantic import ValidationError

from app.agent.models import (
    QueryPlan, PlannerOutput, PlannerToolCall,
    SimulationToolInput, CDEToolInput, SemanticToolInput,
    SemanticPlan, SemanticOperation, FilterCondition
)
from app.agent.prompts import get_planner_prompt
from app.core.config import get_settings
from app.services.statistics_service import StatisticsService
from app.agent.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

# Get settings instance
settings = get_settings()


class QueryPlanner:
    """Plans query execution using LLM."""

    def __init__(self, api_key: Optional[str] = None, db_session=None):
        """Initialize planner with optional LLM and DB access."""
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.db_session = db_session
        if self.api_key:
            if settings.OPENAI_BASE_URL:
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=settings.OPENAI_BASE_URL)
            else:
                self.client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = None
        self._tool_registry: Optional[ToolRegistry] = None
        self._last_planner_output: Optional[PlannerOutput] = None
        self._semantic_plan: Optional[SemanticPlan] = None

    async def plan(self, user_query: str) -> QueryPlan:
        """Plan query execution with Structured Outputs and dynamic tools.

        Returns a legacy QueryPlan for compatibility with current Executor.
        Also stores the last PlannerOutput for use by Executor v2 (if enabled).
        NOTE: A semantic plan is also generated and stored for the executor.
        """
        logger.info(f"Planning query: {user_query}")
        logger.info(f"Planner LLM configured: key={'set' if bool(self.api_key) else 'missing'}, model={settings.OPENAI_MODEL}")

        if not self.client:
            return await self._fallback_plan(user_query)

        # Inject dynamic tool specs
        planner_context = ""
        try:
            db = None
            if self.db_session is not None:
                if hasattr(self.db_session, 'session'):
                    db = await self.db_session.session()
                else:
                    db = self.db_session
            if db is not None:
                self._tool_registry = ToolRegistry(db)
                planner_context = await self._tool_registry.get_planner_context()
        except Exception as e:
            logger.warning(f"Failed to load dynamic tool specs: {e}")

        prompt = get_planner_prompt(user_query) + ("\n\n" + planner_context if planner_context else "")

        try:
            # Prefer Responses API if available on this OpenAI-compatible endpoint.
            # Fallback to Chat Completions tool calling as before.
            try:
                responses = await self.client.responses.create(
                    model=settings.OPENAI_MODEL or "gpt-4o-mini",
                    input=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={
                        "type": "json_schema",
                        "json_schema": PlannerOutput.model_json_schema(),
                    },
                )
                content = responses.output[0].content[0].text if getattr(responses, 'output', None) else None
                if content:
                    # Try semantic plan first
                    sem = self._parse_semantic_plan(content)
                    if sem:
                        self._semantic_plan = sem
                        return self._to_minimal_query_plan_from_semantic(sem)
                    # Fallback to tool-based plan
                    plan_struct = self._parse_planner_output(content)
                    if plan_struct:
                        self._semantic_plan = self._build_semantic_plan(user_query, plan_struct)
                        return self._to_legacy_query_plan(plan_struct, user_query)
            except Exception as e:
                logger.info(f"Responses API unavailable or failed, falling back to Chat Completions: {e}")

            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL or "gpt-4o-mini",
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            choice = response.choices[0]
            # Try parsing JSON directly from content as PlannerOutput
            raw_content = getattr(choice.message, "content", None)
            if raw_content:
                logger.info(f"Planner raw message content: {raw_content[:400]}")
                content_text = raw_content if isinstance(raw_content, str) else str(raw_content)
                # Try direct JSON parse
                plan_data = None
                try:
                    plan_data = json.loads(content_text)
                except Exception:
                    # Extract first JSON object substring and parse
                    try:
                        m = re.search(r"\{[\s\S]*\}", content_text)
                        if m:
                            plan_data = json.loads(m.group(0))
                    except Exception:
                        plan_data = None
                if plan_data is not None:
                    sem = self._parse_semantic_plan(plan_data)
                    if sem:
                        self._semantic_plan = sem
                        return self._to_minimal_query_plan_from_semantic(sem)
                    plan_struct = self._parse_planner_output(plan_data)
                    if plan_struct:
                        self._semantic_plan = self._build_semantic_plan(user_query, plan_struct)
                        return self._to_legacy_query_plan(plan_struct, user_query)

            raise ValueError("No query plan generated via tool call or content")

        except ValidationError as e:
            logger.error(f"Query plan validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            raise

    def _parse_planner_output(self, content: Any) -> Optional[PlannerOutput]:
        """Parse content (str or dict) into PlannerOutput model."""
        try:
            data = content
            if isinstance(content, str):
                data = json.loads(content)
            # Coerce planner tools params into typed inputs
            if isinstance(data, dict) and isinstance(data.get("tools"), list):
                tools: List[Dict[str, Any]] = []
                for t in data["tools"]:
                    name = t.get("tool")
                    params = t.get("parameters", {})
                    if name == "query_simulation_data":
                        params = SimulationToolInput(**params)
                    elif name == "query_cde":
                        params = CDEToolInput(**params)
                    elif name == "semantic_search":
                        params = SemanticToolInput(**params)
                    tools.append({"tool": name, "parameters": params})
                data = {"goal": data.get("goal", ""), "tools": tools}
            po = PlannerOutput(**data)
            logger.info(f"PlannerOutput parsed with {len(po.tools)} tool(s)")
            self._last_planner_output = po
            return po
        except Exception as e:
            logger.warning(f"Failed to parse PlannerOutput: {e}")
            return None

    def _parse_semantic_plan(self, content: Any) -> Optional[SemanticPlan]:
        """Parse content (str or dict) into SemanticPlan model."""
        try:
            data = content
            if isinstance(content, str):
                data = json.loads(content)
            if isinstance(data, dict) and "operations" in data and "intent" in data:
                sp = SemanticPlan(**data)
                logger.info(f"SemanticPlan parsed with {len(sp.operations)} operation(s)")
                return sp
            return None
        except Exception as e:
            logger.warning(f"Failed to parse SemanticPlan: {e}")
            return None

    def _to_minimal_query_plan_from_semantic(self, sp: SemanticPlan) -> QueryPlan:
        """Return a minimal legacy QueryPlan to keep the pipeline compatible. The executor will prefer the semantic plan."""
        metric = None
        aggregation = None
        filters: Dict[str, Any] = {}
        for op in sp.operations:
            if op.operation == "aggregate":
                metric = metric or op.metric
                aggregation = aggregation or op.aggregation
                # Heuristic: map a simple '=' year filter to legacy filters for summary
                for f in op.filters:
                    if f.field == "year" and f.operator == "=" and isinstance(f.value, (int, str)):
                        filters["year"] = f.value
                    if f.field == "crop" and f.operator == "=" and isinstance(f.value, str):
                        filters["crop"] = f.value
                break
        intent = "aggregate" if metric or aggregation else "metadata"
        return QueryPlan(intent=intent, metric=metric, aggregation=aggregation, filters=filters, location=None, comparison=None, time_range=None, required_tools=["statistics"] if intent=="aggregate" else ["metadata"], response_type="summary")

    def _to_legacy_query_plan(self, po: PlannerOutput, user_query: str) -> QueryPlan:
        """Convert PlannerOutput into the existing QueryPlan structure for compatibility."""
        required_tools_map = {
            "query_simulation_data": ["metadata", "statistics", "spatial"],
            "query_cde": ["cde"],
            "semantic_search": ["embedding"],
        }
        required: List[str] = []
        metric: Optional[str] = None
        aggregation: Optional[str] = None
        filters: Dict[str, Any] = {}
        location = None

        for tc in po.tools:
            required.extend([t for t in required_tools_map.get(tc.tool, []) if t not in required])
            if tc.tool == "query_simulation_data":
                params: SimulationToolInput = tc.parameters  # type: ignore
                # adopt first non-empty
                if not metric and params.metrics:
                    metric = params.metrics[0]
                aggregation = aggregation or params.aggregation
                filters.update(params.filters or {})
                # map spatial to legacy location if needed (kept None here to avoid changing existing models)

        # Infer intent
        intent = "metadata"
        if metric or aggregation:
            intent = "aggregate"

        if intent == "aggregate" and "statistics" not in required:
            required.append("statistics")
        if not required:
            required = ["metadata"]

        qp = QueryPlan(
            intent=intent,
            metric=metric,
            aggregation=aggregation,
            filters=filters,
            location=None,
            comparison=None,
            time_range=None,
            required_tools=required,  # executor will still use dynamic dispatch soon
            response_type="summary",
        )
        logger.info(
            f"Generated legacy QueryPlan from PlannerOutput: intent={qp.intent}, metric={qp.metric}, agg={qp.aggregation}, tools={qp.required_tools}"
        )
        return qp

    def get_last_planner_output(self) -> Optional[PlannerOutput]:
        """Expose the last successful PlannerOutput for downstream executor."""
        return getattr(self, "_last_planner_output", None)

    def get_semantic_plan(self) -> Optional[SemanticPlan]:
        return getattr(self, "_semantic_plan", None)

    async def plan_with_fallback(self, user_query: str) -> QueryPlan:
        """Plan query with fallback to simpler parsing and DB-driven enrichment."""
        if not self.client:
            return await self._fallback_plan(user_query)

        try:
            qp = await self.plan(user_query)
            # Enrich if aggregate intent implied but fields missing
            ql = user_query.lower()
            if (not qp.metric or not qp.aggregation) and any(w in ql for w in ["average", "avg", "mean", "total", "sum"]):
                logger.info("Planner: enriching aggregate plan (missing metric/aggregation)")
                qp.aggregation = qp.aggregation or "AVG"
                stats = None
                if self.db_session is not None:
                    try:
                        if hasattr(self.db_session, 'session'):
                            db = await self.db_session.session()
                        else:
                            db = self.db_session
                        stats = StatisticsService(db)
                    except Exception:
                        stats = None
                if stats and not qp.metric:
                    try:
                        qp.metric = await stats.resolve_metric_from_text(user_query)
                        logger.info(f"Planner: resolved metric via DB -> {qp.metric}")
                    except Exception:
                        pass
                # Secondary heuristic: capture explicit uppercase codes in the query (e.g., HWAM)
                if not qp.metric:
                    mcode = re.search(r"\b[A-Z]{3,6}\b", user_query)
                    if mcode:
                        qp.metric = mcode.group(0)
                        logger.info(f"Planner: resolved metric via uppercase heuristic -> {qp.metric}")
                if "year" not in qp.filters:
                    m = re.search(r"\b(19|20)\d{2}\b", ql)
                    if m:
                        try:
                            qp.filters["year"] = int(m.group(0))
                        except Exception:
                            pass
                if stats and "crop" not in qp.filters:
                    try:
                        crop_resolved = await stats.resolve_crop_from_text(user_query)
                        if crop_resolved:
                            qp.filters["crop"] = crop_resolved
                            logger.info(f"Planner: resolved crop via DB -> {crop_resolved}")
                    except Exception:
                        pass
                if "statistics" not in qp.required_tools:
                    qp.required_tools.append("statistics")
                if qp.intent != "aggregate":
                    qp.intent = "aggregate"
            return qp
        except Exception as e:
            logger.warning(f"LLM planning failed, using fallback: {e}")
            qp = await self._fallback_plan(user_query)
            # Build a minimal semantic plan from fallback
            self._semantic_plan = self._fallback_semantic(user_query, qp)
            return qp

    async def _fallback_plan(self, user_query: str) -> QueryPlan:
        """Create fallback plan using data-driven resolution only (no hardcoded maps)."""
        query_lower = user_query.lower()

        intent = "metadata"
        required_tools = ["metadata"]
        filters: Dict[str, Any] = {}

        if any(word in query_lower for word in ["average", "avg", "mean", "total", "sum"]):
            intent = "aggregate"
            required_tools.append("statistics")
            # Resolve metric/crop dynamically using DB if available
            stats = None
            if self.db_session is not None:
                try:
                    if hasattr(self.db_session, 'session'):
                        db = await self.db_session.session()
                    else:
                        db = self.db_session
                    stats = StatisticsService(db)
                except Exception:
                    stats = None
            metric_resolved = None
            if stats:
                try:
                    metric_resolved = await stats.resolve_metric_from_text(user_query)
                except Exception:
                    metric_resolved = None
            # Secondary heuristic: capture explicit uppercase codes in the query (e.g., HWAM)
            if not metric_resolved:
                mcode = re.search(r"\b[A-Z]{3,6}\b", user_query)
                if mcode:
                    metric_resolved = mcode.group(0)
            # Extract a year range or a single year
            range_match = re.search(
                r"\b(?:from|between)\s+((?:19|20)\d{2})\s+(?:to|and|-)\s+((?:19|20)\d{2})\b",
                query_lower,
            )

            if range_match:
                start_year = int(range_match.group(1))
                end_year = int(range_match.group(2))
                filters["year"] = list(range(start_year, end_year + 1))
            else:
                year_match = re.search(r"\b(?:19|20)\d{2}\b", query_lower)
                if year_match:
                    filters["year"] = int(year_match.group(0))

            if self.db_session is not None:
                try:
                    if hasattr(self.db_session, "session"):
                        db = await self.db_session.session()
                    else:
                        db = self.db_session

                    stats = StatisticsService(db)
                    crop_resolved = await stats.resolve_crop_from_text(user_query)

                    if crop_resolved:
                        filters["crop"] = crop_resolved
                except Exception:
                    pass
            # Extract crop dynamically
            crop_resolved = None
            if stats:
                try:
                    crop_resolved = await stats.resolve_crop_from_text(user_query)
                except Exception:
                    crop_resolved = None
            if crop_resolved:
                filters["crop"] = crop_resolved
            # Only produce an aggregate plan when a metric is resolved
            if metric_resolved:
                return QueryPlan(
                    intent=intent,
                    filters=filters,
                    required_tools=required_tools,
                    response_type="summary",
                    metric=metric_resolved,
                    aggregation="AVG",
                )
        elif any(word in query_lower for word in ["within", "near", "radius", "distance"]):
            intent = "spatial_search"
            required_tools.append("spatial")
        elif any(word in query_lower for word in ["compare", "vs", "versus"]):
            intent = "comparison"
            required_tools.append("statistics")

        # Extract a year range or a single year
        range_match = re.search(
            r"\b(?:from|between)\s+((?:19|20)\d{2})\s+(?:to|and|-)\s+((?:19|20)\d{2})\b",
            query_lower,
        )

        if range_match:
            start_year = int(range_match.group(1))
            end_year = int(range_match.group(2))
            filters["year"] = list(range(start_year, end_year + 1))
        else:
            year_match = re.search(r"\b(?:19|20)\d{2}\b", query_lower)
            if year_match:
                filters["year"] = int(year_match.group(0))
        if self.db_session is not None:
            try:
                if hasattr(self.db_session, "session"):
                    db = await self.db_session.session()
                else:
                    db = self.db_session

                stats = StatisticsService(db)
                crop_resolved = await stats.resolve_crop_from_text(user_query)

                if crop_resolved:
                    filters["crop"] = crop_resolved
            except Exception:
                pass

        return QueryPlan(
            intent=intent,
            filters=filters,
            required_tools=required_tools,
            response_type="summary",
        )

    def _build_semantic_plan(self, user_query: str, po: PlannerOutput) -> SemanticPlan:
        """Heuristically convert PlannerOutput into a semantic plan.

        This is an interim refactor: planner emits tools, we coerce into operations.
        """
        ops: List[SemanticOperation] = []
        intent = "metadata"
        for tc in po.tools:
            if tc.tool == "query_simulation_data":
                params: SimulationToolInput = tc.parameters  # type: ignore
                metric = params.metrics[0] if params.metrics else None
                agg = params.aggregation
                filters: List[FilterCondition] = []
                f = params.filters or {}
                # Normalize year filter to operators
                y = f.get("year")
                if isinstance(y, list):
                    # If exactly two numbers and planning intent is compare-ish, we cannot know; use IN for now
                    filters.append(FilterCondition(field="year", operator="IN", value=y))
                elif y is not None:
                    filters.append(FilterCondition(field="year", operator="=", value=y))
                for k in ["crop", "cultivar", "state", "district", "country"]:
                    if f.get(k) is not None:
                        filters.append(FilterCondition(field=k, operator="=", value=f[k]))
                ops.append(SemanticOperation(operation="aggregate", metric=metric, aggregation=agg, filters=filters, group_by=params.group_by or [], independent=True))
                intent = "aggregate"
            elif tc.tool == "query_cde":
                params: CDEToolInput = tc.parameters  # type: ignore
                for v in params.variables or []:
                    ops.append(SemanticOperation(operation="definition", variable=v, independent=True))
                if not params.variables:
                    ops.append(SemanticOperation(operation="definition", independent=True))
                if intent == "metadata":
                    intent = "definition"
            elif tc.tool == "semantic_search":
                params: SemanticToolInput = tc.parameters  # type: ignore
                ops.append(SemanticOperation(operation="semantic_search", query=params.query, independent=True))
                if intent == "metadata":
                    intent = "explanation"
        return SemanticPlan(goal=po.goal or user_query, intent=intent, operations=ops)

    def _fallback_semantic(self, user_query: str, qp: QueryPlan) -> SemanticPlan:
        ops: List[SemanticOperation] = []
        filters: List[FilterCondition] = []
        for k, v in (qp.filters or {}).items():
            if k == "year" and isinstance(v, list):
                filters.append(FilterCondition(field="year", operator="IN", value=v))
            elif k == "year":
                filters.append(FilterCondition(field="year", operator="=", value=v))
            else:
                filters.append(FilterCondition(field=k, operator="=", value=v))
        if qp.intent == "aggregate":
            ops.append(SemanticOperation(operation="aggregate", metric=qp.metric, aggregation=qp.aggregation, filters=filters, independent=True))
        else:
            ops.append(SemanticOperation(operation="metadata", filters=filters, independent=True))
        return SemanticPlan(goal=user_query, intent=qp.intent or "metadata", operations=ops)
