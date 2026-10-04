"""Query Planner - converts natural language to structured execution plan.

Flow for every supported question:
1. The LLM produces the structured plan (Chat Completions on the configured
   OpenAI-compatible gateway; tool specs injected, never DB schemas/SQL).
2. The plan is validated and normalized deterministically: variable aliases
   become DSSAT codes, filter fields/values are checked, and facts stated in
   the question (variables, cultivars, years, method) correct the plan. Every
   correction is recorded in the plan source.
3. Only when no LLM is configured, the LLM call fails, or its plan is invalid,
   the deterministic planners (analysis parser, statistics fallback) plan the
   question instead, and the plan source says so.
The planner never answers; it only plans tools and parameters.
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
    SemanticPlan, SemanticOperation, FilterCondition,
    AnalysisRequest,
)
from app.agent.unsupported_conditions import classify_dropped_filter
from app.agent.analysis_parser import (
    find_filters,
    find_operation,
    parse_analysis_request,
)
from app.agent.statistics_fallback import build_fallback_statistics_plan
from app.agent.prompts import get_planner_prompt
from app.agent.conversation_context import describe_analysis_context
from app.core.config import get_settings
from app.services.statistics_service import StatisticsService
from app.services.cde_service import CDEService
from app.services.analysis_vocabulary import (
    ANALYSIS_VARIABLES,
    FILTER_FIELDS,
    normalize_cultivar,
    normalize_filter_field,
    normalize_variable,
    find_variable_mentions,
    variable_label,
)
from app.agent.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

# Get settings instance
settings = get_settings()


def _variable_name(code: str) -> str:
    definition = CDEService.VARIABLE_DEFINITIONS.get(code)
    return definition["full_name"] if definition else variable_label(code)


def metric_clarification(candidates: List[str]) -> str:
    """Ask which variable an ambiguous metric question means."""
    options = [f"{code} ({_variable_name(code)})" for code in candidates]
    listed = ", ".join(options[:-1]) + f" or {options[-1]}" if len(options) > 1 else options[0]
    return (
        f"Your question could refer to more than one variable: {listed}. "
        "Which one do you mean? For example, ask about "
        f"\"{_variable_name(candidates[0]).lower()}\"."
    )


def explicit_variable_code(query: str) -> Optional[str]:
    """An uppercase token that is a known variable code (never e.g. BASE)."""
    known = set(ANALYSIS_VARIABLES) | set(CDEService.VARIABLE_DEFINITIONS)
    for token in re.findall(r"\b[A-Z]{2,6}\b", query):
        if token in known:
            return token
    return None


class QueryPlanner:
    """Plans query execution using LLM."""

    def _force_definition_plan(
        self,
        user_query: str,
        plan: SemanticPlan,
    ) -> SemanticPlan:
        """Correct definition queries when the LLM misclassifies them."""

        match = re.search(
            r"\b(?:what\s+does|what\s+is|meaning\s+of|define)\s+"
            r"(BASE|LNG|SHT|VLNG|VSHT|"
            r"pfrst-15|pfrst-30|pfrst0|pfrst15|pfrst30|"
            r"HWAM|CWAM|HWAH|GNAM|PRCP|LAI|ET|GDD|NUP|YIELD)\b",
            user_query,
            flags=re.IGNORECASE,
        )

        if not match:
            return plan

        code = match.group(1)

        if code.lower().startswith("pfrst"):
            code = code.lower()
        else:
            code = code.upper()

        return SemanticPlan(
            goal=f"Define {code}",
            intent="definition",
            operations=[
                SemanticOperation(
                    operation="definition",
                    entity=None,
                    metric=None,
                    aggregation=None,
                    filters=[],
                    group_by=[],
                    independent=True,
                    variable=code,
                    query=None,
                )
            ],
            comparison_axis=None,
            comparison_mode=None,
            comparison_values=[],
        )

    def _add_dataset_condition_filters(
        self,
        user_query: str,
        plan: SemanticPlan,
    ) -> SemanticPlan:
        """Add filters based on the current DSSAT dataset naming convention."""

        query_lower = user_query.lower()

        condition_filters = []

        if "rainfed" in query_lower or "rain-fed" in query_lower:
            condition_filters.append(
                FilterCondition(
                    field="irrigation",
                    operator="=",
                    value="RF",
                )
            )

        if (
            "high nitrogen" in query_lower
            or "high n" in query_lower
            or "highn" in query_lower
        ):
            condition_filters.append(
                FilterCondition(
                    field="nitrogen_level",
                    operator="=",
                    value="HighN",
                )
            )

        if not condition_filters:
            return plan

        for operation in plan.operations:
            if operation.operation not in {
                "aggregate",
                "count",
                "metadata",
                "trend",
            }:
                continue

            for condition_filter in condition_filters:
                matching_filter = next(
                    (
                        existing_filter
                        for existing_filter in operation.filters
                        if existing_filter.field == condition_filter.field
                    ),
                    None,
                )

                if matching_filter is not None:
                    matching_filter.value = condition_filter.value
                else:
                    operation.filters.append(condition_filter)

        return plan

    def _analysis_semantic_plan(self, request: AnalysisRequest) -> SemanticPlan:
        variables = " vs ".join(
            code for code in (request.y_variable, request.x_variable) if code
        )
        return SemanticPlan(
            goal=f"{request.operation.replace('_', ' ').capitalize()}: {variables}",
            intent="analysis",
            operations=[
                SemanticOperation(
                    operation="analysis",
                    filters=request.filters,
                    group_by=[request.group_by] if request.group_by else [],
                    independent=True,
                    analysis=request,
                )
            ],
        )

    @staticmethod
    def _analysis_query_plan(request: AnalysisRequest) -> QueryPlan:
        return QueryPlan(
            intent="analysis",
            metric=request.y_variable,
            aggregation=None,
            filters={
                condition.field: condition.value
                for condition in request.filters
            },
            required_tools=["statistics"],
            response_type="summary",
        )

    def _build_analysis_plan(self, request: AnalysisRequest) -> QueryPlan:
        """Store a single-operation analysis plan and return its QueryPlan."""

        self._semantic_plan = self._analysis_semantic_plan(request)
        self._last_planner_output = None

        logger.info(
            "Planner: analysis request %s",
            request.model_dump(exclude={"include_plot", "max_points"}),
        )

        return self._analysis_query_plan(request)

    # -- deterministic validation of LLM plans ---------------------------------

    def _note(self, message: str) -> None:
        """Record a deterministic correction applied to the LLM plan."""
        logger.info("Planner correction: %s", message)
        self._plan_source.setdefault("corrections", []).append(message)

    @staticmethod
    def _filter_years(condition: FilterCondition) -> Optional[frozenset]:
        value = condition.value
        try:
            if condition.operator == "BETWEEN" and isinstance(value, list) and value:
                start, end = sorted((int(value[0]), int(value[-1])))
                return frozenset(range(start, end + 1))
            if isinstance(value, list):
                return frozenset(int(item) for item in value)
            if isinstance(value, str) and "," in value:
                return frozenset(int(item) for item in value.split(",") if item.strip())
            return frozenset([int(value)])
        except (TypeError, ValueError):
            return None

    def _same_filter(self, a: FilterCondition, b: FilterCondition) -> bool:
        if a.field == "year" and b.field == "year":
            years = self._filter_years(a)
            return years is not None and years == self._filter_years(b)
        values = lambda c: sorted(map(str, c.value)) if isinstance(c.value, list) else [str(c.value)]
        return values(a) == values(b)

    def _clean_filters(self, filters: List[FilterCondition], where: str) -> List[FilterCondition]:
        """Canonical field names and cultivar codes; drop unsupported fields."""
        kept: List[FilterCondition] = []
        for condition in filters:
            field = normalize_filter_field(condition.field)
            if field not in FILTER_FIELDS:
                self._note(f"{where}: dropped unsupported filter {condition.field}={condition.value!r}")
                # Reported in the answer as "not applied", never silently ignored.
                self._plan_source.setdefault("unapplied_filters", []).append(
                    classify_dropped_filter(condition.field, condition.value)
                )
                continue
            value = normalize_cultivar(condition.value) if field == "cultivar" else condition.value
            if field != condition.field or value != condition.value:
                self._note(f"{where}: normalized filter {condition.field}={condition.value!r} -> {field}={value!r}")
            kept.append(FilterCondition(field=field, operator=condition.operator, value=value))
        return kept

    def _merge_grounded_filters(
        self,
        filters: List[FilterCondition],
        grounded: List[FilterCondition],
        where: str,
    ) -> List[FilterCondition]:
        """Filters stated in the question win over the LLM's for that field."""
        by_field: Dict[str, FilterCondition] = {c.field: c for c in filters}
        for condition in grounded:
            existing = by_field.get(condition.field)
            if existing is None:
                self._note(f"{where}: added {condition.field} {condition.operator} {condition.value!r} stated in the question")
            elif not self._same_filter(existing, condition):
                self._note(
                    f"{where}: replaced {existing.field} {existing.operator} {existing.value!r} "
                    f"with {condition.operator} {condition.value!r} stated in the question"
                )
            else:
                continue
            by_field[condition.field] = condition.model_copy(deep=True)
        return list(by_field.values())

    def _validate_operation_filters(self, user_query: str, plan: SemanticPlan) -> None:
        """Aggregate/count/metadata/trend filters: canonical and question-grounded."""
        grounded = find_filters(user_query)
        for index, operation in enumerate(plan.operations):
            if operation.operation not in {"aggregate", "count", "metadata", "trend"}:
                continue
            where = f"{operation.operation} #{index + 1}"
            cleaned = self._clean_filters(operation.filters, where)
            operation.filters = self._merge_grounded_filters(cleaned, grounded, where)

    def _correct_analysis(
        self,
        llm: AnalysisRequest,
        parsed: AnalysisRequest,
        user_query: str,
        follow_up: bool,
    ) -> AnalysisRequest:
        """Keep the LLM request; overwrite only fields the question states."""
        request = llm.model_copy(deep=True)

        def correct(field: str, value: Any, reason: str) -> None:
            current = getattr(request, field)
            if value is not None and value != current:
                self._note(f"analysis: {field} {current!r} -> {value!r} ({reason})")
                setattr(request, field, value)

        _, explicit = find_operation(user_query)
        correct("operation", parsed.operation,
                "method named in the question" if explicit else
                "previous analysis" if follow_up else "relationship question defaults to correlation")
        correct("x_variable", parsed.x_variable, "variable stated in the question")
        correct("y_variable", parsed.y_variable, "variable stated in the question")
        if parsed.group_by is not None or follow_up:
            correct("group_by", parsed.group_by, "grouping stated in the question")
            if follow_up and parsed.group_by is None and request.group_by is not None:
                self._note(f"analysis: group_by {request.group_by!r} -> None (not kept by the follow-up)")
                request.group_by = None
        if parsed.requires_group_variation and not request.requires_group_variation:
            request.requires_group_variation = True
        if parsed.spatial is not None and request.spatial is None:
            request.spatial = parsed.spatial

        cleaned = self._clean_filters(request.filters, "analysis")
        if follow_up:
            # The follow-up parser merges the previous filters with the new
            # ones (and applies "all data"/"all cultivars" resets).
            if [c.model_dump() for c in cleaned] != [c.model_dump() for c in parsed.filters]:
                self._note(f"analysis: filters {[(c.field, c.value) for c in cleaned]} -> "
                           f"{[(c.field, c.value) for c in parsed.filters]} (follow-up of the previous analysis)")
            request.filters = [c.model_copy(deep=True) for c in parsed.filters]
        else:
            request.filters = self._merge_grounded_filters(cleaned, parsed.filters, "analysis")
        return request

    def _validate_analysis_operations(
        self,
        user_query: str,
        plan: SemanticPlan,
        analysis_context: Optional[AnalysisRequest],
    ) -> SemanticPlan:
        """Validate analysis operations in an LLM plan against the question.

        The deterministic parser is a corrector here: it only overrides what
        the question itself states, and each change is recorded.
        """
        parsed = parse_analysis_request(user_query, previous=analysis_context)
        follow_up = parsed is not None and parse_analysis_request(user_query) is None
        analysis_ops = [op for op in plan.operations if op.operation == "analysis"]

        if parsed is not None and not analysis_ops:
            planned = ", ".join(op.operation for op in plan.operations) or "no operation"
            self._note(
                f"LLM planned {planned} for an analysis question; replaced with the "
                f"{parsed.operation} request stated in the question"
            )
            return self._analysis_semantic_plan(parsed)

        for operation in analysis_ops:
            request = operation.analysis
            if request is None:
                request = AnalysisRequest(
                    y_variable=operation.metric,
                    filters=operation.filters,
                    group_by=operation.group_by[0] if operation.group_by else None,
                )
            elif not request.filters and operation.filters:
                request.filters = list(operation.filters)
            if parsed is not None:
                request = self._correct_analysis(request, parsed, user_query, follow_up)
            else:
                request.filters = self._clean_filters(request.filters, "analysis")
            operation.analysis = request
            operation.filters = request.filters
            operation.group_by = [request.group_by] if request.group_by else []

        return plan

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
        self._clarification: Optional[str] = None
        self._plan_source: Dict[str, Any] = {}

    def _definition_plan(self, user_query: str) -> Optional[QueryPlan]:
        """Plan "what does X mean?" questions for known codes (no LLM needed)."""
        definition_match = re.search(
            r"\b(?:what\s+does|what\s+is|meaning\s+of|define)\s+"
            r"(BASE|LNG|SHT|VLNG|VSHT|"
            r"pfrst-15|pfrst-30|pfrst0|pfrst15|pfrst30|"
            r"HWAM|CWAM|HWAH|GNAM|PRCP|LAI|ET|GDD|NUP|YIELD)\b",
            user_query,
            flags=re.IGNORECASE,
        )

        if not definition_match:
            return None

        code = definition_match.group(1)
        code = (
            code.lower()
            if code.lower().startswith("pfrst")
            else code.upper()
        )

        self._semantic_plan = SemanticPlan(
            goal=f"Define {code}",
            intent="definition",
            operations=[
                SemanticOperation(
                    operation="definition",
                    entity=None,
                    metric=None,
                    aggregation=None,
                    filters=[],
                    group_by=[],
                    independent=True,
                    variable=code,
                    query=None,
                )
            ],
            comparison_axis=None,
            comparison_mode=None,
            comparison_values=[],
        )

        return QueryPlan(
            intent="definition",
            metric=None,
            aggregation=None,
            filters={},
            location=None,
            comparison=None,
            time_range=None,
            required_tools=["cde"],
            response_type="summary",
        )

    async def _statistics_service(self) -> Optional[StatisticsService]:
        if self.db_session is None:
            return None
        try:
            if hasattr(self.db_session, "session"):
                return StatisticsService(await self.db_session.session())
            return StatisticsService(self.db_session)
        except Exception:
            return None

    async def _canonicalize_plan_metrics(self, user_query: str, qp: QueryPlan) -> None:
        """Map LLM-chosen metrics to canonical DSSAT codes.

        "yield"/"Yield"/"YIELD" -> HWAM, "rainfall" -> PRCP, codes unchanged
        (the same vocabulary the fallback and analysis parser use). A missing
        or unknown metric on a statistic is resolved from the question with
        StatisticsService.resolve_metric; ambiguous wording becomes the same
        clarification question the fallback asks.
        """
        plan = self._semantic_plan
        if plan is None:
            return
        operations = [
            op for op in plan.operations
            if op.operation == "trend"
            or (op.operation == "aggregate" and (op.aggregation or "AVG").upper() != "COUNT")
        ]
        if not operations:
            return

        stats = await self._statistics_service()
        try:
            available = await stats.get_available_variables() if stats else list(ANALYSIS_VARIABLES)
        except Exception:
            available = list(ANALYSIS_VARIABLES)
        by_upper = {code.upper(): code for code in available}

        resolution = None
        requested = list(dict.fromkeys(code for _, code in find_variable_mentions(user_query)))
        explicit = explicit_variable_code(user_query)
        for op in operations:
            if len(operations) == 1 and (explicit or len(requested) == 1):
                code = explicit or requested[0]
                if code.upper() in by_upper:
                    op.metric = by_upper[code.upper()]
                    continue
            if op.metric:
                canonical = normalize_variable(op.metric) or ""
                if canonical.upper() in by_upper:
                    op.metric = by_upper[canonical.upper()]
                    continue
            if resolution is None and stats is not None:
                try:
                    resolution = await stats.resolve_metric(user_query)
                except Exception:
                    resolution = None
            if resolution is not None and resolution.code:
                logger.info("Planner: LLM metric %r -> %s", op.metric, resolution.code)
                op.metric = resolution.code
            elif resolution is not None and resolution.ambiguous:
                self._clarification = metric_clarification(resolution.candidates)

        first = next((op.metric for op in operations if op.metric), None)
        if first:
            qp.metric = first

    async def _no_llm_plan(self, user_query: str) -> QueryPlan:
        """Plan deterministically. Used only when the LLM is absent or fails.

        Order: definition shortcut, then MIN/MAX/AVG/SUM statistics (with
        metric, filters and clarification), then the generic fallback.
        """
        self._semantic_plan = None
        definition_plan = self._definition_plan(user_query)
        if definition_plan is not None:
            return definition_plan

        stats = await self._statistics_service()
        if stats is not None:
            try:
                fallback = await build_fallback_statistics_plan(
                    user_query, stats, metric_clarification
                )
            except Exception as exc:
                logger.warning(f"Deterministic statistics fallback failed: {exc}")
                fallback = None
            if fallback is not None:
                if fallback.clarification:
                    self._clarification = fallback.clarification
                self._semantic_plan = fallback.semantic_plan
                logger.info(
                    "Planner (no LLM): %s",
                    fallback.semantic_plan.goal if fallback.semantic_plan else "clarification",
                )
                return fallback.query_plan

        qp = await self._fallback_plan(user_query)
        # Metadata and count requests need the same domain filters as analysis.
        for condition in find_filters(user_query):
            if condition.operator == "BETWEEN":
                start, end = sorted(map(int, condition.value))
                qp.filters[condition.field] = list(range(start, end + 1))
            else:
                qp.filters[condition.field] = condition.value
        self._semantic_plan = self._fallback_semantic(user_query, qp)
        if re.search(r"\b(?:how many|number of|count)\b", user_query, re.I):
            for operation in self._semantic_plan.operations:
                operation.operation = "count"
                operation.entity = "simulation_outputs" if re.search(r"\boutputs?\b", user_query, re.I) else "simulations"
        return qp

    async def plan(
        self,
        user_query: str,
        analysis_context: Optional[AnalysisRequest] = None,
    ) -> QueryPlan:
        """Ask the LLM for a plan (raw; plan_with_fallback validates it).

        Uses Chat Completions, which every model on the OpenAI-compatible
        gateway supports (the installed SDK has no Responses API). Stores the
        semantic plan (or tool-based PlannerOutput) for the executor.
        """
        logger.info(f"Planning query: {user_query}")
        logger.info(f"Planner LLM configured: key={'set' if bool(self.api_key) else 'missing'}, model={settings.OPENAI_MODEL}")
        if not self.client:
            raise RuntimeError("No LLM client configured")
        self._semantic_plan = None
        self._last_planner_output = None

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

        prompt = get_planner_prompt(user_query)
        if analysis_context is not None:
            prompt += "\n\n" + describe_analysis_context(analysis_context)
        if planner_context:
            prompt += "\n\n" + planner_context

        response = await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL or "gpt-4o-mini",
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_content = response.choices[0].message.content if response.choices else None
        self._plan_source["raw_plan"] = raw_content
        if not raw_content:
            raise ValueError("LLM returned an empty plan")
        logger.info(f"Planner raw message content: {raw_content[:400]}")

        content_text = raw_content if isinstance(raw_content, str) else str(raw_content)
        try:
            plan_data = json.loads(content_text)
        except Exception:
            # Models often wrap JSON in prose or ```json fences.
            match = re.search(r"\{[\s\S]*\}", content_text)
            if not match:
                raise ValueError("LLM plan is not JSON")
            plan_data = json.loads(match.group(0))

        sem = self._parse_semantic_plan(plan_data)
        if sem:
            self._semantic_plan = sem
            return self._to_minimal_query_plan_from_semantic(sem)

        plan_struct = self._parse_planner_output(plan_data)
        if plan_struct:
            self._semantic_plan = self._build_semantic_plan(user_query, plan_struct)
            return self._to_legacy_query_plan(plan_struct, user_query)

        raise ValueError("LLM output is neither a semantic plan nor a tool plan")

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
                    if f.operator == "=" and f.field in {
                        "year",
                        "crop",
                        "cultivar",
                        "planting_stage",
                        "irrigation",
                        "nitrogen",
                        "nitrogen_level",
                        "state",
                        "district",
                        "country",
                    }:
                        filters[f.field] = f.value
                break
        if any(op.operation == "analysis" for op in sp.operations):
            return QueryPlan(intent="analysis", metric=None, aggregation=None, filters={}, location=None, comparison=None, time_range=None, required_tools=["statistics"], response_type="summary")
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

    def get_clarification(self) -> Optional[str]:
        """A question to ask the user instead of executing (e.g. ambiguous metric)."""
        return getattr(self, "_clarification", None)

    def get_semantic_plan(self) -> Optional[SemanticPlan]:
        return getattr(self, "_semantic_plan", None)

    def get_plan_source(self) -> Dict[str, Any]:
        """Who planned the last question: {"planner": "llm"|"fallback", ...}."""
        return dict(self._plan_source)

    async def plan_with_fallback(
        self,
        user_query: str,
        analysis_context: Optional[AnalysisRequest] = None,
    ) -> QueryPlan:
        """Plan with the LLM; fall back to deterministic planning only if needed.

        Args:
            user_query: Natural language question.
            analysis_context: Previous analysis request in this conversation,
                used to complete follow-up analysis questions.
        """
        self._clarification = None
        self._plan_source = {"planner": None, "model": settings.OPENAI_MODEL, "corrections": [],
                             "unapplied_filters": []}

        if not self.client:
            return await self._deterministic_plan(user_query, analysis_context, "no LLM configured")

        try:
            qp = await self.plan(user_query, analysis_context=analysis_context)
            qp = await self._validate_llm_plan(user_query, qp, analysis_context)
        except Exception as exc:
            logger.warning(f"LLM planning failed, using fallback: {type(exc).__name__}: {exc}")
            return await self._deterministic_plan(
                user_query, analysis_context, f"{type(exc).__name__}: {exc}"
            )

        self._plan_source["planner"] = "llm"
        return qp

    async def _deterministic_plan(
        self,
        user_query: str,
        analysis_context: Optional[AnalysisRequest],
        reason: str,
    ) -> QueryPlan:
        """Resilience path: plan without the LLM (analysis parser, then fallback)."""
        self._plan_source.update(planner="fallback", fallback_reason=reason, corrections=[],
                                 unapplied_filters=[])
        self._last_planner_output = None
        self._clarification = None
        analysis_request = parse_analysis_request(user_query, previous=analysis_context)
        if analysis_request is not None:
            return self._build_analysis_plan(analysis_request)
        return await self._no_llm_plan(user_query)

    async def _validate_llm_plan(
        self,
        user_query: str,
        qp: QueryPlan,
        analysis_context: Optional[AnalysisRequest],
    ) -> QueryPlan:
        """Deterministically validate and normalize the LLM plan.

        Raises ValueError when the plan cannot be executed, which sends the
        question to the (labeled) fallback planner.
        """
        sem = self._semantic_plan
        if sem is None:
            raise ValueError("LLM plan has no operations")

        # Locations are resolved before planning; a "spatial" step is not a plan.
        spatial_ops = [op for op in sem.operations if op.operation == "spatial"]
        if spatial_ops:
            self._note("dropped 'spatial' operation (location is resolved before planning)")
            sem.operations = [op for op in sem.operations if op.operation != "spatial"]

        forced = self._force_definition_plan(user_query, sem)
        if forced is not sem:
            if [op.variable for op in sem.operations if op.operation == "definition"] != [forced.operations[0].variable]:
                self._note(f"definition question: planned definition of {forced.operations[0].variable}")
            sem = forced
        sem = self._validate_analysis_operations(user_query, sem, analysis_context)
        sem = self._add_dataset_condition_filters(user_query, sem)
        self._validate_operation_filters(user_query, sem)
        self._semantic_plan = sem

        if not sem.operations:
            raise ValueError("LLM plan has no supported operations")

        analysis = next((op.analysis for op in sem.operations if op.operation == "analysis" and op.analysis), None)
        if analysis is not None:
            return self._analysis_query_plan(analysis)
        if sem.intent == "definition" or all(op.operation == "definition" for op in sem.operations):
            qp = self._to_minimal_query_plan_from_semantic(sem)
            qp.intent, qp.required_tools = "definition", ["cde"]
            return qp

        qp = self._to_minimal_query_plan_from_semantic(sem)
        await self._apply_statistic_rules(user_query, qp)

        missing = [
            op.operation for op in self._semantic_plan.operations
            if op.operation in {"aggregate", "trend"}
            and (op.aggregation or "AVG").upper() != "COUNT"
            and not op.metric
        ]
        if missing and not self._clarification:
            raise ValueError(f"LLM plan has a {missing[0]} operation without a resolvable metric")
        return qp

    async def _apply_statistic_rules(self, user_query: str, qp: QueryPlan) -> None:
        """Metric canonicalization and aggregation rules for statistic plans."""
        # LLM plans may name metrics by alias ("yield") or omit them;
        # canonicalize with the same vocabulary as the fallback path.
        before = [(op.operation, op.metric) for op in self._semantic_plan.operations]
        await self._canonicalize_plan_metrics(user_query, qp)
        for (operation, old), op in zip(before, self._semantic_plan.operations):
            if old != op.metric:
                self._note(f"{operation}: metric {old!r} -> {op.metric!r}")
        ql = user_query.lower()

        def set_aggregation(value: str, operations=("aggregate", "trend")) -> None:
            qp.aggregation = value
            for operation in self._semantic_plan.operations:
                if operation.operation in operations and operation.aggregation != value:
                    self._note(f"{operation.operation}: aggregation {operation.aggregation!r} -> {value!r}")
                    operation.aggregation = value

        # Force explicit minimum/maximum questions to use the
        # requested aggregation instead of the default average.
        if re.search(r"\b(minimum|min|minimal|lowest)\b", ql):
            set_aggregation("MIN")
        elif (
            re.search(r"\b(maximum|max|highest|largest)\b", ql)
            and not re.search(r"\baverage|avg|mean\b", ql)
        ):
            set_aggregation("MAX")
        # A single-year rainfall/precipitation question asks for that
        # year's average, not the sum across all simulation records.
        if (
            ("rainfall" in ql or "precipitation" in ql or "prcp" in ql)
            and re.search(r"\b(?:19|20)\d{2}\b", ql)
            and "total" not in ql
            and "sum" not in ql
            and not re.search(
                r"\b(minimum|minimal|lowest|min|maximum|max|highest|largest)\b",
                ql,
            )
        ):
            set_aggregation("AVG")

        # For "highest average" questions, use only grouped averages.
        if (
            re.search(r"\b(highest|largest|maximal)\b", ql)
            and re.search(r"\b(average|avg|mean)\b", ql)
        ):
            kept = [
                operation
                for operation in self._semantic_plan.operations
                if not (operation.operation == "aggregate" and operation.aggregation == "MAX")
            ]
            if len(kept) != len(self._semantic_plan.operations):
                self._note("dropped ungrouped MAX step from a 'highest average' question")
            self._semantic_plan.operations = kept
            set_aggregation("AVG", operations=("aggregate",))

        if (not qp.metric or not qp.aggregation) and any(w in ql for w in ["average", "avg", "mean", "total", "sum"]):
            logger.info("Planner: enriching aggregate plan (missing metric/aggregation)")
            qp.aggregation = qp.aggregation or "AVG"
            if not qp.metric:
                qp.metric = explicit_variable_code(user_query)
            if "statistics" not in qp.required_tools:
                qp.required_tools.append("statistics")
            if qp.intent != "aggregate":
                qp.intent = "aggregate"

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
                    resolution = await stats.resolve_metric(user_query)
                except Exception:
                    resolution = None
                if resolution is not None and resolution.ambiguous:
                    # Ask instead of guessing between e.g. TMAXA and TMINA.
                    self._clarification = metric_clarification(resolution.candidates)
                    return QueryPlan(
                        intent="metadata",
                        filters={},
                        required_tools=["metadata"],
                        response_type="summary",
                    )
                metric_resolved = resolution.code if resolution else None
            # Secondary heuristic: an explicit variable code in the query (e.g., HWAM)
            if not metric_resolved:
                metric_resolved = explicit_variable_code(user_query)
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
            # Map natural-language hybrid/season terms
            # to the existing cultivar values in the database.
            cultivar_aliases = {
                "very long season": "VLNG",
                "very-long season": "VLNG",
                "long season": "LNG",
                "long-season": "LNG",
                "very short season": "VSHT",
                "very-short season": "VSHT",
                "short season": "SHT",
                "short-season": "SHT",
                "baseline": "BASE",
                "standard": "BASE",
            }

            if "rainfed" in query_lower or "rain-fed" in query_lower:
                filters["irrigation"] = "RF"

            if (
                "high nitrogen" in query_lower
                or "high n" in query_lower
                or "highn" in query_lower
            ):
                filters["nitrogen_level"] = "HighN"
            for phrase, value in cultivar_aliases.items():
                if phrase in query_lower:
                    filters["cultivar"] = value
                    break

            # Map natural-language planting terms.
            # Positive stage values mean delayed planting.
            # Negative stage values mean earlier planting.

            if any(
                phrase in query_lower
                for phrase in [
                    "normal planting",
                    "normal planting date",
                    "reference planting",
                    "reference date",
                    "standard planting",
                    "neither early nor late",
                ]
            ):
                filters["planting_stage"] = "pfrst0"
            else:
                late_match = re.search(
                    r"(30|15)\s*days?\s*(late|later|after|delayed)",
                    query_lower,
                )

                early_match = re.search(
                    r"(30|15)\s*days?\s*(early|earlier|before)",
                    query_lower,
                )

                if late_match:
                    days = late_match.group(1)
                    filters["planting_stage"] = f"pfrst{days}"

                elif early_match:
                    days = early_match.group(1)
                    filters["planting_stage"] = f"pfrst-{days}"

                elif any(
                    phrase in query_lower
                    for phrase in [
                        "two weeks late",
                        "two weeks later",
                        "14 days late",
                        "14 days later",
                        "14 days after",
                    ]
                ):
                    filters["planting_stage"] = "pfrst15"

                elif any(
                    phrase in query_lower
                    for phrase in [
                        "two weeks early",
                        "two weeks earlier",
                        "14 days early",
                        "14 days earlier",
                        "14 days before",
                    ]
                ):
                    filters["planting_stage"] = "pfrst-15"
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
                for k in [
                    "crop",
                    "cultivar",
                    "irrigation",
                    "nitrogen",
                    "nitrogen_level",
                    "planting_stage",
                    "state",
                    "district",
                    "country",
                ]:
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
