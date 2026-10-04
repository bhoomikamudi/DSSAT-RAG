"""Response Generator - generates final answers using LLM."""
import logging
from typing import Optional

from openai import AsyncOpenAI

from app.agent.models import ResponseGeneration, SourceReference
from app.services.display_format import format_stat
from app.agent.prompts import get_response_prompt, SYSTEM_PROMPT
from app.core.config import get_settings

settings = get_settings()

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generates final responses using LLM."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize response generator.

        Args:
            api_key: OpenAI API key (optional)
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        # Who wrote the last answer text: "llm", "fallback_template",
        # or a verbatim notice/clarification/analysis explanation.
        self.last_generated_by: Optional[str] = None
        if self.api_key:
            if settings.OPENAI_BASE_URL:
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=settings.OPENAI_BASE_URL)
            else:
                self.client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = None

    async def generate(
        self,
        user_question: str,
        context: "LLMContext",
        query_plan: Optional["QueryPlan"] = None
    ) -> ResponseGeneration:
        """
        Generate response from context.

        Args:
            user_question: Original user question
            context: Built LLMContext
            query_plan: Query plan (optional)

        Returns:
            Generated response
        """
        logger.info("Generating response")
        logger.info(f"Response LLM configured: key={'set' if bool(self.api_key) else 'missing'}, model={settings.OPENAI_MODEL}")

        # A location that could not be used (failed/ambiguous lookup, no data
        # in the radius) is answered verbatim so the LLM cannot guess.
        if context.spatial_notice:
            self.last_generated_by = "notice"
            return ResponseGeneration(
                answer=context.spatial_notice,
                sources=[],
                confidence="low",
                limitations=["Location filter could not be applied"],
            )

        # An ambiguous question is answered with a question, not a guess.
        if context.clarification:
            self.last_generated_by = "clarification"
            return ResponseGeneration(
                answer=context.clarification,
                sources=[],
                confidence="low",
                limitations=["Clarification needed before querying"],
            )

        # Analyses that could not run are answered with the deterministic
        # explanation so the LLM cannot invent a result.
        if context.analysis and context.analysis.status != "ok":
            self.last_generated_by = "analysis_explanation"
            return ResponseGeneration(
                answer=context.analysis.explanation,
                sources=self._build_sources(context),
                confidence="low",
                limitations=self._identify_limitations(context),
            )

        # If no LLM, build a simple heuristic answer from context
        if not self.client:
            answer = self._heuristic_answer(context, user_question)
            sources = self._build_sources(context)
            confidence = self._assess_confidence(context, answer)
            self.last_generated_by = "fallback_template"
            return ResponseGeneration(
                answer=answer,
                sources=sources,
                confidence=confidence,
                limitations=self._identify_limitations(context)
            )

        # Format context and get prompt
        context_str = self._format_context(context)
        prompt = get_response_prompt(context, user_question)

        try:
            # Chat Completions: supported by every gateway model and by the
            # installed SDK (which has no Responses API).
            cc = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL or "gpt-4o-mini",
                temperature=0.3,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            answer = cc.choices[0].message.content or ""

            sources = self._build_sources(context)
            confidence = self._assess_confidence(context, answer)

            self.last_generated_by = "llm"
            return ResponseGeneration(
                answer=answer,
                sources=sources,
                confidence=confidence,
                limitations=self._identify_limitations(context)
            )

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            # Fall back to the answer built from computed results (statistics,
            # analysis explanation, area) instead of discarding them.
            fallback = self._heuristic_answer(context, user_question)
            if not fallback or fallback == "Query executed successfully":
                fallback = "I encountered an error generating the response. Please try again."
            self.last_generated_by = "fallback_template"
            return ResponseGeneration(
                answer=fallback,
                sources=self._build_sources(context),
                confidence="low",
                limitations=["LLM service unavailable; answer built from computed results"]
                + self._identify_limitations(context),
            )

    def _heuristic_answer(self, context: "LLMContext", user_question: str) -> str:
        """Produce a simple answer without LLM based on available context."""
        management = (context.management_scope or {}).get("sentence")
        if context.analysis:
            explanation = context.analysis.explanation
            if management and context.analysis.status == "ok":
                explanation += "\n\n" + management
            return explanation
        parts = []
        if context.spatial_scope and not context.spatial_scope.get("cleared"):
            parts.append(
                f"Using simulations {context.spatial_scope['description']} "
                f"({context.spatial_scope['locations']} locations)"
            )
        # Primary statistic
        if context.statistics and context.statistics.value is not None:
            parts.append(
                f"{context.statistics.aggregation_type.title()} {context.statistics.metric}: {format_stat(context.statistics.value)}"
            )
        # Additional stats (e.g., MAX and MIN together)
        if context.additional_statistics:
            for s in context.additional_statistics:
                if s.value is not None:
                    parts.append(f"{s.aggregation_type.title()} {s.metric}: {format_stat(s.value)}")
                if s.breakdown and s.breakdown.get("extremum_location"):
                    loc = s.breakdown["extremum_location"]
                    parts.append(
                        f"Location for {s.aggregation_type.title()}: "
                        f"({loc.get('latitude')}, {loc.get('longitude')})"
                        + (f", {loc.get('state') or ''}" if loc.get('state') else "")
                    )
        # Grouped results ("which cultivar has the highest average ..."):
        # report the highest and lowest groups, not only the overall value.
        breakdown = (context.statistics.breakdown or {}) if context.statistics else {}
        groups = [
            v for v in breakdown.get("values") or []
            if v.get("group_value") is not None and v.get("value") is not None
        ]
        if groups:
            ranked = sorted(groups, key=lambda v: v["value"], reverse=True)
            label = {"AVG": "average", "MIN": "minimum", "MAX": "maximum", "SUM": "total"}.get(
                context.statistics.aggregation_type.upper(), context.statistics.aggregation_type.lower()
            )
            parts.append(
                f"By {breakdown.get('group_by')}: highest {label} is "
                f"{ranked[0]['group_value']} ({format_stat(ranked[0]['value'])}), lowest is "
                f"{ranked[-1]['group_value']} ({format_stat(ranked[-1]['value'])})"
            )

        if context.statistics and context.statistics.value is not None:
            parts.append(f"Based on {context.statistics.count:,} records")
        elif context.metadata is not None:
            parts.append(f"Found {context.metadata.total_count:,} matching simulations")
            returned = len(context.metadata.simulations)
            if returned and context.metadata.total_count > returned:
                parts.append(
                    f"Returned a capped sample of {returned:,} simulations "
                    f"(limit {context.metadata.sample_limit or returned:,})"
                )
        if not parts:
            return "Query executed successfully"
        answer = ". ".join(parts)
        return f"{answer}. {management}" if management else answer

    def _format_context(self, context: "LLMContext") -> str:
        """Format context for prompt."""
        parts = []

        # Query summary
        if context.query_summary:
            parts.append(f"Query Summary: {context.query_summary}")

        # Metadata
        if context.metadata and context.metadata.simulations:
            sim_count = len(context.metadata.simulations)
            parts.append(f"\nFound {context.metadata.total_count:,} matching simulations; showing {sim_count:,} sample records")

            # Add sample
            for sim in context.metadata.simulations[:3]:
                parts.append(f"- {sim.get('experiment_name', 'Unknown')}: "
                          f"{sim.get('crop', '')} in {sim.get('country', '')}")

        # Statistics
        if context.statistics and context.statistics.value is not None:
            parts.append(f"\n{context.statistics.aggregation_type.title()} value: "
                        f"{format_stat(context.statistics.value)} (count: {context.statistics.count:,})")
        if context.additional_statistics:
            for s in context.additional_statistics:
                if s.value is not None:
                    parts.append(f"{s.aggregation_type.title()} value: {format_stat(s.value)} (count: {s.count:,})")
                if s.breakdown and s.breakdown.get("extremum_location"):
                    loc = s.breakdown["extremum_location"]
                    parts.append(
                        f"{s.aggregation_type.title()} location: "
                        f"({loc.get('latitude')}, {loc.get('longitude')})"
                    )

        # CDE
        if context.cde and context.cde.variable_definitions:
            parts.append("\nVariable definitions:")
            for var in context.cde.variable_definitions[:3]:
                parts.append(f"- {var.get('full_name', 'Unknown')}: {var.get('description', '')}")

        # Embeddings
        if context.embeddings:
            doc_count = sum(len(e.documents) for e in context.embeddings)
            parts.append(f"\nFound {doc_count} relevant document(s)")

        return "\n".join(parts)

    def _build_sources(self, context: "LLMContext") -> list:
        """Build source references."""
        sources = []

        if context.metadata and context.metadata.total_count > 0:
            sources.append(SourceReference(
                type="metadata",
                description=f"Database: {context.metadata.total_count} simulation records"
            ))

        if context.statistics and context.statistics.count > 0:
            sources.append(SourceReference(
                type="statistics",
                description=f"Aggregated statistics from {context.statistics.count} records"
            ))
        if context.additional_statistics:
            total = sum(s.count for s in context.additional_statistics if s and s.count)
            if total:
                sources.append(SourceReference(
                    type="statistics",
                    description=f"Additional aggregations from {total} records"
                ))

        if context.analysis and context.analysis.sample_size > 0:
            sources.append(SourceReference(
                type="statistics",
                description=(
                    f"Python {context.analysis.analysis_type.replace('_', ' ')} "
                    f"on {context.analysis.sample_size:,} simulation records"
                )
            ))

        if context.cde and context.cde.variable_definitions:
            sources.append(SourceReference(
                type="cde",
                description="Crop Data Exchange definitions"
            ))

        if context.embeddings:
            for emb in context.embeddings:
                if emb.documents:
                    sources.append(SourceReference(
                        type="qdrant",
                        description=f"Document search: {emb.sources[0] if emb.sources else 'unknown'}"
                    ))
                    break  # Only add once

        return sources

    def _assess_confidence(self, context: "LLMContext", answer: str) -> str:
        """Assess confidence in the answer."""
        # Check for uncertainty phrases
        uncertainty_phrases = [
            "I cannot",
            "I don't have enough information",
            "Data not available",
            "Unable to determine"
        ]

        if any(phrase.lower() in answer.lower() for phrase in uncertainty_phrases):
            return "low"

        # Check data quality
        if context.data_quality == "high":
            return "high"
        elif context.data_quality == "medium":
            return "medium"
        else:
            return "low"

    def _identify_limitations(self, context: "LLMContext") -> list:
        """Identify limitations in the response."""
        limitations = []

        # Check for low data quality
        if context.data_quality == "low":
            limitations.append("Limited data available")

        if context.analysis:
            limitations.extend(context.analysis.warnings)
            if context.analysis.chart_data and context.analysis.chart_data.get("sampled"):
                limitations.append(
                    "The chart shows a random sample of points; statistics use all records"
                )
            return limitations

        # Check for missing filters
        if not context.metadata and not context.spatial:
            limitations.append("No matching simulations found")

        return limitations
