"""Dynamic Tool Registry and capability provider for the Planner.

Exposes tool specifications (not DB schemas) to the LLM Planner and
resolves tool handlers for the Executor.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.analysis_vocabulary import (
    ANALYSIS_OPERATIONS,
    ANALYSIS_VARIABLES,
    GROUP_BY_FIELDS,
)
from app.services.statistics_service import StatisticsService


class ToolRegistry:
    """Registry of available tools and their dynamic capabilities."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self._capabilities: Optional[Dict[str, Any]] = None

    async def load_capabilities(self) -> Dict[str, Any]:
        """Load tool capabilities from services/DB and cache them."""
        if self._capabilities is not None:
            return self._capabilities

        stats = StatisticsService(self.db)
        supported_metrics = await stats.get_available_variables()
        supported_crops = await stats.get_available_crops()

        simulation_tool = {
            "name": "query_simulation_data",
            "description": "Retrieve DSSAT simulation information (metadata, outputs, aggregation, filtering, grouping, spatial, trends).",
            "supported_filters": [
                "crop", "cultivar", "year", "country", "state", "district",
                "ecological_zone", "irrigation", "nitrogen", "run_name"
            ],
            "supported_metrics": supported_metrics,
            "supported_crops": supported_crops,
            "supported_aggregations": ["AVG", "MAX", "MIN", "COUNT", "SUM", "GROUP BY"],
            "supported_spatial": [
                "radius", "polygon", "country", "state", "district", "bounding box", "nearest"
            ],
        }

        cde_tool = {
            "name": "query_cde",
            "description": "Retrieve DSSAT definitions and relationships (variables, cultivars, species, ecotypes, synonyms)",
            "supported_entities": ["variables", "cultivars", "species", "ecotypes", "synonyms", "relationships"],
        }

        semantic_tool = {
            "name": "semantic_search",
            "description": "Semantic search over simulation summaries, manuals, research papers, documentation",
            "params": {"query": "string", "top_k": "int"},
        }

        analysis_tool = {
            "name": "analyze_simulation_data",
            "description": "Controlled Python analysis (correlation, linear/quadratic regression, descriptive statistics) on filtered simulation records. Use operation=\"analysis\".",
            "supported_operations": list(ANALYSIS_OPERATIONS),
            "supported_variables": list(ANALYSIS_VARIABLES),
            "supported_group_by": list(GROUP_BY_FIELDS),
        }

        self._capabilities = {
            "tools": [simulation_tool, cde_tool, semantic_tool, analysis_tool]
        }
        return self._capabilities

    async def get_planner_context(self) -> str:
        """Return a concise, human-readable spec string for the planner prompt."""
        caps = await self.load_capabilities()
        lines: List[str] = ["Available Tools (capabilities):"]
        for t in caps.get("tools", []):
            if t["name"] == "query_simulation_data":
                lines.append(f"- {t['name']}: {t['description']}")
                lines.append(f"  Supported Filters: {', '.join(t['supported_filters'])}")
                lines.append(f"  Supported Metrics: {', '.join(t['supported_metrics'][:50])}{' ...' if len(t['supported_metrics'])>50 else ''}")
                if t.get("supported_crops"):
                    lines.append(f"  Supported Crops: {', '.join(t['supported_crops'])}")
                lines.append(f"  Supported Aggregations: {', '.join(t['supported_aggregations'])}")
                lines.append(f"  Supported Spatial: {', '.join(t['supported_spatial'])}")
            elif t["name"] == "analyze_simulation_data":
                lines.append(f"- {t['name']}: {t['description']}")
                lines.append(f"  Operations: {', '.join(t['supported_operations'])}")
                lines.append(f"  Variables: {', '.join(t['supported_variables'])}")
                lines.append(f"  Group by: {', '.join(t['supported_group_by'])}")
            elif t["name"] == "query_cde":
                lines.append(f"- {t['name']}: {t['description']}")
                lines.append(f"  Entities: {', '.join(t['supported_entities'])}")
            else:
                lines.append(f"- {t['name']}: {t['description']}")
        return "\n".join(lines)
