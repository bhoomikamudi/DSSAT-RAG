"""Context Builder - merges tool outputs and prepares for LLM."""
from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional

from app.agent.models import (
    LLMContext,
    MetadataResult,
    StatisticsResult,
    SpatialResult,
    CDEResult,
    EmbeddingResult
)

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds context from tool outputs."""
    
    def build_context(
        self,
        metadata: "Optional[MetadataResult]" = None,
        statistics: "Optional[StatisticsResult]" = None,
        spatial: "Optional[SpatialResult]" = None,
        cde: "Optional[CDEResult]" = None,
        embeddings: "Optional[List[EmbeddingResult]]" = None
    ) -> "LLMContext":
        """
        Build context from tool outputs.
        
        Args:
            metadata: Metadata results
            statistics: Statistics results
            spatial: Spatial results
            cde: CDE results
            embeddings: Embedding search results
            
        Returns:
            Built LLMContext
        """
        # Merge all data
        merged = {
            "metadata": self._process_metadata(metadata),
            "statistics": self._process_statistics(statistics),
            "spatial": self._process_spatial(spatial),
            "cde": self._process_cde(cde),
            "embeddings": self._process_embeddings(embeddings)
        }
        
        # Build query summary
        query_summary = self._build_query_summary(merged)
        
        # Assess data quality
        data_quality = self._assess_data_quality(merged)
        
        return LLMContext(
            metadata=metadata,
            statistics=statistics,
            spatial=spatial,
            cde=cde,
            embeddings=embeddings,
            query_summary=query_summary,
            data_quality=data_quality
        )
    
    def _process_metadata(self, result: "Optional[MetadataResult]") -> Dict[str, Any]:
        """Process metadata results."""
        if not result:
            return {}
        
        # Extract key information
        crops = list(set(result.crops)) if result.crops else []
        cultivars = list(set(result.cultivars)) if result.cultivars else []
        
        # Sample simulations (limit to 5)
        sample_sims = result.simulations[:5] if result.simulations else []
        
        return {
            "total_count": result.total_count,
            "crops": crops,
            "cultivars": cultivars,
            "sample_simulations": sample_sims
        }
    
    def _process_statistics(self, result: "Optional[StatisticsResult]") -> Dict[str, Any]:
        """Process statistics results."""
        if not result:
            return {}
        
        return {
            "aggregation_type": result.aggregation_type,
            "variable_code": result.metric,
            "value": result.value,
            "count": result.count,
            "unit": getattr(result, "unit", None)
        }
    
    def _process_spatial(self, result: "Optional[SpatialResult]") -> Dict[str, Any]:
        """Process spatial results."""
        if not result:
            return {}
        
        # Extract location info
        bounds = result.bounds
        
        # Sample simulations
        sample_sims = result.simulations[:5] if result.simulations else []
        
        return {
            "total_count": result.total_count,
            "returned_count": result.returned_count,
            "sample_limit": result.sample_limit,
            "bounds": bounds,
            "sample_simulations": sample_sims
        }
    
    def _process_cde(self, result: "Optional[CDEResult]") -> Dict[str, Any]:
        """Process CDE results."""
        if not result:
            return {}
        
        # Build variable definitions map
        var_defs = {
            v.get("code", v.get("full_name")): v 
            for v in result.variable_definitions
        }
        
        return {
            "variable_definitions": var_defs,
            "relationships": result.relationships,
            "cultivar_info": result.cultivar_info
        }
    
    def _process_embeddings(self, results: "Optional[List[EmbeddingResult]]") -> List[Dict[str, Any]]:
        """Process embedding search results."""
        if not results:
            return []
        
        processed = []
        
        for result in results:
            docs = [
                {
                    "id": d.get("id"),
                    "title": d.get("title", d.get("summary", "")[:100]),
                    "source": result.sources[i] if i < len(result.sources) else "unknown"
                }
                for i, d in enumerate(result.documents)
            ]
            
            processed.append({
                "collection": result.sources[0] if result.sources else "unknown",
                "documents": docs,
                "scores": result.scores
            })
        
        return processed
    
    def _build_query_summary(self, merged: Dict[str, Any]) -> str:
        """Build natural language summary."""
        parts = []
        
        # Metadata summary
        metadata = merged.get("metadata", {})
        if metadata.get("total_count"):
            count = metadata["total_count"]
            crops = metadata.get("crops", [])
            
            if crops:
                crop_str = " and ".join(crops[:2])
                parts.append(f"Found {count} simulation(s) for {crop_str}")
            else:
                parts.append(f"Found {count} simulation(s)")
        
        # Statistics summary
        stats = merged.get("statistics", {})
        if stats.get("value") is not None:
            value = stats["value"]
            agg_type = stats.get("aggregation_type", "average")
            var_code = stats.get("variable_code", "the metric")
            
            parts.append(f"{agg_type.title()} {var_code}: {value:.2f}")
        
        # Spatial summary
        spatial = merged.get("spatial", {})
        if spatial.get("total_count"):
            count = spatial["total_count"]
            bounds = spatial.get("bounds", {})
            
            if bounds:
                parts.append(f"Spatial coverage: {count} locations")
        
        # CDE summary
        cde = merged.get("cde", {})
        if cde.get("variable_definitions"):
            defs = cde["variable_definitions"]
            parts.append(f"Variable definitions available for {len(defs)} variable(s)")
        
        return ". ".join(parts) if parts else "Query executed successfully"
    
    def _assess_data_quality(self, merged: Dict[str, Any]) -> str:
        """Assess data quality."""
        # Check metadata count
        metadata = merged.get("metadata", {})
        count = metadata.get("total_count", 0)
        
        if count >= 20:
            return "high"
        elif count >= 5:
            return "medium"
        else:
            return "low"
