"""CDE Service - DSSAT definitions and project-specific mappings."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class CDEService:
    """Service for DSSAT CDE information and project mappings."""

    def __init__(self, db: Optional[AsyncSession] = None):
        self.db = db

    # Project-specific labels from the CSV run-name convention
    PROJECT_MAPPINGS: Dict[str, Dict[str, str]] = {
        "BASE": {
            "code": "BASE",
            "meaning": "Baseline cultivar",
            "category": "cultivar",
        },
        "LNG": {
            "code": "LNG",
            "meaning": "Long-season cultivar",
            "category": "cultivar",
        },
        "SHT": {
            "code": "SHT",
            "meaning": "Short-season cultivar",
            "category": "cultivar",
        },
        "VLNG": {
            "code": "VLNG",
            "meaning": "Very-long-season cultivar",
            "category": "cultivar",
        },
        "VSHT": {
            "code": "VSHT",
            "meaning": "Very-short-season cultivar",
            "category": "cultivar",
        },
        "pfrst0": {
            "code": "pfrst0",
            "meaning": "Normal planting",
            "category": "planting_stage",
        },
        "pfrst15": {
            "code": "pfrst15",
            "meaning": "15 days after normal planting",
            "category": "planting_stage",
        },
        "pfrst30": {
            "code": "pfrst30",
            "meaning": "30 days after normal planting",
            "category": "planting_stage",
        },
        "pfrst-15": {
            "code": "pfrst-15",
            "meaning": "15 days before normal planting",
            "category": "planting_stage",
        },
        "pfrst-30": {
            "code": "pfrst-30",
            "meaning": "30 days before normal planting",
            "category": "planting_stage",
        },
    }

    VARIABLE_DEFINITIONS: Dict[str, Dict[str, str]] = {
        "HWAM": {
            "full_name": "Harvested Weight of Dry Matter",
            "description": "Harvested dry matter yield",
            "unit": "kg/ha",
            "category": "yield",
        },
        "CWAM": {
            "full_name": "Crop Weight at Maturity",
            "description": "Above-ground crop biomass at maturity",
            "unit": "kg/ha",
            "category": "biomass",
        },
        "HWAH": {
            "full_name": "Harvested Weight at Harvest",
            "description": "Harvested weight measured at harvest",
            "unit": "kg/ha",
            "category": "harvest",
        },
        "GNAM": {
            "full_name": "Grain Number",
            "description": "Number of grains produced",
            "unit": "count",
            "category": "grain",
        },
        "PRCP": {
            "full_name": "Precipitation",
            "description": "Total rainfall or precipitation",
            "unit": "mm",
            "category": "weather",
        },
        "TMAXA": {
            "full_name": "Average Maximum Temperature",
            "description": "Average daily maximum temperature",
            "unit": "°C",
            "category": "weather",
        },
        "TMINA": {
            "full_name": "Average Minimum Temperature",
            "description": "Average daily minimum temperature",
            "unit": "°C",
            "category": "weather",
        },
        "YIELD": {
            "full_name": "Harvest Yield",
            "description": "Grain yield at harvest",
            "unit": "kg/ha",
            "category": "yield",
        },
        "LAI": {
            "full_name": "Leaf Area Index",
            "description": "Total leaf area per unit ground area",
            "unit": "m²/m²",
            "category": "growth",
        },
        "ET": {
            "full_name": "Evapotranspiration",
            "description": "Water loss from soil and plants to the atmosphere",
            "unit": "mm",
            "category": "water",
        },
        "GDD": {
            "full_name": "Growing Degree Days",
            "description": "Accumulated heat units for crop development",
            "unit": "°C-days",
            "category": "development",
        },
        "NUP": {
            "full_name": "Nitrogen Uptake",
            "description": "Total nitrogen absorbed by the crop",
            "unit": "kg/ha",
            "category": "nutrient",
        },
    }

    VARIABLE_RELATIONSHIPS: Dict[str, List[str]] = {
        "HWAM": ["LAI", "ET", "GDD", "NUP"],
        "CWAM": ["HWAM", "GNAM", "PRCP"],
        "YIELD": ["LAI", "ET", "PRCP"],
        "LAI": ["GDD", "NUP"],
    }

    VARIABLE_SYNONYMS: Dict[str, List[str]] = {
        "HWAM": ["yield", "harvest_yield", "dry_matter", "total_yield"],
        "CWAM": ["biomass", "crop_biomass"],
        "GNAM": ["grain_number"],
        "PRCP": ["precipitation", "rainfall"],
        "TMAXA": ["maximum_temperature", "average_maximum_temperature"],
        "TMINA": ["minimum_temperature", "average_minimum_temperature"],
        "LAI": ["leaf_area_index"],
        "ET": ["evapotranspiration", "water_loss"],
        "GDD": ["growing_degree_days", "heat_units"],
    }

    CULTIVAR_INFO: Dict[str, List[Dict[str, str]]] = {
        "Maize": [
            {
                "name": "BASE",
                "description": "Baseline cultivar",
            },
            {
                "name": "LNG",
                "description": "Long-season cultivar",
            },
            {
                "name": "SHT",
                "description": "Short-season cultivar",
            },
            {
                "name": "VLNG",
                "description": "Very-long-season cultivar",
            },
            {
                "name": "VSHT",
                "description": "Very-short-season cultivar",
            },
        ]
    }

    SPECIES_INFO: Dict[str, Dict[str, Any]] = {
        "Maize": {
            "scientific_name": "Zea mays",
            "growth_cycle_days": "90-120",
            "optimal_temperature": "20-30°C",
            "optimal_ph": "5.5-7.0",
        },
        "Wheat": {
            "scientific_name": "Triticum aestivum",
            "growth_cycle_days": "100-130",
            "optimal_temperature": "15-25°C",
            "optimal_ph": "6.0-7.5",
        },
        "Rice": {
            "scientific_name": "Oryza sativa",
            "growth_cycle_days": "90-150",
            "optimal_temperature": "20-35°C",
            "optimal_ph": "5.5-6.5",
        },
    }

    async def get_project_mapping(
        self,
        code: str,
    ) -> Optional[Dict[str, Any]]:
        """Return a project mapping from the database, with fallback."""

        if self.db is not None:
            try:
                result = await self.db.execute(
                    text(
                        """
                        SELECT code, meaning, category, source_file
                        FROM reference_codes
                        WHERE LOWER(code) = LOWER(:code)
                        LIMIT 1
                        """
                    ),
                    {"code": code},
                )

                row = result.mappings().first()

                if row:
                    return dict(row)

            except Exception as exc:
                logger.warning(
                    "Could not read project mapping from database: %s",
                    exc,
                )

        return self.PROJECT_MAPPINGS.get(code)

    async def get_variable_definition(
        self,
        variable_code: str,
    ) -> Optional[Dict[str, Any]]:
        """Return a variable definition."""

        code = variable_code.upper()

        definition = self.VARIABLE_DEFINITIONS.get(code)

        if definition:
            return {
                "code": code,
                **definition,
            }

        return None

    async def get_variable_relationships(
        self,
        variable_code: str,
    ) -> List[str]:
        """Return related variables."""

        return self.VARIABLE_RELATIONSHIPS.get(
            variable_code.upper(),
            [],
        )

    async def get_cultivar_info(
        self,
        crop: str,
    ) -> List[Dict[str, Any]]:
        """Return cultivar information for a crop."""

        return self.CULTIVAR_INFO.get(crop, [])

    async def get_species_info(
        self,
        crop: str,
    ) -> Optional[Dict[str, Any]]:
        """Return species information for a crop."""

        return self.SPECIES_INFO.get(crop)

    async def find_variable_synonyms(
        self,
        variable_code: str,
    ) -> List[str]:
        """Return synonyms for a variable."""

        return self.VARIABLE_SYNONYMS.get(
            variable_code.upper(),
            [],
        )

    async def search_variables(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        """Search variable definitions."""

        query_lower = query.lower()
        results = []

        for code, info in self.VARIABLE_DEFINITIONS.items():
            if (
                query_lower in code.lower()
                or query_lower in info["full_name"].lower()
                or query_lower in info["description"].lower()
            ):
                results.append(
                    {
                        "code": code,
                        **info,
                    }
                )

        return results

    async def get_all_variables(self) -> List[Dict[str, Any]]:
        """Return all variable definitions."""

        return [
            {
                "code": code,
                **info,
            }
            for code, info in self.VARIABLE_DEFINITIONS.items()
        ]