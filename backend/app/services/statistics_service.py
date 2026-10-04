"""Statistics Service - aggregate calculations and statistical analysis."""

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, func as sql_func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import aliased

from app.models.simulation import Simulation, SimulationOutput
from app.repositories.simulation import (
    SimulationOutputRepository,
    SimulationRepository,
)
from app.services.analysis_vocabulary import find_variable_mentions
from app.services.spatial_sql import spatial_condition

logger = logging.getLogger(__name__)

# Words that never identify a metric on their own ("average" must not match
# "Average Maximum Temperature").
METRIC_STOP_WORDS = {
    "what", "which", "who", "how", "the", "and", "for", "with", "from", "that",
    "this", "was", "were", "are", "is", "show", "give", "tell", "near", "around",
    "average", "avg", "mean", "median", "total", "sum", "minimum", "maximum",
    "min", "max", "highest", "lowest", "value", "values", "daily", "per", "all",
    "data", "simulation", "simulations", "records", "crop", "cultivar", "year",
    "years", "number", "count", "many", "much",
}


@dataclass
class MetricResolution:
    """Outcome of resolving a metric: one code, or competing candidates."""

    code: Optional[str]
    candidates: List[str] = field(default_factory=list)
    source: Optional[str] = None

    @property
    def ambiguous(self) -> bool:
        return self.code is None and len(self.candidates) > 1

    @classmethod
    def from_candidates(cls, candidates: List[str], source: str) -> "MetricResolution":
        unique = list(dict.fromkeys(candidates))
        if len(unique) == 1:
            return cls(code=unique[0], candidates=unique, source=source)
        return cls(code=None, candidates=unique, source=source if unique else None)

# Upper bound on rows transferred for one Python analysis request.
MAX_ANALYSIS_ROWS = 250_000

# Simulation columns that analysis requests may group by or inspect.
ANALYSIS_GROUP_COLUMNS: Dict[str, Any] = {
    "cultivar": Simulation.cultivar,
    "year": Simulation.simulation_year,
    "planting_stage": Simulation.planting_stage,
    "crop": Simulation.crop,
    "irrigation": Simulation.irrigation,
    "nitrogen_level": Simulation.nitrogen_level,
    "state": Simulation.state,
    "district": Simulation.district,
    "country": Simulation.country,
}


class StatisticsService:
    """Service for statistical calculations."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.sim_repo = SimulationRepository(db)
        self.output_repo = SimulationOutputRepository(db)

    async def get_available_variables(self) -> List[str]:
        stmt = (
            select(SimulationOutput.variable_code)
            .distinct()
            .order_by(SimulationOutput.variable_code)
        )
        result = await self.db.execute(stmt)
        return [row[0] for row in result.all()]


    async def get_available_crops(self) -> List[str]:
        stmt = (
            select(Simulation.crop)
            .distinct()
            .order_by(Simulation.crop)
        )
        result = await self.db.execute(stmt)
        return [row[0] for row in result.all()]

    async def resolve_metric_from_text(
        self,
        text_query: str,
    ) -> Optional[str]:
        """Resolve a variable code from natural language.

        Returns None when no variable, or more than one, fits the wording;
        use resolve_metric() to get the competing candidates.
        """

        return (await self.resolve_metric(text_query)).code

    async def resolve_metric(self, text_query: str) -> "MetricResolution":
        """Resolve the metric a question asks about.

        Sources, in order, all restricted to variables present in the data:
        1. The project's canonical synonyms (analysis_vocabulary), e.g.
           "yield" -> HWAM, "rainfall" -> PRCP, codes like "HWAM".
        2. Exact variable codes present in the database (future variables).
        3. CDE definitions (full name and description), ignoring aggregation
           and stop words so "average" cannot select "Average Maximum
           Temperature".
        More than one fitting variable is reported as ambiguous rather than
        picking one.
        """

        variables = await self.get_available_variables()
        available = {variable.upper(): variable for variable in variables}

        explicit = [available[token] for token in re.findall(r"\b[A-Z][A-Z0-9_]+\b", text_query) if token in available]
        if explicit:
            return MetricResolution.from_candidates(explicit, "code")

        mentions = [
            code for _, code in find_variable_mentions(text_query)
            if code in available
        ]
        if mentions:
            return MetricResolution.from_candidates(mentions, "synonym")

        tokens = re.findall(r"[a-z0-9_]+", text_query.lower())
        exact = [available[t.upper()] for t in dict.fromkeys(tokens) if t.upper() in available]
        if exact:
            return MetricResolution.from_candidates(exact, "code")

        words = [t for t in tokens if len(t) >= 3 and t not in METRIC_STOP_WORDS]
        candidates: List[str] = []
        try:
            from app.services.cde_service import CDEService

            for item in await CDEService().get_all_variables():
                code = item.get("code")
                if code not in available or code in candidates:
                    continue
                searchable = set(
                    re.findall(
                        r"[a-z0-9]+",
                        f"{item.get('full_name', '')} {item.get('description', '')}".lower(),
                    )
                )
                if any(word in searchable for word in words):
                    candidates.append(code)
        except Exception:
            logger.exception("Failed to resolve metric through CDE")

        return MetricResolution.from_candidates(candidates, "cde")

    async def resolve_crop_from_text(
        self,
        text_query: str,
    ) -> Optional[str]:
        """Resolve a crop code using values available in the database."""

        query = text_query.lower()
        crops = await self.get_available_crops()

        crop_map = {
            crop.lower(): crop
            for crop in crops
        }

        for word in query.split():
            cleaned_word = word.strip(" ,.:;()[]{}")
            if cleaned_word in crop_map:
                return crop_map[cleaned_word]

        candidates = [
            crop
            for crop in crops
            if crop.lower() in query
        ]

        if len(candidates) == 1:
            return candidates[0]

        return None

    async def resolve_crop_code(
        self,
        crop_text: str,
    ) -> Optional[str]:
        """Resolve a user-provided crop name or code."""

        if not crop_text:
            return None

        crops = await self.get_available_crops()

        for crop in crops:
            if crop.lower() == crop_text.lower():
                return crop

        synonyms = {
            "maize": "MZ",
            "corn": "MZ",
        }

        normalized_text = crop_text.strip().lower()

        if (
            normalized_text in synonyms
            and synonyms[normalized_text] in crops
        ):
            return synonyms[normalized_text]

        normalize = lambda value: "".join(
            character
            for character in value.lower()
            if character.isalpha()
        )

        normalized_crop_text = normalize(crop_text)

        for crop in crops:
            normalized_crop = normalize(crop)

            if (
                normalized_crop == normalized_crop_text
                or crop.lower() in normalized_crop_text
                or normalized_crop_text in crop.lower()
            ):
                return crop

        return None

    @staticmethod
    def _year_filter(year: Optional[object]):
        """Create a year filter for one year or multiple years."""

        if year is None:
            return None

        if isinstance(year, Iterable) and not isinstance(
            year,
            (str, bytes),
        ):
            years = list(year)

            return Simulation.simulation_year.in_(years)

        return Simulation.simulation_year == year

    @staticmethod
    def _equals_or_in(column: Any, value: object):
        """Equality for a scalar value, IN for a list of values."""

        if isinstance(value, (list, tuple, set)):
            return column.in_(list(value))

        return column == value

    def _build_simulation_filters(
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
        spatial: Optional[Any] = None,
    ) -> List[Any]:
        """Build reusable filters for simulation-level fields.

        spatial is an optional radius filter (latitude, longitude, radius_km)
        applied as a geodesic distance condition in SQL.
        """

        filters: List[Any] = []

        if crop is not None:
            filters.append(self._equals_or_in(Simulation.crop, crop))

        if cultivar is not None:
            filters.append(self._equals_or_in(Simulation.cultivar, cultivar))

        if irrigation is not None:
            filters.append(
                self._equals_or_in(Simulation.irrigation, irrigation)
            )

        if nitrogen_level is not None:
            filters.append(
                self._equals_or_in(Simulation.nitrogen_level, nitrogen_level)
            )

        if planting_stage is not None:
            filters.append(
                self._equals_or_in(Simulation.planting_stage, planting_stage)
            )

        year_filter = self._year_filter(year)
        if year_filter is not None:
            filters.append(year_filter)

        if state is not None:
            filters.append(self._equals_or_in(Simulation.state, state))

        if district is not None:
            filters.append(self._equals_or_in(Simulation.district, district))

        if country is not None:
            filters.append(self._equals_or_in(Simulation.country, country))

        if spatial is not None:
            filters.append(spatial_condition(spatial))

        return filters

    async def calculate_aggregation(
        self,
        variable_code: str,
        aggregation: str,
        crop: Optional[str] = None,
        cultivar: Optional[str] = None,
        irrigation: Optional[str] = None,
        nitrogen_level: Optional[str] = None,
        planting_stage: Optional[str] = None,
        year: Optional[object] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
        country: Optional[str] = None,
        spatial: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Calculate an aggregate statistic."""

        simulation_filters = self._build_simulation_filters(
            crop=crop,
            cultivar=cultivar,
            irrigation=irrigation,
            nitrogen_level=nitrogen_level,
            planting_stage=planting_stage,
            year=year,
            state=state,
            district=district,
            country=country,
            spatial=spatial,
        )

        aggregation_functions = {
            "avg": sql_func.avg,
            "max": sql_func.max,
            "min": sql_func.min,
            "count": sql_func.count,
            "sum": sql_func.sum,
        }

        aggregate_function = aggregation_functions.get(
            aggregation.lower(),
            sql_func.avg,
        )

        stmt = (
            select(
                aggregate_function(
                    SimulationOutput.value
                ).label("value"),
                sql_func.count().label("count"),
                sql_func.stddev(
                    SimulationOutput.value
                ).label("stddev"),
                sql_func.min(
                    SimulationOutput.value
                ).label("min"),
                sql_func.max(
                    SimulationOutput.value
                ).label("max"),
            )
            .join(
                Simulation,
                SimulationOutput.simulation_id
                == Simulation.simulation_id,
            )
            .where(
                SimulationOutput.variable_code == variable_code
            )
        )

        if simulation_filters:
            stmt = stmt.where(*simulation_filters)

        logger.info(
            "Calculating aggregation: variable=%s aggregation=%s "
            "crop=%s cultivar=%s irrigation=%s nitrogen_level=%s "
            "planting_stage=%s year=%s state=%s district=%s",
            variable_code,
            aggregation,
            crop,
            cultivar,
            irrigation,
            nitrogen_level,
            planting_stage,
            year,
            state,
            district,
        )


        result = await self.db.execute(stmt)
        row = result.fetchone()

        logger.info(
            "DATABASE RESULT: metric=%s aggregation=%s "
            "crop=%s cultivar=%s planting_stage=%s year=%s "
            "state=%s district=%s value=%s count=%s "
            "stddev=%s min=%s max=%s",
            variable_code,
            aggregation,
            crop,
            cultivar,
            planting_stage,
            year,
            state,
            district,
            row.value if row else None,
            row.count if row else 0,
            row.stddev if row else None,
            row.min if row else None,
            row.max if row else None,
        )


        return {
            "aggregation_type": aggregation,
            "metric": variable_code,
            "value": row.value if row else None,
            "count": row.count if row else 0,
            "stddev": row.stddev if row else None,
            "min": row.min if row else None,
            "max": row.max if row else None,
            "unit": None,
        }

    async def get_extremum_simulation(
        self,
        variable_code: str,
        aggregation: str,
        crop: Optional[str] = None,
        cultivar: Optional[str] = None,
        irrigation: Optional[str] = None,
        nitrogen_level: Optional[str] = None,
        year: Optional[object] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
        planting_stage: Optional[str] = None,
        country: Optional[str] = None,
        spatial: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the simulation corresponding to a MIN or MAX value."""

        simulation_filters = self._build_simulation_filters(
            crop=crop,
            cultivar=cultivar,
            irrigation=irrigation,
            nitrogen_level=nitrogen_level,
            year=year,
            state=state,
            district=district,
            planting_stage=planting_stage,
            country=country,
            spatial=spatial,
        )

        descending = aggregation.upper() == "MAX"

        stmt = (
            select(
                Simulation,
                SimulationOutput.value.label("value"),
            )
            .join(
                Simulation,
                SimulationOutput.simulation_id
                == Simulation.simulation_id,
            )
            .where(
                SimulationOutput.variable_code == variable_code
            )
        )

        if simulation_filters:
            stmt = stmt.where(*simulation_filters)

        stmt = (
            stmt.order_by(
                sql_func.desc(SimulationOutput.value)
                if descending
                else sql_func.asc(SimulationOutput.value)
            )
            .limit(1)
        )

        result = await self.db.execute(stmt)
        row = result.fetchone()

        if not row:
            return None

        simulation = (
            row.Simulation
            if hasattr(row, "Simulation")
            else row[0]
        )

        value = (
            row.value
            if hasattr(row, "value")
            else row[1]
        )

        return {
            "simulation_id": str(simulation.simulation_id),
            "value": value,
            "latitude": simulation.latitude,
            "longitude": simulation.longitude,
            "country": simulation.country,
            "state": simulation.state,
            "district": simulation.district,
            "year": simulation.simulation_year,
        }

    async def calculate_breakdown(
        self,
        variable_code: str,
        aggregation: str,
        group_by: str,
        crop: Optional[str] = None,
        cultivar: Optional[str] = None,
        irrigation: Optional[str] = None,
        nitrogen_level: Optional[str] = None,
        year: Optional[int] = None,
        planting_stage: Optional[str] = None,
        state: Optional[str] = None,
        district: Optional[str] = None,
        country: Optional[str] = None,
        spatial: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Calculate an aggregation grouped by a simulation field."""

        simulation_filters = self._build_simulation_filters(
            crop=crop,
            cultivar=cultivar,
            irrigation=irrigation,
            nitrogen_level=nitrogen_level,
            year=year,
            planting_stage=planting_stage,
            state=state,
            district=district,
            country=country,
            spatial=spatial,
        )

        group_field = getattr(Simulation, group_by, None)

        if group_field is None:
            raise ValueError(
                f"Invalid group field: {group_by}"
            )

        aggregation_functions = {
            "avg": sql_func.avg,
            "max": sql_func.max,
            "min": sql_func.min,
            "count": sql_func.count,
            "sum": sql_func.sum,
        }

        aggregate_function = aggregation_functions.get(
            aggregation.lower(),
            sql_func.avg,
        )

        stmt = (
            select(
                group_field.label("group_value"),
                aggregate_function(
                    SimulationOutput.value
                ).label("value"),
                sql_func.count().label("count"),
            )
            .join(
                Simulation,
                SimulationOutput.simulation_id
                == Simulation.simulation_id,
            )
            .where(
                SimulationOutput.variable_code == variable_code
            )
        )

        if simulation_filters:
            stmt = stmt.where(*simulation_filters)

        stmt = (
            stmt
            .group_by(group_field)
            .order_by(aggregate_function(SimulationOutput.value).desc())
        )

        result = await self.db.execute(stmt)

        return [
            {
                "group_value": row.group_value,
                "value": row.value,
                "count": row.count,
            }
            for row in result.all()
        ]

    async def get_variable_stats(
        self,
        variable_code: str,
        crop: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return comprehensive statistics for a variable."""

        simulation_filters = self._build_simulation_filters(
            crop=crop,
        )

        stmt = (
            select(
                sql_func.min(
                    SimulationOutput.value
                ).label("min"),
                sql_func.max(
                    SimulationOutput.value
                ).label("max"),
                sql_func.avg(
                    SimulationOutput.value
                ).label("avg"),
                sql_func.stddev(
                    SimulationOutput.value
                ).label("stddev"),
                sql_func.count().label("count"),
            )
            .join(
                Simulation,
                SimulationOutput.simulation_id
                == Simulation.simulation_id,
            )
            .where(
                SimulationOutput.variable_code == variable_code
            )
        )

        if simulation_filters:
            stmt = stmt.where(*simulation_filters)

        result = await self.db.execute(stmt)
        row = result.fetchone()

        return {
            "variable_code": variable_code,
            "min": row.min if row else None,
            "max": row.max if row else None,
            "avg": row.avg if row else None,
            "stddev": (
                row.stddev
                if row and row.stddev is not None
                else 0
            ),
            "count": row.count if row else 0,
        }

    async def get_yearly_trend(
        self,
        variable_code: str,
        crop: Optional[str] = None,
        cultivar: Optional[str] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        spatial: Optional[Any] = None,
        **domain_filters: Any,
    ) -> List[Dict[str, Any]]:
        """Return yearly averages with crop, cultivar, and year filters."""

        filters: List[Any] = [
            SimulationOutput.variable_code == variable_code
        ]

        filters.extend(self._build_simulation_filters(
            crop=crop, cultivar=cultivar, spatial=spatial, **domain_filters
        ))

        if start_year is not None:
            filters.append(
                Simulation.simulation_year >= start_year
            )

        if end_year is not None:
            filters.append(
                Simulation.simulation_year <= end_year
            )

        stmt = (
            select(
                Simulation.simulation_year.label("year"),
                sql_func.avg(
                    SimulationOutput.value
                ).label("avg_value"),
                sql_func.count().label("count"),
            )
            .join(
                Simulation,
                SimulationOutput.simulation_id
                == Simulation.simulation_id,
            )
            .where(*filters)
            .group_by(Simulation.simulation_year)
            .order_by(Simulation.simulation_year)
        )

        logger.info(
            "Calculating yearly trend: variable=%s crop=%s "
            "cultivar=%s start_year=%s end_year=%s",
            variable_code,
            crop,
            cultivar,
            start_year,
            end_year,
        )

        result = await self.db.execute(stmt)
        rows = result.all()

        logger.info(
            "DATABASE TREND RESULT: metric=%s crop=%s "
            "cultivar=%s start_year=%s end_year=%s rows=%s",
            variable_code,
            crop,
            cultivar,
            start_year,
            end_year,
            [
                {
                    "year": row.year,
                    "avg_value": row.avg_value,
                    "count": row.count,
                }
                for row in rows
            ],
        )

        return [
            {
                "year": row.year,
                "avg_value": row.avg_value,
                "count": row.count,
            }
            for row in rows
        ]

    async def get_analysis_records(
        self,
        variables: List[str],
        group_by: Optional[str] = None,
        limit: int = MAX_ANALYSIS_ROWS,
        **filters: Any,
    ) -> List[Dict[str, Any]]:
        """Return one row per simulation with the requested output values.

        PostgreSQL applies all simulation filters and pairs the variables by
        joining simulation_outputs once per variable, so only the requested
        columns of matching simulations are transferred. The Python analysis
        layer receives rows like {"PRCP": 512.3, "HWAM": 4873.0, "group": "BASE"}.

        Args:
            variables: Allowlisted DSSAT variable codes (validated upstream).
            group_by: Optional key of ANALYSIS_GROUP_COLUMNS.
            limit: Safety cap on the number of rows returned.
            **filters: Keyword filters accepted by _build_simulation_filters.
        """

        if not variables:
            raise ValueError("At least one variable is required")

        group_column = None
        if group_by is not None:
            group_column = ANALYSIS_GROUP_COLUMNS.get(group_by)
            if group_column is None:
                raise ValueError(f"Invalid group field: {group_by}")

        simulation_filters = self._build_simulation_filters(**filters)

        output_aliases = [
            aliased(SimulationOutput, name=f"output_{index}")
            for index in range(len(variables))
        ]

        columns = [
            alias.value.label(f"value_{index}")
            for index, alias in enumerate(output_aliases)
        ]
        if group_column is not None:
            columns.append(group_column.label("group_value"))

        stmt = select(*columns).select_from(Simulation)

        for alias, variable_code in zip(output_aliases, variables):
            stmt = stmt.join(
                alias,
                and_(
                    alias.simulation_id == Simulation.simulation_id,
                    alias.variable_code == variable_code,
                ),
            )

        if simulation_filters:
            stmt = stmt.where(*simulation_filters)

        stmt = stmt.limit(limit)

        logger.info(
            "Retrieving analysis records: variables=%s group_by=%s filters=%s",
            variables,
            group_by,
            filters,
        )

        result = await self.db.execute(stmt)
        rows = result.all()

        records: List[Dict[str, Any]] = []
        for row in rows:
            record = {
                variable_code: getattr(row, f"value_{index}")
                for index, variable_code in enumerate(variables)
            }
            if group_column is not None:
                record["group"] = row.group_value
            records.append(record)

        logger.info("DATABASE ANALYSIS RESULT: rows=%s", len(records))

        return records

    async def count_simulations(self, **filters: Any) -> int:
        """True number of simulations matching the filters (and radius).

        Accepts the same keyword filters as _build_simulation_filters,
        including spatial. Unsupported filters raise rather than widen scope.
        """
        conditions = self._build_simulation_filters(**filters)
        stmt = select(sql_func.count()).select_from(Simulation)
        if conditions:
            stmt = stmt.where(*conditions)
        return int((await self.db.execute(stmt)).scalar_one())

    async def get_distinct_field_values(
        self,
        field: str,
        **filters: Any,
    ) -> List[Any]:
        """Return distinct values of a simulation field (optionally filtered).

        Used to explain why an analysis is unavailable, e.g. when the dataset
        contains a single irrigation regime or nitrogen level.
        """

        column = ANALYSIS_GROUP_COLUMNS.get(field)
        if column is None:
            raise ValueError(f"Invalid field: {field}")

        stmt = select(column).distinct().order_by(column)

        simulation_filters = self._build_simulation_filters(**filters)
        if simulation_filters:
            stmt = stmt.where(*simulation_filters)

        result = await self.db.execute(stmt)
        return [row[0] for row in result.all() if row[0] is not None]

    async def get_top_simulations(
        self,
        variable_code: str,
        aggregation: str = "max",
        limit: int = 10,
        crop: Optional[str] = None,
        cultivar: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return simulations ordered by maximum or minimum value."""

        filters: List[Any] = [
            SimulationOutput.variable_code == variable_code
        ]

        if crop is not None:
            filters.append(Simulation.crop == crop)

        if cultivar is not None:
            filters.append(Simulation.cultivar == cultivar)

        aggregate_function = (
            sql_func.max
            if aggregation.lower() == "max"
            else sql_func.min
        )

        stmt = (
            select(
                Simulation,
                aggregate_function(
                    SimulationOutput.value
                ).label("value"),
            )
            .join(
                SimulationOutput,
                Simulation.simulation_id
                == SimulationOutput.simulation_id,
            )
            .where(*filters)
            .group_by(Simulation.simulation_id)
            .order_by(aggregate_function(SimulationOutput.value).desc())
            .limit(limit)
        )

        result = await self.db.execute(stmt)

        return [
            {
                "simulation_id": str(row.Simulation.simulation_id),
                "experiment_name": row.Simulation.experiment_name,
                "run_name": row.Simulation.run_name,
                "value": row.value,
                "crop": row.Simulation.crop,
                "cultivar": row.Simulation.cultivar,
            }
            for row in result.all()
        ]
