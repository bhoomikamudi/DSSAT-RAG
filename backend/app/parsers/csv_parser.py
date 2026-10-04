"""CSV parser for DSSAT summary files."""
import calendar
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import date, timedelta

from app.models.canonical import CanonicalSimulation, SimulationModel, LocationModel


class DSSATParser:
    """Parser for DSSAT summary CSV files."""

    # Expected columns in the CSV
    EXPECTED_COLUMNS = [
        "LATITUDE",
        "LONGITUDE",
        "RUN_NAME",
        "HARVEST_AREA",
        "CR",
        "WYEAR",
        "PDAT",
        "MDAT",
        "HDAT",
        "CWAM",
        "HWAM",
        "HWAH",
        "GNAM",
        "TMAXA",
        "TMINA",
        "PRCP",
    ]

    # Output variables to extract
    OUTPUT_VARIABLES = [
        "CWAM",
        "HWAM",
        "HWAH",
        "GNAM",
        "TMAXA",
        "TMINA",
        "PRCP",
    ]

    @staticmethod
    def parse_run_name(run_name: str) -> Dict[str, str]:
        """
        Parse RUN_NAME field.

        Format: Crop_Irrigation_Nitrogen_Crop_Cultivar_PlantingStage

        Example: MZ_RF_HighN_MZ_BASE__pfrst30

        Args:
            run_name: The RUN_NAME string to parse

        Returns:
            Dictionary with extracted fields
        """
        if not run_name or not isinstance(run_name, str):
            return {
                "crop": "",
                "irrigation": "",
                "nitrogen": "",
                "cultivar": "",
                "planting_stage": "",
            }

        # Split by underscore
        parts = run_name.split("_")

        result: Dict[str, str] = {}

        if len(parts) >= 1:
            result["crop"] = parts[0].strip() or ""
        if len(parts) >= 2:
            result["irrigation"] = parts[1].strip() or ""
        if len(parts) >= 3:
            result["nitrogen"] = parts[2].strip() or ""

        # The cultivar can be user-defined and may contain underscores
        # Look for the last part that looks like a planting stage
        # Common patterns: pfrst30, vfrst30, etc.
        if len(parts) >= 4:
            # Try to identify planting stage (ends with numbers or common patterns)
            #cultivar_parts = parts[3:-1] if len(parts) > 4 else [parts[-1]]
            #result["cultivar"] = "_".join(cultivar_parts).strip() or ""
            result["cultivar"] = parts[4].strip() or ""

            if len(parts) >= 5:
                # Last part is usually planting stage
                last_part = parts[-1].strip()
                # Check if it looks like a planting stage (alphanumeric, often starts with letter)
                if last_part and (last_part[0].isalpha() or any(c.isdigit() for c in last_part)):
                    result["planting_stage"] = last_part
                else:
                    #result["cultivar"] = "_".join(parts[3:]).strip()
                    result["cultivar"] = parts[4].strip() or ""
                    result["planting_stage"] = ""
            elif len(parts) == 4:
                # Only one part left, assume it's cultivar
                result["cultivar"] = parts[3].strip() or ""

        return result

    @staticmethod
    def clean_value(value: Any) -> Optional[Any]:
        """
        Clean a CSV value.

        Args:
            value: Raw value from CSV

        Returns:
            Cleaned value (None for missing/empty values)
        """
        if pd.isna(value):
            return None

        if isinstance(value, str):
            value = value.strip()
            if not value or value == ".":
                return None
            # Try to convert to numeric
            try:
                if "." in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                pass
            return value

        return value

    @staticmethod
    def valid_coordinates(latitude: Any, longitude: Any) -> bool:
        """True for numeric WGS84 coordinates within valid ranges."""
        try:
            lat, lon = float(latitude), float(longitude)
        except (TypeError, ValueError):
            return False
        if lat != lat or lon != lon:  # NaN
            return False
        if lat == -99 or lon == -99:  # DSSAT missing-value code
            return False
        return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0

    @classmethod
    def parse_dssat_date(cls, value: Any) -> Optional[date]:
        """Parse a DSSAT YYYYDDD date (e.g. 1984062 -> 1984-03-02).

        Uses the standard library rather than pd.to_datetime, which crashes
        the interpreter with pandas 2.2.2 on Python 3.14.
        """
        cleaned = cls.clean_value(value)
        if cleaned is None:
            return None
        try:
            text = str(int(cleaned))
        except (TypeError, ValueError):
            return None
        if len(text) != 7:
            return None
        year, day_of_year = int(text[:4]), int(text[4:])
        days_in_year = 366 if calendar.isleap(year) else 365
        if not 1 <= day_of_year <= days_in_year:
            return None
        return date(year, 1, 1) + timedelta(days=day_of_year - 1)

    @classmethod
    def parse_csv(
        cls,
        file_path: str,
        source_name: Optional[str] = None,
    ) -> List[CanonicalSimulation]:
        """
        Parse a DSSAT summary CSV file.

        Args:
            file_path: Path to the CSV file
            source_name: Original file name (uploads are stored under temporary
                names); used as the experiment name when given

        Returns:
            List of CanonicalSimulation instances
        """
        simulations, _ = cls.parse_csv_detailed(file_path, source_name)
        return simulations

    @classmethod
    def parse_csv_detailed(
        cls,
        file_path: str,
        source_name: Optional[str] = None,
    ) -> Tuple[List[CanonicalSimulation], int]:
        """Parse a summary CSV and also return the number of data rows read."""
        df = pd.read_csv(file_path, index_col=False)

        # The experiment is named after the original file, not a temp path.
        default_experiment_name = Path(source_name or file_path).stem

        results: List[CanonicalSimulation] = []

        for _, row in df.iterrows():
            model = cls._parse_row(row, default_experiment_name)
            if model:
                results.append(model)

        return results, len(df)

    @classmethod
    def _parse_row(cls, row: pd.Series, default_experiment_name: str) -> Optional[CanonicalSimulation]:
        """
        Parse a single CSV row.

        Args:
            row: Pandas Series representing a row
            default_experiment_name: Fallback experiment name derived from file name

        Returns:
            CanonicalSimulation or None if invalid
        """
        # Helper to read first available column from alternatives
        def get_first(keys: List[str]) -> Any:
            for k in keys:
                if k in row:
                    return row.get(k)
            return None

        # Extract location (support DSSAT variants)
        latitude = cls.clean_value(get_first(["LATITUDE", "LAT", "Latitude", "lat"]))
        longitude = cls.clean_value(get_first(["LONGITUDE", "LONG", "Longitude", "lon", "LON"]))

        # Skip rows without a valid coordinate pair. A missing value used to be
        # stored as 0.0, which silently placed the row on the equator or the
        # prime meridian and would distort spatial filtering.
        if not cls.valid_coordinates(latitude, longitude):
            return None

        # Parse RUN_NAME
        run_name = (
            cls.clean_value(get_first(["RUN_NAME", "RUNNAME"]))
            or cls.clean_value(get_first(["TNAM", "EXNAME"]))
            or default_experiment_name
        ) or ""
        name_parts = cls.parse_run_name(run_name)

        # Extract simulation metadata
        harvest_area = cls.clean_value(get_first(["HARVEST_AREA", "HAREA", "HARVESTAREA"]))
        year = cls.clean_value(get_first(["WYEAR", "YEAR"]))

        pdat = cls.parse_dssat_date(get_first(["PDAT"]))
        mdat = cls.parse_dssat_date(get_first(["MDAT"]))
        hdat = cls.parse_dssat_date(get_first(["HDAT"]))

        try:
            year = int(year) if year is not None else 2024
        except (TypeError, ValueError):
            year = 2024

        # Extract outputs
        outputs: Dict[str, Any] = {}
        for var in cls.OUTPUT_VARIABLES:
            value = cls.clean_value(row.get(var))
            if value is not None:
                outputs[var] = value

        # Build models with sensible fallbacks
        simulation = SimulationModel(
            run_name=run_name,
            experiment_name=default_experiment_name,
            crop=str(cls.clean_value(get_first(["CR", "CROP"])) or name_parts.get("crop", "")),
            cultivar=name_parts.get("cultivar", ""),
            irrigation=name_parts.get("irrigation", ""),
            nitrogen_level=name_parts.get("nitrogen", ""),
            planting_stage=name_parts.get("planting_stage", ""),
            harvest_area=harvest_area,  # may be None
            year=year,
            planting_date=pdat,
            maturity_date=mdat,
            harvest_date=hdat,
        )

        location = LocationModel(
            latitude=latitude,
            longitude=longitude,
            country=None,
            state=None,
            district=None,
            ecological_zone=None,
        )

        return CanonicalSimulation(
            simulation=simulation,
            location=location,
            outputs=outputs,
        )


def parse_summary_csv(file_path: str) -> List[CanonicalSimulation]:
    """
    Parse a DSSAT summary CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        List of CanonicalSimulation instances
    """
    return DSSATParser.parse_csv(file_path)


def parse_run_name(run_name: str) -> Dict[str, str]:
    """
    Parse RUN_NAME field.

    Args:
        run_name: The RUN_NAME string to parse

    Returns:
        Dictionary with extracted fields
    """
    return DSSATParser.parse_run_name(run_name)
