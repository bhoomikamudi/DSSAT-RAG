"""CSV parser for DSSAT summary files."""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import date

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

    @classmethod
    def parse_csv(cls, file_path: str) -> List[CanonicalSimulation]:
        """
        Parse a DSSAT summary CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            List of CanonicalSimulation instances
        """
        # Read CSV with pandas
        df = pd.read_csv(file_path, index_col=False)

        # Derive a sensible default experiment name from the file name
        default_experiment_name = Path(file_path).stem

        results: List[CanonicalSimulation] = []

        for _, row in df.iterrows():
            model = cls._parse_row(row, default_experiment_name)
            if model:
                results.append(model)

        return results

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

        # Skip rows without valid coordinates
        if latitude is None and longitude is None:
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

        # Parse DSSAT day-of-year date fields (PDAT, MDAT, HDAT) in format YYYYDDD -> date
        def parse_yyyydoy(val: Any) -> Optional[date]:
            v = cls.clean_value(val)
            if v is None:
                return None
            try:
                s = str(int(v))
                if len(s) == 7:
                    y = int(s[:4])
                    doy = int(s[4:])
                    return pd.to_datetime(f"{y}-{doy}", format="%Y-%j").date()
            except Exception:
                return None
            return None

        pdat = parse_yyyydoy(get_first(["PDAT"]))
        mdat = parse_yyyydoy(get_first(["MDAT"]))
        hdat = parse_yyyydoy(get_first(["HDAT"]))

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
