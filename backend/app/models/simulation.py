"""SQLAlchemy models for DSSAT simulation data."""
import uuid

from geoalchemy2 import Geometry
from sqlalchemy import (
    Column,
    String,
    UUID,
    Float,
    Date,
    Integer,
    Text,
    Index,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship, Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class Simulation(Base, TimestampMixin):
    """Simulation model with PostGIS geometry support."""

    __tablename__ = "simulations"

    simulation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique identifier for the simulation",
    )

    experiment_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Name of the experiment",
    )

    run_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Name of the simulation run",
    )

    country: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Country where simulation is located",
    )

    state: Mapped[str] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        comment="State or region",
    )

    district: Mapped[str] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        comment="District or county",
    )

    ecological_zone: Mapped[str] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="Ecological zone classification",
    )

    latitude: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Latitude coordinate (WGS84)",
    )

    longitude: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Longitude coordinate (WGS84)",
    )

    location: Mapped[str] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326),
        nullable=False,
        comment="PostGIS geometry point (SRID 4326)",
    )

    geohash: Mapped[str] = mapped_column(
        String(50),
        nullable=True,
        index=True,
        comment="Geohash representation of location",
    )

    crop: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Crop type",
    )

    cultivar: Mapped[str] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="Cultivar name",
    )

    irrigation: Mapped[str] = mapped_column(
        String(100),
        nullable=True,
        comment="Irrigation method",
    )

    nitrogen_level: Mapped[str] = mapped_column(
        String(100),
        nullable=True,
        comment="Nitrogen application level",
    )

    planting_stage: Mapped[str] = mapped_column(
        String(100),
        nullable=True,
        comment="Planting stage description",
    )

    planting_date: Mapped[str] = mapped_column(
        Date,
        nullable=True,
        index=True,
        comment="Date of planting",
    )

    maturity_date: Mapped[str] = mapped_column(
        Date,
        nullable=True,
        index=True,
        comment="Date of maturity",
    )

    harvest_date: Mapped[str] = mapped_column(
        Date,
        nullable=True,
        index=True,
        comment="Date of harvest",
    )

    simulation_year: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
        comment="Year of simulation",
    )

    harvest_area: Mapped[float] = mapped_column(
        Float,
        nullable=True,
        comment="Harvested area in hectares",
    )

    # Relationships
    outputs: Mapped[list["SimulationOutput"]] = relationship(
        back_populates="simulation",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index("idx_simulation_country", "country"),
        Index("idx_simulation_state", "state"),
        Index("idx_simulation_district", "district"),
        Index("idx_simulation_crop", "crop"),
        Index("idx_simulation_cultivar", "cultivar"),
        Index("idx_simulation_year", "simulation_year"),
        Index("idx_simulation_ecological_zone", "ecological_zone"),
        Index(
            "idx_simulation_location",
            "location",
            postgresql_using="gist",
        ),
        # Natural identity of a simulation; created by migration
        # 002_unique_simulation_identity.
        UniqueConstraint(
            "run_name",
            "latitude",
            "longitude",
            "simulation_year",
            name="uq_simulations_run_location_year",
        ),
    )

    def __repr__(self) -> str:
        return f"<Simulation {self.experiment_name} - {self.run_name}>"


class SimulationOutput(Base, TimestampMixin):
    """Simulation output variables model."""

    __tablename__ = "simulation_outputs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Unique identifier for the output record",
    )

    simulation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "simulations.simulation_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
        comment="Foreign key to simulations table",
    )

    variable_code: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Variable code (e.g., YIELD, LAI, ET)",
    )

    value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Value of the variable",
    )

    unit: Mapped[str] = mapped_column(
        String(50),
        nullable=True,
        comment="Unit of measurement",
    )

    # Relationships
    simulation: Mapped[Simulation] = relationship(
        back_populates="outputs",
        lazy="joined",
    )

    __table_args__ = (
        Index("idx_output_simulation", "simulation_id"),
        Index("idx_output_variable", "variable_code"),
        Index(
            "idx_output_simulation_variable",
            "simulation_id",
            "variable_code",
        ),
    )

    def __repr__(self) -> str:
        return f"<SimulationOutput {self.variable_code}: {self.value} {self.unit}>"
