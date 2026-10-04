"""Unique simulation identity: (run_name, latitude, longitude, simulation_year).

A DSSAT summary CSV row is identified by RUN_NAME (treatment), location and
weather year; the application already skips existing keys during ingestion.
This constraint makes the database reject duplicates even if that check is
bypassed (e.g. two uploads racing, or a future code path without the check).

The upgrade first verifies that no duplicate keys exist and stops with a clear
message otherwise, so it never fails half-way on existing data and never
modifies rows. Run scripts/verify_ingestion_integrity.py (and the repair
script if needed) before upgrading.
"""
from alembic import context, op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "unique_simulation_identity"
down_revision: str | None = "initial_migration"
branch_labels: str | None = None
depends_on: str | None = None

CONSTRAINT_NAME = "uq_simulations_run_location_year"
COLUMNS = ["run_name", "latitude", "longitude", "simulation_year"]


def upgrade() -> None:
    if not context.is_offline_mode():
        duplicate_groups = op.get_bind().execute(sa.text(
            """
            SELECT count(*) FROM (
                SELECT 1 FROM simulations
                GROUP BY run_name, latitude, longitude, simulation_year
                HAVING count(*) > 1
            ) duplicates
            """
        )).scalar_one()
        if duplicate_groups:
            raise RuntimeError(
                f"Cannot add {CONSTRAINT_NAME}: {duplicate_groups} duplicate "
                "(run_name, latitude, longitude, simulation_year) groups exist. "
                "No changes were made. Run scripts/verify_ingestion_integrity.py "
                "and repair the duplicates first."
            )

    op.create_unique_constraint(CONSTRAINT_NAME, "simulations", COLUMNS)


def downgrade() -> None:
    op.drop_constraint(CONSTRAINT_NAME, "simulations", type_="unique")
