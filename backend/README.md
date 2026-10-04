# DSSAT RAG Backend

Production-grade Python backend for DSSAT simulation chatbot using FastAPI, SQLAlchemy 2.0, Alembic, PostgreSQL + PostGIS, and Qdrant.

## Features

- **FastAPI** - Modern, fast web framework
- **SQLAlchemy 2.0** - ORM with async support
- **Alembic** - Database migrations
- **PostgreSQL + PostGIS** - Spatial database with geometry support
- **Qdrant** - Vector database for embeddings (ready for future implementation)
- **Repository Pattern** - Clean separation of concerns
- **Service Layer** - Business logic separation
- **Pydantic Models** - Type validation and serialization

## Project Structure

```
backend/
├── app/
│   ├── api/              # API endpoints
│   │   └── v1/          # Version 1 API routes
│   ├── core/            # Core configuration
│   ├── db/              # Database setup
│   ├── models/          # SQLAlchemy models
│   ├── schemas/         # Pydantic schemas
│   ├── repositories/    # Repository pattern implementation
│   ├── services/        # Service layer
│   ├── parsers/         # CSV and data parsers
│   ├── mappers/         # Data mapping utilities
│   └── utils/           # Utility functions
├── alembic/             # Database migrations
├── requirements.txt     # Python dependencies
└── main.py              # Application entry point
```

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ with PostGIS extension
- Qdrant (optional, for future embeddings)

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

Add at minimum for LLM access (used by planner and response generator):

```
OPENAI_API_KEY=YOUR_OPENAI_KEY
OPENAI_MODEL=gpt-4o-mini
# Optional: use a custom base URL (e.g., UF endpoint)
# OPENAI_BASE_URL=https://api.ai.it.ufl.edu/v1
```

Run the app from `backend/`: settings read `.env` relative to the working
directory, so from anywhere else no key is loaded and the planner silently
runs without the LLM.

#### How questions are planned

The LLM is the primary planner for every supported question (averages and
other aggregates, correlations, linear/quadratic regression, grouped
comparisons, trends, metadata/list and count questions, definitions):

1. Location words are resolved first (map point, coordinates, geocoded place).
   "here" with no selected map point is answered with a clarification and
   never reaches the planner, so it cannot become an all-data query.
2. The LLM returns a structured plan (Chat Completions; the installed OpenAI
   SDK has no Responses API).
3. The plan is validated and normalized deterministically: `yield` → `HWAM`,
   rainfall/precipitation → `PRCP`, explicit codes kept, filter fields and
   cultivar codes checked, and facts stated in the question (variables,
   cultivars, years, planting offsets, the named method) correct the plan.
   Every correction is listed in the chat response under `planner.corrections`.
4. Only if no LLM is configured, the LLM call fails, or its plan is unusable,
   the deterministic planners (analysis parser, statistics fallback) plan the
   question. The response then says `planner.planner = "fallback"` with the
   `fallback_reason`.

Each chat response includes `planner`: `{"planner": "llm" | "fallback" |
"not_run", "model", "corrections", "fallback_reason", "answer_by"}`.

#### UF gateway model status (checked 2026-09-30)

On `https://api.ai.it.ufl.edu/v1` the key is accepted and `GET /models` lists
`gpt-4o`, but every Chat Completions request to the OpenAI-routed models
(`gpt-4o`, `gpt-4.1`, `gpt-5`) has its connection reset by the gateway
(`APIConnectionError`, WinError 10054; `curl` exit 56). The Claude models on the
same gateway and key (`claude-4-sonnet`, `claude-4.6-sonnet`, `claude-4.7-opus`)
respond normally. With `OPENAI_MODEL=gpt-4o` every question therefore falls
back to deterministic planning. To develop and verify with a responding model
without editing `.env`:

```bash
python run_dev.py --port 8005 --llm-model claude-4.6-sonnet   # this process only
python scripts/verify_real_llm.py                              # real-LLM end-to-end check (read-only DB)
```

Tests are labeled with pytest markers: `mocked_llm` (scripted LLM stand-in;
proves wiring and validation, not model behavior) and `fallback` (no/failing
LLM). Real-LLM verification is only `scripts/verify_real_llm.py`.

3. Run database migrations:
```bash
alembic upgrade head
```

4. Start the application:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
- `GET /health/status` - Health check endpoint
- `GET /` - Root endpoint

### Ingestion
- `POST /api/v1/ingest/` - Ingest a DSSAT summary CSV file
- `POST /api/v1/ingest/batch` - Ingest multiple CSV files

Summary CSV ingestion is idempotent and atomic:

- **Identity:** a simulation is `(RUN_NAME, latitude, longitude, year)`. RUN_NAME
  encodes the treatment (crop, irrigation, nitrogen, cultivar, planting stage),
  and the four together are unique in the source CSVs. The file name is not part
  of the identity, so the same data uploaded under another name is recognized.
  Files without a RUN_NAME column fall back to TNAM/EXNAME; if their rows do not
  have unique identities, the whole file is rejected rather than merged.
- **Re-uploads:** rows already stored are skipped. A repeat of a complete file
  inserts nothing (`status: already_present`); an overlapping partial upload
  inserts only the missing rows (`status: partial_overlap`) with a warning.
- **Atomicity:** a file's simulations and outputs are committed in one
  transaction; any error rolls everything back. Uploads of the same run are
  serialized with PostgreSQL advisory locks.
- **Provenance:** the original file name is stored as `experiment_name`, and each
  upload is appended to `INGESTION_LOG_PATH` (default `logs/ingestion_log.jsonl`)
  with its SHA-256, row counts, outcome and warnings. Files where locations have
  uneven year coverage are flagged as possibly incomplete.
- **Database backstop:** migration `002_unique_simulation_identity` adds the
  unique constraint `uq_simulations_run_location_year` on
  `(run_name, latitude, longitude, simulation_year)`. If it rejects a row the
  application check missed (e.g. a concurrent upload), ingestion rolls back,
  re-reads existing keys and retries once; otherwise the upload fails cleanly
  with nothing stored. The migration refuses to run (changing nothing) while
  duplicates exist — check first with `scripts/verify_ingestion_integrity.py`.

Apply migrations from `backend/` with `alembic upgrade head` (the target comes
from the `POSTGRES_*` settings / `.env`). Database tests for the constraint are
opt-in: set `MIGRATION_TEST_DATABASE_URL` to a disposable PostGIS database whose
name contains `test` (see `tests/test_unique_identity_migration.py`).

**Python version:** the Docker image uses Python 3.11. pandas 2.2.2 (pinned in
`requirements.txt`) does not support Python 3.14: `pd.to_datetime` crashes the
interpreter there. The CSV parser no longer uses it (DSSAT `YYYYDDD` dates are
parsed with the standard library), but for local development prefer Python 3.11,
or upgrade pandas to a release that supports 3.14 before relying on other pandas
date functions.

## Database Schema

### Simulations Table
- `simulation_id` (UUID, PK)
- `experiment_name`, `run_name`
- `country`, `state`, `district`
- `ecological_zone`
- `latitude`, `longitude`
- `location` (PostGIS Geometry(Point,4326))
- `geohash`
- `crop`, `cultivar`
- `irrigation`, `nitrogen_level`
- `planting_stage`, `planting_date`, `harvest_date`
- `simulation_year`
- `harvest_area`

### Simulation Outputs Table
- `id` (PK)
- `simulation_id` (FK to simulations)
- `variable_code`
- `value`
- `unit`

## Development

### Running Migrations

```bash
# Create new migration
alembic revision -m "migration message"

# Apply migrations
alembic upgrade head

# Downgrade migrations
alembic downgrade -1
```

### Code Structure

The application follows a clean architecture pattern:

- **Models**: Database schema definitions
- **Schemas**: Pydantic models for API validation
- **Repositories**: Data access layer
- **Services**: Business logic layer
- **API**: HTTP endpoints

## Future Enhancements

- Qdrant integration for vector embeddings
- LLM integration for chatbot functionality
- CDE (Crop Data Exchange) support
- Advanced spatial queries with PostGIS
- Real-time data processing
