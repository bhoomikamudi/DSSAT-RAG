# DSSAT RAG - Crop Simulation Chatbot

Production-grade crop simulation chatbot with DSSAT (Decision Support System for Agrotechnology Transfer) integration.

## Overview

This project provides a complete solution for interacting with DSSAT simulation data through a modern chatbot interface. The system ingests DSSAT summary CSV files, stores metadata in PostgreSQL with PostGIS spatial support, and prepares embeddings for future LLM integration.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                        │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │   Chat   │  │  Input   │  │  Output  │  │  UI      │    │
│  │  Interface│  │  Form    │  │  Display │  │  Components│  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                    ↓              ↑
┌─────────────────────────────────────────────────────────────┐
│                   Backend (FastAPI)                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │   API    │  │ Services │  │ Repos    │  │  Models  │    │
│  │  Routes  │→ │  Layer   │→ │  Layer   │→ │  (ORM)   │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                    ↓              ↓
            ┌──────────┐  ┌──────────┐
            │PostgreSQL│  │  Qdrant  │
            │ + PostGIS│  │          │
            └──────────┘  └──────────┘
```

## Project Structure

```
DSSAT-RAG/
├── backend/              # FastAPI Python backend
│   ├── app/
│   │   ├── api/         # API endpoints
│   │   ├── core/        # Core configuration
│   │   ├── db/          # Database setup
│   │   ├── models/      # SQLAlchemy models
│   │   ├── schemas/     # Pydantic schemas
│   │   ├── repositories/# Repository pattern
│   │   ├── services/    # Service layer
│   │   ├── parsers/     # Data parsers
│   │   ├── mappers/     # Data mappers
│   │   └── utils/       # Utilities
│   ├── alembic/         # Database migrations
│   ├── requirements.txt
│   ├── Dockerfile
│   └── docker-compose.yml
├── frontend/            # Next.js frontend
│   ├── pages/
│   ├── components/
│   └── services/
├── n8n/                 # n8n workflow files
├── qdrant_storage/      # Qdrant data storage
└── sample_files/        # Sample DSSAT files
```

## Features

### Backend (FastAPI)

- **Async API**: FastAPI with async support for high performance
- **Spatial Database**: PostgreSQL + PostGIS for location-based queries
- **Repository Pattern**: Clean separation of concerns
- **Service Layer**: Business logic separation
- **Pydantic Models**: Type validation and serialization
- **Database Migrations**: Alembic for schema management
- **RAG Agent Orchestrator** (`app/agent/`): natural-language chat endpoint (`POST /api/v1/chat/`) that plans a query with an OpenAI LLM, fans it out across metadata, spatial, statistics, CDE, and embedding tools, and composes a grounded natural-language answer with cited sources. See [`backend/app/agent/README.md`](backend/app/agent/README.md) for the architecture.
- **Vector Search**: Qdrant-backed embedding service for semantic retrieval over simulation data

### Data Ingestion Pipeline

```
DSSAT → summary.csv → Google Drive → n8n webhook → POST /api/v1/ingest → FastAPI
```

### Database Schema

#### Simulations Table
- UUID primary key
- Spatial data with PostGIS geometry
- Agricultural metadata (crop, cultivar, irrigation)
- Temporal data (planting/harvest dates)

#### Simulation Outputs Table
- Foreign key to simulations
- Variable code, value, unit

## Setup Instructions

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- PostgreSQL 14+ with PostGIS extension
- Docker and Docker Compose (optional)

### Backend Setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your database credentials

# Run migrations
alembic upgrade head

# Start the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Docker Deployment

```bash
docker-compose up -d
```

## API Documentation

Once the backend is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Available Endpoints

#### Health Check
- `GET /health/status` - Health check endpoint
- `GET /` - Root endpoint

#### Ingestion
- `POST /api/v1/ingest/` - Ingest a DSSAT summary CSV file
  - Request: `multipart/form-data` with `file` field
  - Response: Simulation ID and status

- `POST /api/v1/ingest/batch` - Ingest multiple CSV files
  - Request: `multipart/form-data` with `files` field
  - Response: List of results with simulation IDs

#### Chat
- `POST /api/v1/chat/` - Ask a natural-language question about ingested simulation data
  - Request: JSON body with a `message` field
  - Response: LLM-generated answer, cited sources, and any matching simulations/statistics pulled by the agent

## Database Schema

### Simulations Table

| Column | Type | Description |
|--------|------|-------------|
| simulation_id | UUID | Primary key |
| experiment_name | VARCHAR(255) | Experiment name |
| run_name | VARCHAR(255) | Run name |
| country | VARCHAR(100) | Country code |
| state | VARCHAR(100) | State/region |
| district | VARCHAR(100) | District |
| ecological_zone | VARCHAR(255) | Ecological zone |
| latitude | FLOAT | Latitude (WGS84) |
| longitude | FLOAT | Longitude (WGS84) |
| location | Geometry(Point,4326) | PostGIS geometry |
| geohash | VARCHAR(50) | Geohash representation |
| crop | VARCHAR(100) | Crop type |
| cultivar | VARCHAR(255) | Cultivar name |
| irrigation | VARCHAR(100) | Irrigation method |
| nitrogen_level | VARCHAR(100) | Nitrogen level |
| planting_stage | VARCHAR(100) | Planting stage |
| planting_date | DATE | Planting date |
| harvest_date | DATE | Harvest date |
| simulation_year | INT | Simulation year |
| harvest_area | FLOAT | Harvested area (ha) |

### Simulation Outputs Table

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| simulation_id | UUID | Foreign key to simulations |
| variable_code | VARCHAR(100) | Variable code |
| value | FLOAT | Variable value |
| unit | VARCHAR(50) | Unit of measurement |

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

### Code Style

- Python: PEP 8 with type hints
- TypeScript/React: ESLint and Prettier

## Future Enhancements

1. **Frontend chat wiring**: the Next.js chat UI is scaffolded but not yet fully wired to `/api/v1/chat/`
2. **Real-time Processing**: WebSocket support for streaming responses
3. **n8n integration**: promote the workflow definitions in `n8n/` from prototype to a scheduled ingestion pipeline

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- DSSAT team for the crop simulation models
- FastAPI community for the excellent web framework
- SQLAlchemy and Alembic communities for database tools
