"""API routes."""
from fastapi import APIRouter

from app.api.v1 import health, ingest, chat, spatial, dataset

api_router = APIRouter()

# Include versioned routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(ingest.router, prefix="/v1/ingest", tags=["ingestion"])
api_router.include_router(chat.router, prefix="/v1/chat", tags=["chat"])
api_router.include_router(spatial.router, prefix="/v1/spatial", tags=["spatial"])
api_router.include_router(dataset.router, prefix="/v1/dataset", tags=["dataset"])
