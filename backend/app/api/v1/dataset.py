"""Dataset endpoints used by the chat interface."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.services.management_service import ManagementService

router = APIRouter()


@router.get("/management-variables")
async def management_variables(db: AsyncSession = Depends(get_db)) -> dict:
    """Management fields and the values present in the ingested data.

    Only fields with at least one stored value are returned, each value with
    its plain-language meaning and the number of simulations that have it.
    """
    return await ManagementService(db).available()
