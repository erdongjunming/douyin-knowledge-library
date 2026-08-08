from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.security.cookie_store import CookieStore
from douyin_library.sync.creator_sync import CreatorSyncService
from douyin_library.upstream.jiji_gateway import JijiGateway

router = APIRouter(prefix="/api/creators", tags=["creators"])


class CreatorReferenceRequest(BaseModel):
    reference: str = Field(min_length=1)


class CreatorSyncRequest(CreatorReferenceRequest):
    confirmed: bool = False


async def get_creator_gateway() -> AsyncIterator[JijiGateway]:
    cookie = CookieStore().read()
    if not cookie:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="douyin_cookie_not_configured",
        )
    gateway = JijiGateway.from_cookie_header(cookie)
    try:
        yield gateway
    finally:
        await gateway.close()


def get_creator_sync_service(
    gateway: Annotated[JijiGateway, Depends(get_creator_gateway)],
) -> Iterator[CreatorSyncService]:
    database_path = Settings().storage_root / "data" / "library.db"
    if not database_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="storage_not_initialized",
        )
    engine = create_engine(f"sqlite:///{database_path.resolve().as_posix()}")
    with Session(engine) as session:
        yield CreatorSyncService(session, gateway)


@router.post("/preview")
async def preview_creator(
    payload: CreatorReferenceRequest,
    gateway: Annotated[JijiGateway, Depends(get_creator_gateway)],
) -> dict[str, Any]:
    return await gateway.resolve_creator(payload.reference)


@router.post("/sync")
async def sync_creator(
    payload: CreatorSyncRequest,
    service: Annotated[CreatorSyncService, Depends(get_creator_sync_service)],
) -> dict[str, Any]:
    if not payload.confirmed:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="creator_confirmation_required",
        )
    result = await service.sync(payload.reference)
    return {
        "creator_id": result.creator.id,
        "status": result.status,
        "complete": result.complete,
        "new_work_count": result.new_work_count,
        "discovered_count": result.discovered_count,
    }
