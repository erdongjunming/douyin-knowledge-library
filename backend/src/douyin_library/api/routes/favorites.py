from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.api.routes.creators import get_creator_gateway
from douyin_library.config import Settings
from douyin_library.sync.favorites_sync import FavoritesSyncService
from douyin_library.upstream.jiji_gateway import JijiGateway

router = APIRouter(prefix="/api/favorites", tags=["favorites"])


def get_favorites_sync_service(
    gateway: Annotated[JijiGateway, Depends(get_creator_gateway)],
) -> Iterator[FavoritesSyncService]:
    database_path = Settings().storage_root / "data" / "library.db"
    if not database_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="storage_not_initialized",
        )
    engine = create_engine(f"sqlite:///{database_path.resolve().as_posix()}")
    with Session(engine) as session:
        yield FavoritesSyncService(session, gateway)


@router.post("/sync")
async def sync_all_favorites(
    service: Annotated[FavoritesSyncService, Depends(get_favorites_sync_service)],
) -> dict[str, Any]:
    result = await service.sync_all()
    return {
        "status": result.status,
        "complete": result.complete,
        "default_count": result.default_count,
        "folder_count": result.folder_count,
        "unique_work_count": result.unique_work_count,
    }
