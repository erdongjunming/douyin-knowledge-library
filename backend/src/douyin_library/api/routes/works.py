from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.media.service import WorkMediaService
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Work
from douyin_library.security.cookie_store import CookieStore
from douyin_library.upstream.jiji_gateway import JijiGateway

router = APIRouter(prefix="/api/works", tags=["works"])


async def get_work_gateway() -> AsyncIterator[JijiGateway]:
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


def get_work_session() -> Iterator[Session]:
    database_path = Settings().storage_root / "data" / "library.db"
    if not database_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="storage_not_initialized",
        )
    engine = create_engine(f"sqlite:///{database_path.resolve().as_posix()}")
    with Session(engine) as session:
        yield session


@router.post("/{work_id}/process")
async def process_work(
    work_id: int,
    gateway: Annotated[JijiGateway, Depends(get_work_gateway)],
    session: Annotated[Session, Depends(get_work_session)],
) -> dict[str, Any]:
    work = session.scalar(select(Work).where(Work.id == work_id))
    if work is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="work_not_found")
    service = WorkMediaService(
        session,
        gateway,
        StorageLayout(Settings().storage_root),
    )
    try:
        result = await service.process(work)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return {
        "work_id": work.id,
        "status": "succeeded",
        "audio": str(result.audio.path),
        "cover": str(result.cover),
        "metadata": str(result.metadata),
    }
