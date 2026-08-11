from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from douyin_library.api.dependencies import get_runtime_session
from douyin_library.config import Settings
from douyin_library.media.service import WorkMediaService
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Analysis, Asset, Creator, Work
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


def get_work_session(
    session: Annotated[Session, Depends(get_runtime_session)],
) -> Session:
    return session


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


@router.get("")
def list_works(
    session: Annotated[Session, Depends(get_work_session)],
    limit: int = Query(default=100, ge=1, le=500),
    query: str | None = Query(default=None),
) -> dict[str, Any]:
    statement = (
        select(Work, Creator.nickname)
        .outerjoin(Creator, Work.creator_id == Creator.id)
        .order_by(Work.published_at.desc().nullslast(), Work.id.desc())
        .limit(limit)
    )
    if query and query.strip():
        pattern = f"%{query.strip()}%"
        statement = statement.where(Work.description.ilike(pattern))

    rows = session.execute(statement).all()
    work_ids = [work.id for work, _ in rows]
    assets = (
        session.scalars(select(Asset).where(Asset.work_id.in_(work_ids))).all()
        if work_ids
        else []
    )
    assets_by_work: dict[int, list[str]] = {}
    for asset in assets:
        assets_by_work.setdefault(asset.work_id, []).append(asset.kind)

    analyses = session.scalars(
        select(Analysis).where(
            Analysis.work_id.in_(work_ids),
            Analysis.status == "succeeded",
        )
    ).all() if work_ids else []
    topics_by_work: dict[int, list[str]] = {}
    for analysis in analyses:
        try:
            topics = json.loads(analysis.topics_json or "[]")
        except json.JSONDecodeError:
            topics = []
        if isinstance(topics, list):
            topics_by_work.setdefault(analysis.work_id, []).extend(
                str(topic) for topic in topics if str(topic).strip()
            )

    return {
        "items": [
            {
                "id": work.id,
                "aweme_id": work.aweme_id,
                "description": work.description,
                "media_type": work.media_type,
                "availability": work.availability,
                "asset_kinds": sorted(set(assets_by_work.get(work.id, []))),
                "creator_nickname": nickname,
                "published_at": work.published_at.isoformat() if work.published_at else None,
                "topics": sorted(set(topics_by_work.get(work.id, []))),
            }
            for work, nickname in rows
        ]
    }
