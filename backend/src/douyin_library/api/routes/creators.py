from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from douyin_library.api.routes.jobs import get_job_queue, serialize_job
from douyin_library.jobs.queue import DuplicateActiveJobError, JobQueue
from douyin_library.security.cookie_store import CookieStore
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


@router.post("/preview")
async def preview_creator(
    payload: CreatorReferenceRequest,
    gateway: Annotated[JijiGateway, Depends(get_creator_gateway)],
) -> dict[str, Any]:
    return await gateway.resolve_creator(payload.reference)


@router.post("/sync", status_code=status.HTTP_202_ACCEPTED)
def sync_creator(
    payload: CreatorSyncRequest,
    queue: Annotated[JobQueue, Depends(get_job_queue)],
) -> dict[str, Any]:
    if not payload.confirmed:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="creator_confirmation_required",
        )
    try:
        job = queue.create(
            kind="sync_creator",
            payload={"reference": payload.reference.strip()},
        )
    except DuplicateActiveJobError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return serialize_job(job)
