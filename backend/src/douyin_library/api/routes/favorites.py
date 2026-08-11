from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status

from douyin_library.api.routes.jobs import get_job_queue, serialize_job
from douyin_library.jobs.queue import DuplicateActiveJobError, JobQueue

router = APIRouter(prefix="/api/favorites", tags=["favorites"])


@router.post("/sync", status_code=status.HTTP_202_ACCEPTED)
def sync_all_favorites(
    queue: Annotated[JobQueue, Depends(get_job_queue)],
) -> dict[str, Any]:
    try:
        job = queue.create(kind="sync_favorites", payload={})
    except DuplicateActiveJobError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return serialize_job(job)
