from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from douyin_library.api.dependencies import get_runtime_session
from douyin_library.jobs.queue import JobQueue, JobTransitionError
from douyin_library.models import Job

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobCreateRequest(BaseModel):
    kind: str = Field(min_length=1)
    work_id: int | None = None


def get_job_queue(
    session: Annotated[Session, Depends(get_runtime_session)],
) -> JobQueue:
    return JobQueue(session)


@router.get("")
def list_jobs(queue: Annotated[JobQueue, Depends(get_job_queue)]) -> dict[str, Any]:
    jobs = queue.session.scalars(select(Job).order_by(Job.created_at.desc()).limit(100)).all()
    return {"items": [serialize_job(job) for job in jobs]}


@router.post("")
def create_job(
    payload: JobCreateRequest,
    queue: Annotated[JobQueue, Depends(get_job_queue)],
) -> dict[str, Any]:
    try:
        job = queue.create(kind=payload.kind, work_id=payload.work_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return serialize_job(job)


@router.post("/{job_id}/cancel")
def cancel_job(job_id: int, queue: Annotated[JobQueue, Depends(get_job_queue)]) -> dict[str, Any]:
    try:
        job = queue.transition(job_id, "cancelled")
    except (KeyError, JobTransitionError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return serialize_job(job)


def serialize_job(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "kind": job.kind,
        "work_id": job.work_id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "attempts": job.attempts,
        "error_code": None if job.status == "succeeded" else job.error_code,
    }
