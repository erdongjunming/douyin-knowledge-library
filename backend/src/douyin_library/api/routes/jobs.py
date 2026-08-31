from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import exists, func, select
from sqlalchemy.orm import Session

from douyin_library.api.dependencies import get_job_control, get_runtime_session
from douyin_library.jobs.control import JobControl
from douyin_library.jobs.queue import JobQueue, JobTransitionError
from douyin_library.models import Asset, Job, Transcript, Work

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


@router.get("/progress")
def get_job_progress(queue: Annotated[JobQueue, Depends(get_job_queue)]) -> dict[str, Any]:
    session = queue.session
    has_audio = exists().where(Asset.work_id == Work.id, Asset.kind == "audio")
    has_transcript = exists().where(
        Transcript.work_id == Work.id,
        Transcript.status == "succeeded",
    )
    has_running = exists().where(
        Job.work_id == Work.id,
        Job.kind == "transcribe_work",
        Job.status == "running",
    )
    has_pending = exists().where(
        Job.work_id == Work.id,
        Job.kind == "transcribe_work",
        Job.status == "pending",
    )
    has_failed = exists().where(
        Job.work_id == Work.id,
        Job.kind == "transcribe_work",
        Job.status == "failed",
    )
    has_download_running = exists().where(
        Job.work_id == Work.id,
        Job.kind == "download_work",
        Job.status == "running",
    )
    has_download_pending = exists().where(
        Job.work_id == Work.id,
        Job.kind == "download_work",
        Job.status == "pending",
    )
    has_download_failed = exists().where(
        Job.work_id == Work.id,
        Job.kind == "download_work",
        Job.status == "failed",
    )

    def count_works(*conditions: Any) -> int:
        return int(session.scalar(select(func.count(Work.id)).where(*conditions)) or 0)

    collected = count_works()
    audio_ready = count_works(has_audio)
    audio_running = count_works(~has_audio, has_download_running)
    audio_pending = count_works(~has_audio, ~has_download_running, has_download_pending)
    audio_failed = count_works(
        ~has_audio,
        ~has_download_running,
        ~has_download_pending,
        has_download_failed,
    )
    audio_not_queued = max(
        collected - audio_ready - audio_running - audio_pending - audio_failed,
        0,
    )
    audio_percent = round((audio_ready / collected * 100) if collected else 0.0, 1)
    completed = count_works(has_audio, has_transcript)
    incomplete = (has_audio, ~has_transcript)
    running = count_works(*incomplete, has_running)
    pending = count_works(*incomplete, ~has_running, has_pending)
    failed = count_works(*incomplete, ~has_running, ~has_pending, has_failed)
    not_queued = max(audio_ready - completed - running - pending - failed, 0)
    percent = round((completed / audio_ready * 100) if audio_ready else 0.0, 1)

    return {
        "works": {"collected": collected, "audio_ready": audio_ready},
        "audio_extraction": {
            "total": collected,
            "completed": audio_ready,
            "running": audio_running,
            "pending": audio_pending,
            "failed": audio_failed,
            "not_queued": audio_not_queued,
            "percent": audio_percent,
        },
        "transcription": {
            "total": audio_ready,
            "completed": completed,
            "running": running,
            "pending": pending,
            "failed": failed,
            "not_queued": not_queued,
            "percent": percent,
        },
    }


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
def cancel_job(
    job_id: int,
    queue: Annotated[JobQueue, Depends(get_job_queue)],
    control: Annotated[JobControl | None, Depends(get_job_control)],
) -> dict[str, Any]:
    try:
        job = queue.cancel(job_id)
    except (KeyError, JobTransitionError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if control is not None:
        control.request_cancel(job_id)
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
        "heartbeat_at": job.heartbeat_at.isoformat() if job.heartbeat_at else None,
        "worker_id": job.worker_id,
        "error_code": None if job.status == "succeeded" else job.error_code,
    }
