from __future__ import annotations

import json
from datetime import timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import case, exists, func, select
from sqlalchemy.orm import Session

from douyin_library.api.dependencies import get_job_control, get_runtime_session
from douyin_library.config import Settings
from douyin_library.jobs.control import JobControl
from douyin_library.jobs.queue import JobQueue, JobTransitionError
from douyin_library.models import Asset, Job, Transcript, Work

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobCreateRequest(BaseModel):
    kind: str = Field(min_length=1)
    work_id: int | None = None


class JobSelectionRequest(BaseModel):
    job_ids: list[int] = Field(default_factory=list, max_length=100)


JobStatusFilter = Literal[
    "active", "pending", "running", "paused", "failed", "succeeded", "cancelled",
]
JobLaneFilter = Literal["pipeline", "new_note", "legacy_upgrade"]


def get_job_queue(
    session: Annotated[Session, Depends(get_runtime_session)],
) -> JobQueue:
    return JobQueue(session)


@router.get("")
def list_jobs(
    queue: Annotated[JobQueue, Depends(get_job_queue)],
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    status: JobStatusFilter | None = None,
    queue_lane: JobLaneFilter | None = None,
) -> dict[str, Any]:
    return _job_snapshot(queue, limit=limit, offset=offset, status=status, queue_lane=queue_lane)


@router.post("/refresh")
def refresh_jobs(queue: Annotated[JobQueue, Depends(get_job_queue)]) -> dict[str, Any]:
    running_before = {
        job.id: _job_lane(job)
        for job in queue.session.scalars(select(Job).where(Job.status == "running")).all()
    }
    recovered = queue.recover_stale_jobs(
        stale_after=timedelta(seconds=Settings().job_stale_seconds),
    )
    recovered_by_lane = dict.fromkeys(_LANE_LABELS, 0)
    for job_id, lane in running_before.items():
        if queue.get(job_id).status != "running":
            recovered_by_lane[lane] += 1
    legacy_refill = _refresh_legacy_upgrade(queue)
    return {
        "recovered": recovered,
        "recovered_by_lane": recovered_by_lane,
        "legacy_refill": legacy_refill,
        **_job_snapshot(queue),
    }


@router.post("/bulk-pause")
def pause_selected_jobs(
    payload: JobSelectionRequest,
    queue: Annotated[JobQueue, Depends(get_job_queue)],
    control: Annotated[JobControl | None, Depends(get_job_control)],
) -> dict[str, Any]:
    paused: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for job_id in dict.fromkeys(payload.job_ids):
        try:
            job = queue.pause(job_id)
        except (KeyError, JobTransitionError) as exc:
            results.append({"id": job_id, "status": "skipped", "reason": str(exc)})
            continue
        if control is not None:
            control.request_cancel(job_id)
        paused.append(serialize_job(job))
        results.append({"id": job_id, "status": "paused", "reason": None})
    return {
        "paused": len(paused),
        "skipped": len(results) - len(paused),
        "items": paused,
        "results": results,
    }


@router.post("/bulk-resume")
def resume_selected_jobs(
    payload: JobSelectionRequest,
    queue: Annotated[JobQueue, Depends(get_job_queue)],
) -> dict[str, Any]:
    resumed: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for job_id in dict.fromkeys(payload.job_ids):
        try:
            resumed.append(serialize_job(queue.resume(job_id)))
        except (KeyError, JobTransitionError) as exc:
            results.append({"id": job_id, "status": "skipped", "reason": str(exc)})
            continue
        results.append({"id": job_id, "status": "pending", "reason": None})
    return {
        "resumed": len(resumed),
        "skipped": len(results) - len(resumed),
        "items": resumed,
        "results": results,
    }


@router.post("/{job_id}/retry")
def retry_job(
    job_id: int,
    queue: Annotated[JobQueue, Depends(get_job_queue)],
) -> dict[str, Any]:
    try:
        return serialize_job(queue.resume(job_id))
    except (KeyError, JobTransitionError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


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
    lane = _job_lane(job)
    return {
        "id": job.id,
        "kind": job.kind,
        "queue_lane": lane,
        "lane_label": _LANE_LABELS[lane],
        "work_id": job.work_id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "attempts": job.attempts,
        "heartbeat_at": job.heartbeat_at.isoformat() if job.heartbeat_at else None,
        "worker_id": job.worker_id,
        "error_code": None
        if job.status == "succeeded"
        else getattr(job, "error_code", None),
        "error_detail": None
        if job.status == "succeeded"
        else getattr(job, "error_detail", None),
        "next_retry_at": (
            getattr(job, "next_retry_at", None).isoformat()
            if getattr(job, "next_retry_at", None)
            else None
        ),
    }


_LANE_LABELS = {
    "pipeline": "视频录入",
    "new_note": "新笔记录入",
    "legacy_upgrade": "旧笔记升级",
}


def _job_lane(job: Job) -> str:
    lane = getattr(job, "queue_lane", None)
    if lane in _LANE_LABELS:
        return lane
    if job.kind != "generate_note":
        return "pipeline"
    try:
        payload = json.loads(job.payload_json or "{}")
    except (TypeError, json.JSONDecodeError):
        payload = {}
    return (
        "legacy_upgrade"
        if isinstance(payload, dict) and payload.get("legacy_only") is True
        else "new_note"
    )


def _job_snapshot(
    queue: JobQueue,
    *,
    limit: int = 100,
    offset: int = 0,
    status: JobStatusFilter | None = None,
    queue_lane: JobLaneFilter | None = None,
) -> dict[str, Any]:
    # Compute historical lanes in SQL as well, without materializing the full queue.
    valid_payload = case((func.json_valid(Job.payload_json) == 1, Job.payload_json), else_="{}")
    lane = case(
        (Job.queue_lane.in_(tuple(_LANE_LABELS)), Job.queue_lane),
        (Job.kind != "generate_note", "pipeline"),
        (func.json_extract(valid_payload, "$.legacy_only") == 1, "legacy_upgrade"),
        else_="new_note",
    )
    conditions = []
    if status == "active":
        conditions.append(Job.status.in_(("pending", "running", "paused")))
    elif status is not None:
        conditions.append(Job.status == status)
    if queue_lane is not None:
        conditions.append(lane == queue_lane)
    total = int(queue.session.scalar(select(func.count(Job.id)).where(*conditions)) or 0)
    jobs = queue.session.scalars(
        select(Job).where(*conditions).order_by(Job.created_at.desc(), Job.id.desc())
        .limit(limit).offset(offset)
    ).all()
    counts = {
        lane: {"pending": 0, "running": 0, "paused": 0, "failed": 0, "succeeded": 0, "cancelled": 0}
        for lane in _LANE_LABELS
    }
    for lane_name, job_status, count in queue.session.execute(
        select(lane, Job.status, func.count(Job.id)).group_by(lane, Job.status)
    ):
        if job_status in counts[lane_name]:
            counts[lane_name][job_status] = count
    return {
        "items": [serialize_job(job) for job in jobs], "lane_counts": counts,
        "total": total, "limit": limit, "offset": offset, "has_more": offset + len(jobs) < total,
    }


def _refresh_legacy_upgrade(queue: JobQueue) -> dict[str, Any]:
    from douyin_library.jobs.legacy_upgrade import refresh_legacy_upgrade

    return refresh_legacy_upgrade(queue, Settings(), enable=None, limit=3)
