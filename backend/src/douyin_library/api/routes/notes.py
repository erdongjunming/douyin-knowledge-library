from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from douyin_library.api.routes.jobs import get_job_queue
from douyin_library.config import Settings
from douyin_library.jobs.queue import DuplicateActiveJobError, JobQueue
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Job, Work
from douyin_library.notes.ollama_runtime import probe_ollama
from douyin_library.notes.providers.selector import external_is_configured
from douyin_library.notes.service import NoteGenerationService

router = APIRouter(prefix="/api/notes", tags=["notes"])

# Provider readiness is an operator-facing probe.  Keep its last explicit result
# available to the cheap polling endpoint without probing a local service every
# few seconds.
_provider_status_cache: dict[str, Any] | None = None
LEGACY_UPGRADE_LIMIT = 3


class GenerateNotesRequest(BaseModel):
    force: bool = False
    limit: int | None = Field(default=None, ge=1, le=5000)
    dry_run: bool = False
    legacy_only: bool = False
    provider_policy: str = Field(default="auto", pattern="^(auto|local|external|rule)$")
    aweme_ids: list[str] | None = None


@router.post("/generate")
def enqueue_missing_notes(
    payload: GenerateNotesRequest,
    queue: Annotated[JobQueue, Depends(get_job_queue)],
    response: Response,
) -> dict[str, Any]:
    if payload.force and payload.legacy_only:
        raise HTTPException(
            status_code=422,
            detail="force_and_legacy_only_are_mutually_exclusive",
        )
    if payload.legacy_only and payload.limit is not None and payload.limit > LEGACY_UPGRADE_LIMIT:
        raise HTTPException(
            status_code=422,
            detail="legacy_upgrade_limit_exceeded",
        )
    effective_limit = (
        LEGACY_UPGRADE_LIMIT
        if payload.legacy_only and payload.limit is None
        else payload.limit
    )
    effective_provider_policy = "local" if payload.legacy_only else payload.provider_policy
    service = _service(queue)
    aweme_ids = _normalized_aweme_ids(payload.aweme_ids)
    total, completed = service.inventory(aweme_ids=aweme_ids)
    candidates = service.discover(
        force=payload.force,
        aweme_ids=aweme_ids,
        legacy_only=payload.legacy_only,
    )
    selected = candidates[:effective_limit] if effective_limit is not None else candidates
    if payload.dry_run:
        response.status_code = status.HTTP_200_OK
        return {
            "total": total,
            "completed": completed,
            "eligible": len(candidates),
            "selected": len(selected),
            "limit": effective_limit,
            "queued": 0,
            "already_queued": 0,
            "skipped": 0 if payload.force or payload.legacy_only else completed,
            "force": payload.force,
            "dry_run": True,
            "legacy_only": payload.legacy_only,
            "provider_policy": effective_provider_policy,
            "items": [
                {
                    "aweme_id": item.aweme_id,
                    "status": "would_generate",
                    "quality_status": item.quality_status,
                }
                for item in selected[:50]
            ],
            "items_truncated": max(0, len(selected) - 50),
        }
    response.status_code = status.HTTP_202_ACCEPTED
    queued = 0
    already_queued = 0
    for candidate in selected:
        try:
            queue.create(
                kind="generate_note",
                work_id=candidate.work_id,
                payload={
                    "aweme_id": candidate.aweme_id,
                    "force": payload.force,
                    "legacy_only": payload.legacy_only,
                    "provider_policy": effective_provider_policy,
                },
            )
            queued += 1
        except DuplicateActiveJobError:
            already_queued += 1
    return {
        "total": total,
        "completed": completed,
        "eligible": len(candidates),
        "selected": len(selected),
        "limit": effective_limit,
        "queued": queued,
        "already_queued": already_queued,
        "skipped": 0 if payload.force or payload.legacy_only else completed,
        "force": payload.force,
        "dry_run": False,
        "legacy_only": payload.legacy_only,
        "provider_policy": effective_provider_policy,
    }


@router.get("/provider-status")
async def note_provider_status() -> dict[str, Any]:
    global _provider_status_cache
    settings = Settings()
    ollama = await probe_ollama(settings)
    external_configured = external_is_configured(settings)
    _provider_status_cache = {
        "ollama": ollama.as_dict(),
        "external": {
            "configured": external_configured,
            "ready": external_configured,
            "model": settings.note_external_model if external_configured else None,
        },
        "policy": settings.note_provider_policy,
        "provider": "ollama" if ollama.ready else "external" if external_configured else None,
        "model": (
            ollama.model
            if ollama.ready
            else settings.note_external_model
            if external_configured
            else None
        ),
        "ready": bool(ollama.ready or external_configured),
        "external_configured": external_configured,
    }
    return dict(_provider_status_cache)


@router.get("/progress")
def note_progress(
    queue: Annotated[JobQueue, Depends(get_job_queue)],
) -> dict[str, Any]:
    service = _service(queue)
    total, completed = service.inventory()
    jobs = queue.session.scalars(
        select(Job)
        .where(Job.kind == "generate_note")
        .order_by(Job.created_at.desc(), Job.id.desc())
    ).all()
    work_ids = {job.work_id for job in jobs if job.work_id is not None}
    work_map = {
        work.id: work.aweme_id
        for work in queue.session.scalars(select(Work).where(Work.id.in_(work_ids))).all()
    } if work_ids else {}
    latest_normal: dict[str, Job] = {}
    latest_upgrade: dict[str, Job] = {}
    recent_aweme_id: str | None = None
    for job in jobs:
        aweme_id = work_map.get(job.work_id) or _payload_aweme_id(job)
        if aweme_id is None:
            continue
        if _job_is_upgrade(job):
            latest_upgrade.setdefault(aweme_id, job)
        else:
            if recent_aweme_id is None:
                recent_aweme_id = aweme_id
            latest_normal.setdefault(aweme_id, job)

    counts = {"pending": 0, "running": 0, "failed": 0}
    quality_details = _quality_details_snapshot(service)
    quality_by_aweme = {
        aweme_id: detail["quality_status"]
        for aweme_id, detail in quality_details.items()
    }
    failures: list[dict[str, str]] = []
    for job in latest_normal.values():
        if job.status in counts:
            counts[job.status] += 1
        if job.status == "failed" and len(failures) < 5:
            aweme_id = work_map.get(job.work_id) or _payload_aweme_id(job)
            if aweme_id is None:
                continue
            failures.append(
                {
                    "aweme_id": aweme_id,
                    "reason": job.error_detail or job.error_code or "unknown_error",
                }
            )

    upgrade_candidates = {
        aweme_id
        for aweme_id, quality in quality_by_aweme.items()
        if quality in _UPGRADE_QUALITY_STATUSES
    }
    upgrade_cohort = upgrade_candidates | set(latest_upgrade)
    upgrade_completed = 0
    upgrade_pending = 0
    upgrade_running = 0
    upgrade_failed = 0
    upgrade_not_queued = 0
    upgrade_failures: list[dict[str, str]] = []

    # Job order is newest-first, so this preserves the most recent failure
    # details while the remaining unqueued candidates are added deterministically.
    ordered_upgrade_ids = list(latest_upgrade)
    ordered_upgrade_ids.extend(sorted(upgrade_cohort.difference(latest_upgrade)))
    for aweme_id in ordered_upgrade_ids:
        if quality_by_aweme.get(aweme_id) == "semantic_complete":
            upgrade_completed += 1
            continue
        job = latest_upgrade.get(aweme_id)
        if job is None:
            upgrade_not_queued += 1
        elif job.status == "pending":
            upgrade_pending += 1
        elif job.status == "running":
            upgrade_running += 1
        elif job.status == "failed":
            upgrade_failed += 1
            if len(upgrade_failures) < 5:
                upgrade_failures.append(
                    {
                        "aweme_id": aweme_id,
                        "reason": job.error_detail or job.error_code or "unknown_error",
                    }
                )
        elif job.status == "succeeded":
            # Historical workers could mark a preserved/no-op result as
            # succeeded.  Keep it visible as an unresolved semantic failure.
            upgrade_failed += 1
            if len(upgrade_failures) < 5:
                upgrade_failures.append(
                    {
                        "aweme_id": aweme_id,
                        "reason": "upgrade_completed_without_semantic_note",
                    }
                )
        else:
            upgrade_not_queued += 1

    upgrade_total = len(upgrade_cohort)
    upgrade_remaining = max(upgrade_total - upgrade_completed, 0)
    depth_failed = sum(
        1
        for aweme_id in upgrade_cohort
        if (
            quality_details.get(aweme_id, {}).get("coverage", {}).get("quality_gate")
            == "failed"
            and quality_details.get(aweme_id, {}).get("coverage", {}).get("depth_passed")
            is False
        )
    )
    input_insufficient = sum(
        1
        for aweme_id in upgrade_cohort
        if (
            quality_details.get(aweme_id, {}).get("coverage", {}).get("quality_gate")
            == "failed"
            and quality_details.get(aweme_id, {}).get("coverage", {}).get("input_sufficient")
            is False
        )
    )
    upgrade = {
        "total": upgrade_total,
        "completed": upgrade_completed,
        "remaining": upgrade_remaining,
        "pending": upgrade_pending,
        "running": upgrade_running,
        "active": upgrade_pending + upgrade_running,
        "failed": upgrade_failed,
        "not_queued": upgrade_not_queued,
        "eligible": len(upgrade_candidates),
        "queued": len(latest_upgrade),
        "depth_failed": depth_failed,
        "input_insufficient": input_insufficient,
        "percent": round(
            (upgrade_completed / upgrade_total * 100) if upgrade_total else 0.0,
            1,
        ),
        "failures": upgrade_failures,
    }
    not_queued = max(
        total - completed - counts["pending"] - counts["running"] - counts["failed"],
        0,
    )
    quality_counts = _quality_counts(service, snapshot=quality_details)
    return {
        "total": total,
        "completed": completed,
        "running": counts["running"],
        "pending": counts["pending"],
        "failed": counts["failed"],
        "skipped": completed,
        "not_queued": not_queued,
        "percent": round((completed / total * 100) if total else 0.0, 1),
        "recent_aweme_id": recent_aweme_id,
        "failures": failures,
        "quality_counts": quality_counts,
        "upgrade": upgrade,
        "provider_summary": _provider_status_cache,
    }


def _service(queue: JobQueue) -> NoteGenerationService:
    return NoteGenerationService(
        queue.session,
        StorageLayout(Settings().storage_root),
    )


def _payload_aweme_id(job: Job) -> str | None:
    if not job.payload_json:
        return None
    try:
        value = json.loads(job.payload_json).get("aweme_id")
    except (AttributeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, str) else None


def _normalized_aweme_ids(values: list[str] | None) -> set[str] | None:
    if values is None:
        return None
    result = {value.strip() for value in values if isinstance(value, str) and value.strip()}
    return result


def _job_is_upgrade(job: Job) -> bool:
    if not job.payload_json:
        return False
    try:
        payload = json.loads(job.payload_json)
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("legacy_only") is True


def _quality_counts(
    service: NoteGenerationService,
    *,
    snapshot: dict[str, dict[str, Any]] | None = None,
) -> dict[str, int]:
    # Inventory is filesystem-backed, while the work map is database-backed.
    # Count all transcript bundles so an orphaned transcript remains visible.
    counts = {"semantic_complete": 0, "fallback": 0, "legacy": 0, "unknown": 0, "missing": 0}
    details = snapshot if snapshot is not None else _quality_details_snapshot(service)
    for detail in details.values():
        quality = detail["quality_status"]
        counts[quality] = counts.get(quality, 0) + 1
    return counts


_UPGRADE_QUALITY_STATUSES = {"legacy", "fallback", "unknown"}


def _quality_snapshot(service: NoteGenerationService) -> dict[str, str]:
    return {
        aweme_id: detail["quality_status"]
        for aweme_id, detail in _quality_details_snapshot(service).items()
    }


def _quality_details_snapshot(service: NoteGenerationService) -> dict[str, dict[str, Any]]:
    from douyin_library.notes.state import classify_note_bundle

    if not service.layout.transcript.is_dir():
        return {}
    details: dict[str, dict[str, Any]] = {}
    for transcript in service.layout.transcript.glob("*/transcript.txt"):
        state = classify_note_bundle(service.layout.note_dir(transcript.parent.name))
        details[transcript.parent.name] = {
            "quality_status": state.quality_status,
            "coverage": state.coverage or {},
        }
    return details
