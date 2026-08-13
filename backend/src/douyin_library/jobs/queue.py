from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

from sqlalchemy import case, select
from sqlalchemy.orm import Session

from douyin_library.models import Job, ProcessingAudit


class JobTransitionError(ValueError):
    pass


class DuplicateActiveJobError(ValueError):
    pass


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
ALLOWED_TRANSITIONS = {
    "pending": {"running", "cancelled"},
    "running": {"pending", "succeeded", "failed", "cancelled"},
}
RETRY_LIMITS = {
    "sync_creator": 3,
    "sync_favorites": 3,
    "download_work": 3,
    "transcribe_work": 2,
    "analyze_work": 2,
}
RETRY_DELAYS = {
    "sync_creator": (60, 300, 1800),
    "sync_favorites": (60, 300, 1800),
    "download_work": (60, 300, 1800),
    "transcribe_work": (60, 300),
    "analyze_work": (60, 300),
}


class JobQueue:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        *,
        kind: str,
        work_id: int | None = None,
        sync_run_id: int | None = None,
        payload: Mapping[str, object] | None = None,
        status: str = "pending",
        attempts: int = 0,
    ) -> Job:
        if status not in {"pending", "running"}:
            raise JobTransitionError(f"cannot create job in status {status}")
        payload_json = (
            json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)
            if payload is not None
            else None
        )
        if work_id is not None:
            active = self.session.scalar(
                select(Job).where(
                    Job.work_id == work_id,
                    Job.kind == kind,
                    Job.status.not_in(TERMINAL_STATUSES),
                )
            )
            if active is not None:
                raise DuplicateActiveJobError("active_job_already_exists")
        elif payload_json is not None:
            active = self.session.scalar(
                select(Job).where(
                    Job.kind == kind,
                    Job.work_id.is_(None),
                    Job.payload_json == payload_json,
                    Job.status.not_in(TERMINAL_STATUSES),
                )
            )
            if active is not None:
                raise DuplicateActiveJobError("active_job_already_exists")
        job = Job(
            kind=kind,
            work_id=work_id,
            sync_run_id=sync_run_id,
            payload_json=payload_json,
            status=status,
            attempts=attempts,
        )
        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)
        return job

    def get(self, job_id: int) -> Job:
        job = self.session.get(Job, job_id)
        if job is None:
            raise KeyError(f"job_not_found:{job_id}")
        return job

    def claim_next(self) -> Job | None:
        now = datetime.now(UTC)
        job = self.session.scalar(
            select(Job)
            .where(
                Job.status == "pending",
                (Job.next_retry_at.is_(None)) | (Job.next_retry_at <= now),
            )
            .order_by(
                case(
                    (Job.kind.in_(("sync_creator", "sync_favorites")), 0),
                    (Job.kind == "transcribe_work", 1),
                    (Job.kind == "download_work", 2),
                    (Job.kind == "analyze_work", 3),
                    else_=4,
                ),
                Job.created_at,
                Job.id,
            )
        )
        if job is None:
            return None
        job.status = "running"
        job.started_at = now
        self.session.commit()
        self.session.refresh(job)
        return job

    def claim(self, job_id: int, *, ignore_schedule: bool = False) -> Job:
        job = self.get(job_id)
        if job.status != "pending":
            raise JobTransitionError(f"cannot_claim:{job.status}")
        now = datetime.now(UTC)
        if not ignore_schedule and job.next_retry_at is not None:
            retry_at = job.next_retry_at
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            if retry_at > now:
                raise JobTransitionError("cannot_claim_before_retry")
        job.status = "running"
        job.started_at = now
        self.session.commit()
        self.session.refresh(job)
        return job

    def transition(self, job_id: int, status: str) -> Job:
        job = self.get(job_id)
        allowed = ALLOWED_TRANSITIONS.get(job.status, set())
        if status not in allowed:
            raise JobTransitionError(f"invalid_transition:{job.status}->{status}")
        job.status = status
        if status == "succeeded":
            job.stage = "complete"
            job.progress = 1.0
            job.next_retry_at = None
        if status in TERMINAL_STATUSES:
            job.finished_at = datetime.now(UTC)
        self.session.commit()
        self.session.refresh(job)
        return job

    def fail(self, job_id: int, *, error_code: str, error_detail: str) -> Job:
        job = self.get(job_id)
        if job.status != "running":
            raise JobTransitionError(f"cannot_fail:{job.status}")
        job.error_code = error_code
        job.error_detail = error_detail[:500]
        job.attempts += 1
        limit = RETRY_LIMITS.get(job.kind, 1)
        if job.attempts < limit:
            delay_values = RETRY_DELAYS.get(job.kind, (60,))
            delay = delay_values[min(job.attempts - 1, len(delay_values) - 1)]
            job.status = "pending"
            job.next_retry_at = datetime.now(UTC) + timedelta(seconds=delay)
        else:
            job.status = "failed"
            job.finished_at = datetime.now(UTC)
        self.session.commit()
        self.session.refresh(job)
        return job

    def recover_stale_jobs(self, *, stale_after: timedelta = timedelta(minutes=15)) -> int:
        cutoff = datetime.now(UTC) - stale_after
        stale_jobs = self.session.scalars(
            select(Job).where(Job.status == "running", Job.started_at < cutoff)
        ).all()
        stale_job_ids = [job.id for job in stale_jobs]
        for job in stale_jobs:
            job.status = "pending"
            job.started_at = None
        if stale_job_ids:
            audits = self.session.scalars(
                select(ProcessingAudit).where(
                    ProcessingAudit.job_id.in_(stale_job_ids),
                    ProcessingAudit.status == "running",
                )
            ).all()
            finished_at = datetime.now(UTC)
            for audit in audits:
                audit.status = "failed"
                audit.error = "worker_stale_recovered"
                audit.finished_at = finished_at
        self.session.commit()
        return len(stale_jobs)
