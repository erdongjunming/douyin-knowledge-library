from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from datetime import UTC, datetime, timedelta

from sqlalchemy import case, func, or_, select, update
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
    "clean_transcript_work": 2,
    "generate_note": 1,
}
RETRY_DELAYS = {
    "sync_creator": (60, 300, 1800),
    "sync_favorites": (60, 300, 1800),
    "download_work": (60, 300, 1800),
    "transcribe_work": (60, 300),
    "clean_transcript_work": (60, 300),
    "generate_note": (60,),
}

# ``analyze_work`` belonged to an earlier analysis pipeline and has no
# registered handler in the current runtime.  Reject new rows explicitly so
# they cannot sit in a durable queue that no worker can consume.
UNSUPPORTED_JOB_KINDS = {"analyze_work"}


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
        if kind in UNSUPPORTED_JOB_KINDS:
            raise JobTransitionError(f"unsupported_job_kind:{kind}")
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

    def claim_next(
        self,
        *,
        kinds: Collection[str] | None = None,
        worker_id: str | None = None,
    ) -> Job | None:
        for _ in range(5):
            now = datetime.now(UTC)
            candidate = select(Job.id).where(
                Job.status == "pending",
                (Job.next_retry_at.is_(None)) | (Job.next_retry_at <= now),
            )
            if kinds is not None:
                if not kinds:
                    return None
                candidate = candidate.where(Job.kind.in_(tuple(kinds)))
            candidate = candidate.order_by(
                case(
                    (Job.kind.in_(("sync_creator", "sync_favorites")), 0),
                    (Job.kind == "transcribe_work", 1),
                    (Job.kind == "clean_transcript_work", 2),
                    (Job.kind == "generate_note", 3),
                    (Job.kind == "download_work", 4),
                    else_=5,
                ),
                Job.created_at,
                Job.id,
            ).limit(1)
            job_id = self.session.scalar(candidate)
            if job_id is None:
                return None
            result = self.session.execute(
                update(Job)
                .where(Job.id == job_id, Job.status == "pending")
                .values(
                    status="running",
                    started_at=now,
                    heartbeat_at=now,
                    worker_id=worker_id,
                )
            )
            self.session.commit()
            if result.rowcount:
                return self.get(job_id)
            self.session.expire_all()
        return None

    def claim(
        self,
        job_id: int,
        *,
        ignore_schedule: bool = False,
        worker_id: str | None = None,
    ) -> Job:
        # Read first for the same validation/error messages as the queue API,
        # then perform the state change with a conditional UPDATE.  The old
        # ORM assignment could let two workers read ``pending`` and overwrite
        # each other's worker identity on commit.
        with self.session.no_autoflush:
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
        conditions = [Job.id == job_id, Job.status == "pending"]
        if not ignore_schedule:
            conditions.append((Job.next_retry_at.is_(None)) | (Job.next_retry_at <= now))
        try:
            with self.session.no_autoflush:
                result = self.session.execute(
                    update(Job)
                    .where(*conditions)
                    .values(
                        status="running",
                        started_at=now,
                        heartbeat_at=now,
                        worker_id=worker_id,
                    )
                )
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        if not result.rowcount:
            # A concurrent claimant may have won after our initial read.  Do
            # not silently proceed with a stale ORM object or overwrite its
            # ownership; report the observed durable state instead.
            self.session.expire_all()
            current = self.session.get(Job, job_id)
            if current is None:
                raise KeyError(f"job_not_found:{job_id}")
            if current.status != "pending":
                raise JobTransitionError(f"cannot_claim:{current.status}")
            if not ignore_schedule and current.next_retry_at is not None:
                retry_at = current.next_retry_at
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                if retry_at > datetime.now(UTC):
                    raise JobTransitionError("cannot_claim_before_retry")
            raise JobTransitionError("cannot_claim:concurrent_update")
        self.session.expire_all()
        claimed = self.session.get(Job, job_id)
        if claimed is None:  # pragma: no cover - row cannot vanish after UPDATE
            raise KeyError(f"job_not_found:{job_id}")
        return claimed

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

    def cancel(self, job_id: int, *, reason: str = "cancelled_by_user") -> Job:
        job = self.transition(job_id, "cancelled")
        job.error_code = "cancelled_by_user"
        job.error_detail = reason[:500]
        self.session.commit()
        self.session.refresh(job)
        return job

    def heartbeat(self, job_id: int, *, worker_id: str) -> bool:
        result = self.session.execute(
            update(Job)
            .where(
                Job.id == job_id,
                Job.status == "running",
                Job.worker_id == worker_id,
            )
            .values(heartbeat_at=datetime.now(UTC))
        )
        self.session.commit()
        return bool(result.rowcount)

    def release(self, job_id: int, *, worker_id: str) -> bool:
        result = self.session.execute(
            update(Job)
            .where(
                Job.id == job_id,
                Job.status == "running",
                Job.worker_id == worker_id,
            )
            .values(
                status="pending",
                started_at=None,
                heartbeat_at=None,
                worker_id=None,
            )
        )
        self.session.commit()
        return bool(result.rowcount)

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
            select(Job).where(
                Job.status == "running",
                or_(
                    Job.worker_id.is_(None),
                    func.coalesce(Job.heartbeat_at, Job.started_at) < cutoff,
                ),
            )
        ).all()
        stale_job_ids = [job.id for job in stale_jobs]
        for job in stale_jobs:
            job.status = "pending"
            job.started_at = None
            job.heartbeat_at = None
            job.worker_id = None
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
