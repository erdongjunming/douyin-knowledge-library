from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, case, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, aliased

from douyin_library.models import Job, ProcessingAudit, Work


class JobTransitionError(ValueError):
    pass


class DuplicateActiveJobError(ValueError):
    pass


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
ALLOWED_TRANSITIONS = {
    "pending": {"running", "paused", "cancelled"},
    "running": {"pending", "paused", "succeeded", "failed", "cancelled"},
    "paused": {"pending", "cancelled"},
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

# Legacy upgrades are deliberately bounded.  A failed legacy attempt stays
# cooled for the ordinary retry window and terminal failures receive a longer
# marker so a refresh cannot recreate the same work indefinitely.
LEGACY_RETRY_LIMIT = 3
LEGACY_RETRY_DELAYS = (60, 300)
LEGACY_TERMINAL_COOLDOWN = timedelta(hours=1)
LEGACY_PERMANENT_ERRORS = {
    "input_insufficient",
    "legacy_upgrade_input_insufficient",
    "semantic_incomplete",
    "legacy_upgrade_semantic_incomplete",
    "quality_gate_failed",
    "publication_conflict",
    "cancelled_by_user",
    "upgrade_quality_gate",
    "legacy_upgrade_no_semantic_provider_succeeded",
}
QUEUE_LANES = {"pipeline", "new_note", "legacy_upgrade"}
ACTIVE_STATUSES = {"pending", "running", "paused"}

# ``analyze_work`` belonged to an earlier analysis pipeline and has no
# registered handler in the current runtime.  Reject new rows explicitly so
# they cannot sit in a durable queue that no worker can consume.
UNSUPPORTED_JOB_KINDS = {"analyze_work"}


class JobQueue:
    def __init__(self, session: Session) -> None:
        self.session = session

    @staticmethod
    def _payload_dict(payload_json: str | None) -> dict[str, object]:
        if not payload_json:
            return {}
        try:
            value = json.loads(payload_json)
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    def _resolve_lane(
        self,
        *,
        kind: str,
        payload: Mapping[str, object] | None,
        queue_lane: str | None,
    ) -> str:
        if queue_lane is not None:
            if queue_lane not in QUEUE_LANES:
                raise JobTransitionError(f"invalid_queue_lane:{queue_lane}")
            return queue_lane
        if kind != "generate_note":
            return "pipeline"
        return "legacy_upgrade" if bool((payload or {}).get("legacy_only")) else "new_note"

    def _resolve_dedupe_key(
        self,
        *,
        kind: str,
        work_id: int | None,
        payload: Mapping[str, object] | None,
    ) -> str | None:
        if kind != "generate_note":
            return None
        values = payload or {}
        aweme_id = None
        if work_id is not None:
            aweme_id = self.session.scalar(
                select(Work.aweme_id).where(Work.id == work_id)
            )
        if aweme_id is None:
            aweme_id = values.get("aweme_id")
        if aweme_id is None:
            return None
        text_value = str(aweme_id).strip()
        return f"note:{text_value}" if text_value else None

    @staticmethod
    def _effective_lane(job: Job) -> str:
        if job.queue_lane in QUEUE_LANES:
            return job.queue_lane
        # Rows created before the lane migration are interpreted safely until
        # the backfill migration has run.
        if job.kind == "generate_note":
            payload = JobQueue._payload_dict(job.payload_json)
            return "legacy_upgrade" if bool(payload.get("legacy_only")) else "new_note"
        return "pipeline"

    @staticmethod
    def _ownership_conditions(
        job_id: int,
        *,
        worker_id: str | None,
        claim_fence: int | None,
        status: str = "running",
    ) -> list[object]:
        conditions: list[object] = [Job.id == job_id, Job.status == status]
        if worker_id is not None:
            conditions.append(Job.worker_id == worker_id)
        if claim_fence is not None:
            conditions.append(Job.claim_fence == claim_fence)
        return conditions

    @staticmethod
    def _legacy_capacity_conditions(now: datetime) -> list[object]:
        legacy_lane = or_(
            Job.queue_lane == "legacy_upgrade",
            and_(
                Job.queue_lane.is_(None),
                Job.kind == "generate_note",
                func.json_extract(Job.payload_json, "$.legacy_only") == 1,
            ),
        )
        new_note_lane = or_(
            Job.queue_lane == "new_note",
            and_(
                Job.queue_lane.is_(None),
                Job.kind == "generate_note",
                or_(
                    func.json_extract(Job.payload_json, "$.legacy_only").is_(None),
                    func.json_extract(Job.payload_json, "$.legacy_only") != 1,
                ),
            ),
        )
        running_legacy = select(Job.id).where(Job.status == "running", legacy_lane)
        running_new = select(Job.id).where(
            Job.status == "running",
            Job.kind == "generate_note",
            new_note_lane,
        )
        runnable_new = select(Job.id).where(
            Job.status == "pending",
            Job.kind == "generate_note",
            new_note_lane,
            or_(Job.next_retry_at.is_(None), Job.next_retry_at <= now),
        )
        return [~running_legacy.exists(), ~running_new.exists(), ~runnable_new.exists()]

    def create(
        self,
        *,
        kind: str,
        work_id: int | None = None,
        sync_run_id: int | None = None,
        payload: Mapping[str, object] | None = None,
        status: str = "pending",
        attempts: int = 0,
        queue_lane: str | None = None,
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
        lane = self._resolve_lane(
            kind=kind,
            payload=payload,
            queue_lane=queue_lane,
        )
        dedupe_key = self._resolve_dedupe_key(
            kind=kind,
            work_id=work_id,
            payload=payload,
        )
        if dedupe_key is not None:
            active = self.session.scalar(
                select(Job).where(
                    Job.dedupe_key == dedupe_key,
                    Job.status.in_(ACTIVE_STATUSES),
                )
            )
            if active is not None:
                raise DuplicateActiveJobError("active_job_already_exists")
            # A terminal legacy failure is a durable stop reason.  Ordinary
            # refresh/discovery must not manufacture a fresh active row for
            # the same note.  A normal new-note request remains allowed to
            # repair the work.
            if lane == "legacy_upgrade":
                cooled_failure = self.session.scalar(
                    select(Job).where(
                        Job.dedupe_key == dedupe_key,
                        Job.queue_lane == "legacy_upgrade",
                        Job.status == "failed",
                    )
                )
                if cooled_failure is not None:
                    raise DuplicateActiveJobError("legacy_upgrade_cooled")
            # A pre-migration active row may not have a canonical key yet.
            # Keep the work-id guard until the migration backfill is complete.
            if work_id is not None:
                active = self.session.scalar(
                    select(Job).where(
                        Job.work_id == work_id,
                        Job.kind == kind,
                        Job.status.in_(ACTIVE_STATUSES),
                    )
                )
                if active is not None:
                    raise DuplicateActiveJobError("active_job_already_exists")
        elif work_id is not None:
            active = self.session.scalar(
                select(Job).where(
                    Job.work_id == work_id,
                    Job.kind == kind,
                    Job.status.in_(ACTIVE_STATUSES),
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
                    Job.status.in_(ACTIVE_STATUSES),
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
            queue_lane=lane,
            dedupe_key=dedupe_key,
        )
        self.session.add(job)
        try:
            self.session.commit()
        except IntegrityError as exc:
            self.session.rollback()
            # The partial unique index is the authoritative race boundary for
            # concurrent enqueue requests.  Preserve the public queue error.
            if dedupe_key is not None:
                raise DuplicateActiveJobError("active_job_already_exists") from exc
            raise
        self.session.refresh(job)
        return job

    def get(self, job_id: int) -> Job:
        job = self.session.get(Job, job_id)
        if job is None:
            raise KeyError(f"job_not_found:{job_id}")
        return job

    def is_owned(
        self,
        job_id: int,
        *,
        worker_id: str | None,
        claim_fence: int,
    ) -> bool:
        with self.session.no_autoflush:
            return bool(
                self.session.scalar(
                    select(Job.id).where(
                        *self._ownership_conditions(
                            job_id,
                            worker_id=worker_id,
                            claim_fence=claim_fence,
                        )
                    )
                )
            )

    def claim_next(
        self,
        *,
        kinds: Collection[str] | None = None,
        worker_id: str | None = None,
    ) -> Job | None:
        for _ in range(5):
            now = datetime.now(UTC)
            lane = case(
                (
                    Job.queue_lane.is_(None),
                    case((Job.kind == "generate_note", "new_note"), else_="pipeline"),
                ),
                else_=Job.queue_lane,
            )
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
                    (lane == "legacy_upgrade", 6),
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
            selected = self.get(job_id)
            conditions = [Job.id == job_id, Job.status == "pending"]
            if self._effective_lane(selected) == "legacy_upgrade":
                conditions.extend(self._legacy_capacity_conditions(now))
            try:
                result = self.session.execute(
                    update(Job)
                    .where(*conditions)
                    .values(
                        status="running",
                        started_at=now,
                        heartbeat_at=now,
                        worker_id=worker_id,
                        claim_fence=func.coalesce(Job.claim_fence, 0) + 1,
                    )
                )
                self.session.commit()
            except IntegrityError:
                self.session.rollback()
                self.session.expire_all()
                continue
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
            lane = self._effective_lane(job)
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
        if lane == "legacy_upgrade":
            conditions.extend(self._legacy_capacity_conditions(now))
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
                        claim_fence=func.coalesce(Job.claim_fence, 0) + 1,
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

    def transition(
        self,
        job_id: int,
        status: str,
        *,
        worker_id: str | None = None,
        claim_fence: int | None = None,
    ) -> Job:
        with self.session.no_autoflush:
            job = self.get(job_id)
            allowed = ALLOWED_TRANSITIONS.get(job.status, set())
            if status not in allowed:
                raise JobTransitionError(f"invalid_transition:{job.status}->{status}")
        now = datetime.now(UTC)
        conditions = self._ownership_conditions(
            job_id,
            worker_id=worker_id,
            claim_fence=claim_fence,
            status=job.status,
        )
        values: dict[str, object] = {"status": status}
        if status == "succeeded":
            values.update(stage="complete", progress=1.0, next_retry_at=None)
        if status in TERMINAL_STATUSES:
            values["finished_at"] = now
        result = self.session.execute(update(Job).where(*conditions).values(**values))
        self.session.commit()
        if not result.rowcount:
            self.session.expire_all()
            raise JobTransitionError("job_ownership_lost")
        self.session.expire_all()
        transitioned = self.get(job_id)
        return transitioned

    def cancel(
        self,
        job_id: int,
        *,
        reason: str = "cancelled_by_user",
        worker_id: str | None = None,
        claim_fence: int | None = None,
    ) -> Job:
        job = self.transition(
            job_id,
            "cancelled",
            worker_id=worker_id,
            claim_fence=claim_fence,
        )
        job.error_code = "cancelled_by_user"
        job.error_detail = reason[:500]
        self.session.commit()
        self.session.refresh(job)
        return job

    def pause(
        self,
        job_id: int,
        *,
        worker_id: str | None = None,
        claim_fence: int | None = None,
    ) -> Job:
        with self.session.no_autoflush:
            current = self.get(job_id)
            if "paused" not in ALLOWED_TRANSITIONS.get(current.status, set()):
                raise JobTransitionError(f"invalid_transition:{current.status}->paused")
            conditions = self._ownership_conditions(
                job_id,
                worker_id=worker_id,
                claim_fence=claim_fence,
                status=current.status,
            )
        result = self.session.execute(
            update(Job)
            .where(*conditions)
            .values(
                status="paused",
                error_code="paused_by_user",
                error_detail="paused_by_user",
                started_at=None,
                heartbeat_at=None,
                worker_id=None,
                claim_fence=func.coalesce(Job.claim_fence, 0) + 1,
            )
        )
        self.session.commit()
        if not result.rowcount:
            self.session.expire_all()
            raise JobTransitionError("job_ownership_lost")
        self.session.expire_all()
        return self.get(job_id)

    def resume(self, job_id: int) -> Job:
        job = self.get(job_id)
        # Explicit operator retry applies to every job family. Keep automatic
        # terminal-state transitions closed and retain the historical error
        # and attempt count, while invalidating the former worker's claim.
        if job.status == "failed":
            other = aliased(Job)
            identities = []
            if job.dedupe_key is not None:
                identities.append(other.dedupe_key == job.dedupe_key)
            if job.work_id is not None:
                identities.append(and_(other.kind == job.kind, other.work_id == job.work_id))
            elif job.payload_json is not None:
                identities.append(and_(
                    other.kind == job.kind,
                    other.work_id.is_(None),
                    other.payload_json == job.payload_json,
                ))
            conditions = [Job.id == job_id, Job.status == "failed"]
            if identities:
                # Check in the same SQLite write statement as the transition,
                # so two concurrent retries cannot both reopen the same work.
                conditions.append(~select(other.id).where(
                    other.id != job_id,
                    other.status.in_(ACTIVE_STATUSES),
                    or_(*identities),
                ).exists())
            statement = (
                update(Job)
                .where(*conditions)
                .values(
                    status="pending",
                    next_retry_at=None,
                    finished_at=None,
                    started_at=None,
                    heartbeat_at=None,
                    worker_id=None,
                    stage="discover",
                    progress=0.0,
                    claim_fence=func.coalesce(Job.claim_fence, 0) + 1,
                )
            )
            try:
                result = self.session.execute(statement)
                self.session.commit()
            except IntegrityError as exc:
                self.session.rollback()
                if "job.dedupe_key" in str(exc.orig):
                    raise JobTransitionError("duplicate_active_job") from None
                raise
            if not result.rowcount:
                self.session.expire_all()
                if self.get(job_id).status == "failed":
                    raise JobTransitionError("duplicate_active_job")
                raise JobTransitionError("job_ownership_lost")
            self.session.expire_all()
            return self.get(job_id)
        job = self.transition(job_id, "pending")
        job.error_code = None
        job.error_detail = None
        job.next_retry_at = None
        self.session.commit()
        self.session.refresh(job)
        return job

    def heartbeat(
        self,
        job_id: int,
        *,
        worker_id: str,
        claim_fence: int | None = None,
    ) -> bool:
        result = self.session.execute(
            update(Job)
            .where(
                *self._ownership_conditions(
                    job_id,
                    worker_id=worker_id,
                    claim_fence=claim_fence,
                )
            )
            .values(heartbeat_at=datetime.now(UTC))
        )
        self.session.commit()
        return bool(result.rowcount)

    def release(
        self,
        job_id: int,
        *,
        worker_id: str,
        claim_fence: int | None = None,
    ) -> bool:
        result = self.session.execute(
            update(Job)
            .where(
                *self._ownership_conditions(
                    job_id,
                    worker_id=worker_id,
                    claim_fence=claim_fence,
                )
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

    def fail(
        self,
        job_id: int,
        *,
        error_code: str,
        error_detail: str,
        worker_id: str | None = None,
        claim_fence: int | None = None,
    ) -> Job:
        with self.session.no_autoflush:
            job = self.get(job_id)
            if job.status != "running":
                raise JobTransitionError(f"cannot_fail:{job.status}")
            old_attempts = job.attempts
            lane = self._effective_lane(job)
        attempts = old_attempts + 1
        now = datetime.now(UTC)
        if lane == "legacy_upgrade":
            limit = LEGACY_RETRY_LIMIT
            delay_values = LEGACY_RETRY_DELAYS
        else:
            limit = RETRY_LIMITS.get(job.kind, 1)
            delay_values = RETRY_DELAYS.get(job.kind, (60,))
        values: dict[str, object] = {
            "error_code": error_code,
            "error_detail": error_detail[:500],
            "attempts": attempts,
        }
        permanent_failure = lane == "legacy_upgrade" and (
            error_code in LEGACY_PERMANENT_ERRORS
            or any(error_code.endswith(f":{suffix}") for suffix in LEGACY_PERMANENT_ERRORS)
            or any(marker in error_detail for marker in LEGACY_PERMANENT_ERRORS)
        )
        if attempts < limit and not permanent_failure:
            delay = delay_values[min(attempts - 1, len(delay_values) - 1)]
            values.update(status="pending", next_retry_at=now + timedelta(seconds=delay))
        else:
            values.update(status="failed", finished_at=now)
            if lane == "legacy_upgrade":
                values["next_retry_at"] = now + LEGACY_TERMINAL_COOLDOWN
        result = self.session.execute(
            update(Job)
            .where(
                *self._ownership_conditions(
                    job_id,
                    worker_id=worker_id,
                    claim_fence=claim_fence,
                )
            )
            .values(**values)
        )
        self.session.commit()
        if not result.rowcount:
            self.session.expire_all()
            raise JobTransitionError("job_ownership_lost")
        self.session.expire_all()
        return self.get(job_id)

    def recover_stale_jobs(self, *, stale_after: timedelta = timedelta(minutes=15)) -> int:
        cutoff = datetime.now(UTC) - stale_after
        stale_condition = or_(
            Job.worker_id.is_(None),
            func.coalesce(Job.heartbeat_at, Job.started_at) < cutoff,
        )
        with self.session.no_autoflush:
            stale_jobs = self.session.scalars(
                select(Job).where(Job.status == "running", stale_condition)
            ).all()
        stale_job_ids: list[int] = []
        recovered_at = datetime.now(UTC)
        for job in stale_jobs:
            old_fence = job.claim_fence or 0
            values: dict[str, object] = {
                "status": "pending",
                "started_at": None,
                "heartbeat_at": None,
                "worker_id": None,
                "claim_fence": old_fence + 1,
                "error_code": "worker_stale_recovered",
                "error_detail": "worker_stale_recovered",
            }
            if self._effective_lane(job) == "legacy_upgrade":
                attempts = job.attempts + 1
                values["attempts"] = attempts
                if attempts >= LEGACY_RETRY_LIMIT:
                    values.update(
                        status="failed",
                        finished_at=recovered_at,
                        next_retry_at=recovered_at + LEGACY_TERMINAL_COOLDOWN,
                    )
                else:
                    delay = LEGACY_RETRY_DELAYS[
                        min(attempts - 1, len(LEGACY_RETRY_DELAYS) - 1)
                    ]
                    values["next_retry_at"] = recovered_at + timedelta(seconds=delay)
            with self.session.no_autoflush:
                result = self.session.execute(
                    update(Job)
                    .where(
                        Job.id == job.id,
                        Job.status == "running",
                        Job.claim_fence == old_fence,
                        stale_condition,
                    )
                    .values(**values)
                )
            if result.rowcount:
                stale_job_ids.append(job.id)
        if stale_job_ids:
            audits = self.session.scalars(
                select(ProcessingAudit).where(
                    ProcessingAudit.job_id.in_(stale_job_ids),
                    ProcessingAudit.status == "running",
                )
            ).all()
            finished_at = recovered_at
            for audit in audits:
                audit.status = "failed"
                audit.error = "worker_stale_recovered"
                audit.finished_at = finished_at
        self.session.commit()
        self.session.expire_all()
        return len(stale_job_ids)
