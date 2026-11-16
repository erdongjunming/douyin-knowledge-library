"""Durable scheduling for historical note upgrades.

The scheduler only decides which old-note jobs may enter the shared queue.  It
does not generate notes itself; workers still execute the normal fenced
``generate_note`` handler and its semantic publication gates.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.jobs.queue import DuplicateActiveJobError, JobQueue
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Job, Setting
from douyin_library.notes.service import NoteGenerationService

SETTING_KEY = "legacy_upgrade.enabled"
LOCK_SETTING_KEY = "legacy_upgrade.refresh_lock"
LOCK_TTL_SECONDS = 30
DEFAULT_BATCH_SIZE = 3
DISCOVERY_SETTING_KEY = "legacy_upgrade.next_discovery_at"
DISCOVERY_INTERVAL_SECONDS = 30


def set_legacy_upgrade_enabled(queue: JobQueue, enabled: bool = True) -> bool:
    """Persist the operator's choice to keep historical upgrades scheduled."""

    setting = queue.session.execute(
        select(Setting)
        .where(Setting.key == SETTING_KEY)
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()
    if setting is None:
        setting = Setting(key=SETTING_KEY, value_json=json.dumps(bool(enabled)))
        queue.session.add(setting)
    else:
        setting.value_json = json.dumps(bool(enabled))
    queue.session.commit()
    return bool(enabled)


def legacy_upgrade_enabled(session: Session) -> bool:
    setting = session.execute(
        select(Setting)
        .where(Setting.key == SETTING_KEY)
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()
    if setting is None:
        return False
    try:
        return json.loads(setting.value_json) is True
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def refresh_legacy_upgrade(
    queue: JobQueue,
    settings: Settings,
    *,
    limit: int = DEFAULT_BATCH_SIZE,
    enable: bool | None = None,
    throttle: bool = False,
) -> dict[str, Any]:
    """Recover the next bounded legacy batch when the new-note lane is idle.

    ``enable=True`` is used by the explicit legacy-upgrade action.  Once
    enabled, worker idle polls and refresh requests can continue filling one
    small batch at a time.  No terminal failed job is recreated: queue-level
    canonical dedupe and its legacy retry policy own that decision.
    """

    if enable is not None:
        set_legacy_upgrade_enabled(queue, enable)
    if not legacy_upgrade_enabled(queue.session):
        return {"queued": 0, "lane": "legacy_upgrade", "reason": "disabled"}
    bounded_limit = max(0, min(int(limit), DEFAULT_BATCH_SIZE))
    if bounded_limit == 0:
        return {"queued": 0, "lane": "legacy_upgrade", "reason": "limit_zero"}
    lock_token = _acquire_refresh_lock(queue)
    if lock_token is None:
        return {"queued": 0, "lane": "legacy_upgrade", "reason": "refresh_in_progress"}
    try:
        if _legacy_batch_active(queue):
            return {"queued": 0, "lane": "legacy_upgrade", "reason": "batch_active"}
        if _runnable_new_note_exists(queue):
            return {"queued": 0, "lane": "legacy_upgrade", "reason": "new_note_runnable"}
        if throttle and _discovery_is_cooled(queue):
            return {"queued": 0, "lane": "legacy_upgrade", "reason": "discovery_cooldown"}

        service = NoteGenerationService(
            queue.session,
            StorageLayout(settings.storage_root),
            settings,
        )
        candidates = service.discover(legacy_only=True)
        terminal_keys, terminal_work_ids = _terminal_legacy_jobs(queue)
        queued = 0
        skipped = 0
        blocked_failed = 0
        errors: list[dict[str, str]] = []
        for candidate in candidates:
            if queued >= bounded_limit:
                break
            if (
                candidate.work_id is not None
                and candidate.work_id in terminal_work_ids
            ) or f"note:{candidate.aweme_id.strip()}" in terminal_keys:
                blocked_failed += 1
                continue
            try:
                queue.create(
                    kind="generate_note",
                    work_id=candidate.work_id,
                    payload={
                        "aweme_id": candidate.aweme_id,
                        "force": False,
                        "legacy_only": True,
                        "provider_policy": "local",
                    },
                    queue_lane="legacy_upgrade",
                )
            except DuplicateActiveJobError:
                skipped += 1
                continue
            except Exception as exc:  # noqa: BLE001
                errors.append({"aweme_id": candidate.aweme_id, "error": str(exc)[:240]})
                continue
            queued += 1
        # Completed batches can refill immediately.  An empty or permanently
        # blocked inventory needs a bounded scan interval across all workers.
        if not queued:
            _cool_discovery(queue)
        return {
            "queued": queued,
            "skipped": skipped,
            "blocked_failed": blocked_failed,
            "errors": errors,
            "lane": "legacy_upgrade",
            "reason": "queued" if queued else "no_candidates",
            "enabled": True,
            "limit": bounded_limit,
        }
    finally:
        _release_refresh_lock(queue, lock_token)


def _discovery_is_cooled(queue: JobQueue) -> bool:
    setting = queue.session.execute(
        select(Setting)
        .where(Setting.key == DISCOVERY_SETTING_KEY)
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()
    if setting is None:
        return False
    try:
        return float(json.loads(setting.value_json)) > datetime.now(UTC).timestamp()
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def _cool_discovery(queue: JobQueue) -> None:
    setting = queue.session.get(Setting, DISCOVERY_SETTING_KEY)
    value = json.dumps(datetime.now(UTC).timestamp() + DISCOVERY_INTERVAL_SECONDS)
    if setting is None:
        queue.session.add(Setting(key=DISCOVERY_SETTING_KEY, value_json=value))
    else:
        setting.value_json = value
    queue.session.commit()


def _acquire_refresh_lock(queue: JobQueue) -> str | None:
    """Acquire a short durable CAS lock for refill/discovery.

    The lock lives in the existing settings table so the schema migration stays
    backward compatible.  The value contains an owner token and expiry; a
    crashed process therefore cannot block refresh forever.
    """

    now = datetime.now(UTC)
    token = uuid.uuid4().hex
    value = json.dumps({"owner": token, "expires_at": (now.timestamp() + LOCK_TTL_SECONDS)})
    session = queue.session
    setting = session.execute(
        select(Setting)
        .where(Setting.key == LOCK_SETTING_KEY)
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()
    if setting is None:
        try:
            session.add(Setting(key=LOCK_SETTING_KEY, value_json=value))
            session.commit()
            return token
        except IntegrityError:
            session.rollback()
            return None
    old_value = setting.value_json
    try:
        current = json.loads(old_value or "{}")
        expires_at = float(current.get("expires_at", 0)) if isinstance(current, dict) else 0
    except (TypeError, ValueError, json.JSONDecodeError):
        expires_at = 0
    if expires_at > now.timestamp():
        session.rollback()
        return None
    result = session.execute(
        update(Setting)
        .where(Setting.key == LOCK_SETTING_KEY, Setting.value_json == old_value)
        .values(value_json=value)
    )
    session.commit()
    return token if result.rowcount else None


def _release_refresh_lock(queue: JobQueue, token: str) -> None:
    value = json.dumps({"owner": token})
    session = queue.session
    try:
        setting = session.execute(
            select(Setting).where(Setting.key == LOCK_SETTING_KEY)
        ).scalar_one_or_none()
        if setting is not None:
            current = json.loads(setting.value_json or "{}")
            if isinstance(current, dict) and current.get("owner") == token:
                session.execute(
                    update(Setting)
                    .where(
                        Setting.key == LOCK_SETTING_KEY,
                        Setting.value_json == setting.value_json,
                    )
                    .values(value_json=value)
                )
        session.commit()
    except Exception:
        session.rollback()


def _runnable_new_note_exists(queue: JobQueue) -> bool:
    now = datetime.now(UTC)
    jobs = queue.session.scalars(
        select(Job).where(
            Job.kind == "generate_note",
            Job.status == "pending",
            (Job.next_retry_at.is_(None) | (Job.next_retry_at <= now)),
        )
    ).all()
    for job in jobs:
        if job.queue_lane == "legacy_upgrade":
            continue
        if job.queue_lane == "new_note":
            return True
        try:
            payload = json.loads(job.payload_json or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        if payload.get("legacy_only") is not True:
            return True
    return False


def _terminal_legacy_jobs(queue: JobQueue) -> tuple[set[str], set[int]]:
    """Return legacy jobs that reached a terminal failure.

    A failed upgrade is deliberately not recreated by an idle poll or refresh.
    Operators can still resume the existing row explicitly, which preserves
    the failure evidence and avoids silently starting a duplicate generation.
    """

    rows = queue.session.scalars(
        select(Job).where(Job.kind == "generate_note", Job.status == "failed")
    ).all()
    keys: set[str] = set()
    work_ids: set[int] = set()
    for job in rows:
        if JobQueue._effective_lane(job) != "legacy_upgrade":
            continue
        if job.dedupe_key:
            keys.add(job.dedupe_key)
        if job.work_id is not None:
            work_ids.add(job.work_id)
        else:
            payload = JobQueue._payload_dict(job.payload_json)
            aweme_id = payload.get("aweme_id")
            if isinstance(aweme_id, str) and aweme_id.strip():
                keys.add(f"note:{aweme_id.strip()}")
    return keys, work_ids


def _legacy_batch_active(queue: JobQueue) -> bool:
    jobs = queue.session.scalars(
        select(Job).where(
            Job.kind == "generate_note",
            Job.status.in_(("pending", "running", "paused")),
        )
    ).all()
    return any(JobQueue._effective_lane(job) == "legacy_upgrade" for job in jobs)


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "SETTING_KEY",
    "legacy_upgrade_enabled",
    "refresh_legacy_upgrade",
    "set_legacy_upgrade_enabled",
]
