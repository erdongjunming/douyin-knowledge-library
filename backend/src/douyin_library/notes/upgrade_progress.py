"""Read standalone upgrade evidence without loading model request/response blobs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from douyin_library.models import ProcessingAudit, ResourceLease, Work


def utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


@dataclass(frozen=True)
class UpgradeAttempt:
    status: str
    started_at: datetime
    error: str | None


def _ids(row: dict) -> set[str]:
    values = row.get("aweme_ids")
    return {v for v in values if isinstance(v, str) and v} if isinstance(values, list) else set()


def standalone_progress(
    session: Session, root: Path,
) -> tuple[set[str], dict[str, UpgradeAttempt]]:
    cohort: set[str] = set()
    first: dict[str, datetime] = {}
    latest_batch: dict[int, tuple[datetime, set[str]]] = {}
    audit_dir = root / "audit"
    for path in audit_dir.glob("legacy-upgrade-inventory*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if isinstance(row, dict) and row.get("schema") == "douyin-legacy-upgrade-inventory-v1":
            cohort.update(_ids(row))
    for path in audit_dir.glob("legacy-upgrade-*.checkpoints.jsonl"):
        try:
            with path.open(encoding="utf-8-sig") as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict) or row.get("event") != "batch_started":
                            continue
                        at = utc(datetime.fromisoformat(row["at"]))
                        pid = row["pid"]
                        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
                            continue
                    except (ValueError, KeyError, TypeError):
                        # A writer may still be appending the final line.
                        continue
                    ids = _ids(row)
                    cohort.update(ids)
                    for aweme_id in ids:
                        first[aweme_id] = min(first.get(aweme_id, at), at)
                    if pid not in latest_batch or at > latest_batch[pid][0]:
                        latest_batch[pid] = (at, ids)
        except (OSError, UnicodeError):
            continue
    if not first:
        return cohort, {}
    owners = set(session.scalars(select(ResourceLease.owner_worker_id).where(
        ResourceLease.expires_at > datetime.now(UTC),
    )))
    live: dict[str, datetime] = {}
    for pid, (at, ids) in latest_batch.items():
        if f"legacy-upgrade-{pid}" in owners:
            for aweme_id in ids:
                live[aweme_id] = max(live.get(aweme_id, at), at)
    rows = session.execute(select(
        Work.aweme_id, ProcessingAudit.status, ProcessingAudit.started_at, ProcessingAudit.error,
    ).join(Work, ProcessingAudit.work_id == Work.id).where(
        ProcessingAudit.operation == "generate_note",
        ProcessingAudit.job_id.is_(None),
        Work.aweme_id.in_(first),
        ProcessingAudit.started_at >= min(first.values()),
    ).order_by(ProcessingAudit.started_at.desc(), ProcessingAudit.id.desc()))
    attempts: dict[str, UpgradeAttempt] = {}
    for aweme_id, status, started_at, error in rows:
        at = utc(started_at)
        if aweme_id in attempts or at < first[aweme_id]:
            continue
        if status == "running" and (aweme_id not in live or at < live[aweme_id]):
            status = "inactive"
        attempts[aweme_id] = UpgradeAttempt(status, at, error)
    return cohort, attempts
