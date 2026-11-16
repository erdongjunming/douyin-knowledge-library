"""Persist note queue lanes, canonical dedupe keys, and claim fences."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op

revision: str = "0007_note_queue_lane"
down_revision: str | None = "0006_transcript_variants"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ACTIVE_STATUSES = {"pending", "running", "paused"}


def _columns() -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns("job")}


def _indexes() -> set[str]:
    return {index["name"] for index in sa.inspect(op.get_bind()).get_indexes("job")}


def upgrade() -> None:
    bind = op.get_bind()
    if not sa.inspect(bind).has_table("job"):
        return
    columns = _columns()
    if "queue_lane" not in columns:
        op.add_column(
            "job",
            sa.Column("queue_lane", sa.String(length=24), nullable=True),
        )
    if "dedupe_key" not in columns:
        op.add_column(
            "job",
            sa.Column("dedupe_key", sa.String(length=200), nullable=True),
        )
    if "claim_fence" not in columns:
        op.add_column(
            "job",
            sa.Column("claim_fence", sa.Integer(), nullable=True, server_default=sa.text("0")),
        )

    rows = bind.execute(
        sa.text(
            "SELECT id, kind, work_id, payload_json, status, queue_lane, dedupe_key, "
            "claim_fence FROM job ORDER BY id"
        )
    ).mappings().all()
    aweme_by_work = {
        row["id"]: row["aweme_id"]
        for row in bind.execute(sa.text("SELECT id, aweme_id FROM work")).mappings()
    }
    seen_active: dict[str, int] = {}
    now = datetime.now(UTC)
    for row in rows:
        payload: dict[str, object] = {}
        try:
            value = json.loads(row["payload_json"] or "{}")
            if isinstance(value, dict):
                payload = value
        except (TypeError, ValueError):
            pass
        lane = row["queue_lane"]
        if lane not in {"pipeline", "new_note", "legacy_upgrade"}:
            if row["kind"] == "generate_note":
                lane = "legacy_upgrade" if payload.get("legacy_only") is True else "new_note"
            else:
                lane = "pipeline"
        dedupe_key = row["dedupe_key"]
        if dedupe_key is None and row["kind"] == "generate_note":
            aweme_id = payload.get("aweme_id") or aweme_by_work.get(row["work_id"])
            if aweme_id is not None and str(aweme_id).strip():
                dedupe_key = f"note:{str(aweme_id).strip()}"
        fence = int(row["claim_fence"] or 0)
        values: dict[str, object] = {
            "queue_lane": lane,
            "dedupe_key": dedupe_key,
            "claim_fence": fence,
        }
        # A pre-migration database may contain two active jobs for one note.
        # Keep the oldest row canonical and cancel later duplicates while
        # advancing their fence to fence out any late worker.
        if dedupe_key is not None and row["status"] in ACTIVE_STATUSES:
            original = seen_active.get(dedupe_key)
            if original is None:
                seen_active[dedupe_key] = row["id"]
            else:
                values.update(
                    status="cancelled",
                    error_code="dedupe_migration_conflict",
                    error_detail=f"duplicate_active_note:{original}",
                    finished_at=now,
                    next_retry_at=None,
                    started_at=None,
                    heartbeat_at=None,
                    worker_id=None,
                    claim_fence=fence + 1,
                )
        assignments = ", ".join(f"{name} = :{name}" for name in values)
        params = {**values, "id": row["id"]}
        bind.execute(
            sa.text(f"UPDATE job SET {assignments} WHERE id = :id"),
            params,
        )
    indexes = _indexes()
    if "ix_job_queue_lane" not in indexes:
        op.create_index("ix_job_queue_lane", "job", ["queue_lane"])
    if "ix_job_dedupe_key" not in indexes:
        op.create_index("ix_job_dedupe_key", "job", ["dedupe_key"])
    if "uq_job_active_note_dedupe" not in indexes:
        op.create_index(
            "uq_job_active_note_dedupe",
            "job",
            ["dedupe_key"],
            unique=True,
            sqlite_where=sa.text(
                "dedupe_key IS NOT NULL AND status IN ('pending', 'running', 'paused')"
            ),
        )
    if "uq_job_running_legacy" not in indexes:
        op.create_index(
            "uq_job_running_legacy",
            "job",
            ["queue_lane"],
            unique=True,
            sqlite_where=sa.text("queue_lane = 'legacy_upgrade' AND status = 'running'"),
        )


def downgrade() -> None:
    if not sa.inspect(op.get_bind()).has_table("job"):
        return
    indexes = _indexes()
    for name in (
        "uq_job_running_legacy",
        "uq_job_active_note_dedupe",
        "ix_job_dedupe_key",
        "ix_job_queue_lane",
    ):
        if name in indexes:
            op.drop_index(name, table_name="job")
    columns = _columns()
    with op.batch_alter_table("job") as batch_op:
        if "claim_fence" in columns:
            batch_op.drop_column("claim_fence")
        if "dedupe_key" in columns:
            batch_op.drop_column("dedupe_key")
        if "queue_lane" in columns:
            batch_op.drop_column("queue_lane")

