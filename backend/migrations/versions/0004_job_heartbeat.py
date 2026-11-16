"""Add worker ownership and heartbeat timestamps to jobs."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_job_heartbeat"
down_revision: str | None = "0003_processing_audit"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("job")}
    if "heartbeat_at" not in columns:
        op.add_column("job", sa.Column("heartbeat_at", sa.DateTime(), nullable=True))
    if "worker_id" not in columns:
        op.add_column("job", sa.Column("worker_id", sa.String(length=120), nullable=True))
    indexes = {index["name"] for index in inspector.get_indexes("job")}
    if "ix_job_heartbeat_at" not in indexes:
        op.create_index("ix_job_heartbeat_at", "job", ["heartbeat_at"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    indexes = {index["name"] for index in inspector.get_indexes("job")}
    if "ix_job_heartbeat_at" in indexes:
        op.drop_index("ix_job_heartbeat_at", table_name="job")
    columns = {column["name"] for column in inspector.get_columns("job")}
    with op.batch_alter_table("job") as batch_op:
        if "worker_id" in columns:
            batch_op.drop_column("worker_id")
        if "heartbeat_at" in columns:
            batch_op.drop_column("heartbeat_at")
