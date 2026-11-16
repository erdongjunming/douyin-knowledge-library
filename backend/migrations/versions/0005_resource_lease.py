"""Add the process-shared resource lease table."""

from collections.abc import Sequence

from alembic import op

from douyin_library.models import ResourceLease

revision: str = "0005_resource_lease"
down_revision: str | None = "0004_job_heartbeat"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ``0001_initial`` creates the current metadata on a fresh database, so
    # this migration must be harmless when the table is already present.
    ResourceLease.__table__.create(op.get_bind(), checkfirst=True)


def downgrade() -> None:
    ResourceLease.__table__.drop(op.get_bind(), checkfirst=True)
