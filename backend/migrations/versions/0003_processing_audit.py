"""Add complete processing request and response audits."""

from collections.abc import Sequence

from alembic import op

from douyin_library.models import ProcessingAudit

revision: str = "0003_processing_audit"
down_revision: str | None = "0002_job_payload"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    ProcessingAudit.__table__.create(op.get_bind(), checkfirst=True)


def downgrade() -> None:
    ProcessingAudit.__table__.drop(op.get_bind(), checkfirst=True)

