"""Create the initial knowledge library schema."""

from collections.abc import Sequence

from alembic import op

from douyin_library.db import create_fts_schema
from douyin_library.models import Base

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    Base.metadata.create_all(op.get_bind())
    create_fts_schema(op.get_bind())


def downgrade() -> None:
    for trigger in (
        "work_fts_work_insert",
        "work_fts_work_update",
        "work_fts_work_delete",
        "work_fts_transcript_insert",
        "work_fts_transcript_update",
        "work_fts_transcript_delete",
        "work_fts_analysis_insert",
        "work_fts_analysis_update",
        "work_fts_analysis_delete",
    ):
        op.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    op.execute("DROP TABLE IF EXISTS work_fts")
    Base.metadata.drop_all(op.get_bind())
