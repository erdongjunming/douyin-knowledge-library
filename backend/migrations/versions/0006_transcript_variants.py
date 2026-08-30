"""Persist raw and cleaned transcript variants."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from douyin_library.db import FTS_DOCUMENT_SQL, FTS_TRIGGER_NAMES, recreate_fts_schema

revision: str = "0006_transcript_variants"
down_revision: str | None = "0005_resource_lease"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_LEGACY_FTS_DOCUMENT_SQL = """
INSERT OR REPLACE INTO work_fts(
    rowid, aweme_id, description, transcript, summary, keywords, topics
)
SELECT
    work.id,
    work.aweme_id,
    COALESCE(work.description, ''),
    COALESCE((
        SELECT group_concat(transcript.text, ' ')
        FROM transcript
        WHERE transcript.work_id = work.id
          AND transcript.status = 'succeeded'
    ), ''),
    COALESCE((
        SELECT group_concat(analysis.summary, ' ')
        FROM analysis
        WHERE analysis.work_id = work.id
          AND analysis.status = 'succeeded'
    ), ''),
    COALESCE((
        SELECT group_concat(analysis.keywords_json, ' ')
        FROM analysis
        WHERE analysis.work_id = work.id
          AND analysis.status = 'succeeded'
    ), ''),
    COALESCE((
        SELECT group_concat(analysis.topics_json, ' ')
        FROM analysis
        WHERE analysis.work_id = work.id
          AND analysis.status = 'succeeded'
    ), '')
FROM work
"""


def _columns() -> set[str]:
    return {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns("transcript")
    }


def _indexes() -> set[str]:
    return {
        index["name"]
        for index in sa.inspect(op.get_bind()).get_indexes("transcript")
    }


def _rebuild_fts(document_sql: str) -> None:
    bind = op.get_bind()
    recreate_fts_schema(bind, document_sql=document_sql)
    bind.exec_driver_sql("DELETE FROM work_fts")
    bind.exec_driver_sql(document_sql)


def upgrade() -> None:
    columns = _columns()
    if "source_kind" not in columns:
        op.add_column(
            "transcript",
            sa.Column(
                "source_kind",
                sa.String(length=24),
                nullable=False,
                server_default=sa.text("'asr'"),
            ),
        )
    if "variant" not in columns:
        op.add_column(
            "transcript",
            sa.Column(
                "variant",
                sa.String(length=24),
                nullable=False,
                server_default=sa.text("'raw'"),
            ),
        )
    if "parent_transcript_id" not in columns:
        with op.batch_alter_table("transcript") as batch_op:
            batch_op.add_column(
                sa.Column(
                    "parent_transcript_id",
                    sa.Integer(),
                    sa.ForeignKey(
                        "transcript.id",
                        name="fk_transcript_parent_transcript_id_transcript",
                    ),
                    nullable=True,
                )
            )

    op.execute(
        "UPDATE transcript SET source_kind = 'asr' "
        "WHERE source_kind IS NULL OR trim(source_kind) = ''"
    )
    op.execute(
        "UPDATE transcript SET variant = 'raw' "
        "WHERE variant IS NULL OR trim(variant) = ''"
    )

    indexes = _indexes()
    if "uq_transcript_succeeded" in indexes:
        op.drop_index("uq_transcript_succeeded", table_name="transcript")
    op.create_index(
        "uq_transcript_succeeded",
        "transcript",
        ["work_id", "engine", "model", "variant"],
        unique=True,
        sqlite_where=sa.text("status = 'succeeded'"),
    )
    indexes = _indexes()
    if "ix_transcript_parent_transcript_id" not in indexes:
        op.create_index(
            "ix_transcript_parent_transcript_id",
            "transcript",
            ["parent_transcript_id"],
        )
    if "ix_transcript_work_variant_status" not in indexes:
        op.create_index(
            "ix_transcript_work_variant_status",
            "transcript",
            ["work_id", "variant", "status"],
        )
    _rebuild_fts(FTS_DOCUMENT_SQL)


def downgrade() -> None:
    for name in FTS_TRIGGER_NAMES:
        op.execute(f"DROP TRIGGER IF EXISTS {name}")
    indexes = _indexes()
    for name in (
        "ix_transcript_work_variant_status",
        "ix_transcript_parent_transcript_id",
        "uq_transcript_succeeded",
    ):
        if name in indexes:
            op.drop_index(name, table_name="transcript")
    op.create_index(
        "uq_transcript_succeeded",
        "transcript",
        ["work_id", "engine", "model"],
        unique=True,
        sqlite_where=sa.text("status = 'succeeded'"),
    )
    with op.batch_alter_table("transcript") as batch_op:
        batch_op.drop_column("parent_transcript_id")
        batch_op.drop_column("variant")
        batch_op.drop_column("source_kind")
    _rebuild_fts(_LEGACY_FTS_DOCUMENT_SQL)
