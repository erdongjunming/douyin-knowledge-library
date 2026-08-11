"""Add persistent command payloads to jobs."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_job_payload"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("job")}
    if "payload_json" not in columns:
        op.add_column("job", sa.Column("payload_json", sa.Text(), nullable=True))


def downgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("job")}
    if "payload_json" in columns:
        op.drop_column("job", "payload_json")
