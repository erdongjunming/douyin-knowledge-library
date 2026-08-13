from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Creator(Base):
    __tablename__ = "creator"

    id: Mapped[int] = mapped_column(primary_key=True)
    sec_uid: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    nickname: Mapped[str | None] = mapped_column(String(200))
    unique_id: Mapped[str | None] = mapped_column(String(120))
    profile_url: Mapped[str | None] = mapped_column(Text)
    avatar_url: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="active")
    sync_interval: Mapped[str | None] = mapped_column(String(32))
    last_synced_at: Mapped[datetime | None]


class Work(Base):
    __tablename__ = "work"

    id: Mapped[int] = mapped_column(primary_key=True)
    aweme_id: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    creator_id: Mapped[int | None] = mapped_column(ForeignKey("creator.id"))
    media_type: Mapped[str] = mapped_column(String(32), default="video")
    description: Mapped[str | None] = mapped_column(Text)
    share_url: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None]
    duration_ms: Mapped[int | None]
    availability: Mapped[str] = mapped_column(String(32), default="available")
    stats_json: Mapped[str | None] = mapped_column(Text)
    raw_metadata_json: Mapped[str | None] = mapped_column(Text)

    sources: Mapped[list[WorkSource]] = relationship(
        back_populates="work", cascade="all, delete-orphan"
    )


class Source(Base):
    __tablename__ = "source"
    __table_args__ = (UniqueConstraint("kind", "external_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(40), index=True)
    external_id: Mapped[str] = mapped_column(String(240))
    name: Mapped[str | None] = mapped_column(String(240))

    works: Mapped[list[WorkSource]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )


class WorkSource(Base):
    __tablename__ = "work_source"
    __table_args__ = (UniqueConstraint("work_id", "source_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    work_id: Mapped[int] = mapped_column(ForeignKey("work.id"), index=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("source.id"), index=True)
    first_seen_at: Mapped[datetime] = mapped_column(default=utc_now)
    last_seen_at: Mapped[datetime] = mapped_column(default=utc_now)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    folder_external_id: Mapped[str | None] = mapped_column(String(160))

    work: Mapped[Work] = relationship(back_populates="sources")
    source: Mapped[Source] = relationship(back_populates="works")


class FavoriteFolder(Base):
    __tablename__ = "favorite_folder"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(160), unique=True)
    name: Mapped[str] = mapped_column(String(240))
    last_synced_at: Mapped[datetime | None]
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class SyncRun(Base):
    __tablename__ = "sync_run"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_kind: Mapped[str] = mapped_column(String(40))
    source_external_id: Mapped[str] = mapped_column(String(240))
    status: Mapped[str] = mapped_column(String(24), default="running")
    discovered_count: Mapped[int] = mapped_column(Integer, default=0)
    cursor_summary_json: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(default=utc_now)
    finished_at: Mapped[datetime | None]
    error_detail: Mapped[str | None] = mapped_column(Text)


class Asset(Base):
    __tablename__ = "asset"
    __table_args__ = (UniqueConstraint("work_id", "kind"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    work_id: Mapped[int] = mapped_column(ForeignKey("work.id"), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    path: Mapped[str] = mapped_column(Text)
    codec: Mapped[str | None] = mapped_column(String(80))
    mime_type: Mapped[str | None] = mapped_column(String(120))
    bytes: Mapped[int | None]
    sha256: Mapped[str | None] = mapped_column(String(64))
    probe_json: Mapped[str | None] = mapped_column(Text)
    verified_at: Mapped[datetime | None]


class Transcript(Base):
    __tablename__ = "transcript"
    __table_args__ = (
        Index(
            "uq_transcript_succeeded",
            "work_id",
            "engine",
            "model",
            unique=True,
            sqlite_where=text("status = 'succeeded'"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    work_id: Mapped[int] = mapped_column(ForeignKey("work.id"), index=True)
    engine: Mapped[str] = mapped_column(String(80))
    model: Mapped[str] = mapped_column(String(120))
    language: Mapped[str | None] = mapped_column(String(20))
    text: Mapped[str | None] = mapped_column(Text)
    segments_json: Mapped[str | None] = mapped_column(Text)
    word_timestamps_json: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), default="pending")
    error: Mapped[str | None] = mapped_column(Text)


class Analysis(Base):
    __tablename__ = "analysis"

    id: Mapped[int] = mapped_column(primary_key=True)
    work_id: Mapped[int] = mapped_column(ForeignKey("work.id"), index=True)
    provider: Mapped[str] = mapped_column(String(80))
    model: Mapped[str] = mapped_column(String(120))
    prompt_version: Mapped[str] = mapped_column(String(40))
    summary: Mapped[str | None] = mapped_column(Text)
    keywords_json: Mapped[str | None] = mapped_column(Text)
    topics_json: Mapped[str | None] = mapped_column(Text)
    knowledge_points_json: Mapped[str | None] = mapped_column(Text)
    evidence_json: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), default="pending")


class Job(Base):
    __tablename__ = "job"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    work_id: Mapped[int | None] = mapped_column(ForeignKey("work.id"))
    sync_run_id: Mapped[int | None] = mapped_column(ForeignKey("sync_run.id"))
    payload_json: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    stage: Mapped[str] = mapped_column(String(24), default="discover")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_retry_at: Mapped[datetime | None]
    error_code: Mapped[str | None] = mapped_column(String(80))
    error_detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utc_now)
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]


class ProcessingAudit(Base):
    __tablename__ = "processing_audit"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int | None] = mapped_column(ForeignKey("job.id"), index=True)
    work_id: Mapped[int | None] = mapped_column(ForeignKey("work.id"), index=True)
    sync_run_id: Mapped[int | None] = mapped_column(ForeignKey("sync_run.id"), index=True)
    operation: Mapped[str] = mapped_column(String(80), index=True)
    processor: Mapped[str] = mapped_column(String(120))
    model: Mapped[str | None] = mapped_column(String(160))
    version: Mapped[str | None] = mapped_column(String(80))
    request_json: Mapped[str] = mapped_column(Text)
    response_json: Mapped[str | None] = mapped_column(Text)
    stdout: Mapped[str | None] = mapped_column(Text)
    stderr: Mapped[str | None] = mapped_column(Text)
    input_files_json: Mapped[str | None] = mapped_column(Text)
    output_files_json: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), default="running", index=True)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(default=utc_now)
    finished_at: Mapped[datetime | None]


class Setting(Base):
    __tablename__ = "setting"

    key: Mapped[str] = mapped_column(String(120), primary_key=True)
    value_json: Mapped[str] = mapped_column(Text)
