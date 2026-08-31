from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.db import create_schema
from douyin_library.jobs.runtime import GENERAL_JOB_KINDS, build_worker_runtime
from douyin_library.models import Job, ProcessingAudit, Transcript, Work
from douyin_library.transcription.contracts import (
    CleanedTranscript,
    CleanerResult,
    TranscriptSegment,
)


class FakeCleaner:
    name = "fake-cleaner"
    model = "fake-model"
    prompt_version = "test-v1"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    async def clean(self, transcript, *, cancel_event=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("cleaner_down")
        cleaned = CleanedTranscript(
            segments=tuple(
                TranscriptSegment(s.segment_id, s.start, s.end, s.text + "。")
                for s in transcript.segments
            ),
            text="\n".join(s.text + "。" for s in transcript.segments),
            language=transcript.language,
        )
        return CleanerResult(transcript=cleaned, usable=True)


def _setup(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    storage = tmp_path / "storage"
    with Session(engine) as session:
        work = Work(aweme_id="clean-1")
        session.add(work)
        session.flush()
        raw = Transcript(
            work_id=work.id,
            engine="qwen3-asr",
            model="asr-model",
            source_kind="asr",
            variant="raw",
            text="第一句",
            segments_json=json.dumps(
                [{"start": 0, "end": 1, "text": "第一句"}], ensure_ascii=False
            ),
            status="succeeded",
        )
        session.add(raw)
        session.add(Job(kind="clean_transcript_work", work_id=work.id, status="pending"))
        session.commit()
        return engine, storage, work.id


@pytest.mark.asyncio
async def test_clean_job_publishes_derived_files_without_mutating_raw(tmp_path: Path) -> None:
    engine, storage, work_id = _setup(tmp_path)
    cleaner = FakeCleaner()
    runtime = build_worker_runtime(
        engine,
        Settings(storage_root=storage),
        cleaner_factory=lambda settings: cleaner,
        worker_kinds=GENERAL_JOB_KINDS,
    )
    try:
        job = runtime.session.scalar(select(Job).where(Job.kind == "clean_transcript_work"))
        assert job is not None
        assert await runtime.handlers.run_one(kinds=GENERAL_JOB_KINDS) is True
        cleaned = runtime.session.scalar(select(Transcript).where(Transcript.variant == "cleaned"))
        raw = runtime.session.scalar(select(Transcript).where(Transcript.variant == "raw"))
        assert cleaned is not None and raw is not None
        assert cleaned.parent_transcript_id == raw.id
        assert raw.text == "第一句"
        assert (storage / "transcript" / "clean-1" / "transcript.cleaned.md").read_text(
            encoding="utf-8"
        ) == "第一句。"
        assert runtime.queue.get(job.id).status == "succeeded"
        audit = runtime.session.scalar(
            select(ProcessingAudit).where(ProcessingAudit.operation == "clean_transcript_work")
        )
        assert audit is not None and audit.status == "succeeded"
    finally:
        runtime.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_cleaner_failure_keeps_raw_and_allows_note_fallback(tmp_path: Path) -> None:
    engine, storage, work_id = _setup(tmp_path)
    cleaner = FakeCleaner(fail=True)
    runtime = build_worker_runtime(
        engine,
        Settings(storage_root=storage),
        cleaner_factory=lambda settings: cleaner,
        worker_kinds=GENERAL_JOB_KINDS,
    )
    try:
        job = runtime.session.scalar(select(Job).where(Job.kind == "clean_transcript_work"))
        assert job is not None
        await runtime.handlers.run_one(kinds=GENERAL_JOB_KINDS)
        assert runtime.queue.get(job.id).status == "pending"
        assert (
            runtime.session.scalar(select(Transcript).where(Transcript.variant == "cleaned"))
            is None
        )
        assert (
            runtime.session.scalar(select(Transcript).where(Transcript.variant == "raw"))
            is not None
        )
    finally:
        runtime.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_partial_clean_publication_removes_staged_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, storage, _ = _setup(tmp_path)
    cleaner = FakeCleaner()
    import douyin_library.transcription.service as service_module

    original_replace = service_module.os.replace

    def fail_replace(source, target):
        if str(source).endswith(".md"):
            raise OSError("replace_failed")
        return original_replace(source, target)

    monkeypatch.setattr(service_module.os, "replace", fail_replace)
    runtime = build_worker_runtime(
        engine,
        Settings(storage_root=storage, transcription_cleaning_enabled=True),
        cleaner_factory=lambda settings: cleaner,
        worker_kinds=GENERAL_JOB_KINDS,
    )
    try:
        await runtime.handlers.run_one(kinds=GENERAL_JOB_KINDS)
        transcript_dir = storage / "transcript" / "clean-1"
        assert list(transcript_dir.glob(".*transcript.cleaned*")) == []
    finally:
        runtime.close()
        engine.dispose()
