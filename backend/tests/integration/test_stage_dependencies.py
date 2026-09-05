from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.db import create_schema
from douyin_library.jobs.runtime import ASR_JOB_KINDS, GENERAL_JOB_KINDS, build_worker_runtime
from douyin_library.models import Asset, Job, Transcript, Work
from douyin_library.transcription.contracts import (
    CleanedTranscript,
    CleanerResult,
    TranscriptSegment,
)
from douyin_library.transcription.service import SubprocessResult


class FakeTranscriber:
    async def run(self, *, audio_path: Path, result_path: Path) -> SubprocessResult:
        del audio_path
        result_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "text": "第一句",
            "segments": [{"id": "s1", "start": 0.0, "end": 1.0, "text": "第一句"}],
        }
        result_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return SubprocessResult(returncode=0, stdout="", stderr="")


class FakeCleaner:
    model = "qwen3:4b"
    prompt_version = "v1"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    async def clean(self, transcript, *, cancel_event=None):
        del cancel_event
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
    audio_path = storage / "audio" / "stage-1.m4a"
    audio_path.parent.mkdir(parents=True)
    audio = b"audio"
    audio_path.write_bytes(audio)
    with Session(engine) as session:
        work = Work(aweme_id="stage-1")
        session.add(work)
        session.flush()
        session.add(
            Asset(
                work_id=work.id,
                kind="audio",
                path=str(audio_path),
                bytes=len(audio),
                sha256=hashlib.sha256(audio).hexdigest(),
            )
        )
        session.commit()
        return engine, storage, work.id


@pytest.mark.asyncio
async def test_raw_success_waits_for_cleaner_before_note_enqueue(tmp_path: Path) -> None:
    engine, storage, work_id = _setup(tmp_path)
    runtime = build_worker_runtime(
        engine,
        Settings(storage_root=storage, transcription_cleaning_enabled=True),
        transcriber_factory=lambda settings: FakeTranscriber(),
        cleaner_factory=lambda settings: FakeCleaner(),
        worker_kinds=ASR_JOB_KINDS | GENERAL_JOB_KINDS,
    )
    try:
        runtime.queue.create(kind="transcribe_work", work_id=work_id)
        assert await runtime.handlers.run_one(kinds=ASR_JOB_KINDS) is True
        assert runtime.session.scalar(
            select(Job).where(Job.kind == "clean_transcript_work", Job.work_id == work_id)
        ) is not None
        assert runtime.session.scalar(
            select(Job).where(Job.kind == "generate_note", Job.work_id == work_id)
        ) is None

        assert await runtime.handlers.run_one(kinds={"clean_transcript_work"}) is True
        note_jobs = runtime.session.scalars(
            select(Job).where(Job.kind == "generate_note", Job.work_id == work_id)
        ).all()
        assert len(note_jobs) == 1
    finally:
        runtime.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_cleaner_failure_keeps_raw_and_does_not_enqueue_note(tmp_path: Path) -> None:
    engine, storage, work_id = _setup(tmp_path)
    runtime = build_worker_runtime(
        engine,
        Settings(storage_root=storage, transcription_cleaning_enabled=True),
        transcriber_factory=lambda settings: FakeTranscriber(),
        cleaner_factory=lambda settings: FakeCleaner(fail=True),
        worker_kinds=ASR_JOB_KINDS | GENERAL_JOB_KINDS,
    )
    try:
        runtime.queue.create(kind="transcribe_work", work_id=work_id)
        await runtime.handlers.run_one(kinds=ASR_JOB_KINDS)
        clean_job = runtime.session.scalar(
            select(Job).where(Job.kind == "clean_transcript_work", Job.work_id == work_id)
        )
        assert clean_job is not None
        await runtime.handlers.run_one(kinds={"clean_transcript_work"})
        assert runtime.session.scalar(
            select(Transcript).where(
                Transcript.work_id == work_id,
                Transcript.variant == "raw",
                Transcript.status == "succeeded",
            )
        ) is not None
        assert runtime.session.scalar(
            select(Job).where(Job.kind == "generate_note", Job.work_id == work_id)
        ) is None
    finally:
        runtime.close()
        engine.dispose()
