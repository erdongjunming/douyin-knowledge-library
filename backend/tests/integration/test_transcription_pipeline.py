import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.db import create_schema
from douyin_library.jobs.runtime import build_worker_runtime
from douyin_library.models import Asset, ProcessingAudit, Transcript, Work
from douyin_library.transcription.service import SubprocessResult, TranscriptionService


class NoCookieStore:
    def read(self) -> str | None:
        raise AssertionError("transcription must not read the Douyin cookie")


class FakeTranscriber:
    async def run(self, *, audio_path: Path, result_path: Path) -> SubprocessResult:
        payload = {
            "model": "Qwen/Qwen3-ASR-0.6B",
            "backend": "qwen3-asr",
            "language": "Chinese",
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "第一段知识。"},
                {"start": 2.0, "end": 4.0, "text": "第二段理解。"},
            ],
            "text": "第一段知识。\n第二段理解。",
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return SubprocessResult(returncode=0, stdout="full stdout", stderr="full stderr")


@pytest.mark.asyncio
async def test_transcribe_job_writes_assets_database_audit_and_fts(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    storage = tmp_path / "storage"
    audio_path = storage / "audio" / "731.m4a"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"existing-audio")
    with Session(engine) as setup_session:
        work = Work(aweme_id="731", description="口述知识")
        setup_session.add(work)
        setup_session.flush()
        work_id = work.id
        setup_session.add(
            Asset(
                work_id=work.id,
                kind="audio",
                path=str(audio_path),
                bytes=audio_path.stat().st_size,
                sha256="audio-sha",
            )
        )
        setup_session.commit()

    runtime = build_worker_runtime(
        engine,
        Settings(storage_root=storage),
        cookie_store=NoCookieStore(),
        transcriber_factory=lambda settings: FakeTranscriber(),
    )
    try:
        job = runtime.queue.create(kind="transcribe_work", work_id=work_id)

        assert await runtime.handlers.run_one() is True

        transcript = runtime.session.scalar(select(Transcript))
        audit = runtime.session.scalar(select(ProcessingAudit))
        assets = runtime.session.scalars(
            select(Asset).where(Asset.work_id == work_id).order_by(Asset.kind)
        ).all()
        assert runtime.queue.get(job.id).status == "succeeded"
        assert transcript is not None
        assert transcript.status == "succeeded"
        assert transcript.text == "第一段知识。\n第二段理解。"
        assert json.loads(transcript.segments_json or "[]")[1]["start"] == 2.0
        assert audit is not None
        assert audit.status == "succeeded"
        assert audit.stdout == "full stdout"
        assert audit.stderr == "full stderr"
        assert {asset.kind for asset in assets} == {
            "audio",
            "transcript_segments",
            "transcript_text",
        }
        assert (storage / "transcript" / "731" / "transcript.txt").read_text(
            encoding="utf-8"
        ) == transcript.text
        indexed = runtime.session.execute(
            text("SELECT aweme_id FROM work_fts WHERE work_fts MATCH :query"),
            {"query": "第二段理"},
        ).scalar_one()
        assert indexed == "731"
    finally:
        runtime.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_existing_successful_transcript_is_reused_without_subprocess(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    storage = tmp_path / "storage"
    audio_path = storage / "audio" / "732.m4a"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"existing-audio")
    with Session(engine) as setup_session:
        work = Work(aweme_id="732")
        setup_session.add(work)
        setup_session.flush()
        work_id = work.id
        setup_session.add_all(
            [
                Asset(work_id=work.id, kind="audio", path=str(audio_path)),
                Transcript(
                    work_id=work.id,
                    engine="qwen3-asr",
                    model="Qwen/Qwen3-ASR-0.6B",
                    text="已经存在",
                    status="succeeded",
                ),
            ]
        )
        setup_session.commit()

    class MustNotRun:
        async def run(self, **kwargs):
            raise AssertionError("existing transcript should be reused")

    runtime = build_worker_runtime(
        engine,
        Settings(storage_root=storage),
        cookie_store=NoCookieStore(),
        transcriber_factory=lambda settings: MustNotRun(),
    )
    try:
        runtime.queue.create(kind="transcribe_work", work_id=work_id)
        assert await runtime.handlers.run_one() is True
        audits = runtime.session.scalars(select(ProcessingAudit)).all()
        assert len(audits) == 1
        assert json.loads(audits[0].response_json)["reused_existing"] is True
    finally:
        runtime.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_output_failure_rolls_back_successful_transcript_but_keeps_failed_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    storage = tmp_path / "storage"
    audio_path = storage / "audio" / "733.m4a"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"existing-audio")
    with Session(engine) as setup_session:
        work = Work(aweme_id="733")
        setup_session.add(work)
        setup_session.flush()
        work_id = work.id
        setup_session.add(Asset(work_id=work.id, kind="audio", path=str(audio_path)))
        setup_session.commit()

    def fail_write(self, work, payload):
        raise OSError("disk-write-failed")

    monkeypatch.setattr(TranscriptionService, "_write_outputs", fail_write)
    runtime = build_worker_runtime(
        engine,
        Settings(storage_root=storage),
        cookie_store=NoCookieStore(),
        transcriber_factory=lambda settings: FakeTranscriber(),
    )
    try:
        job = runtime.queue.create(kind="transcribe_work", work_id=work_id)
        assert await runtime.handlers.run_one() is True
        assert runtime.queue.get(job.id).status == "pending"
        assert runtime.session.scalar(select(Transcript)) is None
        audit = runtime.session.scalar(select(ProcessingAudit))
        assert audit is not None
        assert audit.status == "failed"
        assert audit.error == "disk-write-failed"
    finally:
        runtime.close()
        engine.dispose()
