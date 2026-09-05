import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select, text, update
from sqlalchemy.orm import Session

from douyin_library.audit import ProcessingAuditRecorder
from douyin_library.config import Settings
from douyin_library.db import create_schema
from douyin_library.jobs.control import JobCancelledError
from douyin_library.jobs.leases import ResourceLeaseStore
from douyin_library.jobs.queue import JobQueue
from douyin_library.jobs.runtime import build_worker_runtime
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Asset, Job, ProcessingAudit, ResourceLease, Transcript, Work
from douyin_library.transcription.contracts import TranscriptDraft, TranscriptSegment
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


class NoOutputTranscriber:
    """Simulate a successful subprocess that forgot to write its output."""

    async def run(self, *, audio_path: Path, result_path: Path) -> SubprocessResult:
        del audio_path, result_path
        return SubprocessResult(returncode=0, stdout="", stderr="")


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
                    sha256=hashlib.sha256(b"existing-audio").hexdigest(),
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
        note_job = runtime.session.scalar(
            select(Job).where(Job.kind == "generate_note", Job.work_id == work_id)
        )
        assert note_job is not None
        assert note_job.status == "pending"
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
                    source_kind="asr",
                    variant="raw",
                    text="已经存在",
                    segments_json=json.dumps(
                        [{"id": "s1", "start": 0.0, "end": 1.0, "text": "已经存在"}],
                        ensure_ascii=False,
                    ),
                    status="succeeded",
                ),
            ]
        )
        setup_session.commit()
    transcript_dir = storage / "transcript" / "732"
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "transcript.txt").write_text("已经存在", encoding="utf-8")
    segments = [{"id": "s1", "start": 0.0, "end": 1.0, "text": "已经存在"}]
    (transcript_dir / "segments.json").write_text(
        json.dumps(segments, ensure_ascii=False),
        encoding="utf-8",
    )
    (transcript_dir / "qwen3_asr.json").write_text(
        json.dumps({"text": "已经存在", "segments": segments}, ensure_ascii=False),
        encoding="utf-8",
    )

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
async def test_existing_raw_asr_wins_before_late_subtitle_discovery(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    storage = tmp_path / "storage"
    audio_path = storage / "audio" / "late-subtitle.m4a"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"existing-audio")
    raw_text = "已有原始转写"
    raw_segments = [{"id": "s1", "start": 0.0, "end": 1.0, "text": raw_text}]
    with Session(engine) as session:
        work = Work(aweme_id="late-subtitle")
        session.add(work)
        session.flush()
        session.add(
            Asset(
                work_id=work.id,
                kind="audio",
                path=str(audio_path),
                bytes=audio_path.stat().st_size,
                sha256="audio-sha",
            )
        )
        raw = Transcript(
            work_id=work.id,
            engine="qwen3-asr",
            model="Qwen/Qwen3-ASR-0.6B",
            source_kind="asr",
            variant="raw",
            text=raw_text,
            segments_json=json.dumps(raw_segments, ensure_ascii=False),
            status="succeeded",
        )
        session.add(raw)
        session.commit()
        work_id = work.id
    transcript_dir = storage / "transcript" / "late-subtitle"
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "transcript.txt").write_text(raw_text, encoding="utf-8")
    (transcript_dir / "segments.json").write_text(
        json.dumps(raw_segments, ensure_ascii=False),
        encoding="utf-8",
    )
    (transcript_dir / "qwen3_asr.json").write_text(
        json.dumps({"text": raw_text, "segments": raw_segments}, ensure_ascii=False),
        encoding="utf-8",
    )

    class LateSubtitle:
        def discover(self, work: Work) -> TranscriptDraft:
            return TranscriptDraft(
                segments=(TranscriptSegment("s1", 0.0, 1.0, "后来的字幕"),),
                text="后来的字幕",
                language="zh",
            )

    class MustNotRun:
        async def run(self, **kwargs):
            raise AssertionError("existing raw ASR must win before subtitle discovery")

    with Session(engine) as session:
        queue = JobQueue(session)
        job = queue.create(kind="transcribe_work", work_id=work_id)
        queue.claim(job.id)
        work = session.get(Work, work_id)
        result = await TranscriptionService(
            session,
            StorageLayout(storage),
            Settings(storage_root=storage),
            MustNotRun(),
            subtitle_provider=LateSubtitle(),
        ).process(job, work)
        assert result.text == raw_text
        assert (
            session.scalar(select(Transcript).where(Transcript.work_id == work_id)).text
            == raw_text
        )
    engine.dispose()


@pytest.mark.asyncio
async def test_audio_hash_mismatch_aborts_before_qwen_and_preserves_source(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    storage = tmp_path / "storage"
    audio_path = storage / "audio" / "changed.m4a"
    audio_path.parent.mkdir(parents=True)
    original = b"original-audio"
    audio_path.write_bytes(original)
    with Session(engine) as session:
        work = Work(aweme_id="changed")
        session.add(work)
        session.flush()
        session.add(
            Asset(
                work_id=work.id,
                kind="audio",
                path=str(audio_path),
                bytes=len(original),
                sha256="wrong-hash",
            )
        )
        session.commit()
        work_id = work.id

    class MustNotRun:
        async def run(self, **kwargs):
            raise AssertionError("audio mismatch must stop before Qwen")

    with Session(engine) as session:
        queue = JobQueue(session)
        job = queue.create(kind="transcribe_work", work_id=work_id)
        queue.claim(job.id)
        work = session.get(Work, work_id)
        with pytest.raises(RuntimeError, match="audio_asset_changed"):
            await TranscriptionService(
                session,
                StorageLayout(storage),
                Settings(storage_root=storage),
                MustNotRun(),
            ).process(job, work)
        assert audio_path.read_bytes() == original
    engine.dispose()


@pytest.mark.asyncio
async def test_existing_cleaned_transcript_is_not_reused_as_asr_raw(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    storage = tmp_path / "storage"
    audio_path = storage / "audio" / "732-cleaned.m4a"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"existing-audio")
    with Session(engine) as setup_session:
        work = Work(aweme_id="732-cleaned")
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
                    source_kind="asr",
                    variant="raw",
                    text="原始转录",
                    status="failed",
                ),
            ]
        )
        setup_session.flush()
        raw = setup_session.scalar(select(Transcript).where(Transcript.work_id == work.id))
        setup_session.add(
            Transcript(
                work_id=work.id,
                engine="qwen3-asr",
                model="Qwen/Qwen3-ASR-0.6B",
                source_kind="asr",
                variant="cleaned",
                parent_transcript_id=raw.id,
                text="清洗转录",
                status="succeeded",
            )
        )
        setup_session.commit()

    class WritesFreshRaw:
        async def run(self, *, audio_path: Path, result_path: Path) -> SubprocessResult:
            del audio_path
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps({"text": "新原始转录", "segments": []}),
                encoding="utf-8",
            )
            return SubprocessResult(returncode=0, stdout="", stderr="")

    runtime = build_worker_runtime(
        engine,
        Settings(storage_root=storage),
        cookie_store=NoCookieStore(),
        transcriber_factory=lambda settings: WritesFreshRaw(),
    )
    try:
        runtime.queue.create(kind="transcribe_work", work_id=work_id)
        assert await runtime.handlers.run_one() is True
        current_job = runtime.session.scalar(select(Job).where(Job.work_id == work_id))
        assert current_job is not None, current_job
        assert current_job.status == "succeeded", current_job.error_detail
        raw_rows = runtime.session.scalars(
            select(Transcript).where(
                Transcript.work_id == work_id,
                Transcript.variant == "raw",
                Transcript.status == "succeeded",
            )
        ).all()
        assert len(raw_rows) == 1
        assert raw_rows[0].text == "新原始转录"
    finally:
        runtime.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_missing_current_asr_output_does_not_reuse_stale_result(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    storage = tmp_path / "storage"
    audio_path = storage / "audio" / "stale-output.m4a"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"existing-audio")
    with Session(engine) as setup_session:
        work = Work(aweme_id="stale-output")
        setup_session.add(work)
        setup_session.flush()
        work_id = work.id
        setup_session.add(Asset(work_id=work.id, kind="audio", path=str(audio_path)))
        setup_session.commit()

    stale_path = storage / "transcript" / "stale-output" / "qwen3_asr.json"
    stale_path.parent.mkdir(parents=True)
    stale_payload = {
        "text": "旧的错误结果",
        "segments": [{"start": 0.0, "end": 1.0, "text": "旧的错误结果"}],
    }
    stale_path.write_text(json.dumps(stale_payload, ensure_ascii=False), encoding="utf-8")

    with Session(engine) as session:
        work = session.get(Work, work_id)
        assert work is not None
        job = Job(kind="transcribe_work", work_id=work.id, status="running", worker_id="asr")
        session.add(job)
        session.commit()
        service = TranscriptionService(
            session,
            StorageLayout(storage),
            Settings(storage_root=storage),
            NoOutputTranscriber(),
            worker_id="asr",
        )

        with pytest.raises(RuntimeError, match="qwen_asr_result_missing"):
            await service.process(job, work)

        assert stale_path.read_text(encoding="utf-8") == json.dumps(
            stale_payload,
            ensure_ascii=False,
        )
        assert session.scalar(select(Transcript)) is None
        assert not (storage / "transcript" / "stale-output" / "transcript.txt").exists()
        assert not (storage / "transcript" / "stale-output" / "segments.json").exists()
    engine.dispose()


@pytest.mark.asyncio
async def test_malformed_current_asr_output_preserves_stable_result_and_audit(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    storage = tmp_path / "storage"
    audio_path = storage / "audio" / "malformed-output.m4a"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"existing-audio")
    with Session(engine) as setup_session:
        work = Work(aweme_id="malformed-output")
        setup_session.add(work)
        setup_session.flush()
        work_id = work.id
        setup_session.add(Asset(work_id=work.id, kind="audio", path=str(audio_path)))
        setup_session.commit()

    transcript_dir = storage / "transcript" / "malformed-output"
    transcript_dir.mkdir(parents=True)
    stable_path = transcript_dir / "qwen3_asr.json"
    stable_path.write_text('{"text":"verified-old","segments":[]}', encoding="utf-8")
    text_path = transcript_dir / "transcript.txt"
    text_path.write_text("verified-old", encoding="utf-8")
    segments_path = transcript_dir / "segments.json"
    segments_path.write_text("[]", encoding="utf-8")

    class MalformedTranscriber:
        async def run(self, *, audio_path: Path, result_path: Path) -> SubprocessResult:
            del audio_path
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text("{", encoding="utf-8")
            return SubprocessResult(returncode=0, stdout="", stderr="qwen stderr")

    with Session(engine) as session:
        work = session.get(Work, work_id)
        assert work is not None
        job = Job(kind="transcribe_work", work_id=work.id, status="running", worker_id="asr")
        session.add(job)
        session.commit()
        service = TranscriptionService(
            session,
            StorageLayout(storage),
            Settings(storage_root=storage),
            MalformedTranscriber(),
            worker_id="asr",
        )

        with pytest.raises(json.JSONDecodeError):
            await service.process(job, work)

        assert stable_path.read_text(encoding="utf-8") == '{"text":"verified-old","segments":[]}'
        assert text_path.read_text(encoding="utf-8") == "verified-old"
        assert segments_path.read_text(encoding="utf-8") == "[]"
        assert session.scalar(select(Transcript)) is None
        audit = session.scalar(select(ProcessingAudit))
        assert audit is not None
        assert audit.status == "failed"
        assert audit.error
    engine.dispose()


@pytest.mark.asyncio
async def test_lease_takeover_cannot_publish_transcript(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    storage = tmp_path / "storage"
    audio_path = storage / "audio" / "lease-takeover.m4a"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"existing-audio")
    with Session(engine) as setup_session:
        work = Work(aweme_id="lease-takeover")
        setup_session.add(work)
        setup_session.flush()
        work_id = work.id
        setup_session.add(Asset(work_id=work.id, kind="audio", path=str(audio_path)))
        setup_session.commit()

    stable_path = storage / "transcript" / "lease-takeover" / "qwen3_asr.json"
    stable_path.parent.mkdir(parents=True)
    stable_path.write_text('{"text":"verified-old","segments":[]}', encoding="utf-8")

    class LeaseStealingStore:
        def __init__(self) -> None:
            self.delegate = ResourceLeaseStore(engine)
            self.stolen = False
            self.ownership_checks = 0

        def try_acquire(self, *args, **kwargs):
            return self.delegate.try_acquire(*args, **kwargs)

        def renew(self, handle, *, ttl):
            return self.delegate.renew(handle, ttl=ttl)

        def is_owned(self, handle):
            self.ownership_checks += 1
            result = self.delegate.is_owned(handle)
            if result and not self.stolen:
                self.take_over()
            return result

        def take_over(self) -> None:
            if self.stolen:
                return
            with Session(engine) as takeover_session:
                takeover_session.execute(
                    update(ResourceLease)
                    .where(ResourceLease.resource_name == "gpu:0")
                    .values(expires_at=datetime.now(UTC) - timedelta(seconds=1))
                )
                takeover_session.commit()
            replacement = self.delegate.try_acquire(
                "gpu:0",
                "replacement-worker",
                999,
                ttl=60,
            )
            assert replacement is not None
            self.stolen = True

        def release(self, handle):
            return self.delegate.release(handle)

    class ValidTranscriber:
        async def run(self, *, audio_path: Path, result_path: Path) -> SubprocessResult:
            del audio_path
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                '{"text":"must-not-publish","segments":[]}',
                encoding="utf-8",
            )
            return SubprocessResult(returncode=0, stdout="", stderr="")

    with Session(engine) as session:
        work = session.get(Work, work_id)
        assert work is not None
        job = Job(kind="transcribe_work", work_id=work.id, status="running", worker_id="asr")
        session.add(job)
        session.commit()
        service = TranscriptionService(
            session,
            StorageLayout(storage),
            Settings(storage_root=storage),
            ValidTranscriber(),
            lease_store=LeaseStealingStore(),
            worker_id="asr",
        )

        with pytest.raises(JobCancelledError, match="gpu_lease_lost"):
            await service.process(job, work)

        session.expire_all()
        assert session.scalar(select(Transcript)) is None
        assert service.lease_store.ownership_checks >= 2
        assert stable_path.read_text(encoding="utf-8") == '{"text":"verified-old","segments":[]}'
        assert not (storage / "transcript" / "lease-takeover" / "transcript.txt").exists()
        assert not (storage / "transcript" / "lease-takeover" / "segments.json").exists()
    engine.dispose()


@pytest.mark.asyncio
async def test_lease_loss_does_not_restore_over_newer_publisher_output(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    storage = tmp_path / "storage"
    audio_path = storage / "audio" / "lease-race.m4a"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"existing-audio")
    with Session(engine) as setup_session:
        work = Work(aweme_id="lease-race")
        setup_session.add(work)
        setup_session.flush()
        work_id = work.id
        setup_session.add(Asset(work_id=work.id, kind="audio", path=str(audio_path)))
        setup_session.commit()

    transcript_dir = storage / "transcript" / "lease-race"
    transcript_dir.mkdir(parents=True)
    stable_path = transcript_dir / "qwen3_asr.json"
    text_path = transcript_dir / "transcript.txt"
    segments_path = transcript_dir / "segments.json"
    stable_path.write_text('{"text":"old","segments":[]}', encoding="utf-8")
    text_path.write_text("old", encoding="utf-8")
    segments_path.write_text("[]", encoding="utf-8")

    class LeaseLostAfterPublish:
        def __init__(self) -> None:
            self.delegate = ResourceLeaseStore(engine)
            self.ownership_checks = 0

        def try_acquire(self, *args, **kwargs):
            return self.delegate.try_acquire(*args, **kwargs)

        def renew(self, handle, *, ttl):
            return self.delegate.renew(handle, ttl=ttl)

        def is_owned(self, handle):
            del handle
            self.ownership_checks += 1
            if self.ownership_checks == 2:
                stable_path.write_text(
                    '{"text":"newer","segments":[]}', encoding="utf-8"
                )
                text_path.write_text("newer", encoding="utf-8")
                segments_path.write_text("[]", encoding="utf-8")
                return False
            return True

        def release(self, handle):
            return self.delegate.release(handle)

    class ValidTranscriber:
        async def run(self, *, audio_path: Path, result_path: Path) -> SubprocessResult:
            del audio_path
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                '{"text":"worker-a","segments":[]}', encoding="utf-8"
            )
            return SubprocessResult(returncode=0, stdout="", stderr="")

    with Session(engine) as session:
        work = session.get(Work, work_id)
        assert work is not None
        job = Job(kind="transcribe_work", work_id=work.id, status="running", worker_id="asr")
        session.add(job)
        session.commit()
        service = TranscriptionService(
            session,
            StorageLayout(storage),
            Settings(storage_root=storage),
            ValidTranscriber(),
            lease_store=LeaseLostAfterPublish(),
            worker_id="asr",
        )

        with pytest.raises(JobCancelledError, match="gpu_lease_lost"):
            await service.process(job, work)

        assert stable_path.read_text(encoding="utf-8") == '{"text":"newer","segments":[]}'
        assert text_path.read_text(encoding="utf-8") == "newer"
        assert segments_path.read_text(encoding="utf-8") == "[]"
        assert session.scalar(select(Transcript)) is None
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
        assert runtime.session.scalar(select(Job).where(Job.kind == "generate_note")) is None
        assert runtime.session.scalar(select(Transcript)) is None
        audit = runtime.session.scalar(select(ProcessingAudit))
        assert audit is not None
        assert audit.status == "failed"
        assert audit.error == "disk-write-failed"
    finally:
        runtime.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_cancellation_committed_during_asr_cannot_publish_transcript(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    storage = tmp_path / "storage"
    audio_path = storage / "audio" / "cancel-race.m4a"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"existing-audio")
    with Session(engine) as setup_session:
        work = Work(aweme_id="cancel-race")
        setup_session.add(work)
        setup_session.flush()
        setup_session.add(Asset(work_id=work.id, kind="audio", path=str(audio_path)))
        job = Job(
            kind="transcribe_work",
            work_id=work.id,
            status="running",
            worker_id="asr-race",
        )
        setup_session.add(job)
        setup_session.commit()
        work_id, job_id = work.id, job.id

    class CancellingTranscriber:
        async def run(self, *, audio_path: Path, result_path: Path) -> SubprocessResult:
            del audio_path
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps({"text": "不应发布", "segments": []}, ensure_ascii=False),
                encoding="utf-8",
            )
            with Session(engine) as api_session:
                JobQueue(api_session).cancel(job_id)
            return SubprocessResult(returncode=0, stdout="", stderr="")

    with Session(engine) as session:
        work = session.get(Work, work_id)
        job = session.get(Job, job_id)
        assert work is not None and job is not None
        service = TranscriptionService(
            session,
            StorageLayout(storage),
            Settings(storage_root=storage),
            CancellingTranscriber(),
            worker_id="asr-race",
        )
        with pytest.raises(JobCancelledError, match="cancelled_by_user"):
            await service.process(job, work, cancel_event=asyncio.Event())

        session.expire_all()
        assert session.scalar(select(Transcript)) is None
        assert not (storage / "transcript" / "cancel-race" / "transcript.txt").exists()
        assert not (storage / "transcript" / "cancel-race" / "segments.json").exists()
    engine.dispose()


@pytest.mark.asyncio
async def test_audit_finalize_failure_keeps_committed_transcript_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    storage = tmp_path / "storage"
    audio_path = storage / "audio" / "audit-failure.m4a"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(b"existing-audio")
    with Session(engine) as setup_session:
        work = Work(aweme_id="audit-failure")
        setup_session.add(work)
        setup_session.flush()
        work_id = work.id
        setup_session.add(Asset(work_id=work.id, kind="audio", path=str(audio_path)))
        setup_session.commit()

    def fail_succeed(self, *args, **kwargs):
        raise RuntimeError("audit-finalize-failed")

    monkeypatch.setattr(ProcessingAuditRecorder, "succeed", fail_succeed)
    with Session(engine) as session:
        work = session.get(Work, work_id)
        assert work is not None
        job = Job(kind="transcribe_work", work_id=work.id, status="running", worker_id="asr")
        session.add(job)
        session.commit()
        service = TranscriptionService(
            session,
            StorageLayout(storage),
            Settings(storage_root=storage),
            FakeTranscriber(),
            worker_id="asr",
        )

        with pytest.raises(RuntimeError, match="audit-finalize-failed"):
            await service.process(job, work)

        transcript = session.scalar(select(Transcript))
        assert transcript is not None
        assert transcript.text == "第一段知识。\n第二段理解。"
        transcript_dir = storage / "transcript" / "audit-failure"
        assert json.loads((transcript_dir / "qwen3_asr.json").read_text(encoding="utf-8"))[
            "text"
        ] == transcript.text
        assert (transcript_dir / "transcript.txt").read_text(encoding="utf-8") == transcript.text
        assert json.loads((transcript_dir / "segments.json").read_text(encoding="utf-8"))[1][
            "text"
        ] == "第二段理解。"
    engine.dispose()
