import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.db import create_schema
from douyin_library.jobs.queue import JobQueue
from douyin_library.jobs.runtime import build_worker_runtime, enqueue_missing_notes
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Analysis, Creator, Job, ProcessingAudit, Work
from douyin_library.notes.generator import SYSTEM_PROMPT
from douyin_library.notes.publisher import PublishError
from douyin_library.notes.service import GenerationResult, NoteGenerationService


def seed_note_source(session: Session, root: Path, aweme_id: str = "731") -> int:
    layout = StorageLayout(root)
    layout.initialize()
    creator = Creator(sec_uid="sec-note", nickname="知识主播", unique_id="note-host")
    session.add(creator)
    session.flush()
    work = Work(
        aweme_id=aweme_id,
        creator_id=creator.id,
        description="建立可复用的表达系统 #表达 #个人成长",
        published_at=None,
        duration_ms=62_000,
    )
    session.add(work)
    session.commit()
    transcript_dir = layout.transcript_dir(aweme_id)
    transcript_dir.mkdir(parents=True)
    transcript = (
        "提升表达能力的关键不是背更多话术，而是建立自己的表达系统。"
        "首先记录真实问题，其次按主题整理案例，最后通过输出验证理解。"
        "如果只是收藏资料却不复述，知识不会自动变成能力。"
    )
    layout.transcript_text_path(aweme_id).write_text(transcript, encoding="utf-8")
    layout.transcript_segments_path(aweme_id).write_text(
        json.dumps(
            [
                {"start": 0, "end": 31, "text": transcript[:37]},
                {"start": 31, "end": 62, "text": transcript[37:]},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    layout.metadata_path(aweme_id).write_text(
        json.dumps({"aweme_id": aweme_id, "verified_at": "2026-08-14T00:00:00Z"}),
        encoding="utf-8",
    )
    layout.audio_path(aweme_id).write_bytes(b"existing-audio")
    layout.cover_path(aweme_id).write_bytes(b"existing-cover")
    return work.id


def test_note_generation_is_audited_incremental_and_forceable(tmp_path: Path) -> None:
    root = tmp_path / "library"
    engine = create_engine(f"sqlite:///{root / 'library.db'}")
    root.mkdir()
    create_schema(engine)

    with Session(engine) as session:
        seed_note_source(session, root)
        service = NoteGenerationService(
            session,
            StorageLayout(root),
            Settings(storage_root=root, note_provider_policy="rule"),
        )

        dry_run = service.run_batch(dry_run=True)
        assert dry_run.would_generate == 1
        assert not (root / "dy-note" / "731").exists()

        generated = service.run_batch(limit=1)
        assert generated.generated == 1
        note_dir = root / "dy-note" / "731"
        expected = {
            "learning_note.md",
            "douyin_ai_brief.md",
            "douyin_ai_brief.json",
            "note_budget.json",
            "note_score.json",
            "page_metadata.json",
            "generation_audit.json",
            "assets/asset_manifest.json",
        }
        assert all((note_dir / relative).is_file() for relative in expected)
        learning_note = (note_dir / "learning_note.md").read_text(encoding="utf-8")
        assert "## 核心观点" in learning_note
        assert "## 可执行行动" in learning_note
        manifest = json.loads(
            (note_dir / "assets" / "asset_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["copies_original_assets"] is False
        assert manifest["assets"]["audio"]["exists"] is True

        audit = session.scalar(
            select(ProcessingAudit).where(ProcessingAudit.operation == "generate_note")
        )
        assert audit is not None
        request = json.loads(audit.request_json)
        assert request["prompt"] == SYSTEM_PROMPT
        assert "提升表达能力" in request["input"]["transcript"]
        response = json.loads(audit.response_json or "{}")
        assert response["raw_output"]
        assert response["parsed_result"]["aweme_id"] == "731"
        assert session.scalar(select(func.count()).select_from(Analysis)) == 1

        incremental = service.run_batch()
        assert incremental.eligible == 0
        assert incremental.skipped == 1

        forced = service.run_batch(force=True, limit=1)
        assert forced.generated == 1
        assert session.scalar(select(func.count()).select_from(Analysis)) == 2

    engine.dispose()


def test_startup_backfill_queues_only_missing_notes(tmp_path: Path) -> None:
    root = tmp_path / "library"
    engine = create_engine(f"sqlite:///{root / 'library.db'}")
    root.mkdir()
    create_schema(engine)

    with Session(engine) as session:
        work_id = seed_note_source(session, root)
        queue = JobQueue(session)

        assert enqueue_missing_notes(queue, Settings(storage_root=root)) == 1
        assert enqueue_missing_notes(queue, Settings(storage_root=root)) == 0
        job = session.scalar(select(Job).where(Job.kind == "generate_note"))
        assert job is not None
        assert job.work_id == work_id
        assert job.status == "pending"

    engine.dispose()


@pytest.mark.asyncio
async def test_publish_failure_marks_generation_audit_failed(tmp_path: Path) -> None:
    root = tmp_path / "library"
    engine = create_engine(f"sqlite:///{root / 'library.db'}")
    root.mkdir()
    create_schema(engine)

    with Session(engine) as session:
        seed_note_source(session, root, aweme_id="734")

        class FailingPublisher:
            def publish(self, *args, **kwargs):
                raise PublishError("manifest_commit_failed")

        service = NoteGenerationService(
            session,
            StorageLayout(root),
            Settings(storage_root=root, note_provider_policy="rule"),
            publisher=FailingPublisher(),
        )
        with pytest.raises(PublishError, match="manifest_commit_failed"):
            await service.generate_one("734", provider_policy="rule")
        audit = session.scalar(
            select(ProcessingAudit).where(ProcessingAudit.operation == "generate_note")
        )
        assert audit is not None
        assert audit.status == "failed"
        assert audit.error and "manifest_commit_failed" in audit.error

    engine.dispose()


@pytest.mark.asyncio
async def test_semantic_publish_checks_gpu_lease_ownership_before_commit(tmp_path: Path) -> None:
    root = tmp_path / "lease-library"
    root.mkdir()
    engine = create_engine(f"sqlite:///{root / 'library.db'}")
    create_schema(engine)
    with Session(engine) as seed_session:
        seed_note_source(seed_session, root, aweme_id="735")
    layout = StorageLayout(root)
    settings = Settings(storage_root=root, note_provider_policy="local")
    ownership_checks = [0]
    released: list[object] = []

    class FakeLeaseStore:
        def try_acquire(self, *args, **kwargs):
            return "lease-handle"

        def is_owned(self, handle):
            ownership_checks[0] += 1
            return handle == "lease-handle"

        def release(self, handle):
            released.append(handle)
            return True

    class CapturingPublisher:
        def publish(self, aweme_id, outputs, *, force=False, ownership_check=None):
            assert ownership_check is not None
            assert ownership_check() is True
            return type("Result", (), {"status": "published"})()

    class SemanticProvider:
        name = "ollama"
        model = "qwen3:8b"
        is_local = True
        context_tokens = 32_768

        async def complete_json(self, **kwargs):
            try:
                from tests.unit.notes.test_pipeline import ScriptedProvider
            except ModuleNotFoundError:
                # Pytest's default import mode exposes the notes test folder
                # directly rather than as a ``tests`` package.
                from test_pipeline import ScriptedProvider

            return await ScriptedProvider().complete_json(**kwargs)

        async def unload(self):
            return None

    with Session(engine) as session:
        service = NoteGenerationService(
            session,
            layout,
            settings,
            provider_factory=lambda name: SemanticProvider(),
            publisher=CapturingPublisher(),
            lease_store=FakeLeaseStore(),
            worker_id="worker-735",
        )
        result = await service.generate_one("735", provider_policy="local")

    assert result.status == "generated"
    assert ownership_checks[0] == 1
    assert released == ["lease-handle"]
    engine.dispose()


@pytest.mark.asyncio
async def test_generate_note_job_uses_the_shared_service(tmp_path: Path) -> None:
    root = tmp_path / "library"
    root.mkdir()
    engine = create_engine(f"sqlite:///{root / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        work_id = seed_note_source(session, root, aweme_id="732")

    runtime = build_worker_runtime(
        engine,
        Settings(storage_root=root, note_provider_policy="rule"),
    )
    try:
        job = runtime.queue.create(
            kind="generate_note",
            work_id=work_id,
            payload={"aweme_id": "732", "force": False},
        )

        assert await runtime.handlers.run_one() is True

        assert runtime.queue.get(job.id).status == "succeeded"
        assert (root / "dy-note" / "732" / "learning_note.md").is_file()
    finally:
        runtime.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_legacy_upgrade_job_fails_for_preserved_result_and_forces_local_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "library"
    root.mkdir()
    engine = create_engine(f"sqlite:///{root / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        work_id = seed_note_source(session, root, aweme_id="734")
    note_dir = root / "dy-note" / "734"
    note_dir.mkdir(parents=True, exist_ok=True)
    original = "# trusted legacy note\n"
    (note_dir / "learning_note.md").write_text(original, encoding="utf-8")

    calls: list[str | None] = []

    async def fake_generate(self, aweme_id, **kwargs):
        del self, aweme_id
        calls.append(kwargs.get("provider_policy"))
        return GenerationResult(
            aweme_id="734",
            status="preserved_existing",
            output_dir=str(note_dir),
            provider="ollama",
            quality_gate="passed",
            error="provider_unavailable",
        )

    monkeypatch.setattr(NoteGenerationService, "generate_one", fake_generate)
    runtime = build_worker_runtime(engine, Settings(storage_root=root))
    try:
        job = runtime.queue.create(
            kind="generate_note",
            work_id=work_id,
            payload={
                "aweme_id": "734",
                "legacy_only": True,
                "provider_policy": "auto",
            },
        )

        assert await runtime.handlers.run_one() is True

        failed = runtime.queue.get(job.id)
        assert failed.status == "failed"
        assert calls == ["local"]
        assert (note_dir / "learning_note.md").read_text(encoding="utf-8") == original
    finally:
        runtime.close()
        engine.dispose()
