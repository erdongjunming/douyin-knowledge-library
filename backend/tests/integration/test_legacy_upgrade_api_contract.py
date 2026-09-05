from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from douyin_library.api.routes.jobs import get_job_queue
from douyin_library.db import create_schema
from douyin_library.jobs.queue import JobQueue
from douyin_library.main import app
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Job


def _seed_upgrade_candidates(root: Path) -> None:
    layout = StorageLayout(root)
    layout.initialize()
    for aweme_id in ("731", "732", "733", "734", "735", "736"):
        transcript_dir = layout.transcript_dir(aweme_id)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        layout.transcript_text_path(aweme_id).write_text(
            f"视频 {aweme_id} 的原始转写。",
            encoding="utf-8",
        )
        note_dir = layout.note_dir(aweme_id)
        note_dir.mkdir(parents=True, exist_ok=True)
        (note_dir / "learning_note.md").write_text(
            f"# old note {aweme_id}\n",
            encoding="utf-8",
        )
        if aweme_id == "731":
            metadata = {"provider": "legacy", "model": "local-rule-v1"}
            name = "generation_audit.json"
        elif aweme_id == "732":
            metadata = {"provider": "rule", "model": "local-rule-v1", "quality_gate": "fallback"}
            name = "generation_manifest.json"
        elif aweme_id == "733":
            metadata = {"provider": "ollama", "model": "qwen3:8b", "quality_gate": "passed"}
            name = "generation_manifest.json"
        elif aweme_id == "734":
            metadata = {"provider": "legacy", "model": "local-rule-v1"}
            name = "generation_audit.json"
        elif aweme_id == "735":
            metadata = {"provider": "legacy", "model": "local-rule-v1"}
            name = "generation_audit.json"
        else:
            metadata = None
            name = ""
        if metadata is not None:
            (note_dir / name).write_text(json.dumps(metadata), encoding="utf-8")


def _client_with_queue(db_path: Path):
    engine = create_engine(f"sqlite:///{db_path}")
    create_schema(engine)
    session = Session(engine)
    queue = JobQueue(session)
    app.dependency_overrides[get_job_queue] = lambda: queue
    transport = httpx.ASGITransport(app=app)
    return engine, session, transport


@pytest.mark.asyncio
async def test_legacy_upgrade_defaults_to_three_and_forces_local_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = tmp_path / "library"
    _seed_upgrade_candidates(storage)
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(storage))
    engine, session, transport = _client_with_queue(tmp_path / "library.db")
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/notes/generate",
                json={"legacy_only": True},
            )

        assert response.status_code == 202
        body = response.json()
        assert body["eligible"] == 5
        assert body["selected"] == 3
        assert body["queued"] == 3
        assert body["provider_policy"] == "local"
        jobs = session.scalars(select(Job).order_by(Job.id)).all()
        assert len(jobs) == 3
        payloads = [json.loads(job.payload_json or "{}") for job in jobs]
        assert all(payload["legacy_only"] is True for payload in payloads)
        assert all(payload["provider_policy"] == "local" for payload in payloads)
        assert [payload["aweme_id"] for payload in payloads] == ["731", "732", "734"]
    finally:
        app.dependency_overrides.pop(get_job_queue, None)
        session.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_legacy_upgrade_limit_above_three_is_rejected_without_queue_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = tmp_path / "library"
    _seed_upgrade_candidates(storage)
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(storage))
    engine, session, transport = _client_with_queue(tmp_path / "library.db")
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/notes/generate",
                json={"legacy_only": True, "limit": 4},
            )

        assert response.status_code == 422
        assert session.scalars(select(Job)).all() == []
    finally:
        app.dependency_overrides.pop(get_job_queue, None)
        session.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_upgrade_progress_uses_current_semantic_state_and_latest_upgrade_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = tmp_path / "library"
    layout = StorageLayout(storage)
    layout.initialize()
    for aweme_id in ("legacy", "fallback", "unknown", "failed", "false-success", "semantic"):
        transcript_dir = layout.transcript_dir(aweme_id)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        layout.transcript_text_path(aweme_id).write_text("原始转写", encoding="utf-8")
        note_dir = layout.note_dir(aweme_id)
        note_dir.mkdir(parents=True, exist_ok=True)
        (note_dir / "learning_note.md").write_text("# old", encoding="utf-8")
    (layout.note_dir("legacy") / "generation_audit.json").write_text(
        json.dumps({"provider": "legacy", "model": "local-rule-v1"}),
        encoding="utf-8",
    )
    (layout.note_dir("fallback") / "generation_manifest.json").write_text(
        json.dumps({"provider": "rule", "model": "local-rule-v1", "quality_gate": "fallback"}),
        encoding="utf-8",
    )
    (layout.note_dir("failed") / "generation_audit.json").write_text(
        json.dumps({"provider": "legacy", "model": "local-rule-v1"}),
        encoding="utf-8",
    )
    (layout.note_dir("false-success") / "generation_audit.json").write_text(
        json.dumps({"provider": "legacy", "model": "local-rule-v1"}),
        encoding="utf-8",
    )
    (layout.note_dir("semantic") / "generation_manifest.json").write_text(
        json.dumps({"provider": "ollama", "model": "qwen3:8b", "quality_gate": "passed"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(storage))
    engine, session, transport = _client_with_queue(tmp_path / "library.db")
    try:
        queue = JobQueue(session)

        pending = queue.create(
            kind="generate_note",
            payload={"aweme_id": "fallback", "legacy_only": True},
        )
        running = queue.create(
            kind="generate_note",
            payload={"aweme_id": "unknown", "legacy_only": True},
            status="running",
        )
        failed = queue.create(
            kind="generate_note",
            payload={"aweme_id": "failed", "legacy_only": True},
        )
        queue.claim(failed.id)
        queue.fail(failed.id, error_code="handler_failed", error_detail="ollama unavailable")
        false_success = queue.create(
            kind="generate_note",
            payload={"aweme_id": "false-success", "legacy_only": True},
        )
        queue.claim(false_success.id)
        queue.transition(false_success.id, "succeeded")
        semantic = queue.create(
            kind="generate_note",
            payload={"aweme_id": "semantic", "legacy_only": True},
        )
        queue.claim(semantic.id)
        queue.transition(semantic.id, "succeeded")
        assert pending.status == "pending"
        assert running.status == "running"

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/notes/progress")

        assert response.status_code == 200
        body = response.json()
        assert body["upgrade"] == {
            "total": 6,
            "completed": 1,
            "remaining": 5,
            "pending": 1,
            "running": 1,
            "active": 2,
            "failed": 2,
            "not_queued": 1,
            "eligible": 5,
            "queued": 5,
            "percent": 16.7,
            "failures": [
                {"aweme_id": "false-success", "reason": "upgrade_completed_without_semantic_note"},
                {"aweme_id": "failed", "reason": "ollama unavailable"},
            ],
        }
    finally:
        app.dependency_overrides.pop(get_job_queue, None)
        session.close()
        engine.dispose()
