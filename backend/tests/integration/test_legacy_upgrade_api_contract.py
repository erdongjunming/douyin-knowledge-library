from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
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


@pytest.mark.asyncio
@pytest.mark.parametrize("lease_live,stale_audit", [(True, False), (False, False), (True, True)])
async def test_progress_includes_standalone_upgrade_and_keeps_completed_in_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lease_live: bool, stale_audit: bool,
) -> None:
    from douyin_library.models import ProcessingAudit, ResourceLease, Work

    storage = tmp_path / "library"
    _seed_upgrade_candidates(storage)
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(storage))
    audit_dir = storage / "audit"
    audit_dir.mkdir()
    (audit_dir / "legacy-upgrade-inventory-broken.json").write_text('{"broken":')
    (audit_dir / "legacy-upgrade-inventory-invalid.json").write_text('["unrelated"]')
    now = datetime.now(UTC)
    (audit_dir / "legacy-upgrade-inventory-full.json").write_text(json.dumps({
        "schema": "douyin-legacy-upgrade-inventory-v1",
        "aweme_ids": ["731", "732", "733", "734", "735", "736"],
    }), encoding="utf-8")
    (audit_dir / "legacy-upgrade-batches-full.checkpoints.jsonl").write_text(
        json.dumps({"event": "batch_started", "pid": 123, "batch_index": 0,
                    "aweme_ids": ["731", "732", "733"],
                    "at": (now - timedelta(minutes=1)).isoformat()}) + '\n{"interrupted":',
        encoding="utf-8",
    )
    engine, session, transport = _client_with_queue(tmp_path / "library.db")
    try:
        for aweme_id, status in (("731", "running"), ("732", "failed")):
            work = Work(aweme_id=aweme_id)
            session.add(work)
            session.flush()
            session.add(ProcessingAudit(
                work_id=work.id, operation="generate_note", processor="note-generator",
                status=status, request_json="{}",
                started_at=(now - timedelta(minutes=2))
                if stale_audit and status == "running" else now,
                error="semantic_incomplete" if status == "failed" else None,
            ))
        session.add(ResourceLease(
            resource_name="gpu:0", lease_token="test", owner_worker_id="legacy-upgrade-123",
            acquired_at=now, heartbeat_at=now,
            expires_at=now + timedelta(minutes=1 if lease_live else -1),
        ))
        session.commit()
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/notes/progress")
            progress = response.json()["upgrade"]
            assert progress["total"] == 6
            assert progress["completed"] == 1
            expected_running = int(lease_live and not stale_audit)
            assert progress["running"] == expected_running
            assert progress["failed"] == 1
            assert progress["not_queued"] == 4 - expected_running
            assert progress["percent"] == 16.7
            # New queued retries supersede earlier standalone failures.
            retry = JobQueue(session).create(
                kind="generate_note", payload={"aweme_id": "732", "legacy_only": True},
            )
            retry.created_at = now + timedelta(seconds=1)
            session.commit()
            retried = (await client.get("/api/notes/progress")).json()["upgrade"]
            assert retried["pending"] == 1
            assert retried["failed"] == 0
            # An old failed queue job must not override a newer standalone result.
            retry.created_at = now - timedelta(minutes=5)
            retry.status = "failed"
            retry.error_detail = "obsolete failure"
            session.commit()
            retried = (await client.get("/api/notes/progress")).json()["upgrade"]
            assert retried["failures"] == [{"aweme_id": "732", "reason": "semantic_incomplete"}]
            # A newly published standalone note must increase completion,
            # rather than disappear from the historical denominator.
            from douyin_library.api.routes import notes as notes_routes
            snapshot = notes_routes._quality_details_snapshot(
                notes_routes._service(JobQueue(session)),
            )
            snapshot["731"]["quality_status"] = "semantic_complete"
            monkeypatch.setattr(notes_routes, "_quality_details_snapshot", lambda _: snapshot)
            updated = (await client.get("/api/notes/progress?refresh=true")).json()["upgrade"]
            assert updated["total"] == 6
            assert updated["completed"] == 2
            assert updated["running"] == 0
            assert updated["percent"] == 33.3
    finally:
        app.dependency_overrides.pop(get_job_queue, None)
        session.close()
        engine.dispose()


def _seed_upgrade_candidates(root: Path) -> None:
    layout = StorageLayout(root)
    layout.initialize()
    for aweme_id in ("731", "732", "733", "734", "735", "736"):
        transcript_dir = layout.transcript_dir(aweme_id)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        transcript_text = (
            f"视频 {aweme_id} 的原始转写内容用于升级验证。"
            if aweme_id == "733"
            else f"视频 {aweme_id} 的原始转写。"
        )
        layout.transcript_text_path(aweme_id).write_text(
            transcript_text,
            encoding="utf-8",
        )
        if aweme_id == "733":
            layout.transcript_segments_path(aweme_id).write_text(
                json.dumps(
                    [{"start": 0, "end": 10, "text": transcript_text}],
                    ensure_ascii=False,
                ),
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
            metadata = {
                "provider": "ollama",
                "model": "qwen3:8b",
                "quality_gate": "passed",
                "semantic_complete": True,
            }
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
        if aweme_id == "733":
            (note_dir / "coverage_report.json").write_text(
                json.dumps(
                    {
                        "quality_gate": "passed",
                        "depth_passed": True,
                        "input_sufficient": True,
                        "semantic_complete": True,
                    }
                ),
                encoding="utf-8",
            )


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
@pytest.mark.parametrize("policy", ["auto", "external", "rule"])
async def test_legacy_upgrade_rejects_non_local_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
) -> None:
    storage = tmp_path / "library"
    _seed_upgrade_candidates(storage)
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(storage))
    engine, session, transport = _client_with_queue(tmp_path / "library.db")
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/notes/generate",
                json={"legacy_only": True, "provider_policy": policy},
            )
        assert response.status_code == 422
        assert response.json()["detail"] == "legacy_upgrade_requires_local_provider"
        assert session.scalars(select(Job)).all() == []
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
        json.dumps(
            {
                "provider": "ollama",
                "model": "qwen3:8b",
                "quality_gate": "failed",
                "semantic_complete": False,
            }
        ),
        encoding="utf-8",
    )
    (layout.note_dir("failed") / "coverage_report.json").write_text(
        json.dumps(
            {
                "quality_gate": "failed",
                "depth_passed": False,
                "input_sufficient": True,
                "semantic_complete": False,
            }
        ),
        encoding="utf-8",
    )
    (layout.note_dir("false-success") / "generation_audit.json").write_text(
        json.dumps({"provider": "legacy", "model": "local-rule-v1"}),
        encoding="utf-8",
    )
    (layout.note_dir("unknown") / "coverage_report.json").write_text(
        json.dumps(
            {
                "quality_gate": "failed",
                "depth_passed": True,
                "input_sufficient": False,
                "semantic_complete": False,
            }
        ),
        encoding="utf-8",
    )
    (layout.note_dir("semantic") / "generation_manifest.json").write_text(
        json.dumps(
            {
                "provider": "ollama",
                "model": "qwen3:8b",
                "quality_gate": "passed",
                "semantic_complete": True,
            }
        ),
        encoding="utf-8",
    )
    (layout.note_dir("semantic") / "coverage_report.json").write_text(
        json.dumps(
            {
                "quality_gate": "passed",
                "depth_passed": True,
                "input_sufficient": True,
                "semantic_complete": True,
            }
        ),
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
        failed = queue.create(
            kind="generate_note",
            payload={"aweme_id": "failed", "legacy_only": True},
        )
        queue.claim(failed.id)
        queue.fail(
            failed.id,
            error_code="input_insufficient",
            error_detail="ollama unavailable",
        )
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
        running = queue.create(
            kind="generate_note",
            payload={"aweme_id": "unknown", "legacy_only": True},
            status="running",
        )
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
            "depth_failed": 1,
            "input_insufficient": 1,
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
