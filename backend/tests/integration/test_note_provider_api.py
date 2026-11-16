from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from douyin_library.api.routes.jobs import get_job_queue
from douyin_library.config import Settings
from douyin_library.db import create_schema
from douyin_library.main import app
from douyin_library.models import Job, Work


@pytest.mark.asyncio
async def test_provider_status_probes_only_when_explicitly_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    calls = 0

    async def fake_probe(settings):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            executable_found=True,
            executable_path="ollama",
            service_status="ready",
            model_available=True,
            ready=True,
            model="qwen3:8b",
            action_required=None,
            as_dict=lambda: {"ready": True, "model": "qwen3:8b"},
        )

    monkeypatch.setattr("douyin_library.api.routes.notes.probe_ollama", fake_probe)
    previous_runtime = getattr(app.state, "runtime", None)
    app.state.runtime = SimpleNamespace(engine=engine)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            progress = await client.get("/api/notes/progress")
            status = await client.get("/api/notes/provider-status")
    finally:
        if previous_runtime is None:
            del app.state.runtime
        else:
            app.state.runtime = previous_runtime
        engine.dispose()

    assert progress.status_code == 200
    assert status.status_code == 200
    assert calls == 1
    assert status.json()["ollama"]["ready"] is True


@pytest.mark.asyncio
async def test_note_generate_dry_run_does_not_create_jobs_and_deduplicates_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = tmp_path / "library"
    transcript_dir = storage / "transcript" / "731"
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "transcript.txt").write_text("正文", encoding="utf-8")
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(storage))
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        session.add(Work(aweme_id="731"))
        session.commit()
        queue = __import__("douyin_library.jobs.queue", fromlist=["JobQueue"]).JobQueue(session)
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/notes/generate",
                    json={"aweme_ids": ["731", "731"], "dry_run": True},
                )
        finally:
            app.dependency_overrides.pop(get_job_queue, None)

        assert response.status_code == 200
        assert response.json()["selected"] == 1
        assert session.scalars(select(Job)).all() == []
    engine.dispose()


@pytest.mark.asyncio
async def test_force_legacy_only_is_unprocessable_entity(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        queue = __import__("douyin_library.jobs.queue", fromlist=["JobQueue"]).JobQueue(session)
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/notes/generate", json={"force": True, "legacy_only": True}
                )
        finally:
            app.dependency_overrides.pop(get_job_queue, None)
    engine.dispose()
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_note_generate_defaults_to_environment_policy_for_new_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = tmp_path / "library"
    transcript_dir = storage / "transcript" / "731"
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "transcript.txt").write_text("正文", encoding="utf-8")
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(storage))
    monkeypatch.setenv("DKL_NOTE_PROVIDER_POLICY", "external")
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        session.add(Work(aweme_id="731"))
        session.commit()
        queue = __import__("douyin_library.jobs.queue", fromlist=["JobQueue"]).JobQueue(session)
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/notes/generate",
                    json={"aweme_ids": ["731"], "dry_run": True},
                )
        finally:
            app.dependency_overrides.pop(get_job_queue, None)
    engine.dispose()
    assert response.status_code == 200
    assert response.json()["provider_policy"] == "external"


@pytest.mark.asyncio
async def test_note_generate_explicit_local_policy_is_honored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = tmp_path / "library"
    transcript_dir = storage / "transcript" / "731"
    transcript_dir.mkdir(parents=True)
    (transcript_dir / "transcript.txt").write_text("正文", encoding="utf-8")
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(storage))
    monkeypatch.setenv("DKL_NOTE_PROVIDER_POLICY", "external")
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        session.add(Work(aweme_id="731"))
        session.commit()
        queue = __import__("douyin_library.jobs.queue", fromlist=["JobQueue"]).JobQueue(session)
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/notes/generate",
                    json={
                        "aweme_ids": ["731"],
                        "dry_run": True,
                        "provider_policy": "local",
                    },
                )
        finally:
            app.dependency_overrides.pop(get_job_queue, None)
    engine.dispose()
    assert response.status_code == 200
    assert response.json()["provider_policy"] == "local"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "ollama_ready", "expected"),
    [("auto", True, "external"), ("auto", False, "external"),
     ("local", True, "ollama"), ("local", False, None),
     ("external", True, "external"), ("rule", True, None)],
)
async def test_provider_status_follows_policy_without_claiming_external_readiness(
    monkeypatch: pytest.MonkeyPatch, policy: str, ollama_ready: bool, expected: str | None,
) -> None:
    from douyin_library.api.routes import notes

    settings = Settings(
        note_provider_policy=policy,
        note_external_base_url="https://synthetic.test/v1",
        note_external_model="synthetic-model",
        note_external_api_key=SecretStr("synthetic-placeholder"),
    )

    async def probe(settings):
        return SimpleNamespace(
            ready=ollama_ready, model="local-backup",
            as_dict=lambda: {"ready": ollama_ready, "model": "local-backup"},
        )

    monkeypatch.setattr(notes, "Settings", lambda: settings)
    monkeypatch.setattr(notes, "probe_ollama", probe)
    monkeypatch.setattr(notes, "_provider_status_cache", None)
    result = await notes.note_provider_status()
    assert result["provider"] == expected
    assert result["available"] is (expected is not None)
    assert result["ready"] is (None if expected == "external" else expected == "ollama")
    assert result["external"]["configured"] is True
    assert result["external"]["ready"] is None
    assert result["external"]["status"] == "not_probed"
