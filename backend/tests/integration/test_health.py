import asyncio
import json
import os
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from douyin_library.config import Settings
from douyin_library.main import app
from douyin_library.worker_main import PoolControl


@pytest.mark.asyncio
async def test_health() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_health_reports_embedded_worker_failure_without_leaking_exception(
    monkeypatch,
) -> None:
    monkeypatch.setattr(app.state, "worker_error", "RuntimeError: private/path", raising=False)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["api"]["status"] == "alive"
    assert response.json()["workers"]["status"] == "failed"
    assert "private/path" not in response.text


@pytest.mark.asyncio
async def test_health_reports_actual_embedded_task_exit_before_monitor_runs(monkeypatch, tmp_path):
    async def exit_worker():
        return

    task = asyncio.create_task(exit_worker())
    await task
    monkeypatch.setattr(app.state, "settings", Settings(storage_root=tmp_path,
        embedded_workers=True), raising=False)
    monkeypatch.setattr(app.state, "worker_tasks", [task], raising=False)
    monkeypatch.setattr(app.state, "worker_error", None, raising=False)
    from douyin_library.main import health

    result = await health()
    assert result["status"] == "degraded"
    assert result["workers"]["status"] == "failed"


@pytest.mark.asyncio
async def test_health_distinguishes_fresh_independent_heartbeat_from_pause_control(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(app.state, "settings", Settings(storage_root=tmp_path), raising=False)
    monkeypatch.setattr(app.state, "worker_error", None, raising=False)
    heartbeat_root = tmp_path / "runtime" / "worker-pools"
    heartbeat_root.mkdir(parents=True)
    for pool in ("general", "asr"):
        (heartbeat_root / f"{pool}.test.heartbeat.json").write_text(json.dumps({
            "schema": "douyin-worker-pool-heartbeat-v1", "pool": pool,
            "pid": os.getpid(), "state": "running", "updated_at": datetime.now(UTC).isoformat(),
        }), encoding="utf-8")
    PoolControl(tmp_path, "asr").set_paused(True)
    from douyin_library.main import health

    result = await health()
    assert result["status"] == "ok"
    assert result["workers"]["mode"] == "independent"
    assert result["workers"]["pools"]["asr"]["status"] == "alive"
    assert result["workers"]["pools"]["asr"]["paused"] is True
    old = datetime.now(UTC) - timedelta(minutes=5)
    (heartbeat_root / "general.test.heartbeat.json").write_text(json.dumps({
        "schema": "douyin-worker-pool-heartbeat-v1", "pool": "general",
        "pid": os.getpid(), "state": "running", "updated_at": old.isoformat(),
    }), encoding="utf-8")
    result = await health()
    assert result["status"] == "degraded"
    assert result["api"]["status"] == "alive"
    assert result["workers"]["pools"]["general"]["status"] == "unavailable"


@pytest.mark.asyncio
async def test_paused_controls_without_heartbeat_do_not_claim_workers_are_alive(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(app.state, "settings", Settings(storage_root=tmp_path), raising=False)
    for pool in ("general", "asr"):
        PoolControl(tmp_path, pool).set_paused(True)
    from douyin_library.main import health

    result = await health()
    assert result["status"] == "ok"
    assert result["workers"]["status"] == "unavailable"
    assert all(pool["paused"] and pool["status"] == "unavailable"
               for pool in result["workers"]["pools"].values())
