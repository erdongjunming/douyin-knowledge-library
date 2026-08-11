from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.api.routes.jobs import get_job_queue
from douyin_library.db import create_schema
from douyin_library.jobs.queue import JobQueue
from douyin_library.main import app


@pytest.mark.asyncio
async def test_creator_sync_endpoint_returns_a_pending_job(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        queue = JobQueue(session)
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/creators/sync",
                    json={"reference": "sec-731", "confirmed": True},
                )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 202
        assert response.json()["kind"] == "sync_creator"
        assert response.json()["status"] == "pending"
        assert queue.get(response.json()["id"]).payload_json == '{"reference": "sec-731"}'


@pytest.mark.asyncio
async def test_favorites_sync_endpoint_returns_a_pending_job(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        queue = JobQueue(session)
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/api/favorites/sync")
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 202
        assert response.json()["kind"] == "sync_favorites"
        assert response.json()["status"] == "pending"
        assert queue.get(response.json()["id"]).payload_json == "{}"


@pytest.mark.asyncio
async def test_duplicate_active_creator_sync_is_rejected(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        queue = JobQueue(session)
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                first = await client.post(
                    "/api/creators/sync",
                    json={"reference": "sec-731", "confirmed": True},
                )
                duplicate = await client.post(
                    "/api/creators/sync",
                    json={"reference": "sec-731", "confirmed": True},
                )
        finally:
            app.dependency_overrides.clear()

        assert first.status_code == 202
        assert duplicate.status_code == 409
        assert duplicate.json() == {"detail": "active_job_already_exists"}
