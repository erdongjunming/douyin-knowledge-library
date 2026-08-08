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
async def test_jobs_api_can_create_and_list_persistent_job(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        queue = JobQueue(session)
        app.dependency_overrides[get_job_queue] = lambda: queue
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                created = await client.post("/api/jobs", json={"kind": "sync_all"})
                listed = await client.get("/api/jobs")
        finally:
            app.dependency_overrides.pop(get_job_queue, None)

    assert created.status_code == 200
    assert created.json()["status"] == "pending"
    assert listed.status_code == 200
    assert listed.json()["items"][0]["kind"] == "sync_all"
