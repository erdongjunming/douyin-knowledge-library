from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.db import create_schema
from douyin_library.jobs.runtime import build_worker_runtime
from douyin_library.models import Analysis, Job, Transcript, Work
from douyin_library.upstream.contracts import (
    DiscoveredAuthor,
    DiscoveredWork,
    DiscoveryPage,
)


def discovered_work(aweme_id: str) -> DiscoveredWork:
    return DiscoveredWork(
        aweme_id=aweme_id,
        author=DiscoveredAuthor(
            sec_uid="sec-knowledge",
            nickname="知识主播",
            unique_id="knowledge-host",
        ),
        media_type="video",
        description=f"知识作品 {aweme_id}",
        share_url=f"https://example.test/{aweme_id}",
        raw_payload={"aweme_id": aweme_id},
    )


class MutableCookieStore:
    def __init__(self, value: str | None = None) -> None:
        self.value = value
        self.read_count = 0

    def read(self) -> str | None:
        self.read_count += 1
        return self.value


class FakeGateway:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    async def resolve_creator(self, reference: str) -> dict[str, str]:
        return {
            "sec_uid": "sec-knowledge",
            "nickname": "知识主播",
            "unique_id": "knowledge-host",
            "profile_url": f"https://www.douyin.com/user/{reference}",
            "avatar_url": "https://example.test/avatar.jpg",
        }

    async def creator_posts(self, sec_uid: str):
        yield DiscoveryPage(
            items=[discovered_work("731")],
            cursor="0",
            has_more=False,
            source_external_id=sec_uid,
        )

    async def all_favorites(self):
        yield DiscoveryPage(
            items=[discovered_work("732")],
            cursor="0",
            has_more=False,
            source_external_id="default",
        )


@pytest.mark.asyncio
async def test_creator_worker_reads_cookie_and_enqueues_new_media(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    cookie_store = MutableCookieStore()
    gateways: list[tuple[str, FakeGateway]] = []

    def gateway_factory(cookie: str) -> FakeGateway:
        gateway = FakeGateway()
        gateways.append((cookie, gateway))
        return gateway

    runtime = build_worker_runtime(
        engine,
        Settings(storage_root=tmp_path / "storage"),
        cookie_store=cookie_store,
        gateway_factory=gateway_factory,
    )
    try:
        sync_job = runtime.queue.create(
            kind="sync_creator",
            payload={"reference": "sec-knowledge"},
        )
        cookie_store.value = "sessionid=runtime-secret"

        assert await runtime.handlers.run_one() is True

        download_job = runtime.session.scalar(
            select(Job).where(Job.kind == "download_work")
        )
        assert runtime.queue.get(sync_job.id).status == "succeeded"
        assert download_job is not None
        assert download_job.status == "pending"
        assert download_job.work_id == runtime.session.scalar(select(Work.id))
        assert cookie_store.read_count == 1
        assert gateways[0][0] == "sessionid=runtime-secret"
        assert gateways[0][1].closed is True
        stored_jobs = runtime.session.scalars(select(Job))
        assert all("runtime-secret" not in (job.payload_json or "") for job in stored_jobs)
    finally:
        runtime.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_favorites_worker_enqueues_new_media(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    gateway = FakeGateway()
    runtime = build_worker_runtime(
        engine,
        Settings(storage_root=tmp_path / "storage"),
        cookie_store=MutableCookieStore("sessionid=favorites"),
        gateway_factory=lambda cookie: gateway,
    )
    try:
        sync_job = runtime.queue.create(kind="sync_favorites", payload={})

        assert await runtime.handlers.run_one() is True

        download_job = runtime.session.scalar(
            select(Job).where(Job.kind == "download_work")
        )
        assert runtime.queue.get(sync_job.id).status == "succeeded"
        assert download_job is not None
        assert download_job.work_id == runtime.session.scalar(select(Work.id))
        assert gateway.closed is True
    finally:
        runtime.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_download_worker_processes_media_without_fake_ai_records(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as setup_session:
        work = Work(aweme_id="733", media_type="video")
        setup_session.add(work)
        setup_session.commit()
        work_id = work.id

    processed: list[int] = []

    class FakeMediaService:
        def __init__(self, session: Session) -> None:
            self.session = session

        async def process(self, work: Work) -> Any:
            assert self.session.in_transaction() is False
            processed.append(work.id)
            return object()

    runtime = build_worker_runtime(
        engine,
        Settings(storage_root=tmp_path / "storage"),
        cookie_store=MutableCookieStore("sessionid=download"),
        gateway_factory=lambda cookie: FakeGateway(),
        media_service_factory=lambda session, collector, layout: FakeMediaService(session),
    )
    try:
        job = runtime.queue.create(kind="download_work", work_id=work_id)

        assert await runtime.handlers.run_one() is True

        assert runtime.queue.get(job.id).status == "succeeded"
        assert processed == [work_id]
        assert runtime.session.scalar(select(func.count()).select_from(Transcript)) == 0
        assert runtime.session.scalar(select(func.count()).select_from(Analysis)) == 0
    finally:
        runtime.close()
        engine.dispose()
