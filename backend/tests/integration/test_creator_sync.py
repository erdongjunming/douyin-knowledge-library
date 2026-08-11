from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from douyin_library.db import create_schema
from douyin_library.models import SyncRun, Work, WorkSource
from douyin_library.sync.creator_sync import CreatorSyncService


class FakeCreatorGateway:
    nickname = "旧昵称"
    fail_after_first_page = False

    async def resolve_creator(self, reference: str) -> dict[str, str]:
        return {
            "sec_uid": "MS4wLjABAAAAtestcreator",
            "nickname": self.nickname,
            "unique_id": "knowledge_host",
            "profile_url": reference,
            "avatar_url": "https://example.test/avatar.jpg",
        }

    async def creator_posts(self, sec_uid: str):
        yield type(
            "Page",
            (),
            {
                "items": [
                    type(
                        "Work",
                        (),
                        {
                            "aweme_id": "731",
                            "author": type(
                                "Author",
                                (),
                                {
                                    "sec_uid": sec_uid,
                                    "nickname": self.nickname,
                                    "unique_id": "knowledge_host",
                                    "avatar_url": None,
                                },
                            )(),
                            "media_type": "video",
                            "description": "第一条知识",
                            "share_url": "https://example.test/731",
                            "published_at": None,
                            "duration_ms": 1000,
                            "cover_url": None,
                            "download_urls": (),
                            "image_urls": (),
                            "stats": {},
                            "raw_payload": {"aweme_id": "731"},
                        },
                    )()
                ],
                "cursor": "20",
                "has_more": False,
                "source_external_id": sec_uid,
                "source_name": None,
            },
        )()


@pytest.mark.asyncio
async def test_creator_sync_updates_nickname_without_changing_identity(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    gateway = FakeCreatorGateway()

    with Session(engine) as session:
        service = CreatorSyncService(session, gateway)
        first = await service.sync("https://www.douyin.com/user/MS4wLjABAAAAtestcreator")
        gateway.nickname = "新昵称"
        second = await service.sync(first.creator.id)

        assert len(first.new_work_ids) == 1
        assert first.new_work_ids[0] == session.scalar(select(Work.id))
        assert second.new_work_ids == ()
        assert second.creator.id == first.creator.id
        assert second.creator.nickname == "新昵称"
        assert second.new_work_count == 0
        assert session.scalar(select(Work.aweme_id)) == "731"


@pytest.mark.asyncio
async def test_incomplete_creator_sync_does_not_deactivate_old_source(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    gateway = FakeCreatorGateway()

    with Session(engine) as session:
        service = CreatorSyncService(session, gateway)
        first = await service.sync("MS4wLjABAAAAtestcreator")
        source = session.scalar(select(WorkSource))
        assert source is not None
        source.active = True
        session.commit()

        gateway.creator_posts = broken_creator_posts
        result = await service.sync(first.creator.id)

        assert result.complete is False
        assert result.status == "incomplete"
        assert session.scalar(select(WorkSource.active)) is True
        assert session.scalar(select(SyncRun.status).order_by(SyncRun.id.desc())) == "incomplete"


async def broken_creator_posts(sec_uid: str):
    raise RuntimeError("risk control")
    yield
