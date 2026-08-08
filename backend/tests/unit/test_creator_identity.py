from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from douyin_library.db import create_schema
from douyin_library.models import Creator
from douyin_library.sync.creator_sync import CreatorSyncService


class FakeCreatorGateway:
    nickname = "旧昵称"

    async def resolve_creator(self, reference: str) -> dict[str, str]:
        return {
            "sec_uid": "MS4wLjABAAAAtestcreator",
            "nickname": self.nickname,
            "unique_id": "knowledge_host",
            "profile_url": reference,
            "avatar_url": "https://example.test/avatar.jpg",
        }

    async def creator_posts(self, sec_uid: str):
        if False:
            yield None


@pytest.mark.asyncio
async def test_creator_identity_is_keyed_by_sec_uid(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    gateway = FakeCreatorGateway()

    with Session(engine) as session:
        service = CreatorSyncService(session, gateway)
        first = await service.ensure_creator(
            "https://www.douyin.com/user/MS4wLjABAAAAtestcreator"
        )
        gateway.nickname = "新昵称"
        second = await service.ensure_creator(first.id)

        assert first.id == second.id
        assert session.scalar(
            select(Creator.nickname).where(Creator.id == first.id)
        ) == "新昵称"
