from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from douyin_library.db import create_schema
from douyin_library.models import FavoriteFolder, Source, Work, WorkSource
from douyin_library.sync.favorites_sync import FavoritesSyncService
from douyin_library.upstream.contracts import (
    DiscoveredAuthor,
    DiscoveredWork,
    DiscoveryPage,
)

AUTHOR = DiscoveredAuthor(
    sec_uid="MS4wLjABAAAAtestfavorite",
    nickname="知识主播",
    unique_id="knowledge_host",
)


def _work(aweme_id: str) -> DiscoveredWork:
    return DiscoveredWork(
        aweme_id=aweme_id,
        author=AUTHOR,
        media_type="video",
        description=f"知识作品 {aweme_id}",
        raw_payload={"aweme_id": aweme_id},
    )


class FakeFavoritesGateway:
    async def all_favorites(self):
        yield DiscoveryPage(
            items=[_work("731"), _work("732")],
            cursor="0",
            has_more=False,
            source_external_id="default",
        )
        yield DiscoveryPage(
            items=[_work("732"), _work("733")],
            cursor="0",
            has_more=False,
            source_external_id="folder-42",
            source_name="重点收藏",
        )


@pytest.mark.asyncio
async def test_full_favorites_sync_deduplicates_across_folders(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        result = await FavoritesSyncService(session, FakeFavoritesGateway()).sync_all()

        assert result.default_count == 2
        assert result.folder_count == 2
        assert result.unique_work_count == 3
        assert result.complete is True
        assert session.scalar(select(func.count()).select_from(Work)) == 3
        assert session.scalar(select(func.count()).select_from(Source)) == 2
        assert session.scalar(select(func.count()).select_from(WorkSource)) == 4
        assert session.scalar(select(FavoriteFolder.name)) == "重点收藏"


class IncompleteFavoritesGateway:
    async def all_favorites(self):
        yield DiscoveryPage(
            items=[_work("731")],
            cursor="20",
            has_more=True,
            source_external_id="default",
        )
        raise RuntimeError("risk control")


@pytest.mark.asyncio
async def test_incomplete_favorites_sync_keeps_unseen_sources_active(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)

    with Session(engine) as session:
        service = FavoritesSyncService(session, FakeFavoritesGateway())
        await service.sync_all()
        service.gateway = IncompleteFavoritesGateway()

        result = await service.sync_all()

        relation_states = session.scalars(
            select(WorkSource.active)
            .join(Source)
            .where(Source.kind == "favorite_default")
            .order_by(WorkSource.id)
        ).all()
        assert result.complete is False
        assert relation_states == [True, True]
