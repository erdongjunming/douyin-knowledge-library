from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from douyin_library.models import (
    Creator,
    FavoriteFolder,
    Source,
    SyncRun,
    Work,
    WorkSource,
)
from douyin_library.sync.reconciliation import SourceReconciler


@dataclass(frozen=True, slots=True)
class FavoritesSyncResult:
    default_count: int
    folder_count: int
    unique_work_count: int
    new_work_ids: tuple[int, ...]
    status: str

    @property
    def complete(self) -> bool:
        return self.status == "complete"


class FavoritesSyncService:
    source_kinds = {"favorite_default", "favorite_folder"}

    def __init__(self, session: Session, gateway: Any) -> None:
        self.session = session
        self.gateway = gateway

    async def sync_all(self) -> FavoritesSyncResult:
        run = SyncRun(
            source_kind="favorites_all",
            source_external_id="self",
            status="running",
        )
        self.session.add(run)
        self.session.commit()

        seen_by_source: dict[int, set[int]] = {}
        seen_folder_ids: set[str] = set()
        unique_work_ids: set[int] = set()
        new_work_ids: set[int] = set()
        default_count = 0
        folder_count = 0
        try:
            async for page in self.gateway.all_favorites():
                is_default = page.source_external_id == "default"
                kind = "favorite_default" if is_default else "favorite_folder"
                source = self._source(kind, page.source_external_id, page.source_name)
                if not is_default:
                    self._favorite_folder(page.source_external_id, page.source_name)
                    seen_folder_ids.add(page.source_external_id)
                source_seen = seen_by_source.setdefault(source.id, set())

                for item in page.items:
                    creator_id = self._creator_id(item.author)
                    work, is_new = self._work(item, creator_id)
                    if is_new:
                        new_work_ids.add(work.id)
                    relation = self.session.scalar(
                        select(WorkSource).where(
                            WorkSource.work_id == work.id,
                            WorkSource.source_id == source.id,
                        )
                    )
                    if relation is None:
                        relation = WorkSource(
                            work_id=work.id,
                            source_id=source.id,
                            folder_external_id=None if is_default else page.source_external_id,
                        )
                        self.session.add(relation)
                    relation.active = True
                    relation.last_seen_at = datetime.now(UTC)
                    source_seen.add(work.id)
                    unique_work_ids.add(work.id)
                    if is_default:
                        default_count += 1
                    else:
                        folder_count += 1

                run.discovered_count = default_count + folder_count
                run.cursor_summary_json = json.dumps(
                    {
                        "source_external_id": page.source_external_id,
                        "cursor": page.cursor,
                        "has_more": page.has_more,
                    }
                )
                self.session.commit()
        except Exception as exc:  # noqa: BLE001
            run.status = "incomplete"
            run.error_detail = str(exc)[:500]
            run.finished_at = datetime.now(UTC)
            self.session.commit()
            return FavoritesSyncResult(
                default_count=default_count,
                folder_count=folder_count,
                unique_work_count=len(unique_work_ids),
                new_work_ids=tuple(sorted(new_work_ids)),
                status="incomplete",
            )

        SourceReconciler(self.session).finish(
            run_status="complete",
            seen_by_source=seen_by_source,
            source_kinds=self.source_kinds,
        )
        now = datetime.now(UTC)
        for folder in self.session.scalars(select(FavoriteFolder)):
            folder.active = folder.external_id in seen_folder_ids
            if folder.active:
                folder.last_synced_at = now
        run.status = "complete"
        run.finished_at = now
        self.session.commit()
        return FavoritesSyncResult(
            default_count=default_count,
            folder_count=folder_count,
            unique_work_count=len(unique_work_ids),
            new_work_ids=tuple(sorted(new_work_ids)),
            status="complete",
        )

    def _source(self, kind: str, external_id: str, name: str | None) -> Source:
        source = self.session.scalar(
            select(Source).where(Source.kind == kind, Source.external_id == external_id)
        )
        if source is None:
            source = Source(kind=kind, external_id=external_id, name=name)
            self.session.add(source)
            self.session.flush()
        elif name:
            source.name = name
        return source

    def _favorite_folder(self, external_id: str, name: str | None) -> FavoriteFolder:
        folder = self.session.scalar(
            select(FavoriteFolder).where(FavoriteFolder.external_id == external_id)
        )
        if folder is None:
            folder = FavoriteFolder(external_id=external_id, name=name or external_id)
            self.session.add(folder)
        elif name:
            folder.name = name
        folder.active = True
        return folder

    def _creator_id(self, author: Any) -> int | None:
        if not author.sec_uid:
            return None
        creator = self.session.scalar(
            select(Creator).where(Creator.sec_uid == author.sec_uid)
        )
        if creator is None:
            creator = Creator(sec_uid=author.sec_uid)
            self.session.add(creator)
            self.session.flush()
        for field in ("nickname", "unique_id", "avatar_url"):
            value = getattr(author, field)
            if value is not None:
                setattr(creator, field, value)
        return creator.id

    def _work(self, item: Any, creator_id: int | None) -> tuple[Work, bool]:
        work = self.session.scalar(select(Work).where(Work.aweme_id == item.aweme_id))
        is_new = work is None
        if work is None:
            work = Work(aweme_id=item.aweme_id)
            self.session.add(work)
            self.session.flush()
        if creator_id is not None:
            work.creator_id = creator_id
        work.media_type = item.media_type
        work.description = item.description
        work.share_url = item.share_url
        work.published_at = item.published_at
        work.duration_ms = item.duration_ms
        work.availability = "available"
        work.stats_json = json.dumps(item.stats, ensure_ascii=False)
        work.raw_metadata_json = json.dumps(item.raw_payload, ensure_ascii=False)
        return work, is_new
