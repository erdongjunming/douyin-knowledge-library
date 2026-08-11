from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from douyin_library.models import Creator, Source, SyncRun, Work, WorkSource


@dataclass(frozen=True, slots=True)
class CreatorSyncResult:
    creator: Creator
    new_work_count: int
    new_work_ids: tuple[int, ...]
    discovered_count: int
    status: str

    @property
    def complete(self) -> bool:
        return self.status == "complete"


class CreatorSyncService:
    def __init__(self, session: Session, gateway: Any) -> None:
        self.session = session
        self.gateway = gateway

    async def ensure_creator(self, reference_or_id: str | int) -> Creator:
        existing: Creator | None = None
        if isinstance(reference_or_id, int):
            existing = self.session.get(Creator, reference_or_id)
            if existing is None:
                raise ValueError("creator_not_found")
            reference = existing.profile_url or existing.sec_uid
        else:
            reference = reference_or_id

        profile = await self.gateway.resolve_creator(reference)
        sec_uid = str(profile.get("sec_uid") or "").strip()
        if not sec_uid:
            raise ValueError("creator_identity_missing_sec_uid")

        creator = self.session.scalar(select(Creator).where(Creator.sec_uid == sec_uid))
        if creator is None:
            creator = Creator(sec_uid=sec_uid)
            self.session.add(creator)
        for field in ("nickname", "unique_id", "profile_url", "avatar_url"):
            value = profile.get(field)
            if value is not None:
                setattr(creator, field, str(value))
        self.session.commit()
        self.session.refresh(creator)
        return creator

    async def sync(self, reference_or_id: str | int) -> CreatorSyncResult:
        creator = await self.ensure_creator(reference_or_id)
        source = self.session.scalar(
            select(Source).where(
                Source.kind == "creator_post",
                Source.external_id == creator.sec_uid,
            )
        )
        if source is None:
            source = Source(
                kind="creator_post",
                external_id=creator.sec_uid,
                name=creator.nickname,
            )
            self.session.add(source)
            self.session.flush()

        run = SyncRun(
            source_kind="creator_post",
            source_external_id=creator.sec_uid,
            status="running",
        )
        self.session.add(run)
        self.session.commit()

        seen_work_ids: set[int] = set()
        new_work_ids: list[int] = []
        new_work_count = 0
        discovered_count = 0
        try:
            async for page in self.gateway.creator_posts(creator.sec_uid):
                for item in page.items:
                    work = self.session.scalar(
                        select(Work).where(Work.aweme_id == item.aweme_id)
                    )
                    if work is None:
                        work = Work(aweme_id=item.aweme_id)
                        self.session.add(work)
                        self.session.flush()
                        new_work_ids.append(work.id)
                        new_work_count += 1
                    self._update_work(work, item, creator.id)
                    relation = self.session.scalar(
                        select(WorkSource).where(
                            WorkSource.work_id == work.id,
                            WorkSource.source_id == source.id,
                        )
                    )
                    if relation is None:
                        relation = WorkSource(work_id=work.id, source_id=source.id)
                        self.session.add(relation)
                    relation.active = True
                    relation.last_seen_at = datetime.now(UTC)
                    seen_work_ids.add(work.id)
                    discovered_count += 1
                run.discovered_count = discovered_count
                run.cursor_summary_json = json.dumps(
                    {"cursor": page.cursor, "has_more": page.has_more}
                )
                self.session.commit()
        except Exception as exc:  # noqa: BLE001
            run.status = "incomplete"
            run.error_detail = str(exc)[:500]
            run.finished_at = datetime.now(UTC)
            self.session.commit()
            return CreatorSyncResult(
                creator=creator,
                new_work_count=new_work_count,
                new_work_ids=tuple(new_work_ids),
                discovered_count=discovered_count,
                status="incomplete",
            )

        for relation in self.session.scalars(
            select(WorkSource).where(WorkSource.source_id == source.id)
        ):
            if relation.work_id not in seen_work_ids:
                relation.active = False
        now = datetime.now(UTC)
        run.status = "complete"
        run.finished_at = now
        creator.last_synced_at = now
        source.name = creator.nickname
        self.session.commit()
        return CreatorSyncResult(
            creator=creator,
            new_work_count=new_work_count,
            new_work_ids=tuple(new_work_ids),
            discovered_count=discovered_count,
            status="complete",
        )

    @staticmethod
    def _update_work(work: Work, item: Any, creator_id: int) -> None:
        work.creator_id = creator_id
        work.media_type = item.media_type
        work.description = item.description
        work.share_url = item.share_url
        work.published_at = item.published_at
        work.duration_ms = item.duration_ms
        work.availability = "available"
        work.stats_json = json.dumps(item.stats, ensure_ascii=False)
        work.raw_metadata_json = json.dumps(item.raw_payload, ensure_ascii=False)
