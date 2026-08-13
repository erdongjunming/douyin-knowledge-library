from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

MediaType = Literal["video", "gallery", "live_photo"]


@dataclass(frozen=True, slots=True)
class DiscoveredAuthor:
    sec_uid: str | None
    nickname: str | None = None
    unique_id: str | None = None
    avatar_url: str | None = None


@dataclass(frozen=True, slots=True)
class DiscoveredWork:
    aweme_id: str
    author: DiscoveredAuthor
    media_type: MediaType
    description: str | None = None
    share_url: str | None = None
    published_at: datetime | None = None
    duration_ms: int | None = None
    cover_url: str | None = None
    download_urls: tuple[str, ...] = ()
    image_urls: tuple[str, ...] = ()
    stats: dict[str, Any] = field(default_factory=dict)
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DiscoveryPage:
    items: list[DiscoveredWork]
    cursor: str
    has_more: bool
    source_external_id: str
    source_name: str | None = None
    raw_summary: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DownloadResult:
    status: Literal["succeeded", "failed"]
    video_path: Path | None = None
    cover_path: Path | None = None
    metadata_path: Path | None = None
    error_code: str | None = None
    error_detail: str | None = None


class Collector(Protocol):
    async def creator_posts(self, sec_uid: str) -> AsyncIterator[DiscoveryPage]: ...

    async def all_favorites(self) -> AsyncIterator[DiscoveryPage]: ...

    async def search_works(self, keyword: str, limit: int) -> list[DiscoveredWork]: ...

    async def hot_board(self, limit: int) -> list[DiscoveredWork]: ...

    async def download(self, work: DiscoveredWork, target: Path) -> DownloadResult: ...
