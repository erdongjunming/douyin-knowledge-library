from __future__ import annotations

import inspect
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

from douyin_library.upstream.contracts import (
    DiscoveredWork,
    DiscoveryPage,
    DownloadResult,
)
from douyin_library.upstream.jiji_normalizer import normalize_page, normalize_work

logger = logging.getLogger(__name__)

EXPECTED_UPSTREAM_VERSION = "2.0.0"
PINNED_UPSTREAM_COMMIT = "ad7338fdc474c14e2063370540a362cd8a953b43"


class UpstreamPaginationError(RuntimeError):
    pass


DownloadHandler = Callable[[DiscoveredWork, Path], Awaitable[DownloadResult]]


class JijiGateway:
    def __init__(
        self,
        api_client: Any,
        *,
        upstream_version: str | None = None,
        download_handler: DownloadHandler | None = None,
        rate_limiter: Any | None = None,
    ) -> None:
        self.api_client = api_client
        self.upstream_version = upstream_version
        self.download_handler = download_handler
        self.rate_limiter = rate_limiter

    @classmethod
    def from_cookie_header(cls, cookie_header: str) -> JijiGateway:
        from control import RateLimiter
        from core.api_client import DouyinAPIClient

        return cls(
            DouyinAPIClient(_parse_cookie_header(cookie_header)),
            upstream_version=version("douyin-downloader"),
            rate_limiter=RateLimiter(max_per_second=1.5),
        )

    def assert_compatible(self) -> None:
        if self.upstream_version and self.upstream_version != EXPECTED_UPSTREAM_VERSION:
            raise RuntimeError(
                "upstream_version_mismatch: "
                f"expected {EXPECTED_UPSTREAM_VERSION} at {PINNED_UPSTREAM_COMMIT}, "
                f"got {self.upstream_version}"
            )

    async def close(self) -> None:
        closer = getattr(self.api_client, "close", None)
        if callable(closer):
            result = closer()
            if inspect.isawaitable(result):
                await result

    async def _acquire(self) -> None:
        acquire = getattr(self.rate_limiter, "acquire", None)
        if callable(acquire):
            await acquire()

    async def _work_pages(
        self,
        fetcher: Callable[..., Awaitable[dict[str, Any]]],
        fetch_id: str,
        *,
        source_external_id: str,
        source_name: str | None = None,
        count: int = 20,
    ) -> AsyncIterator[DiscoveryPage]:
        cursor = 0
        while True:
            await self._acquire()
            payload = await fetcher(fetch_id, max_cursor=cursor, count=count)
            page = normalize_page(
                payload,
                source_external_id=source_external_id,
                source_name=source_name,
            )
            yield page
            if not page.has_more:
                break
            try:
                next_cursor = int(page.cursor)
            except ValueError as exc:
                raise UpstreamPaginationError("invalid_upstream_cursor") from exc
            if next_cursor == cursor:
                raise UpstreamPaginationError("upstream_cursor_stalled")
            cursor = next_cursor

    async def creator_posts(self, sec_uid: str) -> AsyncIterator[DiscoveryPage]:
        self.assert_compatible()
        async for page in self._work_pages(
            self.api_client.get_user_post,
            sec_uid,
            source_external_id=sec_uid,
        ):
            yield page

    async def resolve_creator(self, reference: str) -> dict[str, str | None]:
        self.assert_compatible()
        resolved = reference.strip()
        match = re.search(r"/user/([A-Za-z0-9_-]+)", resolved)
        if resolved.startswith(("http://", "https://")) and match is None:
            await self._acquire()
            final_url = await self.api_client.resolve_short_url(resolved)
            if not final_url:
                raise ValueError("creator_short_url_unresolved")
            resolved = final_url
            match = re.search(r"/user/([A-Za-z0-9_-]+)", resolved)
        sec_uid = match.group(1) if match else resolved
        if not sec_uid or sec_uid.startswith(("http://", "https://")):
            raise ValueError("creator_profile_url_invalid")
        await self._acquire()
        profile = await self.api_client.get_user_info(sec_uid)
        if not isinstance(profile, dict):
            raise ValueError("creator_profile_unavailable")
        stable_sec_uid = str(profile.get("sec_uid") or sec_uid).strip()
        return {
            "sec_uid": stable_sec_uid,
            "nickname": _optional_text(profile.get("nickname")),
            "unique_id": _optional_text(profile.get("unique_id") or profile.get("short_id")),
            "profile_url": f"https://www.douyin.com/user/{stable_sec_uid}",
            "avatar_url": _nested_first_url(
                profile.get("avatar_larger")
                or profile.get("avatar_medium")
                or profile.get("avatar_thumb")
            ),
        }

    async def all_favorites(self) -> AsyncIterator[DiscoveryPage]:
        self.assert_compatible()
        async for page in self._work_pages(
            self.api_client.get_user_collection,
            "self",
            source_external_id="default",
        ):
            yield page

        cursor = 0
        while True:
            await self._acquire()
            payload = await self.api_client.get_user_collects(
                "self", max_cursor=cursor, count=10
            )
            folders = payload.get("items")
            if not isinstance(folders, list):
                folders = payload.get("collects_list")
            if not isinstance(folders, list):
                folders = []
            for folder in folders:
                if not isinstance(folder, dict):
                    continue
                folder_info = folder.get("collects_info")
                if isinstance(folder_info, dict):
                    folder = {**folder_info, **folder}
                folder_id = str(
                    folder.get("collects_id")
                    or folder.get("collects_id_str")
                    or folder.get("id")
                    or ""
                ).strip()
                if not folder_id:
                    logger.warning("invalid_upstream_folder: missing collects_id")
                    continue
                folder_name = str(folder.get("name") or folder.get("collects_name") or "").strip()
                async for page in self._work_pages(
                    self.api_client.get_collect_aweme,
                    folder_id,
                    source_external_id=folder_id,
                    source_name=folder_name or None,
                    count=10,
                ):
                    yield page

            if not bool(payload.get("has_more")):
                break
            next_cursor = int(payload.get("max_cursor") or 0)
            if next_cursor == cursor:
                raise UpstreamPaginationError("upstream_folder_cursor_stalled")
            cursor = next_cursor

    async def search_works(self, keyword: str, limit: int) -> list[DiscoveredWork]:
        self.assert_compatible()
        works: list[DiscoveredWork] = []
        seen: set[str] = set()
        offset = 0
        while len(works) < limit:
            await self._acquire()
            payload = await self.api_client.search_aweme(
                keyword,
                offset=offset,
                count=min(20, limit - len(works)),
            )
            for raw in payload.get("items") or []:
                item = normalize_work(raw)
                if item and item.aweme_id not in seen:
                    seen.add(item.aweme_id)
                    works.append(item)
            if not payload.get("has_more"):
                break
            next_offset = int(payload.get("max_cursor") or 0)
            if next_offset == offset:
                raise UpstreamPaginationError("upstream_search_cursor_stalled")
            offset = next_offset
        return works

    async def hot_board(self, limit: int) -> list[DiscoveredWork]:
        self.assert_compatible()
        await self._acquire()
        payload = await self.api_client.get_hot_search_board()
        works = [normalize_work(raw) for raw in (payload.get("items") or [])[:limit]]
        return [work for work in works if work is not None]

    async def download(self, work: DiscoveredWork, target: Path) -> DownloadResult:
        if self.download_handler is None:
            return DownloadResult(
                status="failed",
                error_code="download_handler_unavailable",
                error_detail="The jiji downloader adapter is not configured",
            )
        return await self.download_handler(work, target)

    async def validate_account(self) -> dict[str, object]:
        self.assert_compatible()
        try:
            await self._acquire()
            profile = await self.api_client.get_self_info()
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            risk_controlled = any(marker in message for marker in ("403", "429", "risk"))
            logger.info("Douyin cookie validation failed: risk_controlled=%s", risk_controlled)
            return {
                "valid": False,
                "account_nickname": None,
                "expires_or_unknown": "unknown",
                "risk_controlled": risk_controlled,
            }
        valid = isinstance(profile, dict) and bool(profile.get("sec_uid") or profile.get("uid"))
        result = {
            "valid": valid,
            "account_nickname": profile.get("nickname") if valid else None,
            "expires_or_unknown": "unknown",
            "risk_controlled": False,
        }
        logger.info("Douyin cookie validation finished: valid=%s", valid)
        return result


def _parse_cookie_header(cookie_header: str) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for part in cookie_header.split(";"):
        name, separator, value = part.strip().partition("=")
        if separator and name and value:
            cookies[name] = value
    return cookies


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _nested_first_url(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    urls = value.get("url_list")
    if not isinstance(urls, list):
        return None
    return next((url for url in urls if isinstance(url, str) and url.strip()), None)
