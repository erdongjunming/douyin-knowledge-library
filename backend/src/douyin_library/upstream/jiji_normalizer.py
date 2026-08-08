from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from douyin_library.upstream.contracts import (
    DiscoveredAuthor,
    DiscoveredWork,
    DiscoveryPage,
    MediaType,
)

logger = logging.getLogger(__name__)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_url(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    mapping = _as_dict(value)
    urls = mapping.get("url_list") or mapping.get("urlList")
    if isinstance(urls, list):
        for url in urls:
            if isinstance(url, str) and url.strip():
                return url.strip()
    return None


def _all_urls(value: Any) -> tuple[str, ...]:
    mapping = _as_dict(value)
    urls = mapping.get("url_list") or mapping.get("urlList")
    if not isinstance(urls, list):
        return ()
    return tuple(url.strip() for url in urls if isinstance(url, str) and url.strip())


def _author(item: dict[str, Any]) -> DiscoveredAuthor:
    payload = _as_dict(item.get("author") or item.get("author_info"))
    avatar = payload.get("avatar_larger") or payload.get("avatar_medium") or payload.get(
        "avatar_thumb"
    )
    return DiscoveredAuthor(
        sec_uid=_text(payload.get("sec_uid") or payload.get("sec_user_id")),
        nickname=_text(payload.get("nickname")),
        unique_id=_text(payload.get("unique_id") or payload.get("short_id")),
        avatar_url=_first_url(avatar),
    )


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _media_type(item: dict[str, Any]) -> MediaType:
    images = item.get("images")
    if isinstance(images, list) and images:
        if any(isinstance(image, dict) and image.get("video") for image in images):
            return "live_photo"
        return "gallery"
    return "video"


def _image_urls(item: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    images = item.get("images")
    if not isinstance(images, list):
        return ()
    for image in images:
        payload = _as_dict(image)
        candidate = payload.get("display_image") or payload.get("origin_image") or payload
        url = _first_url(candidate)
        if url:
            values.append(url)
    return tuple(values)


def _download_urls(item: dict[str, Any]) -> tuple[str, ...]:
    video = _as_dict(item.get("video"))
    values: list[str] = []
    for candidate in (video.get("play_addr"), video.get("download_addr"), video.get("play_api")):
        values.extend(_all_urls(candidate))
    return tuple(dict.fromkeys(values))


def _published_at(value: Any) -> datetime | None:
    try:
        if value is None:
            return None
        return datetime.fromtimestamp(float(value), tz=UTC)
    except (TypeError, ValueError, OverflowError):
        return None


def normalize_work(payload: Any) -> DiscoveredWork | None:
    if not isinstance(payload, dict):
        logger.warning("invalid_upstream_item: expected object")
        return None

    for wrapper_key in ("aweme", "aweme_info", "aweme_detail"):
        wrapped = payload.get(wrapper_key)
        if isinstance(wrapped, dict):
            payload = wrapped
            break

    aweme_id = _text(payload.get("aweme_id") or payload.get("awemeId"))
    if not aweme_id:
        logger.warning("invalid_upstream_item: missing aweme_id")
        return None

    media_type = _media_type(payload)
    video = _as_dict(payload.get("video"))
    images = _image_urls(payload)
    cover = _first_url(video.get("cover") or video.get("origin_cover"))
    if cover is None and images:
        cover = images[0]
    duration = video.get("duration")
    try:
        duration_ms = int(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration_ms = None

    return DiscoveredWork(
        aweme_id=aweme_id,
        author=_author(payload),
        media_type=media_type,
        description=_text(payload.get("desc") or payload.get("description")),
        share_url=_text(payload.get("share_url") or payload.get("shareUrl")),
        published_at=_published_at(payload.get("create_time") or payload.get("created_at")),
        duration_ms=duration_ms,
        cover_url=cover,
        download_urls=_download_urls(payload),
        image_urls=images,
        stats=_as_dict(payload.get("statistics") or payload.get("stats")),
        raw_payload=payload,
    )


def normalize_page(
    payload: Any,
    *,
    source_external_id: str,
    source_name: str | None = None,
) -> DiscoveryPage:
    response = _as_dict(payload)
    raw_items = response.get("items")
    if not isinstance(raw_items, list):
        raw_items = response.get("aweme_list")
    if not isinstance(raw_items, list):
        raw_items = []

    items = [item for raw in raw_items if (item := normalize_work(raw)) is not None]
    cursor_value = response.get("max_cursor")
    if cursor_value is None:
        cursor_value = response.get("cursor", 0)
    return DiscoveryPage(
        items=items,
        cursor=str(cursor_value),
        has_more=bool(response.get("has_more")),
        source_external_id=source_external_id,
        source_name=source_name,
        raw_summary={
            "status_code": response.get("status_code"),
            "item_count": len(raw_items),
            "normalized_count": len(items),
            "has_more": bool(response.get("has_more")),
            "cursor": str(cursor_value),
            "risk_flags": response.get("risk_flags") or {},
        },
    )
