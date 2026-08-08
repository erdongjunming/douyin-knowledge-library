import json
from pathlib import Path

from douyin_library.upstream.jiji_normalizer import normalize_page, normalize_work

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_normalize_work_preserves_identity_and_raw_payload() -> None:
    favorite_fixture = _load("jiji_favorites_page.json")["aweme_list"]

    item = normalize_work(favorite_fixture[0])

    assert item is not None
    assert item.aweme_id == "7370000000000000000"
    assert item.author.sec_uid
    assert item.author.nickname == "心理学讲解"
    assert item.media_type == "gallery"
    assert item.image_urls == (
        "https://example.test/page-1.jpg",
        "https://example.test/page-2.jpg",
    )
    assert item.raw_payload["aweme_id"] == item.aweme_id


def test_normalize_page_filters_invalid_items_and_keeps_page_metadata(caplog) -> None:
    page = normalize_page(_load("jiji_favorites_page.json"), source_external_id="default")

    assert len(page.items) == 1
    assert page.items[0].aweme_id == "7370000000000000000"
    assert page.cursor == "0"
    assert page.has_more is False
    assert page.source_external_id == "default"
    assert page.raw_summary["item_count"] == 2
    assert "invalid_upstream_item" in caplog.text


def test_normalize_creator_video_extracts_author_and_media_fields() -> None:
    creator_fixture = _load("jiji_creator_page.json")["aweme_list"]

    item = normalize_work(creator_fixture[0])

    assert item is not None
    assert item.media_type == "video"
    assert item.duration_ms == 42000
    assert item.cover_url == "https://example.test/cover.jpg"
    assert item.download_urls == ("https://example.test/video.mp4",)
    assert item.published_at is not None


def test_normalize_work_unwraps_collection_entries() -> None:
    favorite_item = _load("jiji_favorites_page.json")["aweme_list"][0]

    item = normalize_work({"aweme_info": favorite_item})

    assert item is not None
    assert item.aweme_id == "7370000000000000000"
    assert item.raw_payload["aweme_id"] == item.aweme_id
