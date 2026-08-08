import json
from pathlib import Path

import pytest

from douyin_library.upstream.jiji_gateway import JijiGateway

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeJiji:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.favorite = _load("jiji_favorites_page.json")["aweme_list"][0]

    async def get_user_collection(self, sec_uid, max_cursor=0, count=20):
        self.calls.append("get_user_collection")
        return {"aweme_list": [self.favorite], "has_more": False, "max_cursor": 0}

    async def get_user_collects(self, sec_uid, max_cursor=0, count=10):
        self.calls.append("get_user_collects")
        return {
            "collects_list": [{"collects_id": "folder-42", "name": "重点收藏"}],
            "has_more": False,
            "max_cursor": 0,
        }

    async def get_collect_aweme(self, collects_id, max_cursor=0, count=10):
        self.calls.append("get_collect_aweme")
        return {"aweme_list": [self.favorite], "has_more": False, "max_cursor": 0}


class FakeRateLimiter:
    def __init__(self) -> None:
        self.acquire_count = 0

    async def acquire(self) -> None:
        self.acquire_count += 1


@pytest.mark.asyncio
async def test_gateway_reads_default_and_custom_favorites() -> None:
    fake_jiji = FakeJiji()
    limiter = FakeRateLimiter()
    pages = [
        page
        async for page in JijiGateway(fake_jiji, rate_limiter=limiter).all_favorites()
    ]

    assert {page.source_external_id for page in pages} == {"default", "folder-42"}
    assert pages[1].source_name == "重点收藏"
    assert fake_jiji.calls == [
        "get_user_collection",
        "get_user_collects",
        "get_collect_aweme",
    ]
    assert limiter.acquire_count == 3


def test_gateway_rejects_an_incompatible_upstream_version() -> None:
    with pytest.raises(RuntimeError, match="upstream_version_mismatch"):
        JijiGateway(FakeJiji(), upstream_version="0.0.0").assert_compatible()
