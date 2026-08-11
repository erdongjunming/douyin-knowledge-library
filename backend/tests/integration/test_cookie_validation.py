import logging

import httpx
import pytest

from douyin_library.api.routes.settings import get_cookie_store, get_cookie_validator
from douyin_library.main import app


class FakeCookieValidator:
    async def validate(self, cookie_header: str | None) -> dict[str, object]:
        logging.getLogger("test-cookie").info("cookie validation requested")
        assert cookie_header == "sessionid=super-secret; ttwid=opaque"
        return {
            "valid": True,
            "account_nickname": "知识主播",
            "expires_or_unknown": "unknown",
            "risk_controlled": False,
        }


class InvalidCookieValidator:
    async def validate(self, cookie_header: str | None) -> dict[str, object]:
        return {
            "valid": False,
            "account_nickname": None,
            "expires_or_unknown": "unknown",
            "risk_controlled": False,
        }


class MemoryCookieStore:
    def __init__(self) -> None:
        self.value: str | None = None

    def save(self, cookie: str) -> None:
        self.value = cookie

    def read(self) -> str | None:
        return self.value

    def delete(self) -> None:
        self.value = None


@pytest.mark.asyncio
async def test_cookie_validation_returns_safe_account_status(caplog) -> None:
    app.dependency_overrides[get_cookie_validator] = lambda: FakeCookieValidator()
    caplog.set_level(logging.INFO)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/settings/douyin-cookie/validate",
                json={"cookie": "sessionid=super-secret; ttwid=opaque"},
            )
    finally:
        app.dependency_overrides.pop(get_cookie_validator, None)

    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "account_nickname": "知识主播",
        "expires_or_unknown": "unknown",
        "risk_controlled": False,
    }
    assert "super-secret" not in caplog.text


@pytest.mark.asyncio
async def test_valid_cookie_is_saved_only_after_validation() -> None:
    store = MemoryCookieStore()
    app.dependency_overrides[get_cookie_validator] = lambda: FakeCookieValidator()
    app.dependency_overrides[get_cookie_store] = lambda: store
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.put(
                "/api/settings/douyin-cookie",
                json={"cookie": "sessionid=super-secret; ttwid=opaque"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "configured": True,
        "valid": True,
        "account_nickname": "\u77e5\u8bc6\u4e3b\u64ad",
        "expires_or_unknown": "unknown",
        "risk_controlled": False,
    }
    assert store.read() == "sessionid=super-secret; ttwid=opaque"
    assert "cookie" not in response.text.lower()


@pytest.mark.asyncio
async def test_invalid_cookie_is_not_saved() -> None:
    store = MemoryCookieStore()
    app.dependency_overrides[get_cookie_validator] = lambda: InvalidCookieValidator()
    app.dependency_overrides[get_cookie_store] = lambda: store
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.put(
                "/api/settings/douyin-cookie",
                json={"cookie": "sessionid=invalid"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    assert response.json() == {"detail": "douyin_cookie_invalid"}
    assert store.read() is None


@pytest.mark.asyncio
async def test_cookie_status_and_delete_do_not_expose_secret() -> None:
    store = MemoryCookieStore()
    store.save("sessionid=super-secret")
    app.dependency_overrides[get_cookie_store] = lambda: store
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            status_response = await client.get("/api/settings/douyin-cookie")
            delete_response = await client.delete("/api/settings/douyin-cookie")
    finally:
        app.dependency_overrides.clear()

    assert status_response.json() == {"configured": True}
    assert delete_response.json() == {"configured": False}
    assert store.read() is None
    assert "super-secret" not in status_response.text
