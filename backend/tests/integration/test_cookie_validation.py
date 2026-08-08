import logging

import httpx
import pytest

from douyin_library.api.routes.settings import get_cookie_validator
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
