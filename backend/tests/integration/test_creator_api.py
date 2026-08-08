import httpx
import pytest

from douyin_library.api.routes.creators import get_creator_sync_service
from douyin_library.main import app


class FakeService:
    called = False

    async def sync(self, reference: str):
        self.called = True
        raise AssertionError("unconfirmed request must not sync")


@pytest.mark.asyncio
async def test_creator_sync_requires_confirmation() -> None:
    fake_service = FakeService()
    app.dependency_overrides[get_creator_sync_service] = lambda: fake_service
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/creators/sync",
                json={"reference": "MS4wLjABAAAAtestcreator", "confirmed": False},
            )
    finally:
        app.dependency_overrides.pop(get_creator_sync_service, None)

    assert response.status_code == 409
    assert response.json()["detail"] == "creator_confirmation_required"
    assert fake_service.called is False
