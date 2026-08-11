import httpx
import pytest

from douyin_library.api.routes.jobs import get_job_queue
from douyin_library.main import app


class FakeQueue:
    called = False

    def create(self, **kwargs):
        self.called = True
        raise AssertionError("unconfirmed request must not sync")


@pytest.mark.asyncio
async def test_creator_sync_requires_confirmation() -> None:
    fake_queue = FakeQueue()
    app.dependency_overrides[get_job_queue] = lambda: fake_queue
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/creators/sync",
                json={"reference": "MS4wLjABAAAAtestcreator", "confirmed": False},
            )
    finally:
        app.dependency_overrides.pop(get_job_queue, None)

    assert response.status_code == 409
    assert response.json()["detail"] == "creator_confirmation_required"
    assert fake_queue.called is False
