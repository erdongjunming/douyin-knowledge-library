from pathlib import Path

import httpx
import pytest

from douyin_library.main import app


@pytest.mark.asyncio
async def test_diagnostics_reports_missing_ffmpeg(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("douyin_library.config.shutil.which", lambda _: None)
    monkeypatch.setenv("DKL_STORAGE_ROOT", str(tmp_path))
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/settings/diagnostics")

    assert response.status_code == 200
    assert response.json()["ffmpeg"]["status"] == "missing"

