from __future__ import annotations

import shutil
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DKL_", extra="ignore")

    storage_root: Path = Path(r"D:\DouyinKnowledgeLibrary")
    job_concurrency: int = 2
    transcription_model: str = "small"


def diagnostics(settings: Settings) -> dict[str, dict[str, str]]:
    storage_status = "ok" if settings.storage_root.exists() else "missing"
    storage_detail = str(settings.storage_root)

    return {
        "ffmpeg": {
            "status": "ok" if shutil.which("ffmpeg") else "missing",
        },
        "ffprobe": {
            "status": "ok" if shutil.which("ffprobe") else "missing",
        },
        "storage": {
            "status": storage_status,
            "path": storage_detail,
        },
    }

