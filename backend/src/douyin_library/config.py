from __future__ import annotations

import os
import shutil
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DKL_", extra="ignore")

    storage_root: Path = Path(r"D:\DouyinKnowledgeLibrary")
    job_concurrency: int = 2
    transcription_model: str = "small"
    ffmpeg_path: Path | None = None
    ffprobe_path: Path | None = None


def resolve_tool(settings: Settings, name: str) -> Path | None:
    explicit = getattr(settings, f"{name}_path")
    if explicit is not None and explicit.is_file():
        return explicit

    system_path = shutil.which(name)
    if system_path:
        return Path(system_path)

    executable = f"{name}.exe" if os.name == "nt" else name
    tool_root = settings.storage_root / "tools" / "ffmpeg"
    candidates = sorted(tool_root.glob(f"*/bin/{executable}"))
    return candidates[0] if candidates else None


def diagnostics(settings: Settings) -> dict[str, dict[str, str]]:
    storage_status = "ok" if settings.storage_root.exists() else "missing"
    storage_detail = str(settings.storage_root)
    ffmpeg = resolve_tool(settings, "ffmpeg")
    ffprobe = resolve_tool(settings, "ffprobe")

    return {
        "ffmpeg": {
            "status": "ok" if ffmpeg else "missing",
            **({"path": str(ffmpeg)} if ffmpeg else {}),
        },
        "ffprobe": {
            "status": "ok" if ffprobe else "missing",
            **({"path": str(ffprobe)} if ffprobe else {}),
        },
        "storage": {
            "status": storage_status,
            "path": storage_detail,
        },
    }
