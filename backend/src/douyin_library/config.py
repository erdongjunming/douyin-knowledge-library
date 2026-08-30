from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DKL_", extra="ignore")

    storage_root: Path = Path(r"D:\DouyinKnowledgeLibrary")
    transcription_subtitle_root: Path | None = None
    job_concurrency: int = Field(default=2, ge=1, le=8)
    transcription_concurrency: int = Field(default=1, ge=1, le=2)
    job_heartbeat_seconds: float = Field(default=10.0, ge=1.0, le=60.0)
    job_stale_seconds: float = Field(default=120.0, ge=30.0, le=3600.0)
    # Production runs the API and worker pools as separate processes.  Tests
    # and small embedded deployments can opt back into the in-process workers.
    embedded_workers: bool = False
    worker_poll_seconds: float = Field(default=1.0, ge=0.05, le=30.0)
    gpu_resource_name: str = "gpu:0"
    gpu_lease_ttl_seconds: float = Field(default=120.0, ge=5.0, le=3600.0)
    gpu_lease_poll_seconds: float = Field(default=0.5, ge=0.01, le=30.0)
    transcription_model: str = "Qwen/Qwen3-ASR-0.6B"
    transcription_language: str = "Chinese"
    qwen_chunk_seconds: float = 60.0
    qwen_python_path: Path | None = None
    qwen_asr_script_path: Path | None = None
    ffmpeg_path: Path | None = None
    ffprobe_path: Path | None = None
    audit_inline_max_bytes: int = Field(default=64 * 1024, ge=4096)
    database_backup_interval_hours: float = Field(default=24.0, ge=1.0)
    database_backup_retention: int = Field(default=7, ge=1, le=90)
    note_provider_policy: Literal["auto", "local", "external", "rule"] = "auto"
    note_ollama_url: str = "http://127.0.0.1:11434"
    # This model is present in the local setup used by the project.  Operators
    # can override it with DKL_NOTE_OLLAMA_MODEL without triggering a download.
    note_ollama_model: str = "qwen3:4b"
    note_ollama_executable: Path | None = None
    note_ollama_autostart: bool = True
    note_ollama_probe_timeout_seconds: float = Field(default=2, ge=0.2, le=30)
    note_ollama_unload_timeout_seconds: float = Field(default=10, ge=0.05, le=60)
    note_request_timeout_seconds: float = Field(default=600, ge=10, le=3600)
    note_context_tokens: int = Field(default=32768, ge=4096, le=131072)
    note_ollama_keep_alive: str = "10m"
    note_external_base_url: str | None = None
    note_external_model: str | None = None
    note_external_api_key: SecretStr | None = None
    note_external_input_cost_per_million: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )
    note_external_output_cost_per_million: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def _default_transcription_subtitle_root(self) -> Settings:
        if self.transcription_subtitle_root is None:
            self.transcription_subtitle_root = self.storage_root / "subtitle"
        return self

    @field_validator("note_ollama_keep_alive")
    @classmethod
    def _strip_note_ollama_keep_alive(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("note_ollama_keep_alive_empty")
        return stripped


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
    qwen_python = settings.qwen_python_path or (
        Path.home() / ".cache" / "rimagination-notes" / "qwen3-asr-venv" / "Scripts" / "python.exe"
    )
    qwen_script = settings.qwen_asr_script_path or (
        Path.home() / ".codex" / "skills" / "dy-note" / "scripts" / "run_qwen_asr.py"
    )
    qwen_ready = qwen_python.is_file() and qwen_script.is_file()

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
        "qwen3_asr": {
            "status": "ok" if qwen_ready else "missing",
            **({"path": str(qwen_python)} if qwen_python.is_file() else {}),
            "model": settings.transcription_model,
        },
    }
