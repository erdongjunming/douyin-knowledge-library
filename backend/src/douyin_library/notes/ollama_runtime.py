from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from douyin_library.config import Settings
from douyin_library.notes.providers.base import parse_base_url, sanitized_base_url

OllamaServiceStatus = Literal["missing", "stopped", "ready", "error"]
AsyncClientFactory = Callable[[], httpx.AsyncClient]
PopenFactory = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class OllamaStatus:
    executable_found: bool
    executable_path: str | None
    service_status: OllamaServiceStatus
    model_available: bool
    ready: bool
    model: str
    action_required: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_ollama_executable(explicit: Path | str | None = None) -> Path | None:
    if explicit is not None:
        explicit_path = Path(explicit)
        if explicit_path.is_file():
            return explicit_path

    path_executable = shutil.which("ollama")
    if path_executable:
        return Path(path_executable)

    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return None
    standard_path = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
    return standard_path if standard_path.is_file() else None


async def probe_ollama(
    settings: Settings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    client_factory: AsyncClientFactory | None = None,
) -> OllamaStatus:
    current_settings = settings or Settings()
    model = current_settings.note_ollama_model.strip()
    executable = resolve_ollama_executable(current_settings.note_ollama_executable)
    if executable is None:
        return OllamaStatus(
            executable_found=False,
            executable_path=None,
            service_status="missing",
            model_available=False,
            ready=False,
            model=model,
            action_required="Install Ollama or configure its executable path.",
        )

    base_url = sanitized_base_url(current_settings.note_ollama_url)
    if not base_url:
        return _status_error(executable, model)
    try:
        if client_factory is None:
            client = httpx.AsyncClient(
                transport=transport,
                timeout=current_settings.note_ollama_probe_timeout_seconds,
            )
        else:
            client = client_factory()
        async with client:
            response = await client.get(f"{base_url}/api/tags")
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return OllamaStatus(
            executable_found=True,
            executable_path=str(executable),
            service_status="stopped",
            model_available=False,
            ready=False,
            model=model,
            action_required="Start Ollama with 'ollama serve'.",
        )
    except Exception:
        return _status_error(executable, model)

    try:
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return _status_error(executable, model)
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return _status_error(executable, model)

    model_available = any(
        isinstance(record, dict)
        and (record.get("name") == model or record.get("model") == model)
        for record in payload["models"]
    )
    return OllamaStatus(
        executable_found=True,
        executable_path=str(executable),
        service_status="ready",
        model_available=model_available,
        ready=model_available,
        model=model,
        action_required=None if model_available else f"ollama pull {model}",
    )


class OllamaProcessManager:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        popen_factory: PopenFactory = subprocess.Popen,
        stop_timeout_seconds: float = 2.0,
    ) -> None:
        if stop_timeout_seconds <= 0:
            raise ValueError("ollama_stop_timeout_invalid")
        self.settings = settings or Settings()
        self._popen_factory = popen_factory
        self._stop_timeout_seconds = stop_timeout_seconds
        self.started_by_app = False
        self.process: Any | None = None

    def start_if_installed(self, executable: Path | str | None = None) -> bool:
        if not self.settings.note_ollama_autostart:
            return False
        ollama_host = _loopback_ollama_host(self.settings.note_ollama_url)
        if ollama_host is None:
            return False
        if self.started_by_app and self.process is not None and self.process.poll() is None:
            return True
        resolved = resolve_ollama_executable(
            executable if executable is not None else self.settings.note_ollama_executable
        )
        if resolved is None:
            return False

        kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "env": {**os.environ, "OLLAMA_HOST": ollama_host},
        }
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            kwargs["startupinfo"] = startupinfo
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        self.process = self._popen_factory([str(resolved), "serve"], **kwargs)
        self.started_by_app = True
        return True

    def stop_if_started(self) -> bool:
        if not self.started_by_app or self.process is None:
            return False
        process = self.process
        if process.poll() is not None:
            self._clear_process(process)
            return True

        process.terminate()
        try:
            process.wait(timeout=self._stop_timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=self._stop_timeout_seconds)
        self._clear_process(process)
        return True

    def _clear_process(self, process: Any) -> None:
        if self.process is process:
            self.started_by_app = False
            self.process = None


def _status_error(executable: Path, model: str) -> OllamaStatus:
    return OllamaStatus(
        executable_found=True,
        executable_path=str(executable),
        service_status="error",
        model_available=False,
        ready=False,
        model=model,
        action_required="Check the Ollama service and configured URL.",
    )


def _loopback_ollama_host(value: str) -> str | None:
    parsed = parse_base_url(value)
    if (
        parsed is None
        or parsed.scheme != "http"
        or not parsed.is_loopback
        or parsed.has_credentials
        or parsed.path
        or parsed.has_query
        or parsed.has_fragment
    ):
        return None
    return parsed.netloc


__all__ = [
    "OllamaProcessManager",
    "OllamaStatus",
    "probe_ollama",
    "resolve_ollama_executable",
]
