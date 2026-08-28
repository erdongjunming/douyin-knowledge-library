import subprocess
from pathlib import Path

import httpx
import pytest

from douyin_library.config import Settings
from douyin_library.notes.ollama_runtime import (
    OllamaProcessManager,
    probe_ollama,
    resolve_ollama_executable,
)


def _executable(tmp_path: Path) -> Path:
    path = tmp_path / "ollama.exe"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"exe")
    return path


class _StopProcess:
    def __init__(
        self,
        *,
        poll_result=None,
        terminate_error: Exception | None = None,
        wait_results=None,
        kill_error: Exception | None = None,
    ) -> None:
        self.poll_result = poll_result
        self.terminate_error = terminate_error
        self.wait_results = list(wait_results or [])
        self.kill_error = kill_error
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_timeouts: list[float] = []

    def poll(self):
        return self.poll_result

    def terminate(self):
        self.terminate_calls += 1
        if self.terminate_error is not None:
            raise self.terminate_error

    def kill(self):
        self.kill_calls += 1
        if self.kill_error is not None:
            raise self.kill_error

    def wait(self, timeout):
        self.wait_timeouts.append(timeout)
        result = self.wait_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def test_resolve_ollama_executable_prefers_existing_explicit_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    explicit = _executable(tmp_path)
    monkeypatch.setattr("douyin_library.notes.ollama_runtime.shutil.which", lambda _: "PATH")

    assert resolve_ollama_executable(explicit) == explicit


def test_resolve_ollama_executable_uses_path_before_standard_location(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path_executable = _executable(tmp_path / "path")
    local_app_data = tmp_path / "local"
    standard = local_app_data / "Programs" / "Ollama" / "ollama.exe"
    standard.parent.mkdir(parents=True)
    standard.write_bytes(b"exe")
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setattr(
        "douyin_library.notes.ollama_runtime.shutil.which",
        lambda _: str(path_executable),
    )

    assert resolve_ollama_executable(tmp_path / "missing.exe") == path_executable


def test_resolve_ollama_executable_uses_standard_windows_location(
    monkeypatch,
    tmp_path: Path,
) -> None:
    standard = tmp_path / "Programs" / "Ollama" / "ollama.exe"
    standard.parent.mkdir(parents=True)
    standard.write_bytes(b"exe")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr("douyin_library.notes.ollama_runtime.shutil.which", lambda _: None)

    assert resolve_ollama_executable() == standard


@pytest.mark.asyncio
async def test_probe_reports_missing_executable_without_contacting_service(
    monkeypatch,
    tmp_path: Path,
) -> None:
    requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json={"models": []})

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr("douyin_library.notes.ollama_runtime.shutil.which", lambda _: None)
    settings = Settings(note_ollama_executable=tmp_path / "missing.exe")

    status = await probe_ollama(settings, transport=httpx.MockTransport(handler))

    assert status.executable_found is False
    assert status.executable_path is None
    assert status.service_status == "missing"
    assert status.model_available is False
    assert status.ready is False
    assert status.model == "qwen3:8b"
    assert status.action_required
    assert requests == 0
    assert status.as_dict()["executable_found"] is False


@pytest.mark.asyncio
async def test_probe_reports_stopped_service_on_connection_failure(tmp_path: Path) -> None:
    executable = _executable(tmp_path)

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    status = await probe_ollama(
        Settings(note_ollama_executable=executable, note_ollama_url="http://ollama.test"),
        transport=httpx.MockTransport(handler),
    )

    assert status.executable_found is True
    assert status.executable_path == str(executable)
    assert status.service_status == "stopped"
    assert status.model_available is False
    assert status.ready is False


@pytest.mark.asyncio
async def test_probe_reports_error_for_http_or_invalid_response(tmp_path: Path) -> None:
    executable = _executable(tmp_path)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="private response")

    status = await probe_ollama(
        Settings(note_ollama_executable=executable, note_ollama_url="http://ollama.test"),
        transport=httpx.MockTransport(handler),
    )

    assert status.service_status == "error"
    assert status.ready is False
    assert "private response" not in repr(status)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://[::1/url-secret",
        "http://127.0.0.1:url-secret",
        "https://api.example.test:url-secret/v1",
        "ollama.test/url-secret",
        "http:///url-secret",
    ],
)
async def test_probe_reports_sanitized_error_for_malformed_url(
    url: str,
    tmp_path: Path,
) -> None:
    requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json={"models": []})

    status = await probe_ollama(
        Settings(note_ollama_executable=_executable(tmp_path), note_ollama_url=url),
        transport=httpx.MockTransport(handler),
    )

    assert status.service_status == "error"
    assert status.ready is False
    assert requests == 0
    assert "url-secret" not in repr(status)
    assert "url-secret" not in repr(status.as_dict())
    assert "url-secret" not in (status.action_required or "")


@pytest.mark.asyncio
async def test_probe_reports_model_missing_and_never_pulls(tmp_path: Path) -> None:
    executable = _executable(tmp_path)
    paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen3:8b-latest"}, {"model": "other:1b"}]},
        )

    status = await probe_ollama(
        Settings(note_ollama_executable=executable, note_ollama_url="http://ollama.test"),
        transport=httpx.MockTransport(handler),
    )

    assert status.service_status == "ready"
    assert status.model_available is False
    assert status.ready is False
    assert status.action_required == "ollama pull qwen3:8b"
    assert paths == ["/api/tags"]
    assert "/api/pull" not in paths


@pytest.mark.asyncio
@pytest.mark.parametrize("model_record", [{"name": "qwen3:8b"}, {"model": "qwen3:8b"}])
async def test_probe_matches_exact_name_or_model_field(
    tmp_path: Path,
    model_record: dict[str, str],
) -> None:
    executable = _executable(tmp_path)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [model_record]})

    status = await probe_ollama(
        Settings(note_ollama_executable=executable, note_ollama_url="http://ollama.test"),
        transport=httpx.MockTransport(handler),
    )

    assert status.service_status == "ready"
    assert status.model_available is True
    assert status.ready is True
    assert status.action_required is None


def test_process_manager_starts_only_hidden_ollama_serve(tmp_path: Path) -> None:
    executable = _executable(tmp_path)
    captured: dict[str, object] = {}

    class FakeProcess:
        def poll(self):
            return None

        def terminate(self):
            raise AssertionError("not stopping during start")

    process = FakeProcess()

    def popen_factory(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return process

    manager = OllamaProcessManager(
        Settings(note_ollama_executable=executable),
        popen_factory=popen_factory,
    )

    assert manager.start_if_installed() is True

    assert captured["command"] == [str(executable), "serve"]
    assert "pull" not in captured["command"]
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW
    assert kwargs["startupinfo"].dwFlags & subprocess.STARTF_USESHOWWINDOW
    assert kwargs["startupinfo"].wShowWindow == subprocess.SW_HIDE
    assert manager.started_by_app is True
    assert manager.process is process


def test_process_manager_custom_loopback_url_sets_independent_ollama_host_env(
    monkeypatch,
    tmp_path: Path,
) -> None:
    executable = _executable(tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setenv("DKL_PROCESS_ENV_SENTINEL", "preserved")

    class FakeProcess:
        def poll(self):
            return None

    def popen_factory(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    manager = OllamaProcessManager(
        Settings(
            note_ollama_executable=executable,
            note_ollama_url="http://localhost:22434",
        ),
        popen_factory=popen_factory,
    )

    assert manager.start_if_installed() is True

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    environment = kwargs["env"]
    assert environment["OLLAMA_HOST"] == "localhost:22434"
    assert environment["DKL_PROCESS_ENV_SENTINEL"] == "preserved"


@pytest.mark.parametrize(
    "url",
    [
        "http://192.0.2.10:11434",
        "https://localhost:11434",
        "not-a-url",
        "http://[::1/url-secret",
        "http://127.0.0.1:url-secret",
        "https://api.example.test:url-secret/v1",
    ],
)
def test_process_manager_rejects_remote_or_invalid_ollama_url(
    url: str,
    tmp_path: Path,
) -> None:
    calls = 0

    def popen_factory(command, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("must not spawn")

    manager = OllamaProcessManager(
        Settings(note_ollama_executable=_executable(tmp_path), note_ollama_url=url),
        popen_factory=popen_factory,
    )

    assert manager.start_if_installed() is False
    assert calls == 0


def test_process_manager_respects_autostart_and_missing_executable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = 0

    def popen_factory(command, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("must not start")

    monkeypatch.setattr("douyin_library.notes.ollama_runtime.shutil.which", lambda _: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "empty-local"))
    disabled = OllamaProcessManager(
        Settings(note_ollama_executable=_executable(tmp_path), note_ollama_autostart=False),
        popen_factory=popen_factory,
    )
    missing = OllamaProcessManager(
        Settings(note_ollama_executable=tmp_path / "missing.exe"),
        popen_factory=popen_factory,
    )

    assert disabled.start_if_installed() is False
    assert missing.start_if_installed() is False
    assert calls == 0


def test_process_manager_does_not_stop_user_owned_process(tmp_path: Path) -> None:
    executable = _executable(tmp_path)
    process = _StopProcess(wait_results=[0])
    manager = OllamaProcessManager(Settings(note_ollama_executable=executable))
    manager.process = process

    assert manager.stop_if_started() is False

    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert process.wait_timeouts == []
    assert manager.process is process
    assert manager.started_by_app is False


def test_process_manager_waits_for_owned_process_before_clearing(tmp_path: Path) -> None:
    process = _StopProcess(wait_results=[0])
    manager = OllamaProcessManager(
        Settings(note_ollama_executable=_executable(tmp_path)),
        stop_timeout_seconds=0.25,
    )
    manager.process = process
    manager.started_by_app = True

    assert manager.stop_if_started() is True

    assert process.terminate_calls == 1
    assert process.wait_timeouts == [0.25]
    assert process.kill_calls == 0
    assert manager.started_by_app is False
    assert manager.process is None


def test_process_manager_clears_already_exited_owned_process(tmp_path: Path) -> None:
    process = _StopProcess(poll_result=0)
    manager = OllamaProcessManager(Settings(note_ollama_executable=_executable(tmp_path)))
    manager.process = process
    manager.started_by_app = True

    assert manager.stop_if_started() is True

    assert process.terminate_calls == 0
    assert process.wait_timeouts == []
    assert manager.started_by_app is False
    assert manager.process is None


def test_process_manager_preserves_tracking_when_terminate_fails(tmp_path: Path) -> None:
    process = _StopProcess(terminate_error=RuntimeError("terminate failed"))
    manager = OllamaProcessManager(Settings(note_ollama_executable=_executable(tmp_path)))
    manager.process = process
    manager.started_by_app = True

    with pytest.raises(RuntimeError, match="terminate failed"):
        manager.stop_if_started()

    assert manager.started_by_app is True
    assert manager.process is process


def test_process_manager_kills_after_stop_timeout_then_confirms_exit(tmp_path: Path) -> None:
    timeout = subprocess.TimeoutExpired("ollama serve", 0.01)
    process = _StopProcess(wait_results=[timeout, 0])
    manager = OllamaProcessManager(
        Settings(note_ollama_executable=_executable(tmp_path)),
        stop_timeout_seconds=0.01,
    )
    manager.process = process
    manager.started_by_app = True

    assert manager.stop_if_started() is True

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_timeouts == [0.01, 0.01]
    assert manager.started_by_app is False
    assert manager.process is None


def test_process_manager_preserves_tracking_when_kill_fails(tmp_path: Path) -> None:
    timeout = subprocess.TimeoutExpired("ollama serve", 0.01)
    process = _StopProcess(
        wait_results=[timeout],
        kill_error=RuntimeError("kill failed"),
    )
    manager = OllamaProcessManager(
        Settings(note_ollama_executable=_executable(tmp_path)),
        stop_timeout_seconds=0.01,
    )
    manager.process = process
    manager.started_by_app = True

    with pytest.raises(RuntimeError, match="kill failed"):
        manager.stop_if_started()

    assert manager.started_by_app is True
    assert manager.process is process
