import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_dev_launcher_starts_and_owns_all_local_processes() -> None:
    script = (PROJECT_ROOT / "scripts" / "dev.ps1").read_text(encoding="utf-8")

    assert "worker_main" in script
    assert re.search(r"worker_main.*--pool.*general", script, re.DOTALL)
    assert re.search(r"worker_main.*--pool.*asr", script, re.DOTALL)
    assert "Start-Process" in script
    assert "Stop-ProcessTree" in script
    assert "taskkill.exe" in script
    assert "finally" in script


def test_cmd_launcher_requests_process_cleanup() -> None:
    script = (PROJECT_ROOT / "start-local.cmd").read_text(encoding="utf-8")

    assert "DKL_STOP_BACKEND_ON_EXIT=1" in script


def test_dev_launcher_honors_backend_cleanup_override() -> None:
    script = (PROJECT_ROOT / "scripts" / "dev.ps1").read_text(encoding="utf-8")

    # The compatibility switch promises to leave a backend started by this
    # launcher alive when explicitly set to 0, while worker/frontend cleanup
    # remains unconditional.
    cleanup = script[script.index("finally") :]
    assert re.search(
        r"\$managed\.Label\s*-eq\s*['\"]backend['\"].*\$StopBackendOnExit|"
        r"\$StopBackendOnExit.*\$managed\.Label\s*-eq\s*['\"]backend['\"]",
        cleanup,
        re.DOTALL,
    )


def test_dev_launcher_binds_its_managed_ollama_process_to_the_probe_url() -> None:
    script = (PROJECT_ROOT / "scripts" / "dev.ps1").read_text(encoding="utf-8")

    assert "$env:OLLAMA_HOST = $OllamaUrl" in script


def test_dev_launcher_waits_for_ollama_api_when_the_spawn_process_exits() -> None:
    script = (PROJECT_ROOT / "scripts" / "dev.ps1").read_text(encoding="utf-8")
    wait_loop = script[
        script.index("function Wait-OllamaReady") : script.index("function Get-PortOwner")
    ]

    assert "if ($ollamaProcess.HasExited)" not in wait_loop
