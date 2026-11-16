import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[3] / "scripts" / "local-runtime.ps1"
pytestmark = pytest.mark.skipif(not shutil.which("powershell.exe"), reason="Windows launcher")


def _run(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
         f"$ErrorActionPreference='Stop'; . '{HELPER}'; {body}"],
        capture_output=True, text=True, timeout=15, env=os.environ.copy(),
    )


def test_launcher_imports_missing_user_config_and_preserves_override() -> None:
    result = _run("""
        $env:DKL_NOTE_EXTERNAL_MODEL = ''
        $env:DKL_NOTE_PROVIDER_POLICY = 'local'
        Import-ProviderUserEnvironment -ReadUserValue { param($name) 'user-value' }
        if ($env:DKL_NOTE_EXTERNAL_MODEL -ne 'user-value') { throw 'missing user config' }
        if ($env:DKL_NOTE_PROVIDER_POLICY -ne 'local') { throw 'override replaced' }
    """)
    assert result.returncode == 0, result.stderr


def test_launcher_rejects_existing_project_workers() -> None:
    result = _run("""
        function Get-CimInstance {
            [pscustomobject]@{ Name='python.exe'; ExecutablePath='C:\\project\\python.exe';
                CommandLine='python -m douyin_library.worker_main --pool general'; ProcessId=42 }
        }
        Assert-NoProjectWorkers -PythonPath 'C:\\project\\python.exe'
    """)
    assert result.returncode != 0
    assert "project_workers_already_running" in result.stderr


def test_launcher_ignores_unrelated_python_processes() -> None:
    result = _run("""
        function Get-CimInstance {
            [pscustomobject]@{ Name='python.exe'; ExecutablePath='C:\\other\\python.exe';
                CommandLine='python -m douyin_library.worker_main --pool general'; ProcessId=42 }
        }
        Assert-NoProjectWorkers -PythonPath 'C:\\project\\python.exe'
    """)
    assert result.returncode == 0, result.stderr


def test_launcher_lock_excludes_concurrent_startup() -> None:
    root = "C:\\synthetic-" + uuid.uuid4().hex
    first = subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
         f"$ErrorActionPreference='Stop'; . '{HELPER}'; "
         f"$guard=Enter-ProjectLauncherLock -ProjectRoot '{root}'; "
         "[Console]::Out.WriteLine('held'); Start-Sleep -Seconds 10"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert first.stdout.readline().strip() == "held"
        second = _run(f"$guard=Enter-ProjectLauncherLock -ProjectRoot '{root}'")
        assert second.returncode != 0
        assert "project_launcher_already_running" in second.stderr
    finally:
        first.terminate()
        first.communicate(timeout=5)
    # Windows releases an abandoned mutex when its owning process exits.
    restarted = _run(f"$guard=Enter-ProjectLauncherLock -ProjectRoot '{root}'")
    assert restarted.returncode == 0, restarted.stderr
