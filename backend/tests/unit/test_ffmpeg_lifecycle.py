import asyncio
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from douyin_library.media.ffmpeg import FfmpegMediaTool, MediaExtractionError


def test_ffmpeg_command_times_out_and_reaps_child(monkeypatch: pytest.MonkeyPatch) -> None:
    children = []
    real_popen = subprocess.Popen

    def capture_child(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", capture_child)
    tool = FfmpegMediaTool()
    tool.operation_timeout_seconds = 0.05
    # Bound the pre-fix command too, so observing RED cannot leave a process behind.
    with pytest.raises(MediaExtractionError, match="timed out"):
        tool._run([sys.executable, "-c", "import time; time.sleep(0.2)"], "probe")
    assert len(children) == 1
    assert children[0].poll() is not None


@pytest.mark.asyncio
async def test_ffmpeg_cancellation_reaps_child(tmp_path: Path, monkeypatch) -> None:
    children = []
    real_popen = subprocess.Popen

    def capture_child(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", capture_child)
    marker = tmp_path / "started"
    tool = FfmpegMediaTool()
    cancel_event = threading.Event()
    tool.cancel_event = cancel_event
    command = [sys.executable, "-c", (
        "from pathlib import Path; import time; "
        f"Path({str(marker)!r}).write_text('started'); time.sleep(0.4)"
    )]
    task = asyncio.create_task(asyncio.to_thread(tool._run, command, "probe"))
    deadline = asyncio.get_running_loop().time() + 2
    while not marker.exists():
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail("synthetic child did not start")
        await asyncio.sleep(0.005)
    cancel_event.set()
    with pytest.raises(MediaExtractionError, match="cancelled"):
        await asyncio.wait_for(task, timeout=2)
    assert len(children) == 1
    assert children[0].poll() is not None
