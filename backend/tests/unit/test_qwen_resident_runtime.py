import asyncio
import sys
from pathlib import Path

import pytest

from douyin_library.jobs.control import JobCancelledError
from douyin_library.transcription.providers.qwen_resident import QwenResidentRunner
from douyin_library.transcription.service import SubprocessResult


def _server_script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "server.py"
    path.write_text(body, encoding="utf-8")
    return path


async def _fake_converter(
    audio_path: Path,
    wav_path: Path,
    cancel_event: asyncio.Event | None = None,
) -> SubprocessResult:
    del audio_path, cancel_event
    wav_path.write_bytes(b"wav")
    return SubprocessResult(returncode=0, stdout="", stderr="")


@pytest.mark.asyncio
async def test_resident_runner_loads_one_server_for_two_requests(tmp_path: Path) -> None:
    starts = tmp_path / "starts.txt"
    offline = tmp_path / "offline.txt"
    server = _server_script(
        tmp_path,
        f"""
import json
import os
from pathlib import Path
Path({str(starts)!r}).open('a', encoding='utf-8').write('start\\n')
Path({str(offline)!r}).write_text(os.environ.get('HF_HUB_OFFLINE', ''), encoding='utf-8')
print(json.dumps({{'event': 'ready'}}, ensure_ascii=False), flush=True)
for line in iter(input, ''):
    request = json.loads(line)
    Path(request['out']).write_text(
        json.dumps({{'text': request['request_id'], 'segments': []}}),
        encoding='utf-8',
    )
    print(
        json.dumps({{'ok': True, 'request_id': request['request_id'], 'returncode': 0}}),
        flush=True,
    )
""",
    )
    runner = QwenResidentRunner(
        python_path=Path(sys.executable),
        server_script_path=server,
        model="Qwen/Qwen3-ASR-0.6B",
        language="Chinese",
        chunk_seconds=60,
        startup_timeout_seconds=5,
        audio_converter=_fake_converter,
    )
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"audio")

    first = await runner.run(audio_path=audio, result_path=tmp_path / "one.json")
    second = await runner.run(audio_path=audio, result_path=tmp_path / "two.json")

    assert first.returncode == second.returncode == 0
    assert (tmp_path / "one.json").is_file()
    assert (tmp_path / "two.json").is_file()
    assert starts.read_text(encoding="utf-8").splitlines() == ["start"]
    assert offline.read_text(encoding="utf-8") == "1"
    await runner.aclose()


@pytest.mark.asyncio
async def test_resident_runner_raises_when_server_does_not_become_ready(tmp_path: Path) -> None:
    server = _server_script(tmp_path, "import time; time.sleep(30)")
    runner = QwenResidentRunner(
        python_path=Path(sys.executable),
        server_script_path=server,
        model="Qwen/Qwen3-ASR-0.6B",
        language="Chinese",
        chunk_seconds=60,
        startup_timeout_seconds=0.1,
        audio_converter=_fake_converter,
    )
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"audio")

    with pytest.raises(RuntimeError, match="qwen_resident_ready_timeout"):
        await runner.run(audio_path=audio, result_path=tmp_path / "result.json")

    assert runner.process is None


@pytest.mark.asyncio
async def test_resident_runner_terminates_server_on_cancellation(tmp_path: Path) -> None:
    server = _server_script(
        tmp_path,
        """
import json
import time
print(json.dumps({'event': 'ready'}), flush=True)
request = json.loads(input())
time.sleep(30)
print(json.dumps({'ok': True, 'request_id': request['request_id'], 'returncode': 0}), flush=True)
""",
    )
    runner = QwenResidentRunner(
        python_path=Path(sys.executable),
        server_script_path=server,
        model="Qwen/Qwen3-ASR-0.6B",
        language="Chinese",
        chunk_seconds=60,
        startup_timeout_seconds=5,
        audio_converter=_fake_converter,
    )
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"audio")
    cancel_event = asyncio.Event()
    task = asyncio.create_task(
        runner.run(
            audio_path=audio,
            result_path=tmp_path / "result.json",
            cancel_event=cancel_event,
        )
    )
    await asyncio.sleep(0.2)
    cancel_event.set()

    with pytest.raises(JobCancelledError, match="cancelled_by_user"):
        await task
    assert runner.process is None


@pytest.mark.asyncio
async def test_resident_runner_sync_close_closes_child_transports(tmp_path: Path) -> None:
    del tmp_path
    class FakeTransport:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

        def is_closing(self) -> bool:
            return self.closed

    class FakeProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.stdin = FakeTransport()
            self.stdout = type("Stream", (), {"_transport": FakeTransport()})()
            self.stderr = type("Stream", (), {"_transport": FakeTransport()})()

        def terminate(self) -> None:
            self.returncode = 0

        async def wait(self) -> int:
            return self.returncode or 0

    runner = QwenResidentRunner(
        python_path=Path(sys.executable),
        server_script_path=Path("server.py"),
        model="Qwen/Qwen3-ASR-0.6B",
        language="Chinese",
        chunk_seconds=60,
        startup_timeout_seconds=5,
        audio_converter=_fake_converter,
    )

    process = FakeProcess()
    runner._process = process  # type: ignore[assignment]

    runner.close()
    await asyncio.sleep(0)

    assert process.stdin.is_closing()
    assert process.stdout._transport.is_closing()
    assert process.stderr._transport.is_closing()
