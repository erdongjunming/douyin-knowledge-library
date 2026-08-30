import asyncio
import json
import sys
from pathlib import Path

import pytest

from douyin_library.jobs.control import JobCancelledError
from douyin_library.transcription.providers.qwen import QwenAsrProvider
from douyin_library.transcription.service import QwenAsrSubprocess, _communicate_or_cancel


@pytest.mark.asyncio
async def test_qwen_subprocess_converts_m4a_to_temporary_16k_wav(
    tmp_path: Path,
    monkeypatch,
) -> None:
    python = tmp_path / "python.exe"
    script = tmp_path / "run_qwen_asr.py"
    ffmpeg = tmp_path / "ffmpeg.exe"
    for executable in (python, script, ffmpeg):
        executable.write_bytes(b"exe")
    audio = tmp_path / "source.m4a"
    audio.write_bytes(b"aac")
    result_path = tmp_path / "transcript" / "qwen3_asr.json"
    commands: list[tuple[str, ...]] = []

    class FakeProcess:
        returncode = 0

        def __init__(self, command: tuple[str, ...]) -> None:
            self.command = command

        async def communicate(self):
            if self.command[0] == str(ffmpeg):
                Path(self.command[-1]).write_bytes(b"wav")
                return b"", b"ffmpeg diagnostics"
            result_path.write_text(
                json.dumps({"text": "知识", "segments": []}, ensure_ascii=False),
                encoding="utf-8",
            )
            return b'qwen stdout', b'qwen stderr'

    async def fake_exec(*command, **kwargs):
        commands.append(command)
        return FakeProcess(command)

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    runner = QwenAsrSubprocess(
        python_path=python,
        script_path=script,
        ffmpeg_path=ffmpeg,
        model="Qwen/Qwen3-ASR-0.6B",
        language="Chinese",
        chunk_seconds=60,
    )

    result = await runner.run(audio_path=audio, result_path=result_path)

    ffmpeg_command, qwen_command = commands
    assert ffmpeg_command[0] == str(ffmpeg)
    audio_channels = ffmpeg_command.index("-ac")
    sample_rate = ffmpeg_command.index("-ar")
    assert ("-ac", "1") == ffmpeg_command[audio_channels : audio_channels + 2]
    assert ("-ar", "16000") == ffmpeg_command[sample_rate : sample_rate + 2]
    wav_path = Path(qwen_command[qwen_command.index("--audio") + 1])
    assert wav_path.suffix == ".wav"
    assert not wav_path.exists()
    assert result.stdout == "qwen stdout"
    assert "ffmpeg diagnostics" in result.stderr
    assert "qwen stderr" in result.stderr


@pytest.mark.asyncio
async def test_cancellation_terminates_running_subprocess() -> None:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    cancel_event = asyncio.Event()
    cancel_event.set()

    with pytest.raises(JobCancelledError, match="cancelled_by_user"):
        await _communicate_or_cancel(process, cancel_event)

    assert process.returncode is not None


@pytest.mark.asyncio
async def test_qwen_provider_delegates_attempt_result_path_and_cancellation(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "audio.m4a"
    result_path = tmp_path / "attempt" / "qwen3_asr.json"
    audio_path.write_bytes(b"audio")
    calls: dict[str, object] = {}

    class FakeRunner:
        async def run(self, **kwargs):
            calls.update(kwargs)
            return "subprocess-result"

    provider = QwenAsrProvider(FakeRunner())
    cancel_event = asyncio.Event()

    result = await provider.run(
        audio_path=audio_path,
        result_path=result_path,
        cancel_event=cancel_event,
    )

    assert result == "subprocess-result"
    assert calls == {
        "audio_path": audio_path,
        "result_path": result_path,
        "cancel_event": cancel_event,
    }
