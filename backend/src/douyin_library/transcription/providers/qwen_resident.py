"""Supervised resident Qwen3-ASR process.

The runner keeps the model process alive across sequential jobs.  It owns only
the local process protocol and temporary WAV conversion; transcript validation,
publication, leases, and audits remain in :mod:`transcription.service`.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from douyin_library.config import Settings, resolve_tool
from douyin_library.jobs.control import JobCancelledError
from douyin_library.transcription.service import (
    DEFAULT_QWEN_PYTHON,
    SubprocessResult,
    _communicate_or_cancel,
)

AudioConverter = Callable[
    [Path, Path, asyncio.Event | None], Awaitable[SubprocessResult]
]


class QwenResidentRunner:
    """Run one request at a time through a resident Qwen server process."""

    def __init__(
        self,
        *,
        python_path: Path,
        server_script_path: Path,
        model: str,
        language: str,
        chunk_seconds: float,
        startup_timeout_seconds: float = 120.0,
        offline: bool = True,
        ffmpeg_path: Path | None = None,
        audio_converter: AudioConverter | None = None,
    ) -> None:
        self.python_path = python_path
        self.server_script_path = server_script_path
        self.model = model
        self.language = language
        self.chunk_seconds = chunk_seconds
        self.startup_timeout_seconds = startup_timeout_seconds
        self.offline = offline
        self.ffmpeg_path = ffmpeg_path
        self.audio_converter = audio_converter or self._convert_audio
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_lines: list[str] = []
        self._lock = asyncio.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> QwenResidentRunner:
        return cls(
            python_path=settings.qwen_python_path or DEFAULT_QWEN_PYTHON,
            server_script_path=settings.qwen_server_script_path,
            model=settings.transcription_model,
            language=settings.transcription_language,
            chunk_seconds=settings.qwen_chunk_seconds,
            startup_timeout_seconds=settings.qwen_runtime_start_timeout_seconds,
            offline=settings.qwen_offline,
            ffmpeg_path=resolve_tool(settings, "ffmpeg"),
        )

    @property
    def process(self) -> asyncio.subprocess.Process | None:
        return self._process

    async def run(
        self,
        *,
        audio_path: Path,
        result_path: Path,
        cancel_event: asyncio.Event | None = None,
    ) -> SubprocessResult:
        if not self.python_path.is_file():
            raise RuntimeError(f"qwen_python_not_found:{self.python_path}")
        if not self.server_script_path.is_file():
            raise RuntimeError(f"qwen_server_script_not_found:{self.server_script_path}")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix="dkl_qwen_resident_",
            suffix=".wav",
            dir=result_path.parent,
        )
        os.close(fd)
        wav_path = Path(temporary_name)
        try:
            converted = await self.audio_converter(audio_path, wav_path, cancel_event)
            if converted.returncode != 0:
                return converted
            async with self._lock:
                await self._ensure_started(cancel_event)
                return await self._request(
                    wav_path,
                    result_path,
                    cancel_event=cancel_event,
                )
        finally:
            wav_path.unlink(missing_ok=True)

    async def _ensure_started(self, cancel_event: asyncio.Event | None) -> None:
        process = self._process
        if process is not None and process.returncode is None:
            return
        await self._stop_process()
        self._stderr_lines.clear()
        child_env = {**os.environ, "PYTHONUTF8": "1"}
        if self.offline:
            child_env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        process = await asyncio.create_subprocess_exec(
            str(self.python_path),
            str(self.server_script_path),
            "--model",
            self.model,
            "--language",
            self.language,
            "--chunk-seconds",
            str(self.chunk_seconds),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_env,
        )
        self._process = process
        self._stderr_task = asyncio.create_task(self._drain_stderr(process))
        try:
            payload = await self._read_json_line(
                timeout=self.startup_timeout_seconds,
                cancel_event=cancel_event,
            )
        except Exception:
            await self._stop_process()
            raise
        if payload.get("event") != "ready":
            await self._stop_process()
            raise RuntimeError(f"qwen_resident_not_ready:{payload.get('error', 'unknown')}")

    async def _request(
        self,
        wav_path: Path,
        result_path: Path,
        *,
        cancel_event: asyncio.Event | None,
    ) -> SubprocessResult:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("qwen_resident_process_missing")
        request_id = uuid4().hex
        request = {
            "request_id": request_id,
            "audio": str(wav_path),
            "out": str(result_path),
            "language": self.language,
            "chunk_seconds": self.chunk_seconds,
        }
        process.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
        await process.stdin.drain()
        try:
            payload = await self._read_json_line(timeout=None, cancel_event=cancel_event)
        except JobCancelledError:
            await self._stop_process()
            raise
        except Exception:
            await self._stop_process()
            raise
        if payload.get("request_id") != request_id:
            await self._stop_process()
            raise RuntimeError("qwen_resident_response_id_mismatch")
        return SubprocessResult(
            returncode=int(payload.get("returncode", 1)),
            stdout=str(payload.get("stdout", "")),
            stderr="\n".join(
                item
                for item in (str(payload.get("stderr", "")), *self._stderr_lines)
                if item
            ),
        )

    async def _read_json_line(
        self,
        *,
        timeout: float | None,
        cancel_event: asyncio.Event | None,
    ) -> dict[str, object]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("qwen_resident_stdout_missing")
        started = asyncio.get_running_loop().time()
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelledError("cancelled_by_user")
            remaining = None
            if timeout is not None:
                remaining = timeout - (asyncio.get_running_loop().time() - started)
                if remaining <= 0:
                    raise RuntimeError("qwen_resident_ready_timeout")
            try:
                poll_timeout = min(0.2, remaining) if remaining else 0.2
                line = await asyncio.wait_for(
                    process.stdout.readline(), timeout=poll_timeout
                )
            except TimeoutError:
                continue
            if not line:
                raise RuntimeError("qwen_resident_process_exited")
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("qwen_resident_protocol_invalid_json") from exc
            if not isinstance(value, dict):
                raise RuntimeError("qwen_resident_protocol_invalid_record")
            return value

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        if process.stderr is None:
            return
        while True:
            line = await process.stderr.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                self._stderr_lines.append(text)
                del self._stderr_lines[:-20]

    async def _convert_audio(
        self,
        audio_path: Path,
        wav_path: Path,
        cancel_event: asyncio.Event | None,
    ) -> SubprocessResult:
        if self.ffmpeg_path is None or not self.ffmpeg_path.is_file():
            raise RuntimeError(f"ffmpeg_not_found:{self.ffmpeg_path}")
        process = await asyncio.create_subprocess_exec(
            str(self.ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(audio_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await _communicate_or_cancel(process, cancel_event)
        return SubprocessResult(
            returncode=int(process.returncode or 0),
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )

    async def _stop_process(self) -> None:
        process = self._process
        self._process = None
        stderr_task = self._stderr_task
        self._stderr_task = None
        if stderr_task is not None:
            stderr_task.cancel()
            with suppress(asyncio.CancelledError):
                await stderr_task
        if process is None:
            return
        if process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except TimeoutError:
            if process.returncode is None:
                process.kill()
            await process.wait()
        self._close_process_transports(process)

    def close(self) -> None:
        """Terminate the resident child during synchronous worker shutdown."""

        process = self._process
        self._process = None
        stderr_task = self._stderr_task
        self._stderr_task = None
        if stderr_task is not None:
            stderr_task.cancel()
        if process is None or process.returncode is not None:
            if process is not None:
                self._close_process_transports(process)
            return
        process.terminate()
        self._close_process_transports(process)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(process.wait())

    @staticmethod
    def _close_process_transports(process: asyncio.subprocess.Process) -> None:
        if process.stdin is not None:
            process.stdin.close()
        for stream in (process.stdout, process.stderr):
            transport = getattr(stream, "_transport", None)
            if transport is not None:
                transport.close()

    async def aclose(self) -> None:
        await self._stop_process()


__all__ = ["QwenResidentRunner"]
