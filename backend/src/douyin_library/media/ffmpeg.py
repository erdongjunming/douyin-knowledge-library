from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from douyin_library.config import Settings, resolve_tool


class MediaExtractionError(RuntimeError):
    """Raised when FFmpeg cannot create a valid media asset."""


class FfmpegMediaTool:
    def __init__(
        self, ffmpeg: str | None = None, ffprobe: str | None = None, *,
        operation_timeout_seconds: float = 600.0,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.ffmpeg = ffmpeg or shutil.which("ffmpeg") or "ffmpeg"
        self.ffprobe = ffprobe or shutil.which("ffprobe") or "ffprobe"
        if operation_timeout_seconds <= 0:
            raise ValueError("media_operation_timeout_must_be_positive")
        self.operation_timeout_seconds = operation_timeout_seconds
        self.cancel_event = cancel_event or threading.Event()

    def with_cancellation(self, cancel_event: threading.Event) -> FfmpegMediaTool:
        return FfmpegMediaTool(
            self.ffmpeg, self.ffprobe,
            operation_timeout_seconds=self.operation_timeout_seconds,
            cancel_event=cancel_event,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> FfmpegMediaTool:
        ffmpeg = resolve_tool(settings, "ffmpeg")
        ffprobe = resolve_tool(settings, "ffprobe")
        return cls(
            ffmpeg=str(ffmpeg) if ffmpeg else None,
            ffprobe=str(ffprobe) if ffprobe else None,
        )

    def extract_audio(self, source: Path, target: Path) -> dict[str, Any]:
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(target),
        ]
        self._run(command, "audio extraction")
        if not target.is_file() or target.stat().st_size == 0:
            raise MediaExtractionError("audio extraction produced an empty file")
        return {"codec": "aac", "command": command}

    def extract_cover(self, source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(target),
        ]
        self._run(command, "cover extraction")
        if not target.is_file() or target.stat().st_size == 0:
            raise MediaExtractionError("cover extraction produced an empty file")

    def probe(self, path: Path) -> dict[str, Any]:
        command = [
            self.ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        completed = self._run(command, "media probe", capture_output=True)
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise MediaExtractionError("ffprobe returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise MediaExtractionError("ffprobe returned an invalid document")
        return result

    def analyze_loudness(self, path: Path) -> dict[str, float | None]:
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-i",
            str(path),
            "-filter_complex",
            "ebur128=framelog=verbose",
            "-f",
            "null",
            "-",
        ]
        completed = self._run(command, "loudness analysis", capture_output=True, allow_failure=True)
        output = f"{completed.stdout}\n{completed.stderr}"
        integrated = _last_float(output, r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS")
        true_peak = _last_float(output, r"Peak:\s*(-?\d+(?:\.\d+)?)\s*dBFS")
        return {"integrated_loudness_lufs": integrated, "true_peak_dbfs": true_peak}

    def _run(
        self,
        command: list[str],
        operation: str,
        *,
        capture_output: bool = False,
        allow_failure: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if self.cancel_event.is_set():
            raise MediaExtractionError(f"{operation} cancelled")
        child = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE if capture_output else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + self.operation_timeout_seconds
        try:
            while True:
                if self.cancel_event.is_set():
                    raise MediaExtractionError(f"{operation} cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MediaExtractionError(
                        f"{operation} timed out after {self.operation_timeout_seconds:g} seconds"
                    )
                try:
                    stdout, stderr = child.communicate(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            self._terminate(child)
            raise
        completed = subprocess.CompletedProcess(command, child.returncode, stdout, stderr)
        if completed.returncode and not allow_failure:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise MediaExtractionError(f"{operation} failed: {detail[-500:]}")
        return completed

    @staticmethod
    def _terminate(child: subprocess.Popen[str]) -> None:
        if child.poll() is None:
            child.terminate()
        try:
            child.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate(timeout=2.0)


def _last_float(value: str, pattern: str) -> float | None:
    matches = re.findall(pattern, value, flags=re.IGNORECASE)
    return float(matches[-1]) if matches else None
