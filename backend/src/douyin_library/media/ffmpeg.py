from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


class MediaExtractionError(RuntimeError):
    """Raised when FFmpeg cannot create a valid media asset."""


class FfmpegMediaTool:
    def __init__(self, ffmpeg: str | None = None, ffprobe: str | None = None) -> None:
        self.ffmpeg = ffmpeg or shutil.which("ffmpeg") or "ffmpeg"
        self.ffprobe = ffprobe or shutil.which("ffprobe") or "ffprobe"

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
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=capture_output,
        )
        if completed.returncode and not allow_failure:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise MediaExtractionError(f"{operation} failed: {detail[-500:]}")
        return completed


def _last_float(value: str, pattern: str) -> float | None:
    matches = re.findall(pattern, value, flags=re.IGNORECASE)
    return float(matches[-1]) if matches else None
