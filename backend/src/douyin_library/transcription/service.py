from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from douyin_library.audit import ProcessingAuditRecorder
from douyin_library.config import Settings, resolve_tool
from douyin_library.media.metadata import file_sha256
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Asset, Job, Transcript, Work

DEFAULT_QWEN_PYTHON = (
    Path.home()
    / ".cache"
    / "rimagination-notes"
    / "qwen3-asr-venv"
    / "Scripts"
    / "python.exe"
)
DEFAULT_QWEN_SCRIPT = (
    Path.home() / ".codex" / "skills" / "dy-note" / "scripts" / "run_qwen_asr.py"
)


@dataclass(frozen=True, slots=True)
class SubprocessResult:
    returncode: int
    stdout: str
    stderr: str


class Transcriber(Protocol):
    async def run(self, *, audio_path: Path, result_path: Path) -> SubprocessResult: ...


class QwenAsrSubprocess:
    def __init__(
        self,
        *,
        python_path: Path,
        script_path: Path,
        ffmpeg_path: Path,
        model: str,
        language: str,
        chunk_seconds: float,
    ) -> None:
        self.python_path = python_path
        self.script_path = script_path
        self.ffmpeg_path = ffmpeg_path
        self.model = model
        self.language = language
        self.chunk_seconds = chunk_seconds

    @classmethod
    def from_settings(cls, settings: Settings) -> QwenAsrSubprocess:
        ffmpeg = resolve_tool(settings, "ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg_not_found")
        return cls(
            python_path=settings.qwen_python_path or DEFAULT_QWEN_PYTHON,
            script_path=settings.qwen_asr_script_path or DEFAULT_QWEN_SCRIPT,
            ffmpeg_path=ffmpeg,
            model=settings.transcription_model,
            language=settings.transcription_language,
            chunk_seconds=settings.qwen_chunk_seconds,
        )

    async def run(self, *, audio_path: Path, result_path: Path) -> SubprocessResult:
        if not self.python_path.is_file():
            raise RuntimeError(f"qwen_python_not_found:{self.python_path}")
        if not self.script_path.is_file():
            raise RuntimeError(f"qwen_script_not_found:{self.script_path}")
        if not self.ffmpeg_path.is_file():
            raise RuntimeError(f"ffmpeg_not_found:{self.ffmpeg_path}")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix="dkl_qwen_",
            suffix=".wav",
            dir=result_path.parent,
        )
        os.close(fd)
        wav_path = Path(temporary_name)
        try:
            ffmpeg_process = await asyncio.create_subprocess_exec(
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
            ffmpeg_stdout, ffmpeg_stderr = await ffmpeg_process.communicate()
            if ffmpeg_process.returncode:
                detail = ffmpeg_stderr.decode("utf-8", errors="replace")
                raise RuntimeError(f"ffmpeg_asr_conversion_failed:{detail}")
            process = await asyncio.create_subprocess_exec(
                str(self.python_path),
                str(self.script_path),
                "--audio",
                str(wav_path),
                "--out",
                str(result_path),
                "--model",
                self.model,
                "--language",
                self.language,
                "--chunk-seconds",
                str(self.chunk_seconds),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONUTF8": "1"},
            )
            stdout, stderr = await process.communicate()
        finally:
            wav_path.unlink(missing_ok=True)
        return SubprocessResult(
            returncode=int(process.returncode or 0),
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr="\n".join(
                part
                for part in (
                    ffmpeg_stdout.decode("utf-8", errors="replace"),
                    ffmpeg_stderr.decode("utf-8", errors="replace"),
                    stderr.decode("utf-8", errors="replace"),
                )
                if part
            ),
        )


class TranscriptionService:
    engine = "qwen3-asr"
    audit_version = "1"

    def __init__(
        self,
        session: Session,
        layout: StorageLayout,
        settings: Settings,
        transcriber: Transcriber,
    ) -> None:
        self.session = session
        self.layout = layout
        self.settings = settings
        self.transcriber = transcriber

    async def process(self, job: Job, work: Work) -> Transcript:
        self.layout.initialize()
        audio = self.session.scalar(
            select(Asset).where(Asset.work_id == work.id, Asset.kind == "audio")
        )
        if audio is None:
            raise RuntimeError("audio_asset_missing")
        audio_path = Path(audio.path)
        if not audio_path.is_file():
            raise RuntimeError(f"audio_file_missing:{audio_path}")

        recorder = ProcessingAuditRecorder(self.session)
        result_path = self._result_path(work)
        request = {
            "audio_path": str(audio_path),
            "result_path": str(result_path),
            "model": self.settings.transcription_model,
            "language": self.settings.transcription_language,
            "chunk_seconds": self.settings.qwen_chunk_seconds,
        }
        audit = recorder.start(
            operation="transcribe_work",
            processor=self.engine,
            model=self.settings.transcription_model,
            version=self.audit_version,
            job=job,
            request=request,
            input_files=[audio_path],
        )
        existing = self.session.scalar(
            select(Transcript).where(
                Transcript.work_id == work.id,
                Transcript.engine == self.engine,
                Transcript.model == self.settings.transcription_model,
                Transcript.status == "succeeded",
            )
        )
        if existing is not None:
            recorder.succeed(
                audit,
                response={"reused_existing": True, "transcript_id": existing.id},
            )
            return existing

        run_result = SubprocessResult(returncode=-1, stdout="", stderr="")
        try:
            run_result = await self.transcriber.run(
                audio_path=audio_path,
                result_path=result_path,
            )
            if run_result.returncode != 0:
                raise RuntimeError(f"qwen_asr_exit_code:{run_result.returncode}")
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            transcript = self._save_transcript(work, payload)
            output_paths = self._write_outputs(work, payload)
            self.session.commit()
            recorder.succeed(
                audit,
                response=payload,
                stdout=run_result.stdout,
                stderr=run_result.stderr,
                output_files=[result_path, *output_paths],
            )
            return transcript
        except Exception as exc:
            self.session.rollback()
            recorder.fail(
                audit,
                error=str(exc),
                stdout=run_result.stdout,
                stderr=run_result.stderr,
                output_files=[result_path] if result_path.exists() else [],
            )
            raise

    def _result_path(self, work: Work) -> Path:
        return self.layout.transcript_dir(work.aweme_id) / "qwen3_asr.json"

    def _save_transcript(self, work: Work, payload: dict[str, Any]) -> Transcript:
        text = str(payload.get("text") or "").strip()
        segments = payload.get("segments")
        if not text or not isinstance(segments, list):
            raise RuntimeError("qwen_asr_result_invalid")
        transcript = Transcript(
            work_id=work.id,
            engine=self.engine,
            model=str(payload.get("model") or self.settings.transcription_model),
            language=str(payload.get("language") or self.settings.transcription_language),
            text=text,
            segments_json=json.dumps(segments, ensure_ascii=False),
            status="succeeded",
        )
        self.session.add(transcript)
        self.session.flush()
        return transcript

    def _write_outputs(self, work: Work, payload: dict[str, Any]) -> list[Path]:
        text_path = self.layout.transcript_text_path(work.aweme_id)
        segments_path = self.layout.transcript_segments_path(work.aweme_id)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(str(payload["text"]).strip(), encoding="utf-8")
        segments_path.write_text(
            json.dumps(payload["segments"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._upsert_asset(work.id, "transcript_text", text_path, "text/plain")
        self._upsert_asset(
            work.id,
            "transcript_segments",
            segments_path,
            "application/json",
        )
        return [text_path, segments_path]

    def _upsert_asset(
        self,
        work_id: int,
        kind: str,
        path: Path,
        mime_type: str,
    ) -> None:
        asset = self.session.scalar(
            select(Asset).where(Asset.work_id == work_id, Asset.kind == kind)
        )
        if asset is None:
            asset = Asset(work_id=work_id, kind=kind)
            self.session.add(asset)
        asset.path = str(path)
        asset.mime_type = mime_type
        asset.bytes = path.stat().st_size
        asset.sha256 = file_sha256(path)
