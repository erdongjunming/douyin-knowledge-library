from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .metadata import can_delete_source_video, duration_seconds, file_sha256, first_audio_stream


class MediaTool(Protocol):
    def extract_audio(self, source: Path, target: Path) -> dict[str, Any]: ...

    def extract_cover(self, source: Path, target: Path) -> None: ...

    def probe(self, path: Path) -> dict[str, Any]: ...

    def analyze_loudness(self, path: Path) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class StoredAsset:
    path: Path
    sha256: str
    bytes: int
    codec: str | None
    probe: dict[str, Any]
    verified_at: datetime | None


@dataclass(frozen=True, slots=True)
class MediaProcessResult:
    audio: StoredAsset
    cover: Path
    metadata: Path


class StorageLayout:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.data = self.root / "data"
        self.audio = self.root / "audio"
        self.covers = self.root / "cover"
        self.metadata = self.root / "metadata"
        self.subtitle = self.root / "subtitle"
        self.transcript = self.root / "transcript"
        self.raw = self.root / "raw"

    def initialize(self) -> None:
        for directory in (
            self.root,
            self.data,
            self.audio,
            self.covers,
            self.metadata,
            self.subtitle,
            self.transcript,
            self.raw,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def audio_path(self, aweme_id: str) -> Path:
        return self.audio / f"{aweme_id}.m4a"

    def cover_path(self, aweme_id: str) -> Path:
        return self.covers / f"{aweme_id}.jpg"

    def metadata_path(self, aweme_id: str) -> Path:
        return self.metadata / f"{aweme_id}.json"

    def transcript_dir(self, aweme_id: str) -> Path:
        return self.transcript / aweme_id

    def transcript_text_path(self, aweme_id: str) -> Path:
        return self.transcript_dir(aweme_id) / "transcript.txt"

    def transcript_segments_path(self, aweme_id: str) -> Path:
        return self.transcript_dir(aweme_id) / "segments.json"

    def transcript_qwen_path(self, aweme_id: str) -> Path:
        return self.transcript_dir(aweme_id) / "qwen3_asr.json"

    def subtitle_dir(self, aweme_id: str) -> Path:
        return self.subtitle / aweme_id

    def subtitle_candidate_paths(self, aweme_id: str) -> tuple[Path, ...]:
        directory = self.subtitle_dir(aweme_id)
        return tuple(
            directory / name
            for name in (
                "subtitle.srt",
                "subtitle.vtt",
                "subtitle.ass",
                "subtitle.ssa",
                "subtitle.txt",
            )
        )

    def transcript_cleaned_text_path(self, aweme_id: str) -> Path:
        return self.transcript_dir(aweme_id) / "transcript.cleaned.md"

    def transcript_cleaned_json_path(self, aweme_id: str) -> Path:
        return self.transcript_dir(aweme_id) / "transcript.cleaned.json"

    def note_dir(self, aweme_id: str) -> Path:
        return self.dy_note / aweme_id


class MediaPipeline:
    def __init__(self, layout: StorageLayout, tool: MediaTool) -> None:
        self.layout = layout
        self.tool = tool

    def process(
        self,
        aweme_id: str,
        source: Path,
        *,
        cover_source: Path | None = None,
    ) -> MediaProcessResult:
        self.layout.initialize()
        source_probe = self.tool.probe(source)
        audio_path = self.layout.audio_path(aweme_id)
        cover_path = self.layout.cover_path(aweme_id)
        metadata_path = self.layout.metadata_path(aweme_id)

        self.tool.extract_audio(source, audio_path)
        audio_probe = self.tool.probe(audio_path)
        audio_stream = first_audio_stream(audio_probe)
        audio_bytes = audio_path.stat().st_size if audio_path.exists() else 0
        digest = file_sha256(audio_path) if audio_bytes else ""
        audio_duration = duration_seconds(audio_probe)
        source_duration = duration_seconds(source_probe)
        verified = bool(audio_stream and audio_duration and audio_bytes and digest)
        verified_at = datetime.now(UTC) if verified else None
        if not verified:
            raise RuntimeError("extracted audio failed validation")

        if (
            cover_source is not None
            and cover_source.is_file()
            and cover_source.stat().st_size > 0
        ):
            shutil.copyfile(cover_source, cover_path)
        else:
            self.tool.extract_cover(source, cover_path)
        loudness = self.tool.analyze_loudness(audio_path)
        metadata = {
            "aweme_id": aweme_id,
            "source": str(source),
            "source_probe": source_probe,
            "audio_probe": audio_probe,
            "audio_sha256": digest,
            "audio_bytes": audio_bytes,
            "loudness": loudness,
            "verified_at": verified_at.isoformat() if verified_at else None,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if can_delete_source_video(
            source_duration=source_duration,
            audio_duration=audio_duration,
            audio_bytes=audio_bytes,
            sha256=digest,
            verified=verified,
        ):
            source.unlink()

        return MediaProcessResult(
            audio=StoredAsset(
                path=audio_path,
                sha256=digest,
                bytes=audio_bytes,
                codec=audio_stream.get("codec_name") if audio_stream else None,
                probe=audio_probe,
                verified_at=verified_at,
            ),
            cover=cover_path,
            metadata=metadata_path,
        )
