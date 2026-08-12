from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.db import create_schema
from douyin_library.media.ffmpeg import MediaExtractionError
from douyin_library.media.service import WorkMediaService
from douyin_library.media.storage import MediaPipeline, StorageLayout
from douyin_library.models import Work
from douyin_library.upstream.contracts import DownloadResult


class FakeMediaTool:
    def extract_audio(self, source: Path, target: Path) -> dict:
        if source.read_bytes() == b"corrupt":
            raise MediaExtractionError("source has no valid audio stream")
        target.write_bytes(b"verified-audio")
        return {"copy_mode": "stream_copy"}

    def extract_cover(self, source: Path, target: Path) -> None:
        target.write_bytes(b"jpeg-cover")

    def probe(self, path: Path) -> dict:
        if path.suffix == ".mp4":
            return {
                "format": {"duration": "42.0"},
                "streams": [{"codec_type": "audio", "codec_name": "aac"}],
            }
        return {
            "format": {"duration": "41.95"},
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "48000",
                    "channels": 2,
                }
            ],
        }

    def analyze_loudness(self, path: Path) -> dict:
        return {"integrated_loudness_lufs": -16.2, "true_peak_dbfs": -1.1}


def test_video_deleted_only_after_verified_audio(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path / "library")
    layout.initialize()
    source = tmp_path / "sample_voice.mp4"
    source.write_bytes(b"valid-video")

    result = MediaPipeline(layout, FakeMediaTool()).process("731", source)

    assert result.audio.verified_at is not None
    assert result.audio.sha256
    assert result.cover.exists()
    assert result.metadata.exists()
    assert not source.exists()


def test_failed_audio_keeps_source_video(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path / "library")
    layout.initialize()
    source = tmp_path / "corrupt.mp4"
    source.write_bytes(b"corrupt")

    with pytest.raises(MediaExtractionError):
        MediaPipeline(layout, FakeMediaTool()).process("731", source)

    assert source.exists()


def test_upstream_cover_is_used_when_download_has_no_video_stream(tmp_path: Path) -> None:
    class AudioOnlyMediaTool(FakeMediaTool):
        def extract_cover(self, source: Path, target: Path) -> None:
            raise MediaExtractionError("source has no video stream")

    layout = StorageLayout(tmp_path / "library")
    layout.initialize()
    source = tmp_path / "audio-only.mp4"
    source.write_bytes(b"valid-audio-container")
    upstream_cover = tmp_path / "upstream-cover.jpg"
    upstream_cover.write_bytes(b"upstream-jpeg-cover")

    result = MediaPipeline(layout, AudioOnlyMediaTool()).process(
        "732",
        source,
        cover_source=upstream_cover,
    )

    assert result.cover.read_bytes() == b"upstream-jpeg-cover"
    assert not source.exists()


def test_empty_upstream_cover_falls_back_to_frame_extraction(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path / "library")
    layout.initialize()
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"valid-video")
    upstream_cover = tmp_path / "empty-cover.jpg"
    upstream_cover.touch()

    result = MediaPipeline(layout, FakeMediaTool()).process(
        "734",
        source,
        cover_source=upstream_cover,
    )

    assert result.cover.read_bytes() == b"jpeg-cover"


@pytest.mark.asyncio
async def test_media_service_passes_downloaded_cover_to_pipeline(tmp_path: Path) -> None:
    class AudioOnlyMediaTool(FakeMediaTool):
        def extract_cover(self, source: Path, target: Path) -> None:
            raise MediaExtractionError("source has no video stream")

    class AudioOnlyCollector:
        async def download(self, work: Any, target: Path) -> DownloadResult:
            target.write_bytes(b"valid-audio-container")
            cover = target.with_name(f"{target.stem}_cover.jpg")
            cover.write_bytes(b"downloaded-cover")
            metadata = target.with_name(f"{target.stem}_data.json")
            metadata.write_text("{}", encoding="utf-8")
            return DownloadResult(
                status="succeeded",
                video_path=target,
                cover_path=cover,
                metadata_path=metadata,
            )

    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        work = Work(aweme_id="733", media_type="gallery")
        session.add(work)
        session.commit()

        result = await WorkMediaService(
            session,
            AudioOnlyCollector(),
            StorageLayout(tmp_path / "storage"),
            media_tool=AudioOnlyMediaTool(),
        ).process(work)

        assert result.cover.read_bytes() == b"downloaded-cover"
        assert not (tmp_path / "storage" / "raw" / "733").exists()
    engine.dispose()
