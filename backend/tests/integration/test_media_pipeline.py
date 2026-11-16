import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from douyin_library.db import create_schema
from douyin_library.jobs.control import JobCancelledError
from douyin_library.media import storage
from douyin_library.media.ffmpeg import MediaExtractionError
from douyin_library.media.service import WorkMediaService
from douyin_library.media.storage import MediaPipeline, StorageLayout
from douyin_library.models import Asset, Work
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


@pytest.mark.asyncio
async def test_media_service_preserves_source_when_cancelled_at_deletion_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    cancellation_delivered = threading.Event()
    can_delete = storage.can_delete_source_video

    def cancel_at_deletion_check(**kwargs: Any) -> bool:
        def deliver_cancellation() -> None:
            cancel_event.set()
            cancellation_delivered.set()

        loop.call_soon_threadsafe(deliver_cancellation)
        assert cancellation_delivered.wait(timeout=1)
        return can_delete(**kwargs)

    monkeypatch.setattr(storage, "can_delete_source_video", cancel_at_deletion_check)

    class Collector:
        async def download(self, work: Any, target: Path) -> DownloadResult:
            target.write_bytes(b"valid-video")
            return DownloadResult(status="succeeded", video_path=target)

    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        work = Work(aweme_id="late-cancel")
        session.add(work)
        session.commit()
        layout = StorageLayout(tmp_path / "storage")

        with pytest.raises(JobCancelledError):
            await WorkMediaService(
                session, Collector(), layout, media_tool=FakeMediaTool(),
            ).process(work, cancel_event=cancel_event)

        assert (layout.raw / work.aweme_id / f"{work.aweme_id}.mp4").is_file()
        assert session.query(Asset).count() == 0
    engine.dispose()


@pytest.mark.asyncio
async def test_media_service_preserves_source_when_asset_commit_fails(tmp_path: Path) -> None:
    class Collector:
        async def download(self, work: Any, target: Path) -> DownloadResult:
            target.write_bytes(b"valid-video")
            return DownloadResult(status="succeeded", video_path=target)

    def fail_commit(session: Session) -> None:
        raise RuntimeError("asset commit failed")

    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        work = Work(aweme_id="failed-commit")
        session.add(work)
        session.commit()
        layout = StorageLayout(tmp_path / "storage")
        event.listen(session, "before_commit", fail_commit)

        with pytest.raises(RuntimeError, match="asset commit failed"):
            await WorkMediaService(
                session, Collector(), layout, media_tool=FakeMediaTool(),
            ).process(work)

        assert (layout.raw / work.aweme_id / f"{work.aweme_id}.mp4").is_file()
        with Session(engine) as observer:
            assert observer.query(Asset).count() == 0
    engine.dispose()


@pytest.mark.asyncio
async def test_media_service_deletes_source_only_after_assets_are_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Collector:
        async def download(self, work: Any, target: Path) -> DownloadResult:
            target.write_bytes(b"valid-video")
            return DownloadResult(status="succeeded", video_path=target)

    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    layout = StorageLayout(tmp_path / "storage")
    source = layout.raw / "committed" / "committed.mp4"
    unlink = Path.unlink
    deleted = []

    def verify_assets_before_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == source:
            with Session(engine) as observer:
                assert observer.query(Asset).count() == 3
            deleted.append(path)
        unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", verify_assets_before_unlink)
    with Session(engine) as session:
        work = Work(aweme_id="committed")
        session.add(work)
        session.commit()

        await WorkMediaService(
            session, Collector(), layout, media_tool=FakeMediaTool(),
        ).process(work)

        assert deleted == [source]
        assert not source.exists()
    engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("task_cancel", [False, True, "repeated"])
async def test_media_cancellation_waits_for_pipeline_and_preserves_source(
    tmp_path: Path, task_cancel: bool | str,
) -> None:
    started = threading.Event()
    finished = threading.Event()
    cancel_event = asyncio.Event()

    class SlowMediaTool(FakeMediaTool):
        def probe(self, path: Path) -> dict:
            started.set()
            time.sleep(0.1)
            finished.set()
            return super().probe(path)

    class Collector:
        async def download(self, work: Any, target: Path) -> DownloadResult:
            target.write_bytes(b"valid-video")
            return DownloadResult(status="succeeded", video_path=target)

    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        work = Work(aweme_id="cancelled")
        session.add(work)
        session.commit()
        layout = StorageLayout(tmp_path / "storage")
        task = asyncio.create_task(WorkMediaService(
            session, Collector(), layout, media_tool=SlowMediaTool(),
        ).process(work, cancel_event=cancel_event))
        deadline = asyncio.get_running_loop().time() + 1
        while not started.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                pytest.fail("media pipeline did not start")
            await asyncio.sleep(0.005)
        if task_cancel:
            task.cancel()
            if task_cancel == "repeated":
                await asyncio.sleep(0.005)
                task.cancel()
            expected = asyncio.CancelledError
        else:
            cancel_event.set()
            expected = JobCancelledError
        with pytest.raises(expected):
            await asyncio.wait_for(task, timeout=2)
        assert finished.is_set(), "pipeline continued writing after cancellation returned"
        assert (layout.raw / "cancelled" / "cancelled.mp4").is_file()
        assert not layout.audio_path("cancelled").exists()
        assert not layout.metadata_path("cancelled").exists()
        assert session.query(Asset).count() == 0
    engine.dispose()


@pytest.mark.asyncio
async def test_media_pipeline_keeps_event_loop_responsive_and_database_on_original_thread(
    tmp_path: Path,
) -> None:
    main_thread = threading.get_ident()
    media_threads = []
    database_threads = []

    class SlowMediaTool(FakeMediaTool):
        def probe(self, path: Path) -> dict:
            media_threads.append(threading.get_ident())
            time.sleep(0.08)
            return super().probe(path)

    class Collector:
        async def download(self, work: Any, target: Path) -> DownloadResult:
            target.write_bytes(b"valid-video")
            return DownloadResult(status="succeeded", video_path=target)

    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    with Session(engine) as session:
        work = Work(aweme_id="responsive")
        session.add(work)
        session.commit()
        event.listen(engine, "before_cursor_execute", lambda *args: database_threads.append(
            threading.get_ident()
        ))
        task = asyncio.create_task(WorkMediaService(
            session, Collector(), StorageLayout(tmp_path / "storage"),
            media_tool=SlowMediaTool(),
        ).process(work))
        try:
            await asyncio.sleep(0.02)
            assert not task.done(), "the event loop could not tick during media processing"
            await asyncio.wait_for(task, timeout=2)
            assert media_threads and all(t != main_thread for t in media_threads)
            assert database_threads and all(t == main_thread for t in database_threads)
        finally:
            await task
    engine.dispose()
