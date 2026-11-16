from __future__ import annotations

import asyncio
import json
import threading
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.jobs.control import JobCancelledError
from douyin_library.models import Asset, Work
from douyin_library.upstream.contracts import Collector

from .ffmpeg import FfmpegMediaTool
from .metadata import file_sha256
from .storage import MediaPipeline, MediaProcessResult, StorageLayout


class WorkMediaService:
    def __init__(
        self,
        session: Session,
        collector: Collector,
        layout: StorageLayout,
        *,
        media_tool: Any | None = None,
    ) -> None:
        self.session = session
        self.collector = collector
        self.layout = layout
        self.media_tool = media_tool or FfmpegMediaTool.from_settings(Settings())

    async def process(
        self, work: Work, *, cancel_event: asyncio.Event | None = None,
    ) -> MediaProcessResult:
        if cancel_event is not None and cancel_event.is_set():
            raise JobCancelledError("cancelled_by_user")
        self.layout.initialize()
        raw_dir = self.layout.raw / work.aweme_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        source = raw_dir / f"{work.aweme_id}.mp4"
        discovered = _work_to_discovered(work)
        download = await self.collector.download(discovered, source)
        if download.status != "succeeded" or download.video_path is None:
            raise RuntimeError(download.error_detail or download.error_code or "download_failed")

        cancelled = threading.Event()
        tool = (
            self.media_tool.with_cancellation(cancelled)
            if isinstance(self.media_tool, FfmpegMediaTool) else self.media_tool
        )
        # Capture ORM values before offloading; the pipeline receives only
        # paths/strings and its SQLAlchemy Session stays on this thread.
        pipeline_task = asyncio.create_task(asyncio.to_thread(
            MediaPipeline(self.layout, tool).process,
            work.aweme_id, download.video_path,
            cover_source=download.cover_path, cancel_event=cancelled,
            defer_source_deletion=True,
        ))
        cancel_waiter = (
            asyncio.create_task(cancel_event.wait()) if cancel_event is not None else None
        )
        try:
            if cancel_waiter is not None:
                done, _ = await asyncio.wait(
                    (pipeline_task, cancel_waiter), return_when=asyncio.FIRST_COMPLETED
                )
                if cancel_waiter in done:
                    cancelled.set()
                    await self._finish_cancelled_pipeline(pipeline_task)
                    raise JobCancelledError("cancelled_by_user")
            result = await asyncio.shield(pipeline_task)
        except asyncio.CancelledError:
            cancelled.set()
            # Do not release the job while its media thread still owns a
            # subprocess or can publish files. FFmpeg cleanup is bounded.
            await self._finish_cancelled_pipeline(pipeline_task)
            raise
        finally:
            if cancel_waiter is not None:
                cancel_waiter.cancel()
                with suppress(asyncio.CancelledError):
                    await cancel_waiter
        self._raise_if_cancelled(cancel_event)
        try:
            self._save_assets(work.id, result)
            self._raise_if_cancelled(cancel_event)
            self.session.commit()
        except (Exception, asyncio.CancelledError):
            self.session.rollback()
            raise
        # Let cancellation requested during the synchronous commit reach this
        # task before removing its only source video.
        await asyncio.sleep(0)
        self._raise_if_cancelled(cancel_event)
        if result.source_to_delete is not None:
            result.source_to_delete.unlink()
        self._remove_download_artifacts(
            raw_dir,
            download.cover_path,
            download.metadata_path,
        )
        return result

    @staticmethod
    def _raise_if_cancelled(cancel_event: asyncio.Event | None) -> None:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise asyncio.CancelledError
        if cancel_event is not None and cancel_event.is_set():
            raise JobCancelledError("cancelled_by_user")

    @staticmethod
    async def _finish_cancelled_pipeline(task: asyncio.Task[MediaProcessResult]) -> None:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # Shutdown may issue a second cancel while FFmpeg is being
                # reaped. Retain ownership until that cleanup completes.
                continue
            except Exception:
                break
        with suppress(Exception, asyncio.CancelledError):
            task.result()

    @staticmethod
    def _remove_download_artifacts(
        raw_dir: Path,
        cover_path: Path | None,
        metadata_path: Path | None,
    ) -> None:
        for path in (cover_path, metadata_path):
            if path is not None and path.is_file() and path.parent == raw_dir:
                path.unlink()
        try:
            raw_dir.rmdir()
        except OSError:
            pass

    def _save_assets(self, work_id: int, result: MediaProcessResult) -> None:
        self._upsert(
            work_id,
            kind="audio",
            path=result.audio.path,
            codec=result.audio.codec,
            mime_type="audio/mp4",
            size=result.audio.bytes,
            digest=result.audio.sha256,
            probe=result.audio.probe,
            verified_at=result.audio.verified_at,
        )
        self._upsert(
            work_id,
            kind="cover",
            path=result.cover,
            codec=None,
            mime_type="image/jpeg",
            size=result.cover.stat().st_size,
            digest=file_sha256(result.cover),
            probe=None,
            verified_at=result.audio.verified_at,
        )
        self._upsert(
            work_id,
            kind="metadata",
            path=result.metadata,
            codec=None,
            mime_type="application/json",
            size=result.metadata.stat().st_size,
            digest=file_sha256(result.metadata),
            probe=None,
            verified_at=result.audio.verified_at,
        )

    def _upsert(
        self,
        work_id: int,
        *,
        kind: str,
        path: Path,
        codec: str | None,
        mime_type: str,
        size: int,
        digest: str,
        probe: dict[str, Any] | None,
        verified_at: datetime | None,
    ) -> None:
        asset = self.session.scalar(
            select(Asset).where(Asset.work_id == work_id, Asset.kind == kind)
        )
        if asset is None:
            asset = Asset(work_id=work_id, kind=kind)
            self.session.add(asset)
        asset.path = str(path)
        asset.codec = codec
        asset.mime_type = mime_type
        asset.bytes = size
        asset.sha256 = digest
        asset.probe_json = json.dumps(probe, ensure_ascii=False) if probe is not None else None
        asset.verified_at = verified_at


def _work_to_discovered(work: Work) -> Any:
    """Build the smallest collector work object needed for downloading."""
    from douyin_library.upstream.contracts import DiscoveredAuthor, DiscoveredWork

    return DiscoveredWork(
        aweme_id=work.aweme_id,
        author=DiscoveredAuthor(sec_uid=None),
        media_type=(
            work.media_type
            if work.media_type in {"video", "gallery", "live_photo"}
            else "video"
        ),
        description=work.description,
        share_url=work.share_url,
        published_at=work.published_at,
        duration_ms=work.duration_ms,
    )
