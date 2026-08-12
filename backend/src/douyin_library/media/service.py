from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from douyin_library.config import Settings
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

    async def process(self, work: Work) -> MediaProcessResult:
        self.layout.initialize()
        raw_dir = self.layout.raw / work.aweme_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        source = raw_dir / f"{work.aweme_id}.mp4"
        discovered = _work_to_discovered(work)
        download = await self.collector.download(discovered, source)
        if download.status != "succeeded" or download.video_path is None:
            raise RuntimeError(download.error_detail or download.error_code or "download_failed")

        result = MediaPipeline(self.layout, self.media_tool).process(
            work.aweme_id,
            download.video_path,
            cover_source=download.cover_path,
        )
        self._save_assets(work.id, result)
        self.session.commit()
        self._remove_download_artifacts(
            raw_dir,
            download.cover_path,
            download.metadata_path,
        )
        return result

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
