"""Controlled capability test: run the production external path on substantial transcripts."""
import asyncio
import json

from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.media.storage import StorageLayout
from douyin_library.notes.service import NoteGenerationService
from douyin_library.runtime import initialize_runtime

TEST_AWEMES = [
    "7480733954736262415",  # 61KB transcript
    "7530270018546715914",  # 50KB
    "7513159867662650676",  # 39KB
]


async def run_one(service: NoteGenerationService, aweme_id: str) -> None:
    cancel_event = asyncio.Event()
    try:
        result = await service.generate_one(
            aweme_id,
            force=False,
            job=None,
            cancel_event=cancel_event,
            provider_policy="external",
            legacy_only=False,
        )
        print(
            json.dumps(
                {
                    "aweme_id": aweme_id,
                    "status": result.status,
                    "provider": result.provider,
                    "model": result.model,
                    "quality_gate": result.quality_gate,
                    "error": result.error,
                    "output_dir": result.output_dir,
                },
                ensure_ascii=False,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"aweme_id": aweme_id, "exception": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))


async def main() -> None:
    settings = Settings()
    print("policy:", settings.note_provider_policy, "| model:", settings.note_external_model)
    runtime = initialize_runtime(settings)
    try:
        with Session(runtime.engine) as session:
            service = NoteGenerationService(
                session,
                StorageLayout(settings.storage_root),
                settings,
            )
            for aweme_id in TEST_AWEMES:
                await run_one(service, aweme_id)
    finally:
        runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
