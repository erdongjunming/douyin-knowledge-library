from fastapi import APIRouter

from douyin_library.config import Settings, diagnostics

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("/diagnostics")
async def get_diagnostics() -> dict[str, dict[str, str]]:
    return diagnostics(Settings())

