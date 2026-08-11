from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from douyin_library.api.routes.creators import router as creators_router
from douyin_library.api.routes.favorites import router as favorites_router
from douyin_library.api.routes.jobs import router as jobs_router
from douyin_library.api.routes.settings import router as settings_router
from douyin_library.api.routes.works import router as works_router
from douyin_library.config import Settings
from douyin_library.runtime import initialize_runtime


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    runtime = initialize_runtime(Settings())
    app.state.runtime = runtime
    try:
        yield
    finally:
        runtime.engine.dispose()


app = FastAPI(title="Douyin Knowledge Library", lifespan=lifespan)
app.include_router(settings_router)
app.include_router(creators_router)
app.include_router(favorites_router)
app.include_router(jobs_router)
app.include_router(works_router)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
