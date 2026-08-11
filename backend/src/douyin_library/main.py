import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

from douyin_library.api.routes.creators import router as creators_router
from douyin_library.api.routes.favorites import router as favorites_router
from douyin_library.api.routes.jobs import router as jobs_router
from douyin_library.api.routes.settings import router as settings_router
from douyin_library.api.routes.works import router as works_router
from douyin_library.config import Settings
from douyin_library.jobs.runtime import build_worker_runtime
from douyin_library.runtime import initialize_runtime


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    runtime = initialize_runtime(settings)
    worker_runtime = build_worker_runtime(runtime.engine, settings)
    worker_task = asyncio.create_task(
        worker_runtime.worker.run(),
        name="douyin-library-job-worker",
    )
    app.state.runtime = runtime
    app.state.worker_runtime = worker_runtime
    app.state.worker_task = worker_task
    try:
        yield
    finally:
        worker_runtime.worker.stop()
        try:
            await asyncio.wait_for(worker_task, timeout=10)
        except TimeoutError:
            worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await worker_task
        finally:
            worker_runtime.close()
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
