from fastapi import FastAPI

from douyin_library.api.routes.settings import router as settings_router

app = FastAPI(title="Douyin Knowledge Library")
app.include_router(settings_router)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
