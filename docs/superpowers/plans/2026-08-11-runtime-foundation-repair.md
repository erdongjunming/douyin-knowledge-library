# Runtime Foundation Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a fresh local installation initialize itself, recognize bundled media tools, persist a validated Douyin Cookie, report backend health accurately, and execute synchronization/media jobs through the persistent queue.

**Architecture:** Keep FastAPI, SQLite, React, the current SQLAlchemy models, and the existing jiji gateway. Add a small runtime bootstrap boundary that owns storage/database startup, a media-tool resolver, safe credential endpoints, and a worker composition module. Synchronization becomes a queued command; successful discovery enqueues media work without yet pretending that ASR or AI note generation exists.

**Tech Stack:** Python 3.14, FastAPI, SQLAlchemy, Alembic, APScheduler-compatible job model, keyring, React 19, TypeScript, Vitest, Playwright, FFmpeg.

---

### Task 1: Initialize Runtime Storage And Database

**Files:**
- Create: `backend/src/douyin_library/runtime.py`
- Modify: `backend/src/douyin_library/main.py`
- Test: `backend/tests/integration/test_runtime_initialization.py`

- [ ] **Step 1: Write the failing initialization test**

```python
from pathlib import Path

from sqlalchemy import inspect

from douyin_library.config import Settings
from douyin_library.runtime import initialize_runtime


def test_runtime_initializes_database_on_first_start(tmp_path: Path) -> None:
    settings = Settings(storage_root=tmp_path / "library")

    runtime = initialize_runtime(settings)

    assert runtime.database_path.is_file()
    assert "work" in inspect(runtime.engine).get_table_names()
    assert "alembic_version" in inspect(runtime.engine).get_table_names()
```

- [ ] **Step 2: Run the test and verify the missing module failure**

Run: `backend\.venv\Scripts\python.exe -m pytest tests/integration/test_runtime_initialization.py -v`

Expected: FAIL because `douyin_library.runtime` does not exist.

- [ ] **Step 3: Implement Alembic-backed runtime initialization**

```python
@dataclass(frozen=True, slots=True)
class Runtime:
    database_path: Path
    engine: Engine


def initialize_runtime(settings: Settings) -> Runtime:
    data_dir = settings.storage_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    database_path = data_dir / "library.db"
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url(database_path))
    command.upgrade(config, "head")
    return Runtime(database_path, create_engine(database_url(database_path)))
```

Use a FastAPI lifespan in `main.py` to call `initialize_runtime(Settings())`, store the result on `app.state.runtime`, and dispose the engine on shutdown. Do not catch storage creation errors or fall back to the system drive.

- [ ] **Step 4: Run the focused test and runtime API smoke test**

Run: `backend\.venv\Scripts\python.exe -m pytest tests/integration/test_runtime_initialization.py tests/integration/test_health.py -v`

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/douyin_library/runtime.py backend/src/douyin_library/main.py backend/tests/integration/test_runtime_initialization.py
git commit -m "fix: initialize runtime database on startup"
```

### Task 2: Resolve Bundled FFmpeg Tools

**Files:**
- Modify: `backend/src/douyin_library/config.py`
- Modify: `backend/src/douyin_library/media/ffmpeg.py`
- Modify: `backend/src/douyin_library/media/service.py`
- Modify: `backend/tests/integration/test_diagnostics.py`
- Create: `backend/tests/unit/test_media_tool_resolution.py`

- [ ] **Step 1: Write failing resolver tests**

```python
def test_diagnostics_finds_ffmpeg_bundled_under_storage(tmp_path: Path) -> None:
    bin_dir = tmp_path / "tools" / "ffmpeg" / "ffmpeg-build" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "ffmpeg.exe").write_bytes(b"exe")
    (bin_dir / "ffprobe.exe").write_bytes(b"exe")

    result = diagnostics(Settings(storage_root=tmp_path))

    assert result["ffmpeg"] == {"status": "ok", "path": str(bin_dir / "ffmpeg.exe")}
    assert result["ffprobe"] == {"status": "ok", "path": str(bin_dir / "ffprobe.exe")}
```

Also assert that `FfmpegMediaTool.from_settings(settings)` uses the two resolved absolute paths.

- [ ] **Step 2: Run the tests and verify they fail because only PATH is checked**

Run: `backend\.venv\Scripts\python.exe -m pytest tests/integration/test_diagnostics.py tests/unit/test_media_tool_resolution.py -v`

Expected: FAIL with `status == "missing"`.

- [ ] **Step 3: Implement deterministic tool resolution**

```python
def resolve_tool(settings: Settings, name: str) -> Path | None:
    explicit = getattr(settings, f"{name}_path")
    if explicit and explicit.is_file():
        return explicit
    system_path = shutil.which(name)
    if system_path:
        return Path(system_path)
    executable = f"{name}.exe" if os.name == "nt" else name
    candidates = sorted((settings.storage_root / "tools" / "ffmpeg").glob(f"*/bin/{executable}"))
    return candidates[0] if candidates else None
```

Add optional `ffmpeg_path` and `ffprobe_path` settings. Add `FfmpegMediaTool.from_settings()` and make `WorkMediaService` use it when no media tool is injected.

- [ ] **Step 4: Run focused tests**

Run: `backend\.venv\Scripts\python.exe -m pytest tests/integration/test_diagnostics.py tests/unit/test_media_tool_resolution.py tests/integration/test_media_pipeline.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/douyin_library/config.py backend/src/douyin_library/media backend/tests/integration/test_diagnostics.py backend/tests/unit/test_media_tool_resolution.py
git commit -m "fix: resolve bundled ffmpeg executables"
```

### Task 3: Validate And Persist The Douyin Cookie

**Files:**
- Modify: `backend/src/douyin_library/security/cookie_store.py`
- Modify: `backend/src/douyin_library/api/routes/settings.py`
- Modify: `backend/tests/integration/test_cookie_validation.py`
- Modify: `backend/tests/unit/test_cookie_store.py`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`
- Modify: `frontend/src/styles.css`

- [ ] **Step 1: Write failing backend credential tests**

```python
@pytest.mark.asyncio
async def test_valid_cookie_is_saved_only_after_validation() -> None:
    store = MemoryCookieStore()
    app.dependency_overrides[get_cookie_validator] = lambda: ValidCookieValidator()
    app.dependency_overrides[get_cookie_store] = lambda: store
    response = await client.put("/api/settings/douyin-cookie", json={"cookie": "sessionid=test"})
    assert response.status_code == 200
    assert response.json()["configured"] is True
    assert store.read() == "sessionid=test"
```

Add a second test proving an invalid Cookie returns 422 and is not stored. Add a unit test for `CookieStore.delete()`.

- [ ] **Step 2: Run backend credential tests and verify RED**

Run: `backend\.venv\Scripts\python.exe -m pytest tests/integration/test_cookie_validation.py tests/unit/test_cookie_store.py -v`

Expected: FAIL because the endpoint, dependency, and delete method do not exist.

- [ ] **Step 3: Implement safe credential endpoints**

Add:

```text
GET    /api/settings/douyin-cookie -> {configured: boolean}
PUT    /api/settings/douyin-cookie -> validate, then save, never echo secret
DELETE /api/settings/douyin-cookie -> remove stored secret
```

The PUT response includes the safe account nickname and validation status only. Never log or return the Cookie.

- [ ] **Step 4: Write the failing React settings test**

Render Settings, enter `sessionid=test`, click `验证并保存`, and assert the saved account state appears while the Cookie value is cleared from the input.

- [ ] **Step 5: Run the React test and verify RED**

Run: `npm test -- --run src/App.test.tsx`

Expected: FAIL because the settings form does not exist.

- [ ] **Step 6: Implement the settings credential form**

Use a password input, a clear save command, configured/not-configured status, disabled busy state, and the existing status-message pattern. Do not store the Cookie in browser storage.

- [ ] **Step 7: Run backend and frontend credential tests**

Run: `backend\.venv\Scripts\python.exe -m pytest tests/integration/test_cookie_validation.py tests/unit/test_cookie_store.py -v`

Run: `npm test -- --run src/App.test.tsx`

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add backend/src/douyin_library/security backend/src/douyin_library/api/routes/settings.py backend/tests frontend/src
git commit -m "feat: save validated douyin credentials"
```

### Task 4: Report Backend Connectivity And Parse API Errors Safely

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`
- Modify: `frontend/src/styles.css`

- [ ] **Step 1: Write failing connection and empty-response tests**

```typescript
it("shows disconnected when health check fails", async () => {
  vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("fetch failed"))
  render(<App />)
  expect(await screen.findByText("本地服务未连接")).toBeVisible()
})

it("uses an actionable message for an empty non-json error", async () => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(null, { status: 502 }))
  render(<App />)
  fireEvent.click(screen.getByRole("button", { name: "任务中心" }))
  expect(await screen.findByText("本地服务请求失败 (502)")).toBeVisible()
})
```

- [ ] **Step 2: Run the React tests and verify RED**

Run: `npm test -- --run src/App.test.tsx`

Expected: the hard-coded connected label remains and empty JSON parsing leaks an exception.

- [ ] **Step 3: Implement health polling and safe response parsing**

Read response text first, parse JSON only when non-empty, and fall back to `本地服务请求失败 (<status>)`. Poll `/api/health` without overlapping timers and show connecting, connected, or disconnected state from real results.

- [ ] **Step 4: Run React tests and production build**

Run: `npm test -- --run src/App.test.tsx`

Run: `npm run build`

Expected: tests and build PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/App.tsx frontend/src/App.test.tsx frontend/src/styles.css
git commit -m "fix: report local service health accurately"
```

### Task 5: Queue Synchronization And Start The Worker

**Files:**
- Modify: `backend/src/douyin_library/models.py`
- Create: `backend/migrations/versions/0002_job_payload.py`
- Modify: `backend/src/douyin_library/jobs/queue.py`
- Create: `backend/src/douyin_library/jobs/runtime.py`
- Modify: `backend/src/douyin_library/runtime.py`
- Modify: `backend/src/douyin_library/main.py`
- Modify: `backend/src/douyin_library/api/routes/creators.py`
- Modify: `backend/src/douyin_library/api/routes/favorites.py`
- Modify: `backend/src/douyin_library/sync/creator_sync.py`
- Modify: `backend/src/douyin_library/sync/favorites_sync.py`
- Modify: `frontend/src/App.tsx`
- Test: `backend/tests/integration/test_queued_sync.py`
- Test: `backend/tests/unit/test_job_state_machine.py`
- Test: `frontend/src/App.test.tsx`

- [ ] **Step 1: Write failing queue payload and sync enqueue tests**

```python
def test_job_queue_persists_json_payload(session: Session) -> None:
    job = JobQueue(session).create(kind="sync_creator", payload={"reference": "sec-731"})
    assert json.loads(job.payload_json or "{}") == {"reference": "sec-731"}


@pytest.mark.asyncio
async def test_creator_sync_endpoint_returns_a_pending_job() -> None:
    response = await client.post(
        "/api/creators/sync",
        json={"reference": "sec-731", "confirmed": True},
    )
    assert response.status_code == 202
    assert response.json()["kind"] == "sync_creator"
    assert response.json()["status"] == "pending"
```

- [ ] **Step 2: Run focused backend tests and verify RED**

Run: `backend\.venv\Scripts\python.exe -m pytest tests/unit/test_job_state_machine.py tests/integration/test_queued_sync.py -v`

Expected: FAIL because jobs have no payload and sync executes inline.

- [ ] **Step 3: Add the job payload migration and queue support**

Add nullable `payload_json` to `Job`, serialize mappings with `ensure_ascii=False`, and include only safe fields in API responses. Never put Cookies in job payloads.

- [ ] **Step 4: Build and register worker handlers**

Register `sync_creator`, `sync_favorites`, and `download_work` handlers. Each sync handler reads the Cookie from keyring at execution time, runs the existing service, and creates deduplicated `download_work` jobs for newly discovered work IDs. The download handler runs `WorkMediaService`; it does not create a fake transcript or note.

Start one `JobWorker` task in FastAPI lifespan and stop it cleanly during shutdown. Keep the worker session private to the worker.

- [ ] **Step 5: Change sync APIs and frontend messages to queued semantics**

Return HTTP 202 with the serialized pending job. The frontend should say `同步任务已加入队列` and direct the user to Task Center rather than claiming the full sync already completed.

- [ ] **Step 6: Run focused queue and API tests**

Run: `backend\.venv\Scripts\python.exe -m pytest tests/unit/test_job_state_machine.py tests/integration/test_queued_sync.py tests/integration/test_jobs_api.py -v`

Run: `npm test -- --run src/App.test.tsx`

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/src backend/migrations backend/tests frontend/src
git commit -m "feat: run douyin synchronization through the job queue"
```

### Task 6: Full Verification And Rendered QA

**Files:**
- Modify only if a test exposes a defect in files already listed above.

- [ ] **Step 1: Run repository verification**

Run: `powershell -ExecutionPolicy Bypass -File scripts/verify.ps1`

Expected: 0 failures, clean Ruff output, successful Vite build.

- [ ] **Step 2: Run real startup smoke checks against an empty storage root**

Start the backend with an isolated `DKL_STORAGE_ROOT`, verify `/api/health`, `/api/jobs`, `/api/works`, `/api/settings/diagnostics`, and credential status all return JSON rather than `storage_not_initialized`.

- [ ] **Step 3: Run Playwright desktop and mobile checks**

Use system Playwright with Edge at 1440x900 and 390x844. Verify page identity, no framework overlay, no relevant console errors, all five navigation controls, truthful connected/disconnected labels, Cookie form layout, and initialized empty states.

- [ ] **Step 4: Review the final diff and document deferred scope**

Confirm that ASR, AI note generation, audit payload retention, and Windows scheduled wake-up remain explicit follow-up plans rather than silent stubs.

