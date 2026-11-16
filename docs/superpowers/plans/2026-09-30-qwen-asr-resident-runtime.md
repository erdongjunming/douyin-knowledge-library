# Qwen3-ASR Resident Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse one locally loaded Qwen3-ASR model process across sequential ASR jobs without weakening raw transcript integrity or job recovery.

**Architecture:** Add a local JSON-lines Qwen server and a supervised resident runner. Keep the existing subprocess runner as an explicit fallback. Construct the selected provider once per worker runtime and close it during worker shutdown; `TranscriptionService` continues to own temporary audio, validation, audit, lease fencing, and publication.

**Tech Stack:** Python 3.14 backend, isolated Qwen Python environment, asyncio subprocesses, JSON-lines IPC, pytest/pytest-asyncio, existing SQLAlchemy queue and lease store.

---

### Task 1: Define runtime settings and provider lifetime

**Files:**
- Modify: `backend/src/douyin_library/config.py`
- Modify: `backend/src/douyin_library/jobs/runtime.py`
- Modify: `backend/src/douyin_library/worker_main.py`
- Test: `backend/tests/unit/test_config.py`
- Test: `backend/tests/integration/test_job_runtime.py`

- [x] **Step 1: Write the failing settings test.** Add assertions that the runtime mode defaults to `resident`, the server script path is configurable, and the startup timeout has a bounded positive default.
- [x] **Step 2: Run the focused test and verify it fails** because the settings fields do not exist.
- [x] **Step 3: Add the settings fields** with `DKL_TRANSCRIPTION_RUNTIME`, `DKL_QWEN_SERVER_SCRIPT_PATH`, and `DKL_QWEN_RUNTIME_START_TIMEOUT_SECONDS`.
- [x] **Step 4: Write the provider lifetime test.** Build a worker runtime with a factory that increments a counter and assert two transcription handler invocations construct one provider; closing the runtime calls its `close()` method.
- [x] **Step 5: Run the focused tests and verify they pass.**

### Task 2: Add the resident JSON-lines server

**Files:**
- Create: `scripts/qwen_asr_server.py`
- Test: `backend/tests/unit/test_qwen_resident_runtime.py`

- [x] **Step 1: Write the failing protocol tests.** Cover a `ready` record, one request/response pair, malformed request error output, and a server response that reports inference failure without writing a stable output file.
- [x] **Step 2: Run the focused tests and verify they fail** because the server module does not exist.
- [x] **Step 3: Implement the server.** Parse CLI model/language/chunk settings, import `torch`, `soundfile`, and `qwen_asr`, load the model once, emit a readiness JSON line, process one request at a time, write only the caller-provided attempt output path, and flush every response.
- [x] **Step 4: Run the focused protocol tests and verify they pass.**

### Task 3: Add the supervised resident runner

**Files:**
- Create: `backend/src/douyin_library/transcription/providers/qwen_resident.py`
- Modify: `backend/src/douyin_library/transcription/providers/qwen.py`
- Test: `backend/tests/unit/test_qwen_resident_runtime.py`

- [x] **Step 1: Write failing runner tests.** Assert lazy startup, one server process for two sequential requests, readiness timeout, process exit, and cancellation cleanup.
- [x] **Step 2: Run the focused tests and verify they fail** because the resident runner is absent.
- [x] **Step 3: Implement the runner.** Use an asyncio lock, launch the configured Qwen Python/server script, wait for `ready`, send request IDs, read matching responses, convert source audio with the existing FFmpeg contract, return `SubprocessResult`, terminate and clear the process on cancellation or protocol failure, and expose `close()`.
- [x] **Step 4: Add provider selection.** `QwenAsrProvider.from_settings()` selects resident or legacy subprocess mode from settings; both implement the existing `Transcriber` protocol.
- [x] **Step 5: Run focused resident and legacy provider tests and verify they pass.**

### Task 4: Integrate worker ownership and diagnostics

**Files:**
- Modify: `backend/src/douyin_library/jobs/runtime.py`
- Modify: `backend/src/douyin_library/config.py`
- Modify: `backend/src/douyin_library/worker_main.py`
- Modify: `backend/src/douyin_library/api/routes/settings.py`
- Test: `backend/tests/integration/test_worker_process_contract.py`
- Test: `backend/tests/integration/test_diagnostics.py`

- [x] **Step 1: Write failing integration assertions.** Assert one provider is retained for the worker lifetime, `close()` is called on shutdown, and diagnostics report runtime mode plus server script status without probing production audio.
- [x] **Step 2: Run the focused integration tests and verify they fail.**
- [x] **Step 3: Construct the provider once** when `build_worker_runtime()` creates the worker runtime, use it for every ASR handler invocation, and close it before the runtime session closes.
- [x] **Step 4: Extend diagnostics** with `runtime`, `server_script`, and `model` fields while preserving existing response keys.
- [x] **Step 5: Run the focused integration tests and verify they pass.**

### Task 5: Verify the complete code path

**Files:**
- Modify: `README.md`
- Modify: `docs/operations/semantic-notes.md`

- [x] **Step 1: Run backend tests, Ruff, frontend tests, and frontend build.** Backend tests and frontend checks pass; full Ruff still reports the pre-existing import-order issue in `tests/unit/notes/test_pipeline.py`.
- [x] **Step 2: Run a synthetic resident smoke using a temporary server and temporary storage; assert raw output publication and failed-attempt rollback.** The smoke publishes two raw transcripts through the existing service and confirms one resident process; existing rollback tests cover failed-attempt preservation.
- [x] **Step 3: Document runtime selection, readiness states, cancellation behavior, and the explicit fallback setting.**
- [x] **Step 4: Run `git diff --check` and inspect the final worktree.**
