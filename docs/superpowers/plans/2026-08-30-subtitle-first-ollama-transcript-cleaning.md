# Subtitle-First Transcription and Ollama Transcript Cleaning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local subtitle-first routing, immutable raw transcripts, Ollama text cleaning, and evidence-complete note generation for future transcriptions without changing existing source media or historical outputs.

**Architecture:** Keep Qwen3-ASR as the only production audio-to-text provider. A local subtitle provider produces a candidate first; a quality gate selects it or falls back to Qwen. A separate durable `clean_transcript_work` job sends raw segments to Ollama, publishes a validated cleaned variant atomically, and lets note generation prefer cleaned while retaining raw fallback.

**Tech Stack:** Python 3.12, SQLAlchemy/Alembic, SQLite FTS5, asyncio subprocesses, FastAPI, React/Vite, Ollama `/api/chat`, pytest, Ruff.

---

## File Map

Create focused modules:

- `backend/src/douyin_library/transcription/contracts.py`: transcript source, ASR, cleaner, segment, and validation result types.
- `backend/src/douyin_library/transcription/providers/subtitle_local.py`: local subtitle discovery, normalization, scoring, and source draft creation.
- `backend/src/douyin_library/transcription/providers/qwen.py`: adapter around the existing Qwen subprocess implementation.
- `backend/src/douyin_library/transcription/providers/ollama_cleaner.py`: deterministic Ollama JSON request, batching, and response validation.
- `backend/src/douyin_library/transcription/selection.py`: subtitle-first selection policy.
- `backend/migrations/versions/0006_transcript_variants.py`: transcript variant columns and indexes.
- `backend/tests/unit/transcription/test_subtitle_provider.py`: subtitle parsing and quality tests.
- `backend/tests/unit/transcription/test_cleaner.py`: cleaner contract and validation tests.
- `backend/tests/integration/test_transcript_cleaning.py`: job chaining, persistence, fallback, and immutability tests.

Modify existing responsibilities:

- `backend/src/douyin_library/transcription/service.py`: delegate to providers while retaining subprocess cancellation and fencing.
- `backend/src/douyin_library/media/storage.py`: expose subtitle and cleaned transcript paths.
- `backend/src/douyin_library/models.py`: add transcript variant metadata.
- `backend/src/douyin_library/jobs/queue.py` and `backend/src/douyin_library/jobs/runtime.py`: register and chain `clean_transcript_work`.
- `backend/src/douyin_library/notes/input.py` and note service modules: select cleaned input and run evidence extraction.
- `backend/src/douyin_library/db.py`: make FTS prefer cleaned text without losing raw availability.
- `backend/src/douyin_library/api/routes/works.py`: expose source/variant/cleaning state.
- `backend/src/douyin_library/cli.py`: add explicit dry-run-first historical cleaning command.
- `frontend/src/App.tsx`, `frontend/src/NoteReader.tsx`, and `frontend/src/styles.css`: display provenance and cleaning fallback state.

## Task 1: Define Contracts and Storage Paths

**Files:** Create `backend/src/douyin_library/transcription/contracts.py`; modify `backend/src/douyin_library/media/storage.py`; create `backend/tests/unit/transcription/test_contracts.py`.

- [ ] **Step 1: Write failing contract tests.** Add tests asserting `TranscriptSegment`, `TranscriptDraft`, `CleanedTranscript`, and `CleanerResult` are immutable dataclasses and that `StorageLayout` returns `subtitle/<id>/`, `transcript.cleaned.md`, and `transcript.cleaned.json` paths.

- [ ] **Step 2: Run the focused tests.**

  Run: `uv run --group dev pytest backend/tests/unit/transcription/test_contracts.py -q`

  Expected: FAIL because the contract module and path methods do not exist.

- [ ] **Step 3: Implement the minimal contracts and paths.** Define:

  ```python
  @dataclass(frozen=True, slots=True)
  class TranscriptSegment:
      segment_id: str
      start: float | None
      end: float | None
      text: str

  class TranscriptSourceProvider(Protocol):
      def discover(self, work: Work) -> TranscriptDraft | None: ...

  class ASRProvider(Protocol):
      async def transcribe(self, audio_path: Path, *, cancel_event: asyncio.Event | None = None) -> TranscriptDraft: ...

  class TranscriptPostprocessor(Protocol):
      async def clean(self, transcript: TranscriptDraft, *, cancel_event: asyncio.Event | None = None) -> CleanedTranscript: ...
  ```

  Add `subtitle_dir`, `subtitle_candidate_paths`, `transcript_cleaned_text_path`, and `transcript_cleaned_json_path` methods without changing existing path methods.

- [ ] **Step 4: Run the focused tests.** Expected: PASS.

- [ ] **Step 5: Commit.**

  ```bash
  git add backend/src/douyin_library/transcription/contracts.py backend/src/douyin_library/media/storage.py backend/tests/unit/transcription/test_contracts.py
  git commit -m "feat: define transcript provider contracts and paths"
  ```

## Task 2: Implement Local Subtitle Discovery and Quality Gate

**Files:** Create `backend/src/douyin_library/transcription/providers/subtitle_local.py` and `backend/src/douyin_library/transcription/selection.py`; create `backend/tests/unit/transcription/test_subtitle_provider.py` and `backend/tests/unit/transcription/test_selection.py`.

- [ ] **Step 1: Write failing parser and gate tests.** Cover SRT, VTT, ASS dialogue, plain text, duplicate cues, empty files, malformed timestamps, placeholder/noise text, and a valid one-sentence short subtitle. Assert `select_transcript_source(subtitle, asr)` chooses subtitle only when `SubtitleQuality.usable` is true.

- [ ] **Step 2: Run tests to verify failure.**

  Run: `uv run --group dev pytest backend/tests/unit/transcription/test_subtitle_provider.py backend/tests/unit/transcription/test_selection.py -q`

  Expected: FAIL with missing provider functions.

- [ ] **Step 3: Implement normalization and quality scoring.** Parse cues into `TranscriptSegment` objects, strip format markup/timecode headers, collapse consecutive duplicates, and calculate a score containing `text_chars`, `segment_count`, `timestamp_valid`, `placeholder_ratio`, and `usable`. Reject unreadable/empty content, invalid monotonic timing, and obvious placeholders while allowing a short valid sentence.

- [ ] **Step 4: Implement deterministic selection.** `select_transcript_source` returns the subtitle draft when usable; otherwise invokes the ASR factory. The provider must read only `Settings.transcription_subtitle_root` and work-associated local asset paths. Do not call network APIs or media download helpers.

- [ ] **Step 5: Run focused tests and the existing transcription tests.**

  Run: `uv run --group dev pytest backend/tests/unit/transcription backend/tests/integration/test_transcription_pipeline.py -q`

  Expected: PASS; existing Qwen behavior remains unchanged.

- [ ] **Step 6: Commit.**

  ```bash
  git add backend/src/douyin_library/transcription/providers/subtitle_local.py backend/src/douyin_library/transcription/selection.py backend/tests/unit/transcription
  git commit -m "feat: add local subtitle-first selection"
  ```

## Task 3: Extract the Qwen ASR Adapter Without Behavioral Regression

**Files:** Create `backend/src/douyin_library/transcription/providers/qwen.py`; modify `backend/src/douyin_library/transcription/service.py` and `backend/src/douyin_library/jobs/runtime.py`; extend `backend/tests/unit/test_qwen_subprocess.py` and `backend/tests/integration/test_transcription_pipeline.py`.

- [ ] **Step 1: Add adapter contract tests.** Assert the adapter passes an attempt-specific result path, preserves FFmpeg diagnostics, terminates on cancellation, rejects missing/invalid JSON, and never publishes a stale stable result.

- [ ] **Step 2: Run the focused tests and confirm the new adapter fails.**

  Run: `uv run --group dev pytest backend/tests/unit/test_qwen_subprocess.py backend/tests/integration/test_transcription_pipeline.py -q`

  Expected: FAIL for the new adapter import/constructor.

- [ ] **Step 3: Move only provider-specific construction.** Implement `QwenAsrProvider` by delegating to the existing `QwenAsrSubprocess`; keep `_communicate_or_cancel`, GPU lease acquisition, ownership checks, attempt paths, rollback, and audit behavior in the service layer.

- [ ] **Step 4: Wire the static provider factory.** `build_worker_runtime` creates `QwenAsrProvider` by default, while existing injected `Transcriber` test doubles continue to satisfy the protocol.

- [ ] **Step 5: Run focused and full backend tests.** Expected: all existing transcription and job recovery tests pass.

- [ ] **Step 6: Commit.**

  ```bash
  git add backend/src/douyin_library/transcription/providers/qwen.py backend/src/douyin_library/transcription/service.py backend/src/douyin_library/jobs/runtime.py backend/tests/unit/test_qwen_subprocess.py backend/tests/integration/test_transcription_pipeline.py
  git commit -m "refactor: isolate qwen asr provider"
  ```

## Task 4: Add Transcript Variants and Atomic Cleaned Paths

**Files:** Modify `backend/src/douyin_library/models.py`, `backend/src/douyin_library/db.py`, `backend/src/douyin_library/media/storage.py`; create `backend/migrations/versions/0006_transcript_variants.py`; create `backend/tests/integration/test_transcript_variants.py`.

- [ ] **Step 1: Write migration and persistence tests.** Verify old rows backfill to `source_kind=asr, variant=raw`, new subtitle/ASR raw rows persist their source, cleaned rows point to `parent_transcript_id`, and raw rows/files are byte-identical after a cleaned publish.

- [ ] **Step 2: Run tests to verify failure.**

  Run: `uv run --group dev pytest backend/tests/integration/test_transcript_variants.py -q`

  Expected: FAIL because columns and migration are absent.

- [ ] **Step 3: Add model columns and migration.** Add non-null `source_kind` and `variant` with safe server defaults, nullable self-referential `parent_transcript_id`, and indexes supporting one raw and one cleaned variant per work/provider/model. Backfill existing rows without changing text.

- [ ] **Step 4: Update FTS refresh.** Select the highest-priority cleaned transcript when valid and raw transcripts otherwise; add/update triggers for variant inserts and updates. Keep raw text available through direct transcript queries.

- [ ] **Step 5: Implement atomic cleaned publication.** Add a per-work publication lock or compare-and-publish token in the publisher path. The lock must cover guard check through archive/replacement so a concurrent semantic/fallback publisher cannot be overwritten. Adapt custom publishers that do not accept `publish_guard`.

- [ ] **Step 6: Run migration, persistence, and existing database tests.** Expected: PASS with no changes to existing raw files.

- [ ] **Step 7: Commit.**

  ```bash
  git add backend/src/douyin_library/models.py backend/src/douyin_library/db.py backend/src/douyin_library/media/storage.py backend/migrations/versions/0006_transcript_variants.py backend/tests/integration/test_transcript_variants.py
  git commit -m "feat: persist raw and cleaned transcript variants"
  ```

## Task 5: Implement the Ollama Transcript Cleaner

**Files:** Create `backend/src/douyin_library/transcription/providers/ollama_cleaner.py`; modify `backend/src/douyin_library/config.py`; create `backend/tests/unit/transcription/test_cleaner.py`.

- [ ] **Step 1: Write failing cleaner tests.** Cover deterministic request options (`temperature=0`, thinking disabled, configured keep-alive), valid JSON output, malformed JSON, changed timestamps, changed segment count/order, Markdown-wrapped output, empty text, extreme drift, batching and cancellation.

- [ ] **Step 2: Run tests to verify failure.**

  Run: `uv run --group dev pytest backend/tests/unit/transcription/test_cleaner.py -q`

  Expected: FAIL because the cleaner and settings do not exist.

- [ ] **Step 3: Add settings.** Add `transcription_cleaning_enabled`, `transcription_cleaner_model`, timeout, batch character limit, drift ratio, subtitle enabled/root, and cleaner prompt version with validation bounds. Default model is `qwen3:4b`; do not auto-download models.

- [ ] **Step 4: Implement request and response validation.** Use the existing local HTTP client conventions with `trust_env=False`. Require the output segment IDs/order/timestamps to match exactly; accept text edits only. Return a typed failure reason for transport, timeout, schema, drift, cancellation, or lease loss.

- [ ] **Step 5: Implement long-text batching.** Split on segment boundaries with one-segment overlap, include stable IDs in every request, merge by ID, and reject duplicate/missing IDs. Never concatenate model prose outside the JSON payload.

- [ ] **Step 6: Run focused tests and Ollama-independent backend tests.** Expected: PASS without requiring a live Ollama server.

- [ ] **Step 7: Commit.**

  ```bash
  git add backend/src/douyin_library/transcription/providers/ollama_cleaner.py backend/src/douyin_library/config.py backend/tests/unit/transcription/test_cleaner.py
  git commit -m "feat: add ollama transcript cleaner"
  ```

## Task 6: Add Durable Cleaning Jobs and Source Routing

**Files:** Modify `backend/src/douyin_library/jobs/queue.py`, `backend/src/douyin_library/jobs/runtime.py`, `backend/src/douyin_library/jobs/worker.py`, `backend/src/douyin_library/transcription/service.py`; create `backend/tests/integration/test_transcript_cleaning.py` and extend `backend/tests/integration/test_worker_process_contract.py`.

- [ ] **Step 1: Write failing job tests.** Verify a successful `transcribe_work` enqueues exactly one `clean_transcript_work`, subtitle success skips the ASR subprocess, failed subtitle gate invokes Qwen, duplicate clean jobs are deduplicated by raw fingerprint/model/prompt version, and cleaning failure leaves `generate_note` eligible with raw fallback.

- [ ] **Step 2: Run tests to verify failure.**

  Run: `uv run --group dev pytest backend/tests/integration/test_transcript_cleaning.py backend/tests/integration/test_worker_process_contract.py -q`

  Expected: FAIL because the job kind and handler are absent.

- [ ] **Step 3: Register the job kind.** Add `clean_transcript_work` to general-worker kinds, retry limits, stage/progress labels, API job filters, and stale recovery coverage. Keep it disjoint from `ASR_JOB_KINDS`.

- [ ] **Step 4: Implement the handler.** Load the raw transcript and audio-independent source metadata, acquire Ollama's GPU lease, run the cleaner, validate output, publish cleaned files/row atomically, and record a processing audit. On any cleaner failure, mark the clean job failed while preserving raw.

- [ ] **Step 5: Chain jobs with explicit ownership.** After raw publication, enqueue cleaning once. `generate_note` must not require a successful cleaner job; it selects cleaned when present and otherwise records `transcript_variant=raw` and `cleaning_fallback=<reason>`.

- [ ] **Step 6: Test cancellation, heartbeat, lease loss, stale recovery, and retry idempotency.** Expected: no half-written cleaned files and no raw mutation.

- [ ] **Step 7: Commit.**

  ```bash
  git add backend/src/douyin_library/jobs/queue.py backend/src/douyin_library/jobs/runtime.py backend/src/douyin_library/jobs/worker.py backend/src/douyin_library/transcription/service.py backend/tests/integration/test_transcript_cleaning.py backend/tests/integration/test_worker_process_contract.py
  git commit -m "feat: chain durable transcript cleaning jobs"
  ```

## Task 7: Make Note Input Evidence-Complete

**Files:** Modify `backend/src/douyin_library/notes/input.py`, `backend/src/douyin_library/notes/service.py`, `backend/src/douyin_library/notes/prompts.py`, `backend/src/douyin_library/notes/coverage.py`, `backend/src/douyin_library/notes/render.py`; create `backend/tests/unit/notes/test_evidence_extract.py`; extend `backend/tests/integration/test_note_generation.py`.

- [ ] **Step 1: Write failing input and evidence tests.** Assert the loader prefers a valid cleaned variant, reports raw fallback metadata when cleaned is missing/failed, and evidence extraction retains examples, conditions, steps, numbers, names, quotations, segment IDs, timestamps, and uncertainty markers.

- [ ] **Step 2: Run tests to verify failure.**

  Run: `uv run --group dev pytest backend/tests/unit/notes/test_evidence_extract.py backend/tests/integration/test_note_generation.py -q`

  Expected: FAIL for cleaned selection/evidence map behavior.

- [ ] **Step 3: Implement variant selection.** Extend `NoteInput` with `transcript_variant`, `transcript_source_kind`, and `cleaning_fallback`; load `transcript.cleaned.json` only after validating its parent raw fingerprint and segment identity, otherwise load current raw files.

- [ ] **Step 4: Implement chunked evidence extraction.** Add an Ollama-backed structured extraction pass that writes `evidence_map.json` and keeps every extracted item linked to segment IDs. Use chunk overlap and deterministic merge by evidence ID.

- [ ] **Step 5: Update composition prompts and coverage gates.** Require complete outline, examples/cases, limitations/exceptions, ordered steps, parameters, timeline/evidence references, quotations, takeaways, and unconfirmed content. Reject notes that omit required evidence categories or present unsupported inference as fact.

- [ ] **Step 6: Add the “姜学长” regression fixture.** Use the existing transcript/note pair (identified by its stored work metadata) and assert its examples, conditions, and steps appear in the new evidence map and composed note. Do not regenerate the historical note during the test.

- [ ] **Step 7: Run note unit/integration tests.** Expected: PASS; existing rule fallback remains available when Ollama is unavailable.

- [ ] **Step 8: Commit.**

  ```bash
  git add backend/src/douyin_library/notes backend/tests/unit/notes/test_evidence_extract.py backend/tests/integration/test_note_generation.py
  git commit -m "feat: preserve evidence details in note generation"
  ```

## Task 8: Expose Provenance and Add Explicit Historical CLI

**Files:** Modify `backend/src/douyin_library/api/routes/works.py`, `backend/src/douyin_library/api/routes/jobs.py`, `backend/src/douyin_library/cli.py`, `frontend/src/App.tsx`, `frontend/src/NoteReader.tsx`, `frontend/src/styles.css`; create `backend/tests/integration/test_transcript_provenance_api.py` and extend `backend/tests/unit/test_cli.py`.

- [ ] **Step 1: Write failing API/CLI tests.** Assert work detail returns raw/cleaned variant, source kind, cleaner model, and fallback reason. Assert `clean-transcripts --aweme-id X --limit 3 --dry-run` performs no writes and normal startup does not enqueue historical cleaning.

- [ ] **Step 2: Implement API fields and job filters.** Include `clean_transcript_work` in relevant job summaries and expose transcript provenance without returning cookies, signed URLs, or raw model payloads.

- [ ] **Step 3: Implement the explicit CLI.** Add dry-run, selected IDs, limit, provider/model override, and per-item result reporting. Default command mode only selects transcripts lacking a valid cleaned variant; it never uses `--force --all` implicitly.

- [ ] **Step 4: Update UI states.** Show whether the note used `subtitle/raw`, `asr/raw`, or `cleaned`, plus a concise cleaning fallback reason. Keep full note reading and long-text scrolling intact.

- [ ] **Step 5: Run backend API/CLI and frontend tests/build.**

  Run: `uv run --group dev pytest backend/tests/integration/test_transcript_provenance_api.py backend/tests/unit/test_cli.py -q`

  Run: `npm test -- --run` and `npm run build` from `frontend`.

  Expected: PASS and a successful production build.

- [ ] **Step 6: Commit.**

  ```bash
  git add backend/src/douyin_library/api/routes/works.py backend/src/douyin_library/api/routes/jobs.py backend/src/douyin_library/cli.py frontend/src/App.tsx frontend/src/NoteReader.tsx frontend/src/styles.css backend/tests/integration/test_transcript_provenance_api.py backend/tests/unit/test_cli.py
  git commit -m "feat: expose transcript provenance and safe cleanup cli"
  ```

## Task 9: End-to-End Verification and Controlled Rollout

**Files:** Modify `docs/operations/semantic-notes.md`, `README.md`, and `.env.example` only after implementation tests pass; create no production data files.

- [ ] **Step 1: Run the complete backend quality gates.**

  ```bash
  uv run --group dev pytest -q
  uv run --group dev ruff check backend/src backend/tests
  uv run python -m compileall -q backend/src backend/tests
  git diff --check
  ```

  Expected: all tests pass, Ruff is clean, compileall succeeds, and no whitespace errors are reported.

- [ ] **Step 2: Run a synthetic subtitle-first integration case.** Create a temporary test workspace containing a valid local SRT and an audio asset; assert no ASR subprocess is invoked, raw source is `subtitle`, and cleaning is queued.

- [ ] **Step 3: Run a synthetic ASR fallback case.** Use an empty/invalid subtitle candidate and a fake Qwen provider; assert raw source is `asr`, raw files are immutable after cleaning failure, and note generation reports raw fallback.

- [ ] **Step 4: Run one authorized live Ollama cleaning smoke test.** Use one newly transcribed work only, verify `qwen3:4b` is already installed, inspect audit/output hashes, and stop on any schema, drift, or provenance failure. Do not enqueue historical work.

- [ ] **Step 5: Verify backup and publication safety.** Confirm the database backup includes transcript variant rows, cleaned files, audits, and external payloads. Exercise concurrent publish tests to confirm the TOCTOU guard cannot replace a newer bundle.

- [ ] **Step 6: Document operational rollout.** Record feature flags, model requirements, fallback semantics, dry-run command, and the prohibition on video downloads/source overwrites in `docs/operations/semantic-notes.md`.

- [ ] **Step 7: Commit operational documentation.**

  ```bash
  git add docs/operations/semantic-notes.md
  git commit -m "docs: document subtitle-first cleaning rollout"
  ```

## Plan Self-Review

- Spec coverage: provider contracts (Tasks 1-3), local-only subtitle-first routing (Task 2), raw/cleaned persistence and atomic publication (Task 4), Ollama validation/batching (Task 5), durable jobs and fallback (Task 6), evidence-complete notes and the “姜学长” regression (Task 7), provenance/UI/CLI (Task 8), and rollout verification (Task 9) are all mapped.
- Completeness scan: every code change names files, tests, and expected outcomes; no step relies on an unspecified follow-up.
- Type consistency: `TranscriptSegment`, provider protocols, variant names, job kind `clean_transcript_work`, and settings names are defined once in earlier tasks and reused consistently.
- Safety: no task downloads videos, reads browser credentials, starts historical queues, or overwrites raw D-drive assets. The known publisher TOCTOU race is an explicit Task 4 prerequisite.
