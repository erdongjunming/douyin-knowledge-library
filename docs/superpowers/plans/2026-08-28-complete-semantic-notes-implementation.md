# Complete Semantic Learning Notes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the lossy `local-rule-v1` note path with an auditable local-first semantic pipeline that preserves the full transcript, publishes only validated bundles, and can incrementally upgrade existing notes without risking source assets.

**Architecture:** The note path is split into typed evidence extraction, theme synthesis, independent source review, one repair pass, deterministic coverage validation, rendering, and transactional publishing. Ollama is the default semantic provider, an explicitly configured external OpenAI-compatible endpoint is optional, and an all-segment deterministic fallback is always available. SQLite jobs, externalized audit payloads, a process-shared GPU lease, and separate general/ASR worker processes provide cancellation, heartbeat, resource control, and recovery.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2, Alembic, Pydantic 2, httpx, SQLite WAL, pytest/pytest-asyncio, React 19, TypeScript, TanStack Query, Vitest.

---

## Safety And Compatibility Invariants

- Work only in `C:\Users\CJM\Documents\Codex\2026-08-06\hermes-deepseek-3\outputs\douyin-knowledge-library`.
- Never download or save full videos and never rewrite `audio`, `cover`, `metadata`, or `transcript` source assets.
- Treat the existing dirty worktree as user-owned. Read each overlapping file immediately before patching; do not reset, checkout, clean, or stage unrelated changes.
- Keep existing API response fields and CLI behavior; new fields are additive.
- Never include an external API key, Authorization value, cookie, or signed query string in jobs, diagnostics, prompts, note files, or SQLite.
- Never replace an existing note with fallback output or with semantic output that failed its quality gate.
- Never auto-install Ollama or auto-pull a model. Model download is an explicit operator action.

## File Responsibility Map

- `notes/contracts.py`: strict structured schemas shared across providers, pipeline, coverage, and rendering.
- `notes/state.py`: classify current bundles as semantic, fallback, legacy, unknown, or missing.
- `notes/input.py`: read complete source assets without mutation or truncation.
- `notes/prompts.py`: versioned extraction, synthesis, review, and repair prompts.
- `notes/providers/`: provider protocol plus Ollama, external, rule, and policy selection.
- `notes/coverage.py`: deterministic segment/atom/source coverage checks and quality score inputs.
- `notes/pipeline.py`: multi-pass semantic orchestration with exactly one repair pass.
- `notes/render.py`: complete learning note, concise brief, budget, score, metadata, and manifest rendering.
- `notes/publisher.py`: staging, archive, manifest-last publication, fencing, and rollback.
- `jobs/leases.py`: atomic cross-process SQLite resource lease operations.
- `worker_main.py`: standalone general or ASR worker process entry point.
- `notes/service.py`: candidate selection, provider fallback, auditing, Analysis persistence, and publishing.
- `api/routes/notes.py`, `api/routes/works.py`, `cli.py`: compatible controls and status surfaces.
- `frontend/src/NoteReader.tsx`: safe in-page Markdown reader and theme/timeline navigation.

### Task 1: Strict Note Contracts, Bundle State, And Deterministic Coverage

**Files:**
- Create: `backend/src/douyin_library/notes/contracts.py`
- Create: `backend/src/douyin_library/notes/state.py`
- Create: `backend/src/douyin_library/notes/coverage.py`
- Create: `backend/tests/unit/notes/test_contracts.py`
- Create: `backend/tests/unit/notes/test_coverage.py`
- Create: `backend/tests/unit/notes/test_state.py`

- [ ] **Step 1: Write failing contract and coverage tests**

```python
def test_every_source_segment_requires_a_disposition() -> None:
    extraction = ExtractionResult(
        atoms=[atom(id="A-0001", source_segment=0)],
        dispositions=[disposition(segment_index=0, atom_ids=["A-0001"])],
    )
    report = CoverageValidator().validate(
        segment_count=2,
        extraction=extraction,
        synthesis=synthesis(included_atom_ids=["A-0001"]),
        review=clean_review(reviewed_segment_indexes=[0, 1]),
        semantic_generator="ollama/qwen3:8b",
    )
    assert report.quality_gate == "failed"
    assert report.missing_segment_indexes == [1]


def test_unknown_bundle_is_not_eligible_for_legacy_upgrade(tmp_path: Path) -> None:
    (tmp_path / "learning_note.md").write_text("# old", encoding="utf-8")
    state = classify_note_bundle(tmp_path)
    assert state.quality_status == "unknown"
    assert state.legacy_eligible is False
```

- [ ] **Step 2: Verify the tests fail for missing modules**

Run: `cd backend; uv run pytest tests/unit/notes/test_contracts.py tests/unit/notes/test_coverage.py tests/unit/notes/test_state.py -v`

Expected: collection fails because `notes.contracts`, `notes.coverage`, and `notes.state` do not exist.

- [ ] **Step 3: Implement strict schemas and bundle classification**

Use `ConfigDict(extra="forbid")` on all Pydantic models. Define these exact public types:

```python
AtomKind = Literal[
    "claim", "reason", "definition", "step", "example", "counterexample",
    "condition", "warning", "number", "quote", "recommendation", "promotion",
]
DispositionKind = Literal[
    "included", "duplicate", "transition", "promotion", "unintelligible_asr"
]
QualityGate = Literal["passed", "failed", "fallback"]

class EvidenceAtom(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: AtomKind
    topic: str
    statement: str
    source_segment: int = Field(ge=0)
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    source_excerpt: str
    relations: list[str] = Field(default_factory=list)
    sequence: int | None = Field(default=None, ge=1)
    is_promotional: bool = False
    asr_uncertainty: list[str] = Field(default_factory=list)

class SegmentDisposition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segment_index: int = Field(ge=0)
    disposition: DispositionKind
    atom_ids: list[str] = Field(default_factory=list)
    reason: str | None = None

class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    atoms: list[EvidenceAtom]
    dispositions: list[SegmentDisposition]

class ThemeSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    heading: str
    content: str
    atom_ids: list[str]
    timestamps: list[float] = Field(default_factory=list)

class TimelineEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segment_index: int = Field(ge=0)
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, ge=0)
    disposition: DispositionKind
    atom_ids: list[str]
    content: str

class SynthesisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    overview: str
    structure: list[str]
    themes: list[ThemeSection]
    methods_and_steps: list[ThemeSection]
    examples_and_counterexamples: list[ThemeSection]
    boundaries_and_risks: list[ThemeSection]
    concepts: list[str]
    quotes: list[str]
    implications: list[str]
    actions: list[str]
    timeline: list[TimelineEntry]
    promotions: list[ThemeSection]
    tags: list[str]
    included_atom_ids: list[str]

class SourceOmissionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segment_index: int = Field(ge=0)
    source_excerpt: str
    missing_statement: str
    suggested_kind: AtomKind

class CoverageReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewed_segment_indexes: list[int]
    source_omission_candidates: list[SourceOmissionCandidate]
    unsupported_draft_claims: list[str]
    disposition_corrections: list[SegmentDisposition]

class ProviderUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    api_cost: float | None = Field(default=None, ge=0)
    cost_status: Literal["zero_local", "reported", "estimated_from_config", "unknown"]

class CoverageReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_name: Literal["douyin-note-coverage-v1"] = Field(
        default="douyin-note-coverage-v1", serialization_alias="schema"
    )
    segment_total: int
    segment_disposition_count: int
    missing_segment_indexes: list[int]
    evidence_atom_total: int
    evidence_atom_included: int
    missing_atom_ids: list[str]
    traceable_atom_count: int
    source_segments_independently_reviewed: int
    source_omission_candidates: list[SourceOmissionCandidate]
    unsupported_draft_claims: list[str]
    steps_verified: bool
    examples_verified: bool
    conditions_verified: bool
    promotional_content_preserved: bool
    semantic_generator: str
    quality_gate: QualityGate
    repairable: bool
```

`classify_note_bundle(note_dir)` must inspect, in order, `generation_manifest.json`, `generation_audit.json`, and `note_budget.json`; malformed JSON returns `unknown`. Only an explicit `local-rule-v1` model/generator is `legacy`.

- [ ] **Step 4: Implement deterministic validation**

`CoverageValidator.validate(...)` must fail when any segment lacks exactly one disposition, any atom is untraceable or absent from synthesis, an ordered step sequence has gaps, example/condition relations are unresolved, a promotional atom is absent, any raw segment was not independently reviewed, or the review contains omission candidates/unsupported claims. Coverage percentages return `None` when their denominator is zero.

- [ ] **Step 5: Run focused tests and commit**

Run: `cd backend; uv run pytest tests/unit/notes/test_contracts.py tests/unit/notes/test_coverage.py tests/unit/notes/test_state.py -v`

Expected: all tests pass.

Commit: `git commit -m "feat: define semantic note coverage contracts"`

### Task 2: Provider Configuration, Ollama Runtime, And Provider Policy

**Files:**
- Modify: `backend/src/douyin_library/config.py`
- Create: `backend/src/douyin_library/notes/providers/__init__.py`
- Create: `backend/src/douyin_library/notes/providers/base.py`
- Create: `backend/src/douyin_library/notes/providers/ollama.py`
- Create: `backend/src/douyin_library/notes/providers/external.py`
- Create: `backend/src/douyin_library/notes/providers/selector.py`
- Create: `backend/src/douyin_library/notes/ollama_runtime.py`
- Create: `backend/tests/unit/notes/test_provider_selection.py`
- Create: `backend/tests/unit/notes/test_ollama_provider.py`
- Create: `backend/tests/unit/notes/test_external_provider.py`
- Modify: `backend/tests/integration/test_diagnostics.py`

- [ ] **Step 1: Write failing policy, transport, cancellation, and diagnostics tests**

```python
def test_auto_never_uses_external_without_complete_explicit_configuration() -> None:
    chain = select_provider_chain(settings(provider_policy="auto", external_key=None))
    assert [provider.name for provider in chain] == ["ollama", "rule"]


@pytest.mark.asyncio
async def test_ollama_cancel_aborts_request_and_never_pulls_model() -> None:
    requested_paths: list[str] = []
    provider = OllamaNoteProvider(settings(), transport=recording_transport(requested_paths))
    cancel_event = asyncio.Event()
    task = asyncio.create_task(provider.complete_json(request(), cancel_event=cancel_event))
    cancel_event.set()
    with pytest.raises(JobCancelledError):
        await task
    assert "/api/pull" not in requested_paths
```

- [ ] **Step 2: Verify red**

Run: `cd backend; uv run pytest tests/unit/notes/test_provider_selection.py tests/unit/notes/test_ollama_provider.py tests/unit/notes/test_external_provider.py tests/integration/test_diagnostics.py -v`

Expected: failures for missing provider settings and modules.

- [ ] **Step 3: Add settings and provider-neutral response types**

Add settings with these defaults and bounds:

```python
note_provider_policy: Literal["auto", "local", "external", "rule"] = "auto"
note_ollama_url: str = "http://127.0.0.1:11434"
note_ollama_model: str = "qwen3:8b"
note_ollama_executable: Path | None = None
note_request_timeout_seconds: float = Field(default=600, ge=10, le=3600)
note_context_tokens: int = Field(default=32768, ge=4096, le=131072)
note_ollama_keep_alive: str = "10m"
note_external_base_url: str | None = None
note_external_model: str | None = None
note_external_api_key: SecretStr | None = None
note_external_input_cost_per_million: float | None = Field(default=None, ge=0)
note_external_output_cost_per_million: float | None = Field(default=None, ge=0)
```

Define `ProviderResponse(raw_output, parsed_json, usage, diagnostics)` and an async `NoteProvider` protocol with `complete_json(...)` and `unload()` methods. Usage must report token counts independently from `api_cost`; unknown external monetary cost is `None` with `cost_status="unknown"`.

- [ ] **Step 4: Implement providers and Ollama lifecycle**

`OllamaNoteProvider` must use async `httpx` against `/api/chat` with `stream: false`, a supplied JSON schema in `format`, temperature zero, and the configured short keep-alive while the service owns `gpu:0`. Race the HTTP request against `cancel_event.wait()` and cancel the request task when cancellation wins. `unload()` sends an explicit Ollama request with `keep_alive: 0`; the service awaits that attempt before releasing the GPU lease. `ExternalNoteProvider` must read the revealed secret only while creating the Authorization header and must return sanitized diagnostics.

`resolve_ollama_executable()` checks explicit path, `PATH`, then `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`. `probe_ollama()` distinguishes executable missing, service stopped, model missing, and ready. `start_if_installed()` uses a hidden window on Windows, tracks whether this app started the process, and never calls `pull`.

- [ ] **Step 5: Verify and commit**

Run: `cd backend; uv run pytest tests/unit/notes/test_provider_selection.py tests/unit/notes/test_ollama_provider.py tests/unit/notes/test_external_provider.py tests/integration/test_diagnostics.py -v`

Expected: all tests pass and no response contains the configured key.

Commit: `git commit -m "feat: add local-first note providers"`

### Task 3: Complete Input Loading, Prompts, Semantic Pipeline, And Rendering

**Files:**
- Create: `backend/src/douyin_library/notes/input.py`
- Create: `backend/src/douyin_library/notes/prompts.py`
- Create: `backend/src/douyin_library/notes/pipeline.py`
- Create: `backend/src/douyin_library/notes/render.py`
- Modify: `backend/src/douyin_library/notes/generator.py`
- Create: `backend/tests/unit/notes/test_input.py`
- Create: `backend/tests/unit/notes/test_pipeline.py`
- Create: `backend/tests/unit/notes/test_render.py`
- Create: `backend/tests/unit/notes/test_rule_fallback.py`

- [ ] **Step 1: Write failing end-to-end unit tests around a synthetic transcript**

The fixture must contain three ordered steps, an example with result, a condition, a warning, and a promotional closing. Assert that extraction batches cover every segment, source review sees raw segments independently, repair occurs once, every item appears in theme review and timeline, and fallback includes every segment in source order.

```python
@pytest.mark.asyncio
async def test_pipeline_repairs_one_source_omission_then_passes() -> None:
    provider = ScriptedProvider([extraction_json, synthesis_json, review_with_gap, repaired_json])
    result = await SemanticNotePipeline(provider).generate(note_input, asyncio.Event())
    assert provider.stages == ["extract", "synthesize", "review", "repair"]
    assert result.coverage.quality_gate == "passed"
    assert result.repair_count == 1


def test_rule_fallback_timeline_contains_all_segments() -> None:
    bundle = generate_rule_note(note_input, source_files={}, generated_at=NOW)
    assert [item["segment_index"] for item in bundle.parsed["timeline"]] == [0, 1, 2, 3]
```

- [ ] **Step 2: Verify red**

Run: `cd backend; uv run pytest tests/unit/notes/test_input.py tests/unit/notes/test_pipeline.py tests/unit/notes/test_render.py tests/unit/notes/test_rule_fallback.py -v`

Expected: failures because the complete pipeline and all-segment fallback do not exist.

- [ ] **Step 3: Implement complete, non-truncating input and versioned prompts**

`NoteInputLoader.load(aweme_id)` must return `NoteInput`, a dataclass with `aweme_id`, `transcript`, `segments`, `metadata`, `description`, `author`, `author_unique_id`, `published_at`, `duration_ms`, `source_url`, `source_files`, `asset_records`, and `warnings`. Each `InputSegment` has `index`, `start`, `end`, and `text`. Invalid `segments.json` becomes one synthetic segment containing the full transcript and an audit warning; source text is never cut. Batch boundaries must be recorded in `note_budget.json`.

Prompt constants must explicitly prohibit invention, require atom IDs and source excerpts, preserve examples/conditions/steps/promotions, distinguish author claims from generated interpretation, and require JSON-only output. Give each stage its own prompt version.

- [ ] **Step 4: Implement the four-stage pipeline and one repair pass**

```python
class SemanticNotePipeline:
    async def generate(
        self,
        note_input: NoteInput,
        cancel_event: asyncio.Event,
        audit_sink: AuditSink,
    ) -> SemanticGeneration:
        extraction = await self._extract_all_batches(note_input, cancel_event, audit_sink)
        synthesis = await self._synthesize(extraction, cancel_event, audit_sink)
        review = await self._review_raw_source(note_input, synthesis, cancel_event, audit_sink)
        coverage = self.validator.validate(
            segment_count=len(note_input.segments),
            extraction=extraction,
            synthesis=synthesis,
            review=review,
            semantic_generator=f"{self.provider.name}/{self.provider.model}",
        )
        repair_count = 0
        if coverage.quality_gate == "failed" and coverage.repairable:
            synthesis = await self._repair(note_input, extraction, synthesis, coverage, cancel_event, audit_sink)
            repair_count = 1
            review = await self._review_raw_source(note_input, synthesis, cancel_event, audit_sink)
            coverage = self.validator.validate(
                segment_count=len(note_input.segments),
                extraction=extraction,
                synthesis=synthesis,
                review=review,
                semantic_generator=f"{self.provider.name}/{self.provider.model}",
            )
        return SemanticGeneration(extraction, synthesis, review, coverage, repair_count)
```

The second review after repair is validation, not a second repair opportunity. If it still fails, return a failed semantic candidate to the service; do not render it as passed.

- [ ] **Step 5: Render the complete note and improve fallback**

Render all required sections from the specification, including `## 主题复盘`, `## 方法与步骤`, `## 案例与反例`, `## 前提、边界与风险`, `## 我的理解与启发`, `## 完整逐段时间线`, and `## 推广与行动号召`. Theme items contain timestamp and atom references. Timeline entries contain source order, time range, disposition, atom IDs, and faithful content.

The deterministic fallback must also render every segment, set `quality_gate="fallback"`, state that it did not pass semantic completeness review, and never claim semantic coverage.

- [ ] **Step 6: Verify and commit**

Run: `cd backend; uv run pytest tests/unit/notes/test_input.py tests/unit/notes/test_pipeline.py tests/unit/notes/test_render.py tests/unit/notes/test_rule_fallback.py -v`

Expected: all tests pass.

Commit: `git commit -m "feat: generate complete evidence-linked notes"`

### Task 4: Versioned, Atomic, And Downgrade-Safe Bundle Publishing

**Files:**
- Create: `backend/src/douyin_library/notes/publisher.py`
- Create: `backend/tests/unit/notes/test_publisher.py`

- [ ] **Step 1: Write failing publisher tests**

Cover first publish, archive-on-replace, required output validation, manifest-last ordering, simulated replace failure rollback, semantic ownership fencing, and rejection of fallback over any existing note.

```python
def test_failed_replace_restores_previous_bundle(tmp_path: Path) -> None:
    publisher = NoteBundlePublisher(tmp_path, replace=fail_on_third_replace())
    seed_current_bundle(tmp_path, learning_note="# previous")
    with pytest.raises(PublishError):
        publisher.publish("731", passed_semantic_candidate(), force=True)
    assert (tmp_path / "dy-note/731/learning_note.md").read_text() == "# previous"


def test_fallback_never_replaces_existing_note(tmp_path: Path) -> None:
    seed_current_bundle(tmp_path, learning_note="# previous")
    result = NoteBundlePublisher(tmp_path).publish("731", fallback_candidate(), force=True)
    assert result.status == "preserved_stronger_existing"
```

- [ ] **Step 2: Verify red**

Run: `cd backend; uv run pytest tests/unit/notes/test_publisher.py -v`

Expected: import fails because `notes.publisher` does not exist.

- [ ] **Step 3: Implement staging, archive, fencing, and rollback**

Write all required files under `dy-note/<id>/.staging/<uuid>/`, fsync files, validate JSON and required sections, then archive the current generated bundle to `versions/<UTC timestamp>-<generator-version>/`. Do not copy source assets. Replace generated files individually and write `generation_manifest.json` last. On any error restore the archive, remove only the generation's staging directory, and record the failure. Before committing the manifest, invoke an ownership callback that proves the job still owns its GPU lease and remains running.

- [ ] **Step 4: Verify and commit**

Run: `cd backend; uv run pytest tests/unit/notes/test_publisher.py -v`

Expected: all tests pass, including injected failure after two replacements.

Commit: `git commit -m "feat: publish versioned note bundles safely"`

### Task 5: Cross-Process GPU Resource Lease

**Files:**
- Modify: `backend/src/douyin_library/models.py`
- Create: `backend/migrations/versions/0005_resource_lease.py`
- Create: `backend/src/douyin_library/jobs/leases.py`
- Create: `backend/tests/unit/test_resource_lease.py`
- Modify: `backend/tests/integration/test_migrations.py`

- [ ] **Step 1: Write failing lease and migration tests**

```python
def test_expired_lease_can_be_taken_but_old_token_cannot_release_it(session: Session) -> None:
    first = store(session).try_acquire("gpu:0", "worker-a", 1, ttl=SECOND)
    expire(first, session)
    second = store(session).try_acquire("gpu:0", "worker-b", 2, ttl=MINUTE)
    assert second is not None
    assert store(session).release(first) is False
    assert store(session).is_owned(second) is True
```

Migration coverage must create a database at `0004_job_heartbeat`, upgrade to head, and assert table columns plus unique/token and heartbeat/expiry indexes. It must also prove repeated `upgrade head` is harmless because `0001_initial` invokes current metadata.

- [ ] **Step 2: Verify red**

Run: `cd backend; uv run pytest tests/unit/test_resource_lease.py tests/integration/test_migrations.py -v`

Expected: missing `ResourceLease` and migration failures.

- [ ] **Step 3: Add the model, idempotent migration, and atomic store**

```python
class ResourceLease(Base):
    __tablename__ = "resource_lease"
    resource_name: Mapped[str] = mapped_column(String(80), primary_key=True)
    lease_token: Mapped[str] = mapped_column(String(64), unique=True)
    owner_worker_id: Mapped[str] = mapped_column(String(120), index=True)
    owner_job_id: Mapped[int | None] = mapped_column(ForeignKey("job.id"), index=True)
    acquired_at: Mapped[datetime]
    heartbeat_at: Mapped[datetime] = mapped_column(index=True)
    expires_at: Mapped[datetime] = mapped_column(index=True)
```

`ResourceLeaseStore.try_acquire()` must execute one SQLite `INSERT ... ON CONFLICT(resource_name) DO UPDATE ... WHERE resource_lease.expires_at <= excluded.acquired_at`. Renewal and release must match resource name, token, worker, and job. Every method uses a short transaction and never holds a transaction while sleeping.

- [ ] **Step 4: Verify and commit**

Run: `cd backend; uv run pytest tests/unit/test_resource_lease.py tests/integration/test_migrations.py -v`

Expected: all tests pass.

Commit: `git commit -m "feat: coordinate GPU work with SQLite leases"`

### Task 6: True Standalone Workers, Cross-Process Cancellation, And ASR Lease Use

**Files:**
- Create: `backend/src/douyin_library/worker_main.py`
- Modify: `backend/src/douyin_library/jobs/worker.py`
- Modify: `backend/src/douyin_library/jobs/runtime.py`
- Modify: `backend/src/douyin_library/jobs/queue.py`
- Modify: `backend/src/douyin_library/transcription/service.py`
- Modify: `backend/src/douyin_library/main.py`
- Modify: `backend/src/douyin_library/cli.py`
- Modify: `scripts/dev.ps1`
- Modify: `start-local.cmd`
- Modify: `backend/tests/unit/test_job_cancellation.py`
- Modify: `backend/tests/integration/test_job_recovery.py`
- Modify: `backend/tests/integration/test_transcription_pipeline.py`
- Create: `backend/tests/integration/test_worker_process_contract.py`

- [ ] **Step 1: Write failing tests for DB cancellation and GPU waiting**

```python
@pytest.mark.asyncio
async def test_remote_cancel_sets_event_and_terminates_asr_child(tmp_path: Path) -> None:
    task = asyncio.create_task(asr_worker.run_one())
    await child_started.wait()
    with Session(engine) as api_session:
        JobQueue(api_session).cancel(job.id)
    await asyncio.wait_for(task, timeout=3)
    assert fake_process.terminate_called is True
    assert read_job(engine, job.id).status == "cancelled"


@pytest.mark.asyncio
async def test_asr_waits_without_starting_transcriber_while_gpu_is_leased() -> None:
    blocker = acquire_gpu_for_other_job(engine)
    task = asyncio.create_task(service.process(job, work, cancel_event=cancel_event))
    await wait_until(lambda: read_job(engine, job.id).stage == "waiting_gpu")
    assert transcriber.calls == 0
    release(blocker)
    await task
    assert transcriber.calls == 1
```

- [ ] **Step 2: Verify red**

Run: `cd backend; uv run pytest tests/unit/test_job_cancellation.py tests/integration/test_job_recovery.py tests/integration/test_transcription_pipeline.py tests/integration/test_worker_process_contract.py -v`

Expected: cancellation is not observed across processes and ASR does not use a lease.

- [ ] **Step 3: Make heartbeat drive cancellation and lease renewal**

`JobWorker._heartbeat(job_id, cancel_event)` must use an independent session. When `JobQueue.heartbeat()` returns false because the row is cancelled, requeued, or owned by another worker, it sets `cancel_event`. On each successful job heartbeat it renews every lease belonging to that job and worker. Stale recovery may purge only already-expired leases; token matching prevents an old worker from releasing a replacement owner's lease.

- [ ] **Step 4: Hold the GPU only around actual ASR execution**

After checking for a reusable transcript, `TranscriptionService.process()` loops on `try_acquire("gpu:0", ...)` with a short configured interval, setting job stage to `waiting_gpu`. Each loop checks both the local event and database job status. Once acquired, stage becomes `transcribe`; `transcriber.run()` is wrapped in `try/finally` and the exact handle is released on success, error, or cancellation.

- [ ] **Step 5: Add process entry points and remove production embedded workers**

`python -m douyin_library.worker_main --pool general --concurrency 2` runs only `GENERAL_JOB_KINDS`; `--pool asr --concurrency 1` runs only `ASR_JOB_KINDS`. FastAPI initializes storage and exposes the API but defaults `embedded_workers` to false. Tests may enable embedded workers explicitly. `scripts/dev.ps1` and `start-local.cmd` launch backend, general worker, ASR worker, and frontend as separate hidden/managed processes and stop the whole process tree on shutdown.

- [ ] **Step 6: Verify and commit**

Run: `cd backend; uv run pytest tests/unit/test_job_cancellation.py tests/integration/test_job_recovery.py tests/integration/test_transcription_pipeline.py tests/integration/test_worker_process_contract.py -v`

Expected: all tests pass and a cancelled ASR job terminates its child within the test timeout.

Commit: `git commit -m "feat: run ASR in a cancellable worker process"`

### Task 7: Audit Payload Integrity And Resilient Automatic Backups

**Files:**
- Modify: `backend/src/douyin_library/audit.py`
- Modify: `backend/src/douyin_library/maintenance.py`
- Modify: `backend/src/douyin_library/runtime.py`
- Modify: `backend/src/douyin_library/main.py`
- Modify: `backend/src/douyin_library/cli.py`
- Modify: `backend/tests/unit/test_audit.py`
- Modify: `backend/tests/unit/test_maintenance.py`
- Modify: `backend/tests/integration/test_runtime_initialization.py`

- [ ] **Step 1: Write failing migration, integrity, and backup resilience tests**

```python
def test_compaction_externalizes_all_large_audit_fields(session: Session, tmp_path: Path) -> None:
    audit = seed_large_legacy_audit(session, fields=("request_json", "response_json", "stdout", "stderr"))
    result = compact_large_audit_payloads(session, audit_root=tmp_path, inline_max_bytes=4096)
    session.refresh(audit)
    assert result["migrated_fields"] == 4
    for value in (audit.request_json, audit.response_json, audit.stdout, audit.stderr):
        envelope = json.loads(value)
        assert envelope["kind"] == "external-audit-payload-v1"
        assert AuditPayloadStore(tmp_path, 4096).load(value)


@pytest.mark.asyncio
async def test_backup_loop_continues_after_one_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = iter([RuntimeError("disk busy"), {"backup": "ok"}])
    monkeypatch.setattr(main_module, "create_database_backup", lambda *a, **k: next(calls))
    await run_two_backup_iterations()
    assert next_backup_iteration_was_reached()
```

- [ ] **Step 2: Verify red**

Run: `cd backend; uv run pytest tests/unit/test_audit.py tests/unit/test_maintenance.py tests/integration/test_runtime_initialization.py -v`

Expected: legacy compaction ignores request/stdout/stderr, integrity loading is absent, and the backup loop exits on error.

- [ ] **Step 3: Generalize compaction and validate references**

Replace the response-only migration with `compact_large_audit_payloads()`. Iterate `request_json`, `response_json`, `stdout`, and `stderr` with the correct JSON/text suffix, sanitize legacy values before writing, commit each migrated field independently, skip valid existing envelopes, and support safe restart after a partial failure. `AuditPayloadStore.load(reference)` must reject paths outside the configured root and verify the declared byte length and SHA-256 before returning content. Keep `compact_large_audit_responses()` as a compatible wrapper for existing callers.

- [ ] **Step 4: Make backups migration-aware and failure-resilient**

Before Alembic upgrade, compare current and head revisions. If a database exists and migration is required, force a verified online backup with `min_interval=timedelta(0)`; otherwise retain the normal interval. Catch and log each periodic backup exception inside the loop, then continue to the next interval. Serialize backup creation across processes with a short `maintenance:backup` SQLite/file lock and recheck the interval after acquiring it. Retention removes backup and matching manifest together.

- [ ] **Step 5: Verify and commit**

Run: `cd backend; uv run pytest tests/unit/test_audit.py tests/unit/test_maintenance.py tests/integration/test_runtime_initialization.py -v`

Expected: all tests pass, compaction is idempotent, corrupt references are rejected, and backup retries continue.

Commit: `git commit -m "fix: harden audit storage and database backups"`

### Task 8: Async Note Service, Provider Fallback, Audit, And Safe Analysis Persistence

**Files:**
- Modify: `backend/src/douyin_library/notes/service.py`
- Modify: `backend/src/douyin_library/jobs/runtime.py`
- Modify: `backend/src/douyin_library/audit.py`
- Modify: `backend/tests/integration/test_note_generation.py`
- Modify: `backend/tests/integration/test_job_runtime.py`
- Modify: `backend/tests/unit/test_audit.py`

- [ ] **Step 1: Write failing service integration tests**

Cover local success, local invalid output to one repair, configured external fallback, no external configuration to rule fallback, cancellation, existing semantic preservation, legacy archive and upgrade, actual provider in `Analysis`, and externalized input/raw output audit references.

```python
@pytest.mark.asyncio
async def test_failed_semantic_chain_preserves_existing_note(tmp_path: Path) -> None:
    seed_semantic_note(tmp_path, "731", "# trusted")
    result = await service(failing_local(), failing_external()).generate_one(
        "731", force=True, provider_policy="auto", cancel_event=asyncio.Event()
    )
    assert result.status == "preserved_existing"
    assert learning_note(tmp_path, "731") == "# trusted"


def test_large_note_audit_stores_hash_summary_and_path(session: Session, tmp_path: Path) -> None:
    reference = generate_large_audit(session, tmp_path)
    envelope = json.loads(reference)
    assert envelope["kind"] == "external-audit-payload-v1"
    assert len(envelope["sha256"]) == 64
    assert Path(envelope["path"]).is_file()
```

- [ ] **Step 2: Verify red**

Run: `cd backend; uv run pytest tests/integration/test_note_generation.py tests/integration/test_job_runtime.py tests/unit/test_audit.py -v`

Expected: service is synchronous/rule-only and semantic scenarios fail.

- [ ] **Step 3: Implement async service orchestration**

Change `generate_one()` and `run_batch()` to async. Recheck bundle state at execution time. Provider policy is `auto: local -> configured external -> rule`, `local: local -> rule`, `external: external -> rule`, and `rule: rule only`. Acquire `gpu:0` for the complete local semantic attempt plus unload, set visible stages, release in `finally`, and verify lease ownership plus job ownership before publishing. A failed semantic candidate is audited but never published.

Use `ProcessingAuditRecorder(..., inline_max_bytes=settings.audit_inline_max_bytes)` and store every stage prompt/raw/parsed result in the recorder response so large values are externalized. `generation_audit.json` stores sanitized summaries, hashes, external payload paths, provider decisions, timing, coverage, and error details instead of duplicating multi-megabyte payloads.

A successful bundle publishes its root `generation_audit.json`. A failed or cancelled attempt writes only a compact audit descriptor under `dy-note/<id>/attempts/<UTC timestamp>/generation_audit.json`; it must not touch current generated files. The descriptor points to verified external audit payloads for full prompts, source input, raw responses, and parsed structures.

- [ ] **Step 4: Persist search Analysis only after successful publication**

Before inserting a new current `Analysis`, mark prior succeeded analyses for the same work as `superseded`; the existing FTS query then indexes only the current succeeded result instead of concatenating every historical generation. Insert the replacement after the generation manifest is committed, with actual final provider, model, prompt version, concise summary, tags/topics/key points, evidence and coverage. Fallback Analysis is marked provider `rule` and must not masquerade as semantic output.

- [ ] **Step 5: Await service from the job handler and verify**

The `generate_note` handler validates `force`, `legacy_only`, and `provider_policy`, then awaits the service with the job's `cancel_event`. It reclassifies legacy eligibility immediately before generation.

Run: `cd backend; uv run pytest tests/integration/test_note_generation.py tests/integration/test_job_runtime.py tests/unit/test_audit.py -v`

Expected: all tests pass.

Commit: `git commit -m "feat: orchestrate audited semantic note generation"`

### Task 9: Compatible API, CLI, Progress, And Work Detail Contracts

**Files:**
- Modify: `backend/src/douyin_library/api/routes/notes.py`
- Modify: `backend/src/douyin_library/api/routes/works.py`
- Modify: `backend/src/douyin_library/cli.py`
- Modify: `backend/tests/integration/test_jobs_api.py`
- Modify: `backend/tests/integration/test_works_transcript_api.py`
- Modify: `backend/tests/unit/test_cli.py`
- Create: `backend/tests/integration/test_note_provider_api.py`

- [ ] **Step 1: Write failing API and CLI tests**

Assert old `POST /api/notes/generate` still returns 202; dry-run returns 200 without jobs; `force && legacy_only` returns 422; only explicit legacy bundles are selected; duplicate IDs are harmless; keys never enter payloads; progress includes legacy upgrade jobs even when `learning_note.md` exists; malformed coverage does not cause a 500.

```python
response = client.post("/api/notes/generate", json={
    "legacy_only": True, "provider_policy": "auto", "limit": 20
})
assert response.status_code == 202
payload = json.loads(latest_job.payload_json)
assert payload == {
    "aweme_id": "731", "force": False,
    "legacy_only": True, "provider_policy": "auto",
}
assert "key" not in latest_job.payload_json.lower()
```

- [ ] **Step 2: Verify red**

Run: `cd backend; uv run pytest tests/integration/test_jobs_api.py tests/integration/test_works_transcript_api.py tests/integration/test_note_provider_api.py tests/unit/test_cli.py -v`

Expected: new request fields, status endpoint, and note quality fields are absent.

- [ ] **Step 3: Add request, status, progress, and work detail fields**

`GenerateNotesRequest` adds `dry_run`, `legacy_only`, `provider_policy`, and `aweme_ids`. Dry-run performs classification without creating jobs. Add `GET /api/notes/provider-status`; it probes only on explicit request, not from the five-second progress endpoint. Extend progress with `quality_counts`, `upgrade`, and cached `provider_summary`. Extend work rows/detail with `note_quality_status`, `note_job_status`, provider/model/gate, nullable segment/atom coverage, full coverage report, and generation metadata. Preserve existing `note_status` and `note_available` meanings.

- [ ] **Step 4: Add CLI controls using the shared classifier**

Support `--legacy-only`, `--provider auto|local|external|rule`, repeated `--aweme-id`, `--dry-run`, `--force`, and mutually exclusive `--limit/--all`. Reject `--force --legacy-only`. Run async service code with `asyncio.run` for direct CLI execution, while the web API continues to queue jobs.

- [ ] **Step 5: Verify and commit**

Run: `cd backend; uv run pytest tests/integration/test_jobs_api.py tests/integration/test_works_transcript_api.py tests/integration/test_note_provider_api.py tests/unit/test_cli.py -v`

Expected: all tests pass.

Commit: `git commit -m "feat: expose semantic note controls and status"`

### Task 10: Provider-Aware Progress And Complete In-Page Note Reader

**Files:**
- Modify: `frontend/package.json`
- Create: `frontend/src/NoteReader.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/src/App.test.tsx`

- [ ] **Step 1: Write failing frontend tests**

Test Ollama-ready/model-missing states, upgrade request body, disabled upgrade when no semantic provider is ready, fallback warning, nullable coverage, single-item generation, and theme/timeline navigation. Both complete sections must remain in the DOM; navigation scrolls the modal body instead of hiding content.

```tsx
fireEvent.click(await screen.findByRole("button", { name: "升级旧笔记" }))
expect(api).toHaveBeenCalledWith("/api/notes/generate", expect.objectContaining({
  method: "POST",
  body: JSON.stringify({ legacy_only: true, provider_policy: "auto" }),
}))

fireEvent.click(screen.getByRole("button", { name: "逐段时间线" }))
expect(screen.getByRole("heading", { name: "主题复盘" })).toBeVisible()
expect(screen.getByRole("heading", { name: "完整逐段时间线" })).toBeVisible()
expect(scrollIntoView).toHaveBeenCalled()
```

- [ ] **Step 2: Verify red**

Run: `cd frontend; npm test -- --run`

Expected: new controls, fields, and reader navigation assertions fail.

- [ ] **Step 3: Add safe Markdown rendering and navigation**

Add `react-markdown` without `rehype-raw`. `NoteReader` parses level-two headings only for navigation, renders the complete Markdown once, and shows sticky icon/text navigation buttons for available `主题复盘` and `完整逐段时间线` sections. Links receive safe `target`/`rel` handling; raw HTML is never executed.

- [ ] **Step 4: Add provider, quality, coverage, and upgrade UI**

Keep new response fields optional so existing mocks remain compatible. Show provider/model, semantic/fallback/legacy/unknown counts, coverage when non-null, and an explicit fallback warning. Keep “生成缺失笔记”; add “升级旧笔记” and a per-work “生成完整笔记” action. Disable legacy upgrade if Ollama is not ready and external is not explicitly enabled because fallback cannot replace current notes.

- [ ] **Step 5: Verify responsive layout and commit**

Run: `cd frontend; npm test -- --run`

Run: `cd frontend; npm run build`

Expected: tests pass; TypeScript and Vite production build pass without warnings.

Commit: `git commit -m "feat: review complete semantic notes in the library"`

### Task 11: Full Verification, Ollama Setup, And Jiang Sample Acceptance

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Create: `docs/operations/semantic-notes.md`
- Modify: relevant backend/frontend tests only when an acceptance failure is reproduced first

- [ ] **Step 1: Document explicit setup and cost semantics**

Document `ollama serve`, the deliberate `ollama pull qwen3:8b` command, provider policies, external opt-in variables, GPU lease behavior, standalone worker commands, audit payload paths, backup retention, dry-run, 3-item sample, legacy upgrade, retry, and rollback locations. State that local Ollama API cost is zero but consumes local GPU/electricity; external cost is unknown unless returned or explicit rates are configured.

- [ ] **Step 2: Run static and automated verification**

Run: `cd backend; uv run ruff check src tests`

Run: `cd backend; uv run pytest -v`

Run: `cd frontend; npm test -- --run`

Run: `cd frontend; npm run build`

Expected: every command exits zero.

- [ ] **Step 3: Verify no source mutation and inspect candidate counts**

Hash the representative videos' audio, cover, metadata, transcript, segments, and ASR JSON before generation. Run:

```powershell
cd backend
uv run python -m douyin_library.cli generate-notes --legacy-only --provider local --dry-run
```

Expected: reports candidates without creating jobs, versions, or note files. Re-hash source assets and assert all hashes match.

- [ ] **Step 4: Start Ollama deliberately and validate readiness**

Only after explicit operator approval, run the installed executable's `serve` command and then `ollama pull qwen3:8b`. Verify `GET /api/notes/provider-status` reports `ready: true`, model `qwen3:8b`, and does not expose secrets.

- [ ] **Step 5: Generate three to five Jiang Xuezhang samples**

Select a long tutorial, a list video, and a viewpoint video from existing metadata. Run with explicit IDs and provider `local`; do not use `--force` on unknown bundles. Confirm each old legacy bundle is archived, current manifest is semantic/passed, both note views are complete, `api_cost` is zero, and every source asset hash remains unchanged.

For `7636753241384695225`, manually and programmatically assert all acceptance facts listed in the design specification, including the three construction phases, four semantic-alignment techniques, examples, conditions, rollback warning, error-report template, final recommendations, and promotional content.

- [ ] **Step 6: Run the incremental legacy upgrade only after sample approval**

```powershell
cd backend
uv run python -m douyin_library.cli generate-notes --legacy-only --provider local --all
```

Expected: already semantic notes are skipped; failed items retain current bundles and auditable retry reasons; no paid provider is contacted.

- [ ] **Step 7: Final review and commit**

Inspect `git diff --check`, the complete backend/frontend test logs, three sample bundles, SQLite audit envelopes, and source hashes.

Commit: `git commit -m "docs: operate and verify semantic note generation"`

## Final Acceptance Checklist

- Every semantic note has complete theme review and chronological timeline content.
- Every source segment has a disposition and independent review result.
- Every extracted atom is included and traceable; unsupported claims and omission candidates are empty.
- Existing notes are archived and replaced only after a passed quality gate.
- Rule fallback contains all segments and never overwrites an existing note.
- Ollama is local-first, uses `qwen3:8b`, runs at concurrency one, unloads after use, and has zero API cost.
- External API use is explicit, secret-safe, and cost reporting is honest.
- ASR and Ollama cannot concurrently own `gpu:0`; leases heartbeat, expire, fence publishing, and release on cancellation.
- ASR runs in a separate OS worker process and remote cancellation terminates its subprocess.
- Large prompts/responses are files; SQLite stores summaries, hashes, and paths; backups run automatically.
- CLI, API, web progress, upgrade controls, and in-page note reading remain backward compatible.
- All source asset hashes are unchanged after generation.
