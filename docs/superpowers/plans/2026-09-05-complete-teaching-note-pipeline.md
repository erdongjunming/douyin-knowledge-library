# Complete Teaching Note Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing evidence-linked Ollama note pipeline so each historical or newly generated note is a complete teaching document whose claims, steps, examples, results, conditions, risks, and actions remain together and traceable to source atoms and timestamps.

**Architecture:** Keep the current lossless extraction, source review, bounded repair, deterministic timeline, and atomic publisher. Add typed teaching outlines and section drafts after extraction, generate each fixed teaching section independently within its own context budget, run a depth review in addition to coverage review, and deterministically merge/render the sections. A semantic-complete result requires coverage, depth, and input-sufficiency gates; degraded results remain auditable and never replace a protected historical bundle.

**Tech Stack:** Python 3.12, Pydantic 2, FastAPI, SQLAlchemy 2, Ollama-compatible JSON provider, pytest/pytest-asyncio, Ruff, React 19, TypeScript, Vitest.

---

## Scope And Invariants

- Work in `C:\Users\CJM\Documents\Codex\2026-08-06\hermes-deepseek-3\outputs\douyin-knowledge-library`.
- Treat the existing dirty worktree as user-owned. Read overlapping files immediately before editing; never reset, checkout, clean, or stage unrelated changes.
- Reuse `EvidenceAtom`, `SegmentDisposition`, existing extraction batching, source review, `CoverageValidator`, `NoteBundlePublisher`, and `NoteInputLoader`; do not create a second note-generation path.
- Keep subtitle/ASR as the factual spine and local Ollama as the primary semantic provider. Do not add a new remote provider, browser automation, CAPTCHA bypass, or credential handling.
- Never invent information absent from the input. Missing categories render the exact phrase `视频未提供这一类信息。`; visual or ASR gaps remain explicit.
- Keep `learning_note.md`, `evidence_map.json`, `coverage_report.json`, `note_budget.json`, `note_score.json`, and existing API fields backward compatible. Additive fields are allowed.
- Existing source audio, cover, metadata, raw transcript, segments, and cleaned transcript assets are read-only inputs.
- Historical upgrade publication is accepted only for a passed Ollama semantic bundle. Provider errors, failed gates, cancellation, ownership loss, and publication rejection preserve the old bundle and remain retryable.

## File Responsibility Map

- `backend/src/douyin_library/notes/contracts.py`: strict outline, section, claim, depth-review, and additive coverage contracts.
- `backend/src/douyin_library/notes/depth.py`: deterministic depth checks for compression, adjacency, literals, and unsupported claims.
- `backend/src/douyin_library/notes/prompts.py`: versioned outline, section, depth-review, and bounded-repair instructions.
- `backend/src/douyin_library/notes/providers/base.py`: provider stage literals and protocol compatibility.
- `backend/src/douyin_library/notes/pipeline.py`: outline scheduling, per-section calls/retries, deterministic merges, depth review, and one repair pass.
- `backend/src/douyin_library/notes/coverage.py`: combine coverage, depth, and input-sufficiency gates without weakening existing foundational checks.
- `backend/src/douyin_library/notes/render.py`: render nested teaching content while preserving the existing timeline, brief, score, budget, and evidence map.
- `backend/src/douyin_library/notes/generator.py`: carry the new outline/section/depth artifacts in the existing `NoteBundle` without changing required output names.
- `backend/src/douyin_library/notes/service.py`: select passed semantic candidates, publish allowed degraded new notes, and protect legacy upgrades.
- `backend/src/douyin_library/notes/state.py`: classify additive quality/depth metadata and keep failed/degraded bundles upgrade-eligible.
- `backend/src/douyin_library/api/routes/notes.py`: preserve progress wire compatibility and expose additive depth/input fields.
- `frontend/src/App.tsx`, `frontend/src/App.test.tsx`: show upgrade counts and compatibility fallbacks without conflating normal generation with semantic upgrades.
- `backend/tests/unit/notes/`, `backend/tests/integration/`, `frontend/src/`: red-green regression coverage and the Jiang Xuezhang fixture.

### Task 0: Verify The Existing Dirty Baseline

**Files:**
- Read-only: `backend/src/douyin_library/transcription/inventory.py`, `backend/src/douyin_library/notes/`, `backend/tests/`, `frontend/src/`

- [ ] **Step 1: Record the current worktree without changing it.** Run `git status --short --branch` and `git diff --stat`. Confirm the existing runtime, transcript-cleaning, inventory/backfill, semantic-v2, and frontend changes are user-owned baseline changes; do not stage or modify them.

- [ ] **Step 2: Run the focused baseline tests.**

Run: `cd backend; uv run pytest tests/unit/notes/test_contracts.py tests/unit/notes/test_coverage.py tests/unit/notes/test_pipeline.py tests/unit/notes/test_render.py tests/unit/notes/test_service_semantic.py tests/integration/test_note_generation.py tests/integration/test_upgrade_job_contract.py -q`

Expected: the baseline either passes or its pre-existing failures are recorded before Task 1. Do not attribute a baseline failure to the teaching-note work until the same test is rerun after the corresponding task.

- [ ] **Step 3: Run static checks before the first teaching change.**

Run: `cd backend; uv run ruff check src tests; uv run python -m compileall -q src tests; git diff --check`

Expected: baseline diagnostics are captured; no file is reset or cleaned.

### Task 1: Add Strict Teaching Contracts And Depth Findings

**Files:**
- Modify: `backend/src/douyin_library/notes/contracts.py`
- Create: `backend/src/douyin_library/notes/depth.py`
- Modify: `backend/src/douyin_library/notes/coverage.py`
- Modify: `backend/tests/unit/notes/test_contracts.py`
- Modify: `backend/tests/unit/notes/test_coverage.py`
- Create: `backend/tests/unit/notes/test_depth.py`

- [ ] **Step 1: Write failing contract tests.** Add these exact cases to `test_contracts.py`:

```python
def test_teaching_outline_rejects_unknown_section_and_missing_atom() -> None:
    with pytest.raises(ValidationError):
        TeachingOutline.model_validate({
            "sections": [{
                "section_id": "not-a-section",
                "title": "bad",
                "order": 1,
                "atom_ids": ["A-1"],
                "required_evidence_types": ["claim"],
                "missing_evidence_policy": "视频未提供这一类信息。",
                "timestamps": [],
            }],
            "included_atom_ids": ["A-1"],
        })


def test_section_claim_requires_atom_and_timestamp() -> None:
    with pytest.raises(ValidationError):
        SectionClaim.model_validate({
            "text": "无来源主张",
            "atom_ids": [],
            "timestamps": [],
        })


def test_depth_review_is_strict_and_has_explicit_repair_sections() -> None:
    review = DepthReview(
        abstracted_atom_ids=["A-step-1"],
        example_without_result_atom_ids=[],
        condition_without_scope_atom_ids=[],
        reordered_step_atom_ids=[],
        lost_literal_atom_ids=["A-number"],
        unsupported_claims=[],
        repair_section_ids=["methods_steps"],
        input_sufficient=True,
    )
    assert review.model_config["extra"] == "forbid"
    assert review.repair_section_ids == ["methods_steps"]
```

- [ ] **Step 2: Run the red tests.**

Run: `cd backend; uv run pytest tests/unit/notes/test_contracts.py::test_teaching_outline_rejects_unknown_section_and_missing_atom tests/unit/notes/test_contracts.py::test_section_claim_requires_atom_and_timestamp tests/unit/notes/test_depth.py -q`

Expected: collection or validation failures because the teaching contracts and `notes.depth` do not exist.

- [ ] **Step 3: Define the strict public types.** Add `ConfigDict(extra="forbid", strict=True)` models and exact literals:

```python
TeachingSectionId = Literal[
    "overview", "core_points", "methods_steps", "examples_results",
    "boundaries_risks", "actions", "timeline", "evidence_limits",
]
EvidenceRequirement = Literal[
    "claim", "reason", "step", "example", "result", "condition",
    "risk", "number", "named_entity", "uncertainty", "promotion", "action",
]

class TeachingOutlineSection(BaseModel):
    model_config = _STRICT_MODEL_CONFIG
    section_id: TeachingSectionId
    title: NonEmptyString
    order: int = Field(ge=1)
    atom_ids: list[NonEmptyString]
    required_evidence_types: list[EvidenceRequirement]
    missing_evidence_policy: Literal["视频未提供这一类信息。"]
    timestamps: list[NonNegativeFloat] = Field(default_factory=list)

class TeachingOutline(BaseModel):
    model_config = _STRICT_MODEL_CONFIG
    sections: list[TeachingOutlineSection]
    included_atom_ids: list[NonEmptyString]
    unassigned_atom_ids: list[NonEmptyString] = Field(default_factory=list)
    input_sufficient: bool = True

class SectionClaim(BaseModel):
    model_config = _STRICT_MODEL_CONFIG
    text: NonEmptyString
    atom_ids: list[NonEmptyString]
    timestamps: list[NonNegativeFloat]

class SectionDraft(BaseModel):
    model_config = _STRICT_MODEL_CONFIG
    section_id: TeachingSectionId
    title: NonEmptyString
    content: NonEmptyString
    claims: list[SectionClaim] = Field(default_factory=list)
    steps: list[SectionClaim] = Field(default_factory=list)
    examples: list[SectionClaim] = Field(default_factory=list)
    results: list[SectionClaim] = Field(default_factory=list)
    conditions: list[SectionClaim] = Field(default_factory=list)
    risks: list[SectionClaim] = Field(default_factory=list)
    actions: list[SectionClaim] = Field(default_factory=list)
    included_atom_ids: list[NonEmptyString]
    unprovided_items: list[NonEmptyString] = Field(default_factory=list)
    generated_interpretations: list[NonEmptyString] = Field(default_factory=list)

class DepthReview(BaseModel):
    model_config = _STRICT_MODEL_CONFIG
    abstracted_atom_ids: list[NonEmptyString]
    example_without_result_atom_ids: list[NonEmptyString]
    condition_without_scope_atom_ids: list[NonEmptyString]
    reordered_step_atom_ids: list[NonEmptyString]
    lost_literal_atom_ids: list[NonEmptyString]
    lost_warning_or_promotion_atom_ids: list[NonEmptyString]
    lost_uncertainty_markers: list[NonEmptyString]
    unsupported_claims: list[NonEmptyString]
    repair_section_ids: list[TeachingSectionId]
    input_sufficient: bool
```

Keep `SynthesisResult` and `CoverageReport` field names intact. Extend `CoverageReport` additively with `depth_passed: bool = False`, `input_sufficient: bool = False`, `depth_review: dict[str, Any] | None = None`, and `semantic_complete: bool = False`; old JSON remains readable because all new fields have defaults.

- [ ] **Step 4: Implement deterministic depth validation.** In `depth.py`, define:

```python
class DepthValidator:
    def validate(
        self,
        *,
        atoms: list[EvidenceAtom],
        outline: TeachingOutline,
        sections: list[SectionDraft],
        review: DepthReview,
    ) -> DepthReview:
        """Return a fail-closed review with IDs not represented in section bodies."""
```

Build a searchable body from every section’s `content` and structured item text. Mark an atom abstracted when its `statement` and `source_excerpt` are absent from the body and it is not explicitly listed in `unprovided_items`. Require every example/counterexample to have a result relation or a `results` item; require every condition/warning/risk to appear in the same section as at least one related claim/step; require `sequence` values to appear in order; and require number, named-entity, promotion, warning, and `asr_uncertainty` text to survive. Unknown atom IDs or unsupported claims are non-repairable. Only findings limited to missing section content are repairable.

- [ ] **Step 5: Combine quality gates without changing old validator semantics.** Change `CoverageValidator.validate` to accept optional `depth_review: DepthReview | None = None` and `input_sufficient: bool = True`. Preserve all existing foundational/repairable checks, then set:

```python
depth_passed = depth_review is not None and not any(
    (depth_review.abstracted_atom_ids,
     depth_review.example_without_result_atom_ids,
     depth_review.condition_without_scope_atom_ids,
     depth_review.reordered_step_atom_ids,
     depth_review.lost_literal_atom_ids,
     depth_review.lost_warning_or_promotion_atom_ids,
     depth_review.lost_uncertainty_markers,
     depth_review.unsupported_claims)
)
semantic_complete = passed and depth_passed and input_sufficient
quality_gate = "passed" if semantic_complete else "failed"
```

When old callers omit `depth_review`, keep `quality_gate` based on the prior coverage result and set `depth_passed=False`, `semantic_complete=False` so legacy tests can be migrated explicitly rather than silently claiming completeness.

- [ ] **Step 6: Run focused tests and commit.**

Run: `cd backend; uv run pytest tests/unit/notes/test_contracts.py tests/unit/notes/test_coverage.py tests/unit/notes/test_depth.py -q`

Expected: all new and existing contract/coverage tests pass.

Commit: `git add backend/src/douyin_library/notes/contracts.py backend/src/douyin_library/notes/depth.py backend/src/douyin_library/notes/coverage.py backend/tests/unit/notes/test_contracts.py backend/tests/unit/notes/test_coverage.py backend/tests/unit/notes/test_depth.py; git commit -m "feat: add teaching note depth contracts"`

### Task 2: Add Versioned Outline, Section, And Depth Prompts

**Files:**
- Modify: `backend/src/douyin_library/notes/providers/base.py`
- Modify: `backend/src/douyin_library/notes/prompts.py`
- Modify: `backend/tests/unit/notes/test_pipeline.py`
- Create: `backend/tests/unit/notes/test_teaching_prompts.py`

- [ ] **Step 1: Write prompt and stage tests.** Assert the provider stage type accepts `outline`, `section`, and `depth_review`; each prompt contains `视频未提供这一类信息`, `atom_ids`, timestamps, examples/results, conditions/risks, no-invention language, and JSON-only output; and prompt versions are distinct.

- [ ] **Step 2: Run the red tests.**

Run: `cd backend; uv run pytest tests/unit/notes/test_teaching_prompts.py -q`

Expected: failures because the stage literals and prompt keys are absent.

- [ ] **Step 3: Extend the provider stage contract.** Change `NoteStage` to:

```python
NoteStage = Literal[
    "extract", "outline", "synthesize", "section", "review", "depth_review", "repair",
]
```

Keep `NoteProvider.complete_json` unchanged so existing Ollama/external providers and test doubles remain protocol-compatible.

- [ ] **Step 4: Add exact prompt versions and rules.** Add `OUTLINE_PROMPT_VERSION = "teaching-outline-v1"`, `SECTION_PROMPT_VERSION = "teaching-section-v1"`, and `DEPTH_REVIEW_PROMPT_VERSION = "teaching-depth-review-v1"`. The outline prompt must emit the eight fixed sections in order and assign every atom to at least one section. The section prompt must require a `SectionDraft`, preserve source order/atom IDs/timestamps, keep each example with its result, and place conditions/risks next to the linked claim or step. The depth prompt must inspect raw segments and section drafts independently, report all loss categories, and never repair in the review call. Add `REPAIR_PROMPT_VERSION = "teaching-bounded-repair-v2"` while keeping the old key readable in audit records.

- [ ] **Step 5: Verify prompt tests and commit.**

Run: `cd backend; uv run pytest tests/unit/notes/test_teaching_prompts.py tests/unit/notes/test_pipeline.py -q`

Expected: prompt assertions pass; existing provider fixtures still collect.

Commit: `git add backend/src/douyin_library/notes/providers/base.py backend/src/douyin_library/notes/prompts.py backend/tests/unit/notes/test_teaching_prompts.py backend/tests/unit/notes/test_pipeline.py; git commit -m "feat: add teaching note provider prompts"`

### Task 3: Schedule Outline And Independent Section Synthesis

**Files:**
- Modify: `backend/src/douyin_library/notes/pipeline.py`
- Modify: `backend/tests/unit/notes/test_pipeline.py`
- Modify: `backend/tests/unit/notes/test_input.py`

- [ ] **Step 1: Add red pipeline fixtures.** Extend the scripted provider with `TeachingOutline`, `SectionDraft`, and `DepthReview` JSON responses. Add tests that use a transcript containing three ordered steps, a named example and result, a condition, a warning, a number, and a promotion. Assert `SemanticGeneration.outline.sections` has the fixed order, each section has a bounded atom list, and the section stage call list contains one call per section.

- [ ] **Step 2: Test section-local retry and long-input preservation.** Make one section provider response invalid once, then valid; assert only that section is retried and previously successful sections are not called again. Build a transcript larger than the context budget and assert all atom IDs, source segment indexes, cross-batch relations, and order survive outline and section merging.

- [ ] **Step 3: Run red tests.**

Run: `cd backend; uv run pytest tests/unit/notes/test_pipeline.py -k "teaching or section or long_input" -q`

Expected: failures because `SemanticGeneration` has no outline/section fields and the pipeline has no outline or section stages.

- [ ] **Step 4: Extend `SemanticGeneration` without removing old fields.** Add:

```python
outline: TeachingOutline
section_drafts: tuple[SectionDraft, ...]
depth_review: DepthReview | None
section_attempts: dict[str, int]
input_sufficient: bool
```

Keep `synthesis: SynthesisResult` as the compatibility projection consumed by `render_semantic_note` and `NoteBundle`.

- [ ] **Step 5: Implement `_build_outline`.** After `_merge_extractions`, build a compact atom catalog containing `id`, `kind`, `topic`, `statement`, `source_segment`, timestamps, relations, and uncertainty. Call the provider with `stage="outline"` and `TeachingOutline.model_json_schema()`. If the catalog does not fit, split only between atoms, call one outline slice at a time, and deterministically union sections by fixed `section_id`, deduplicate atom IDs in source order, and put unassigned atoms in `evidence_limits`. Reject unknown IDs, duplicate section IDs, or missing fixed sections before continuing.

- [ ] **Step 6: Implement `_synthesize_sections`.** For each outline section in `order`, pass only its assigned atoms plus the corresponding raw segments and metadata. Use `_provider_request_budget` with `stage="section"` and `SectionDraft` schema. Split an oversized section payload on atom boundaries and merge its drafts by `section_id`, preserving atom/source order and concatenating structured lists with deterministic ID/text de-duplication. Retry only a failed section once; record `section_attempts[section_id]` and a `StageRecord` for every attempt. A failed section becomes a draft with `unprovided_items` and causes a failed quality gate; it does not trigger a full-video rerun.

- [ ] **Step 7: Build a compatibility `SynthesisResult`.** Deterministically map section drafts to the existing fields: overview from `overview`, themes from `core_points`, methods from `methods_steps`, examples from `examples_results`, boundaries from `boundaries_risks`, actions from `actions`, promotions from atoms marked promotion, and timeline from the existing source-order projection. Do not let this adapter discard structured claims, results, conditions, risks, atom IDs, or timestamps.

- [ ] **Step 8: Verify focused pipeline tests and commit.**

Run: `cd backend; uv run pytest tests/unit/notes/test_pipeline.py tests/unit/notes/test_input.py -q`

Expected: all existing extraction/merge tests and new outline/section tests pass.

Commit: `git add backend/src/douyin_library/notes/pipeline.py backend/tests/unit/notes/test_pipeline.py backend/tests/unit/notes/test_input.py; git commit -m "feat: synthesize teaching sections within budget"`

### Task 4: Add Depth Review, One Repair, And Honest Quality State

**Files:**
- Modify: `backend/src/douyin_library/notes/pipeline.py`
- Modify: `backend/src/douyin_library/notes/coverage.py`
- Modify: `backend/src/douyin_library/notes/input.py`
- Modify: `backend/src/douyin_library/notes/prompts.py`
- Modify: `backend/tests/unit/notes/test_pipeline.py`
- Modify: `backend/tests/unit/notes/test_coverage.py`
- Modify: `backend/tests/unit/notes/test_input.py`

- [ ] **Step 1: Add red depth/retry tests.** Cover an abstracted step, an example without its result, a condition detached from its claim, a lost number/name/warning/promotion/uncertainty marker, and an unsupported section claim. Assert only the flagged section is repaired once; a second failed depth review returns `quality_gate="failed"`, `semantic_complete=False`, and `repair_count == 1`.

- [ ] **Step 2: Implement `_review_depth`.** Call `stage="depth_review"` with raw source segments, the outline, and all section drafts. For long inputs, send one section group per request and merge findings by atom/section ID. Run `DepthValidator` after provider parsing so omitted bookkeeping fails closed and unknown IDs are non-repairable.

- [ ] **Step 3: Extend bounded repair.** Change the existing repair payload to include `outline`, `section_drafts`, `coverage_report`, and `depth_review`, and request only `repair_section_ids`/`repairable` atoms. Materialize the returned `RepairResult` into the existing extraction/synthesis merge plus the affected `SectionDraft`s. Increment `repair_count` once; re-run source review, depth review, and coverage validation, but never call repair a second time.

- [ ] **Step 4: Define input sufficiency.** Add `_input_is_sufficient(note_input)` in `pipeline.py` using `NoteInput.warnings`, `transcript_variant`, and optional `metadata["visual_dependency"]`/`note_budget.json` risk. Missing segments, sparse transcript warnings, or `visual_dependency.risk in {"medium", "high"}` make the result readable but not semantic-complete. This gate does not invent visual evidence.

- [ ] **Step 5: Set final gate.** Pass `depth_review` and `input_sufficient` into `CoverageValidator`; only `coverage passed + depth passed + input_sufficient` yields `semantic_complete=True` and `quality_gate="passed"`. Provider errors remain exceptions; evidence insufficiency returns a failed semantic candidate for service-level handling.

- [ ] **Step 6: Verify and commit.**

Run: `cd backend; uv run pytest tests/unit/notes/test_pipeline.py tests/unit/notes/test_coverage.py tests/unit/notes/test_input.py -q`

Expected: all current coverage, one-repair, cancellation, context-budget, and new depth tests pass.

Commit: `git add backend/src/douyin_library/notes/pipeline.py backend/src/douyin_library/notes/coverage.py backend/src/douyin_library/notes/input.py backend/src/douyin_library/notes/prompts.py backend/tests/unit/notes/test_pipeline.py backend/tests/unit/notes/test_coverage.py backend/tests/unit/notes/test_input.py; git commit -m "feat: gate semantic notes on depth and input sufficiency"`

### Task 5: Render Nested Teaching Content And Preserve Existing Artifacts

**Files:**
- Modify: `backend/src/douyin_library/notes/render.py`
- Modify: `backend/src/douyin_library/notes/generator.py`
- Modify: `backend/tests/unit/notes/test_render.py`
- Modify: `backend/tests/unit/notes/test_rule_fallback.py`
- Modify: `backend/tests/unit/notes/test_evidence_extract.py`

- [ ] **Step 1: Write red rendering assertions.** Assert the topic body contains, in order, `观点`, `原文依据`, `解释`, `例子/结果`, `适用条件/限制`, and `风险或行动`, with an atom reference and timestamp on each claim. Assert a section with no relevant atoms renders `视频未提供这一类信息。`, not a generic model sentence. Assert the full timeline remains source ordered and `evidence_map.json` contains section IDs.

- [ ] **Step 2: Implement deterministic section rendering.** Add `_render_section_draft(lines, draft, atoms_by_id)` that emits each claim and then its linked source/time, explanation, examples/results, conditions/risks, and actions. Use source excerpts only for author evidence; label `generated_interpretations` as `生成理解`. Render unprovided categories exactly as specified. Keep the existing headings and old `SynthesisResult` fallback path for fixtures that do not yet carry `section_drafts`.

- [ ] **Step 3: Extend bundle payloads additively.** Add `teaching_outline`, `section_drafts`, `depth_review`, `semantic_complete`, and `input_sufficient` to `NoteBundle.parsed`, `brief_json`, `page_metadata`, `coverage_report`, and `note_budget` where present. Do not add a required new file to the publisher; keep all machine-readable teaching evidence in existing `evidence_map.json` and `coverage_report.json`.

- [ ] **Step 4: Keep rule fallback honest.** The deterministic fallback continues to include every source segment in the timeline and uses `quality_gate="fallback"`, `semantic_complete=False`, and an explicit “未通过语义完整度审查” line. It must never emit a teaching section that claims examples, results, conditions, or risks absent from the source.

- [ ] **Step 5: Verify rendering and commit.**

Run: `cd backend; uv run pytest tests/unit/notes/test_render.py tests/unit/notes/test_rule_fallback.py tests/unit/notes/test_evidence_extract.py -q`

Expected: complete/failed/fallback bundles all preserve existing paths and pass the new nested-content assertions.

Commit: `git add backend/src/douyin_library/notes/render.py backend/src/douyin_library/notes/generator.py backend/tests/unit/notes/test_render.py backend/tests/unit/notes/test_rule_fallback.py backend/tests/unit/notes/test_evidence_extract.py; git commit -m "feat: render complete teaching note sections"`

### Task 6: Enforce Publication And Historical Upgrade Semantics

**Files:**
- Modify: `backend/src/douyin_library/notes/service.py`
- Modify: `backend/src/douyin_library/notes/state.py`
- Modify: `backend/src/douyin_library/notes/publisher.py`
- Modify: `backend/src/douyin_library/api/routes/notes.py`
- Modify: `backend/tests/unit/notes/test_service_semantic.py`
- Modify: `backend/tests/integration/test_note_generation.py`
- Modify: `backend/tests/integration/test_legacy_upgrade_api_contract.py`
- Modify: `backend/tests/unit/notes/test_upgrade_batch_contract.py`

- [ ] **Step 1: Add failing service tests.** Cover:

```python
async def test_failed_depth_preserves_existing_upgrade_bundle(tmp_path):
    result = await service.generate_one("731", legacy_only=True, cancel_event=asyncio.Event())
    assert result.status == "preserved_existing"
    assert old_learning_note.read_text(encoding="utf-8") == old_text


async def test_new_input_insufficiency_publishes_readable_failed_gate(tmp_path):
    result = await service.generate_one("new", cancel_event=asyncio.Event())
    assert result.status == "generated"
    assert result.quality_gate == "failed"
    assert json.loads((note_dir / "generation_manifest.json").read_text())["semantic_complete"] is False
```

Also assert a provider transport error never publishes the failed candidate and never replaces an existing note.

- [ ] **Step 2: Separate provider failure, degraded semantic output, and upgrade acceptance.** In `_generate_one_async`:

  - Treat provider construction/transport/schema/cancellation/publish errors as failed attempts and continue the configured chain or preserve the existing bundle.
  - A parsed semantic generation with `quality_gate="failed"` may be rendered and published only when `learning_note.md` did not previously exist and the provider produced a complete deterministic bundle; mark it `semantic_complete=False`, `quality_gate="failed"`, and upgrade-eligible.
  - When `legacy_only=True`, accept only `provider.name == "ollama"`, `semantic_complete=True`, and `quality_gate="passed"`; otherwise preserve the old bundle and return `preserved_existing`/failed audit.
  - Keep `NoteBundlePublisher` ownership and legacy-state guards around the final replacement.

- [ ] **Step 3: Make bundle classification additive and fail closed.** `classify_note_bundle` must classify only passed + semantic-complete Ollama/external identities as `semantic_complete`; failed-gate or missing-depth metadata is `unknown`/upgrade-eligible. Malformed JSON remains `unknown`. Add tests for old manifests without new fields and new failed-gate manifests.

- [ ] **Step 4: Fix progress compatibility.** In `api/routes/notes.py` preserve current top-level fields and expose `upgrade.depth_failed`/`upgrade.input_insufficient` counts if available. In the frontend contract used by the route, `eligible` falls back to the server’s legacy `upgrade.eligible` when `quality_counts` is absent; `already_queued` is reported; and enqueueing updates the local progress lane immediately. Do not change the three-item hard cap or local Ollama-only upgrade policy.

- [ ] **Step 5: Verify service/API tests and commit.**

Run: `cd backend; uv run pytest tests/unit/notes/test_service_semantic.py tests/unit/notes/test_upgrade_batch_contract.py tests/integration/test_note_generation.py tests/integration/test_legacy_upgrade_api_contract.py -q`

Expected: old bundles are byte-for-byte preserved on upgrade/provider failure; degraded new notes are readable but non-complete; semantic upgrades count only passed Ollama publications.

Commit: `git add backend/src/douyin_library/notes/service.py backend/src/douyin_library/notes/state.py backend/src/douyin_library/notes/publisher.py backend/src/douyin_library/api/routes/notes.py backend/tests/unit/notes/test_service_semantic.py backend/tests/unit/notes/test_upgrade_batch_contract.py backend/tests/integration/test_note_generation.py backend/tests/integration/test_legacy_upgrade_api_contract.py; git commit -m "fix: enforce teaching quality and upgrade publication gates"`

### Task 7: Update Frontend Progress And Reader Regression Coverage

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`
- Modify: `frontend/src/NoteReader.tsx`
- Modify: `frontend/src/NoteReader.test.tsx`

- [ ] **Step 1: Add failing UI tests.** Cover a progress response containing only `upgrade.eligible`, a response with `queued=0, already_queued=3`, a local progress update immediately after clicking upgrade, and rendering of nested teaching headings/quality warnings.

- [ ] **Step 2: Implement compatibility and state updates.** Use `upgrade.remaining ?? upgrade.eligible ?? 0` for button visibility, display queued plus already-active counts, update `noteProgress.upgrade` from the enqueue response before polling, and keep the button disabled unless local Ollama is ready. Do not remove existing normal-generation fields.

- [ ] **Step 3: Keep the reader source-safe.** Render the complete Markdown through the existing safe reader, expose `主题复盘`, `方法与步骤`, `案例与反例`, `前提、边界与风险`, and `完整逐段时间线` navigation, and retain explicit degraded-quality text. No raw HTML or external credential-bearing links are introduced.

- [ ] **Step 4: Verify and commit.**

Run: `cd frontend; npm test -- --run src/App.test.tsx src/NoteReader.test.tsx; npm run build`

Expected: all frontend tests pass and the production build succeeds.

Commit: `git add frontend/src/App.tsx frontend/src/App.test.tsx frontend/src/NoteReader.tsx frontend/src/NoteReader.test.tsx; git commit -m "fix: expose teaching note quality and upgrade progress"`

### Task 8: Jiang Xuezhang Regression, Historical Reuse, And Full Verification

**Files:**
- Modify: `backend/tests/unit/notes/test_evidence_extract.py`
- Modify: `backend/tests/integration/test_note_generation.py`
- Create: `backend/tests/integration/test_teaching_note_regression.py`
- Modify: `docs/operations/semantic-notes.md`
- Modify: `README.md`

- [ ] **Step 1: Add a deterministic regression fixture.** Use the existing synthetic Jiang fixture shape and the stored work metadata for `7636753241384695225` when present. The fixture must contain and assert in the topic body, not only the timeline: three ordered steps, the case and its result, `100元`, the weekly-review condition, the ASR/name risk, and the promotion. Assert every item has atom IDs and timestamps and the old note remains unchanged until publication succeeds.

- [ ] **Step 2: Add historical reuse tests.** Seed an existing raw/cleaned transcript, metadata, note, and asset manifest; run `legacy_only` with a fake Ollama provider; assert no downloader/ASR/transcript-cleaning call occurs, section retries are bounded, and a failed upgrade leaves every source and old note hash unchanged.

- [ ] **Step 3: Run the complete verification gates.**

Run:

```powershell
cd backend
uv run pytest -q
uv run ruff check src tests
uv run python -m compileall -q src tests
git diff --check
cd ..\frontend
npm test -- --run
npm run build
```

Expected: all backend/frontend tests pass, Ruff and compileall are clean, and no whitespace errors are reported.

- [ ] **Step 4: Document the operator workflow.** Update `docs/operations/semantic-notes.md` and `README.md` with the fixed section order, `semantic_complete` gate, degraded-note semantics, local Ollama setup, asset reuse during historical upgrades, three-item API cap, dry-run/retry behavior, and the prohibition on overwriting source assets or bypassing CAPTCHA. Do not run a live historical upgrade in this task.

- [ ] **Step 5: Commit the regression/docs changes.**

Run: `git add backend/tests/unit/notes/test_evidence_extract.py backend/tests/integration/test_note_generation.py backend/tests/integration/test_teaching_note_regression.py docs/operations/semantic-notes.md README.md; git commit -m "test: verify complete teaching notes on historical fixtures"`

## Plan Self-Review

- **Spec coverage:** strict data contracts (Task 1), chapter outline and independent section generation (Tasks 2-3), depth review and one bounded repair (Task 4), deterministic nested rendering and explicit missing evidence (Task 5), semantic/degraded/upgrade publication state (Task 6), UI progress compatibility (Task 7), and historical reuse plus Jiang acceptance (Task 8) map to every section of `docs/superpowers/specs/2026-09-05-learning-note-optimization-design.md`.
- **Placeholder scan:** every implementation step names exact files, tests, commands, and expected results; no step relies on an unspecified follow-up.
- **Type consistency:** `TeachingSectionId`, `TeachingOutline`, `SectionDraft`, `DepthReview`, `SemanticGeneration.outline`, `SemanticGeneration.section_drafts`, and `CoverageReport.semantic_complete` are defined before they are consumed. Existing `SynthesisResult` remains the compatibility projection.
- **Failure semantics:** provider failures preserve existing bundles; evidence insufficiency is readable but non-complete; legacy upgrades require passed Ollama semantic publication; rule fallback remains non-semantic.
- **Source safety:** no task downloads videos, reads browser credentials, mutates raw assets, or starts historical work automatically.
