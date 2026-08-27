# Complete Semantic Learning Notes Design

## Context

The current note pipeline uses `local-rule-v1`. It splits the ASR transcript into
sentences, selects at most six high-scoring sentences, keeps at most two sentences
from each of the first twelve ASR segments, and renders those selections into a
fixed Markdown template. The recorded system prompt is audit metadata; no language
model currently interprets it.

This produces readable briefs quickly and without API cost, but it cannot preserve
semantic structures such as ordered steps, an example's background and outcome,
contrasting cases, conditions, or an author's complete argument. Existing scores
also infer completeness from list sizes rather than comparing the note with its
source. A nine-minute Jiang Xuezhang sample received a completeness score of 90
while omitting multiple steps, examples, constraints, and final recommendations.

The learning note must become a complete review artifact: after reading it, the
user should normally not need to replay the video. The short `douyin_ai_brief.*`
files remain the quick-reading layer.

## Goals

- Generate evidence-linked Chinese learning notes with a local language model by
  default.
- Preserve every substantive claim, explanation, step, example, counterexample,
  condition, risk, number, proper noun, and promotional call to action found in
  the ASR evidence.
- Provide both a theme-organized review and a chronological segment timeline.
- Make omissions measurable through a machine-readable coverage report.
- Use an optional external API when configured and retain a deterministic rule
  fallback when semantic providers are unavailable.
- Never replace an existing note with a failed, incomplete, or lower-quality
  generation.
- Archive the current note bundle before a validated semantic note replaces it.
- Reuse the existing job queue, independent ASR worker, heartbeat, audit payload
  externalization, database backup, note viewer, and incremental execution model.
- Keep external API use opt-in and expose token and cost information accurately.

## Non-Goals

- Redownloading videos or retaining full source videos.
- Re-running ASR when usable transcript assets already exist.
- Treating ASR as evidence for text or actions that appear only on screen.
- Fact-checking claims against external sources in the first implementation.
- Silently downloading a model, installing Ollama, or enabling a paid API.
- Replacing `douyin_ai_brief.md` with the long-form note.

## Confirmed Product Decisions

- `learning_note.md` is a complete review note, not another summary.
- The note contains both a theme review and a full chronological timeline.
- Promotional, community, course, follow, and comment calls to action are retained
  faithfully in a separate section.
- The provider order is local model first, optional external API second, and rule
  fallback last.
- The default local model is the quantized Ollama `qwen3:8b` model, with one local
  note generation at a time on the available 8 GB GPU.
- Existing notes are archived, then replaced only after the new bundle passes its
  validation and coverage gates.
- Rollout begins with three to five representative Jiang Xuezhang videos before
  upgrading all legacy rule notes.

## Considered Approaches

### 1. Increase rule-selection limits

This is inexpensive and fast, but it still cannot recognize that several
sentences form one procedure or that a story is evidence for a conclusion. It
reduces visible compression without solving semantic omission.

### 2. Preserve every ASR segment chronologically

This reliably retains source text and is suitable for the final fallback, but it
mostly creates a compressed transcript. It does not produce the theme structure,
conceptual relationships, or reusable actions expected from a learning note.

### 3. Evidence atoms plus semantic synthesis (selected)

A language model first extracts typed, source-linked evidence atoms from every
segment. A second pass organizes all atoms into the complete note. A coverage
validator then proves that every segment and atom has an explicit disposition and
repairs missing material before publishing. This costs more local compute but
directly addresses the observed failure mode and remains auditable.

## Architecture

The note subsystem will be split into clear units:

- `NoteInputLoader` reads transcript, segments, metadata, work data, and asset
  paths without mutating source assets.
- `NoteProvider` is a provider-neutral async interface returning structured JSON,
  usage, model identity, raw output, and diagnostics.
- `OllamaNoteProvider` calls the local Ollama HTTP API with structured-output
  instructions and no secret material.
- `ExternalNoteProvider` calls an explicitly configured OpenAI-compatible HTTP
  endpoint. Its API key is read from settings or the credential mechanism and is
  never included in prompts or audits.
- `RuleNoteProvider` provides deterministic last-resort output. Its timeline
  includes every source segment even when theme-level semantic quality is weak.
- `EvidenceExtractor` converts source segments into typed evidence atoms.
- `NoteSynthesizer` renders theme content from the complete atom set while
  preserving atom identifiers and timestamps.
- `CoverageValidator` checks segment disposition, atom inclusion, ordered steps,
  examples, constraints, promotional content, schema validity, and traceability.
- `SourceCoverageReviewer` independently compares each raw source segment with the
  synthesized note. It can identify material the extractor failed to turn into an
  atom, so atom-set consistency is not mistaken for source completeness.
- `NoteBundlePublisher` stages output, archives a previous bundle, and publishes
  only a validated generation.
- `GpuLease` coordinates the independent ASR and general workers through a
  process-shared, heartbeating lease stored in SQLite.

Provider selection and publishing remain orchestration concerns in
`NoteGenerationService`; provider-specific transport and parsing do not leak into
the API routes or job handler.

## Evidence Atom Schema

Each atom represents one independently preservable fact from the source:

```json
{
  "id": "A-0012",
  "kind": "step",
  "topic": "semantic alignment",
  "statement": "Give the model positive instructions focused on the desired result.",
  "source_segment": 5,
  "start": 240.0,
  "end": 300.0,
  "source_excerpt": "专注于要什么，少说不要什么",
  "relations": ["A-0010"],
  "sequence": 3,
  "is_promotional": false,
  "asr_uncertainty": []
}
```

Supported `kind` values include `claim`, `reason`, `definition`, `step`,
`example`, `counterexample`, `condition`, `warning`, `number`, `quote`,
`recommendation`, and `promotion`. Atom IDs are generation-local and stable
through synthesis and validation.

The extractor must not invent missing context. Apparent ASR errors are retained
in `source_excerpt`; a normalized interpretation may be recorded separately with
an uncertainty marker.

## Generation Flow

1. Validate all required source paths and load the complete transcript and segment
   list.
2. Create a processing audit containing sanitized input metadata, prompt version,
   provider policy, and file manifests.
3. Acquire the shared GPU lease when using the local provider.
4. Extract evidence atoms from every segment. Short and medium transcripts may be
   sent in one request; long transcripts are processed in complete, ordered
   segment batches without truncation.
5. Validate the atom schema and ensure every segment has a disposition:
   `included`, `duplicate`, `transition`, `promotion`, or `unintelligible_asr`.
6. Synthesize the theme review from all atoms, retaining atom IDs and timestamps.
7. Build the chronological timeline directly from ordered atoms and dispositions.
8. Run deterministic atom validation, then perform an independent source-coverage
   review against every raw segment and the synthesized draft. The reviewer emits
   missing atom candidates, unsupported draft claims, and disposition corrections.
9. If material is missing, send one focused repair request containing the missing
   atom IDs or candidates and the relevant source evidence.
10. Render Markdown and JSON outputs, then run structural, atom, and raw-source
    coverage validation again.
11. Tell Ollama to unload the model (`keep_alive: 0`) and release the GPU lease
    only after the request and unload attempt complete.
12. Stage the complete output bundle. If validation passed, archive the old bundle
    and atomically publish the new files. Otherwise leave the current bundle
    untouched.
13. Finish the audit with raw response paths, structured atoms, coverage, usage,
    timing, and any repair or fallback decisions.

No source input is truncated silently. When a provider context limit requires
batching, `note_budget.json` records the batch count and input coverage.

## Learning Note Structure

`learning_note.md` will contain:

1. Title and evidence/model status.
2. Video basic information.
3. One-sentence overview.
4. Complete video structure.
5. Theme review.
6. Methods and ordered steps.
7. Examples and counterexamples, retaining background, action, result, and author
   conclusion when present.
8. Preconditions, boundaries, risks, and cautions.
9. Key concepts.
10. Reusable original expressions, marked as ASR-derived quotations.
11. Understanding and implications, clearly separated from the author's claims.
12. Executable actions derived from actual procedures rather than template text.
13. Full chronological timeline.
14. Closing promotion and calls to action.
15. Archive tags.
16. Evidence limitations and ASR terms requiring review.

Theme items contain timestamps and atom references. The timeline remains in source
order and includes every segment disposition so that content cannot disappear
between the transcript and the organized review.

`douyin_ai_brief.md` and `douyin_ai_brief.json` remain concise. They may use the
same atom set but are not used as the factual source for the complete note.

## Coverage Report And Quality Gate

Each successful or fallback bundle adds `coverage_report.json`:

```json
{
  "schema": "douyin-note-coverage-v1",
  "segment_total": 10,
  "segment_disposition_count": 10,
  "evidence_atom_total": 34,
  "evidence_atom_included": 34,
  "missing_atom_ids": [],
  "traceable_atom_count": 34,
  "source_segments_independently_reviewed": 10,
  "source_omission_candidates": [],
  "unsupported_draft_claims": [],
  "steps_verified": true,
  "examples_verified": true,
  "conditions_verified": true,
  "promotional_content_preserved": true,
  "semantic_generator": "ollama/qwen3:8b",
  "quality_gate": "passed"
}
```

Publishing a semantic note requires:

- valid schemas for extraction and synthesis;
- a disposition for every source segment;
- inclusion and source traceability for every extracted atom;
- an independent comparison of every raw segment with the synthesized draft;
- no unresolved source-omission candidate or unsupported draft claim;
- no missing step in any detected ordered sequence;
- complete example and condition relationships where present;
- preserved promotional atoms;
- all required Markdown sections and non-empty timeline entries; and
- no unresolved repair request.

`note_score.json` will calculate completeness from these measured properties, not
from note length or list counts. Clarity and usefulness remain separate dimensions
so a complete but poorly organized note cannot receive a misleading overall score.

## Adaptive Note Budget

The complete note has no fixed character ceiling. `note_budget.json` records:

- transcript characters and estimated input tokens;
- segment and evidence-atom counts;
- provider context limit, batch count, and whether any input was omitted;
- recommended detail based on evidence types, not a broad transcript-length band;
- provider/model, generation and repair token usage;
- local versus external execution; and
- API cost status.

Local generation reports `api_cost: 0`. External monetary cost is recorded only
when the provider returns it or the user configures explicit input/output token
rates. Otherwise cost is `null` with `cost_status: "unknown"`; the application
must not present a token estimate as a billed amount.

## Provider Configuration And Ollama Lifecycle

The application detects Ollama in this order:

1. explicitly configured executable;
2. system `PATH`;
3. the standard Windows user installation path
   `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`.

The current machine has Ollama at the standard user path but no running service or
downloaded model. Startup may start `ollama serve` as a hidden child process when
the executable exists. It must not install Ollama or pull a model automatically.
Diagnostics report service status, selected model, model availability, and the
action needed to become ready.

Default settings include the local base URL `http://127.0.0.1:11434`, model
`qwen3:8b`, request timeout, provider policy, and local concurrency of one. The
external base URL, model, and optional token rates are absent by default. The
external key is accepted only through `DKL_NOTE_EXTERNAL_API_KEY`; it is never
returned by settings APIs or copied into a job payload. Sensitive values are
sanitized through the existing audit layer.

## Shared GPU Lease

ASR and local note generation run in different worker processes, so an in-process
lock is insufficient. A SQLite-backed resource lease coordinates them.

The GPU lease records resource name, owning worker and job, acquisition time,
heartbeat, and expiry. Acquisition is atomic. The owner renews the lease on the
existing worker heartbeat cadence, and stale leases are recoverable after the
configured stale interval. Job cancellation and normal completion release the
lease in `finally` blocks.

Local note generation holds the lease for the full Ollama request and unload
attempt. ASR holds it for the ASR subprocess lifetime. External and rule note
providers do not acquire the GPU lease. The job remains in a visible waiting stage
while the GPU is busy rather than failing or competing for memory.

## Versioning And Publish Safety

Before replacing a valid current bundle, the publisher archives all generated note
artifacts under:

```text
dy-note/<aweme_id>/versions/<UTC timestamp>-<generator-version>/
```

The archive includes the prior learning note, briefs, structured summary, budget,
score, page metadata, coverage report when present, asset manifest, and generation
audit. Original audio, cover, metadata, and transcript assets are referenced, not
copied.

New files are written to a staging directory and individually replaced with the
existing atomic-write mechanism. A generation manifest is committed last. A
publish failure restores the previous files from the just-created version archive
and records the error. Version creation happens only when there is an existing
bundle and a new validated bundle is ready to publish.

Normal incremental generation still skips an existing current semantic note.
`force` always creates a version before replacement. A dedicated legacy-upgrade
mode selects notes whose recorded generator is `local-rule-v1` and skips already
upgraded semantic notes.

## Failure And Fallback Policy

1. Invalid or incomplete local output receives one focused semantic repair.
2. If local generation still fails and an external provider is configured, retry
   the complete semantic operation externally.
3. If no semantic provider succeeds, generate a rule fallback whose chronological
   timeline contains every source segment.
4. A fallback never replaces an existing semantic note or another current note
   that passed a stronger quality gate.
5. If no note exists, publish the fallback with `quality_gate: "fallback"` and a
   visible low-quality warning.
6. Cancellation leaves source assets and the current note untouched, terminates or
   cancels active provider work where supported, unloads the local model, and
   releases the GPU lease.

All errors retain provider, stage, sanitized message, start/end time, and audit
payload references. Network failures do not cause an automatic paid-provider call
unless the external provider was explicitly configured and enabled in the policy.

## Audit And Storage

The processing audit records:

- complete input text and file manifests;
- extraction, synthesis, repair, and fallback prompts;
- raw provider responses;
- parsed evidence atoms and segment dispositions;
- coverage reports and validation problems;
- provider/model identity, token usage, cost status, timing, and fallback reason;
- output file manifests; and
- cancellation or failure details.

Large request, response, stdout, stderr, and structured payloads use the existing
external audit payload store. SQLite retains sanitized summaries, hashes, and
paths. API keys, authorization headers, signed media URLs, and cookies are never
stored in note or audit payloads.

## API, CLI, And Web UI

Existing note-generation endpoints and jobs remain compatible. Add provider and
quality information without removing current response fields.

CLI/API controls support:

- normal incremental generation;
- a specific aweme ID;
- explicit `force` regeneration;
- legacy-only upgrade;
- provider policy `auto`, `local`, `external`, or `rule`;
- dry-run counts; and
- bounded limit/all selection.

The web interface shows:

- selected provider/model and Ollama readiness;
- complete semantic, fallback, pending upgrade, running, and failed states;
- segment and atom coverage;
- recent video ID and failure reason;
- actions for one complete note and batch legacy upgrade; and
- the existing in-page note viewer with theme-review and timeline navigation.

The UI never describes fallback output as a complete semantic note. Model download
requires a deliberate user action or documented setup command.

## Tests

Unit tests cover:

- provider selection and opt-in external fallback;
- strict parsing and schema validation;
- complete segment disposition and atom traceability;
- detection of material omitted during atom extraction by the independent raw
  source review;
- rejection of unsupported claims introduced during synthesis;
- detection of missing ordered steps, examples, conditions, and promotions;
- one-repair limit;
- rule fallback's all-segment timeline;
- adaptive budget and honest cost status;
- Ollama executable discovery and no silent model pull;
- GPU lease exclusivity, heartbeat, stale recovery, cancellation, and release;
- version archive creation, atomic publish, and rollback; and
- secret sanitization and external audit payload references.

Integration tests cover:

- successful local semantic generation through the job queue;
- independent ASR and general workers waiting on the same GPU lease;
- local failure to optional external provider to rule fallback transitions;
- preserving an existing note when all semantic providers fail;
- legacy-only batch selection and incremental skip behavior;
- API progress and coverage fields; and
- complete bundle output and viewer content.

Frontend tests cover provider readiness, progress/error/fallback states, batch
upgrade controls, coverage display, and theme/timeline navigation. Backend tests,
Ruff, frontend tests, and the production frontend build must pass.

## Runtime Acceptance

Acceptance begins with three to five representative videos by Qinghua Jiang
Xuezhang: a long tutorial, a list-style video, and a viewpoint video. Existing note
bundles are archived before replacement.

For video `7636753241384695225`, acceptance explicitly verifies that the note
contains and traces:

- the role shift from programmer to product manager/director;
- the three construction phases;
- all three parts of the initial prompt and the AI HR example;
- discussion-mode duration and its cost rationale;
- all four semantic-alignment techniques and their examples;
- atomic updates, the three-change rollback example, and the interruption warning;
- the structured error-report template;
- both learning-by-doing examples;
- all three final recommendations; and
- the community/course/comment call to action.

The sample passes only when every source segment has a disposition, every evidence
atom is included and traceable, the theme review and timeline are both readable,
the archived old note remains available, and API cost is reported as zero for the
local run. After user review of these samples, upgrade all remaining legacy notes
incrementally, skipping already upgraded results and preserving failures for retry.
