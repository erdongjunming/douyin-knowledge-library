# Douyin Asset and Local ASR Workflow Design

**Status:** Draft for user review
**Date:** 2026-09-05

## Context

The Douyin Knowledge Library already has the most important factual safeguards:
local subtitle-first selection, Qwen3-ASR, separate raw and cleaned transcript
variants, Ollama semantic extraction, coverage review, GPU leases, audit records,
and atomic note publication. The next problem is operational: historical works
have audio that can be reused, but queue state and stage dependencies make it
easy to repeat expensive work or report a no-op as success.

The current library snapshot previously contained about 2,510 works, 1,131 audio
assets (about 7.03 GB), 751 successful raw transcripts, 380 audio assets without a
successful raw transcript, and 727 learning notes. These numbers are planning
inputs, not a promise of current state or completion time. Earlier queue waits
were measured in hours or days, so throughput must be measured from a controlled
canary rather than inferred from model speed alone.

Public GitHub projects provide useful patterns but are not dependencies for this
change:

- `jiji262/douyin-downloader` (MIT) informs manifest, `aweme_id` deduplication,
  integrity checks, rate limits, retries, and incremental synchronization.
- `zwj-3193655211/video-transcriber` (MIT) informs local/cloud/auto ASR routing,
  cached audio and text, and timestamped segments.
- `HYPERVAPOR/dysub` informs audio chunking and timestamp-offset merging; its
  repository license is not verified and its cloud API default is not adopted.
- `like-attract/video-to-note` informs transcript-only reuse and the separation
  of transcription from note generation.
- `f2`, `MediaCrawler`, `TikTokDownloader`, `ReelLoom`, and
  `douyin-knowledge-rag` inform capability matrices, checkpoints, and failure
  handling only. Their licenses, platform assumptions, or cloud/web-AI behavior
  prevent direct code reuse in the local factual pipeline.

The design must preserve user-authorized access and platform boundaries. A
login challenge or CAPTCHA is a manual stop, never an automation target. Cookies,
tokens, signed media URLs, and external provider credentials are not copied into
jobs, notes, or logs.

## Goals

1. Make the processing stages independently reusable and observable.
2. Prioritize existing local subtitles and audio before any new acquisition.
3. Make historical raw-transcript backfill bounded, repeatable, and zero-write in
   dry-run mode.
4. Ensure a cleaner or note job cannot race ahead of its required transcript.
5. Make provider failures, preserved bundles, and no-op results visibly fail
   rather than becoming successful queue jobs.
6. Keep Qwen3-ASR as the only default ASR path and keep Ollama local for semantic
   note upgrades.
7. Preserve details that were lost in overly concise notes: examples, ordered
   steps, conditions, limitations, risks, context, numbers, named entities,
   timeline references, and uncertainty markers.
8. Measure queue wait, GPU wait, runtime, retries, and failure classes before
   expanding a batch.

## Non-goals

- No direct reuse or embedding of third-party crawler code or request-signing
  logic.
- No automatic CAPTCHA solving, login-state extraction, or credential export.
- No automatic full-library run, queue drain, or modification of the real
  `D:\DouyinKnowledgeLibrary` data during implementation or acceptance tests.
- No database migration for content-addressed asset versions in Phase A.
- No automatic cloud ASR, Doubao, visual OCR, or key-frame analysis in the
  default raw-transcript path.
- No replacement of a valid existing raw transcript merely to change provider,
  model, or prompt.
- No forced replacement of a stronger semantic note with rule output or an
  incomplete provider result.

## Target architecture

The long-term architecture has four explicit layers. Phase A uses the existing
tables and files to enforce the boundaries; Phase B can make the boundaries
content-addressed and persistent without changing their meaning.

### Work and source layer

`Work` identifies an `aweme_id` and stores descriptive metadata and source
relations. A synchronized work is not assumed to have media, a transcript, or a
note. Creator and favorites synchronization remains incremental and records
page/cursor summaries without storing credentials.

### Media and asset layer

`Asset` identifies retained audio, cover, metadata, subtitle, or temporary source
artifacts. A usable audio asset must be a readable non-empty file with a recorded
byte count, hash, probe result, and verification time. A missing, zero-byte, or
hash-mismatched asset is `blocked`, not pending. Downloaders may use temporary
paths, content-length checks, append-only manifests, and atomic promotion; note
or transcript generation never downloads or overwrites source media.

### Transcript layer

Transcript sources are selected in this order:

1. A local SRT/VTT/ASS/SSA/TXT candidate that passes the existing quality gate.
2. The retained audio sent to the local Qwen3-ASR provider in bounded chunks.
3. A separately configured cloud/OpenAI-compatible provider only when an
   operator explicitly opts in; this is not part of Phase A.

Every raw result records `source_kind` (`subtitle` or `asr`), provider/engine,
model, language, segment IDs, start/end times when available, input audio hash,
and a sanitized failure reason. A cleaned transcript is a derived child of one
raw fingerprint and may change only text, never segment identity or timing.
Existing successful raw files are immutable for Phase A: a retry skips them and
reports the existing fingerprint rather than rewriting the fixed transcript
path. True content-addressed versions and multiple provider variants are Phase B
work.

### Note layer

Notes consume the current valid transcript and its fingerprint. Semantic note
generation is a two-pass process: evidence extraction, then composition and
independent coverage/claim review. A note is publishable only when the semantic
quality gate passes. Rule output remains useful for normal missing-note fallback,
but is not an acceptable replacement for a legacy upgrade. The published note
must retain source provenance and distinguish video facts, model inference,
visual supplements, and unverified ASR terms.

## Phase A: safety gates and queue contracts

Phase A is deliberately narrow. It makes the existing pipeline safe to observe
and run in small batches before introducing a new schema.

### Historical inventory and dry-run

Add a read-only inventory operation that emits a deterministic JSON snapshot,
sorted by `aweme_id`. Each row reports:

- audio path, existence, non-zero status, bytes, SHA-256, probe/verification;
- raw transcript rows and files, source kind, variant, status, segment count,
  and DB/file fingerprint consistency;
- cleaned transcript validity and parent fingerprint match;
- current note classification (`semantic_complete`, `fallback`, `legacy`,
  `unknown`, or `missing`);
- active, stale, failed, and retry-scheduled jobs by stage; and
- a machine-readable `reason`/`blocked_reason` and the selected cohort.

The operation must not insert or update jobs, audits, database rows, files, or
provider calls. Repeating it against an unchanged library must produce the same
cohort, ordering, and counts. It is the authoritative input for deciding which
historical assets need work; `learning_note.md` existence alone is insufficient.

### Raw-ASR-only backfill

Provide an explicitly bounded entry point using `transcribe_work` semantics:

- `--dry-run` is the default review step;
- exact `--aweme-id` selection takes precedence when supplied;
- otherwise `--limit` selects a deterministic prefix;
- only verified, readable local audio is eligible;
- successful raw subtitle or ASR results are skipped and reused;
- malformed transcript JSON, missing audio, or failed verification is blocked;
- no cleaner, note, download, or external provider job is created; and
- each accepted job records `phase=raw_asr`, `provider_policy=local`, provider,
  model, input hash, and prompt/config version where applicable.

The first execution is a three-work canary containing a short item, a medium
item, and a longer instructional item. If it passes inspection, the next batch
is at most ten works. Expansion to 25 or more requires reviewing canary metrics
and output quality; the workflow never infers a full-library ETA from a single
run.

### Stage dependency

The runtime must not enqueue `generate_note` before an enabled cleaner has
completed. A successful raw transcription may enqueue a cleaner; a successful
cleaner may then enqueue a note. If cleaning is disabled, a raw transcript may
enqueue a note directly. If cleaning fails, raw remains available and the result
is explicitly `raw_fallback`/`clean_failed`; it must not be silently presented as
cleaned input. A note handler may consume raw only when the policy explicitly
allows that fallback.

This dependency is enforced by durable state or by enqueue-after-success, not
only by worker priority. Multiple general workers must not allow a note to race
ahead of a cleaner.

### Job success contract

The generic worker currently treats a non-throwing handler as success. Phase A
changes the semantic boundary: a handler must return or expose an outcome that
the runtime validates. For note upgrades, only `generated` with a published,
coverage-passed semantic bundle yields `succeeded`. `preserved_existing`,
`preserved_stronger_existing`, `skipped`, rule fallback, provider failure,
coverage failure, lease loss, ownership loss, and publication rejection yield a
failed or retryable job, preserve the previous bundle byte-for-byte, and record a
sanitized audit reason.

Upgrade policy remains local Ollama even for older payloads that say `auto`.
Upgrade candidates are `legacy`, `fallback`, and `unknown`; `semantic_complete`
and `missing` are excluded. The Web/API upgrade cap remains three, with an
omitted limit defaulting to three and a value above three rejected before queue
mutation. Existing normal missing-note controls are not silently changed.

### Recovery and observability

Inventory and job responses expose queue wait, GPU lease wait, execution time,
attempt number, retry schedule, and a bounded sanitized failure reason. Stale
recovery is visible and does not authorize an unbounded historical retry storm.
Old payloads remain readable; runtime normalization supplies safe local defaults
without rewriting existing rows.

ASR and Ollama inference continue to use the shared `gpu:0` lease. A lease wait
or loss fails closed and cannot publish an output. I/O, hashing, and inventory
may run concurrently, but GPU inference is serialized according to the lease.

## Phase B: durable asset versioning

After Phase A has stable metrics and contracts, introduce content-addressed
transcript assets and a capability registry without changing the evidence rules:

- immutable raw/cleaned paths keyed by input fingerprint and provider version;
- explicit provider capabilities (timestamps, language, chunk limits, GPU need,
  maximum input, and fallback policy);
- per-chunk checkpoints and offset-aware segment merge;
- DB records that bind note input to an exact asset path and hash; and
- a manifest that is the stable entry point for later note regeneration.

Schema changes must preserve all existing raw files and provide a read-only
backfill before changing canonical selection. Multiple raw providers must not be
silently collapsed into the current fixed `transcript.txt` path.

## Phase C: value-based throughput and visual supplements

Once raw coverage is reliable, rank work by learning value, duration, transcript
density, and existing note quality. Use cheap metadata and existing subtitles
for screening, then spend Qwen/Ollama time on high-value works. Visual review is
separate and opt-in for low transcript density, on-screen text, prices, or
operation details. Douyin built-in AI or Doubao can be recorded as supplemental
evidence with `weak`, `search-derived`, `visual-claimed`, or `blocked` status;
none can replace the raw transcript fact spine.

## Failure and preservation semantics

Failure is a first-class result. The following outcomes preserve current source
assets and existing notes:

- provider transport/model/runtime error;
- invalid or incomplete structured output;
- malformed segments or timestamp drift;
- cancellation, GPU lease loss, or stale ownership;
- incomplete claim/coverage review;
- rule fallback during a semantic upgrade; and
- atomic publication or compare-before-publish rejection.

Each attempt retains a compact audit descriptor with operation, provider/model,
input and output hashes, stage timings, retry count, and a sanitized error. Raw
subtitles and ASR files remain usable even when cleaning or note generation
fails. A future retry is allowed only after the reason is visible and the
candidate remains eligible.

## Verification strategy

All tests use temporary directories and databases, never the real library.

### Phase A tests

- Inventory dry-run writes zero rows/files/audits and makes zero provider calls;
  repeated snapshots are deterministic.
- Selection honors exact IDs and limits, skips valid raw transcripts, blocks bad
  audio/JSON, and excludes semantic-complete notes from upgrades.
- Valid local subtitles bypass Qwen; invalid subtitles invoke Qwen; successful
  ASR leaves source audio, metadata, and hashes unchanged.
- Repeated ASR selection reuses a successful raw result and creates no duplicate
  active or successful transcript.
- Cleaner success precedes note enqueue; cleaner failure leaves raw available and
  reports explicit raw fallback.
- Provider failure, no-op/preserved result, rule fallback, coverage failure, and
  publication rejection become failed jobs rather than successful no-ops.
- Upgrade requests enforce the three-item cap, local policy, candidate classes,
  and progress partition `completed + remaining = total`.
- Two workers cannot hold `gpu:0` concurrently; cancellation and stale recovery
  cannot publish or duplicate an asset.

### Canary acceptance

The operator reviews three representative works before any larger run. Each must
show complete segment disposition and preserve examples, steps, conditions,
numbers, named entities, timeline references, and uncertainty markers. The
operator records P50/P90 ASR runtime, queue wait, GPU wait, retry count, and
failure rate. These measurements determine the next batch size; they do not
constitute a guarantee for all historical works.

## Rollout and authorization

1. Implement and test Phase A against temporary fixtures.
2. Run inventory only against the real library, with zero-write verification.
3. Review the inventory and select three explicit canary IDs.
4. Run raw-ASR-only canary and inspect hashes, segments, audits, and metrics.
5. Obtain separate operator approval before enabling cleaner or note generation
   for historical works.
6. Increase batch size only after the preceding batch is reviewed.

The implementation must not start workers, call Ollama, or enqueue historical
jobs as part of code verification. Full historical processing remains an
explicit operational action after this design and its implementation have been
reviewed.

## Acceptance criteria

The design is implemented when:

1. Historical inventory is deterministic and zero-write in dry-run mode.
2. Raw-ASR-only backfill is bounded, local-only, idempotent, and reuses existing
   audio/subtitles without rewriting valid raw assets.
3. Cleaner and note stages obey durable dependencies and expose raw fallback.
4. Only a published, coverage-passed semantic result can make an upgrade job
   succeed.
5. Legacy, fallback, and unknown notes are upgradeable; semantic-complete and
   missing notes are protected.
6. Source hashes and old notes remain unchanged after every failed attempt.
7. Queue, GPU, retry, and failure metrics are available before batch expansion.
8. The detailed-note quality gate retains examples, steps, limitations,
   conditions, risks, context, timeline, and source traceability.
9. No CAPTCHA is bypassed, no credential is exported, and no unapproved external
   ASR/provider call occurs.
