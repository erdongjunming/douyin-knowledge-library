# Subtitle-First Transcription and Ollama Transcript Cleaning

**Status:** Approved design

**Date:** 2026-08-30

## Goal

Adopt the useful parts of MediaScribe's separation of transcription, provider
adapters, and video/text orchestration while preserving this project's durable
jobs, audit trail, GPU leases, cancellation, and atomic publication.

The new default for future work is:

```text
local subtitle candidate
  -> subtitle quality gate
  -> Qwen3-ASR fallback when needed
  -> immutable raw transcript
  -> Ollama text cleanup
  -> cleaned transcript variant
  -> evidence-preserving note generation
```

Qwen3-ASR remains the production audio-to-text engine. Ollama is a text
postprocessor and note generator; it is not treated as an ASR engine. The
current Ollama API/model inventory does not provide a stable audio-input
contract for the installed models, so production transcription must not depend
on Ollama receiving audio.

## Scope and non-goals

In scope:

- local-only subtitle discovery and normalization;
- deterministic subtitle-first routing with a content quality gate;
- an explicit ASR provider contract around the existing Qwen subprocess;
- an Ollama transcript-cleaner provider that preserves segment structure;
- durable cleaning jobs with retries, cancellation, lease fencing, and audit;
- cleaned/raw selection for note generation;
- a two-pass evidence-preserving note workflow;
- explicit, dry-run-first historical cleanup commands.

Out of scope:

- remote subtitle retrieval or new video downloads;
- saving complete videos;
- automatic historical reprocessing;
- replacing Qwen3-ASR with an experimental Ollama audio path;
- dynamic third-party plugin discovery in the first implementation.

## Architecture

### Provider contracts

The transcription package will expose small, testable protocols:

- `TranscriptSourceProvider`: discovers or produces a raw transcript draft and
  reports `source_kind` (`subtitle` or `asr`).
- `ASRProvider`: converts an existing local audio asset into text and timed
  segments. `QwenAsrProvider` adapts the current FFmpeg + Qwen subprocess and
  cancellation behavior.
- `TranscriptPostprocessor`: receives a raw transcript and returns a cleaned
  transcript with the same segment identity and timing.

Providers are selected through explicit factories/static mappings. The design
does not copy MediaScribe's import-time package walking because the current
application needs predictable startup, configuration, and audit behavior.

### Orchestration and workers

`TranscriptPipeline` owns source selection and raw publication:

1. Find local subtitle candidates.
2. Normalize and score the candidate.
3. Publish subtitle/raw when the gate passes; otherwise run Qwen3-ASR.
4. Publish the immutable raw transcript.
5. Enqueue `clean_transcript_work` when cleaning is enabled.

`clean_transcript_work` runs in the general worker, obtains the shared GPU
lease for Ollama, and publishes a cleaned variant only after validation.
`generate_note` selects cleaned when available and valid, otherwise raw with an
explicit fallback status. ASR and Ollama therefore do not hold the GPU lease
at the same time.

## Local subtitle strategy

No network call is made by the subtitle provider. It checks the configured
subtitle root (default `D:\DouyinKnowledgeLibrary\subtitle`) and any existing
subtitle asset paths associated with the work. Supported formats are `.srt`,
`.vtt`, `.ass`, `.ssa`, and plain `.txt`.

Normalization removes format headers, timecode lines, style markup, and
consecutive duplicate cues. The internal draft retains segment IDs and timing
when present. A candidate is usable only when it is readable, non-empty, has
at least one valid text segment, has monotonic timestamps when timestamps are
present, and is not obvious placeholder/noise content. Short videos use a
duration-aware threshold so a single legitimate sentence is not discarded.

When no candidate passes, the pipeline uses the existing local audio asset and
Qwen3-ASR. It never downloads a replacement video or overwrites source media.

## Persistence model

Raw and cleaned transcripts are separate variants of the same work:

```text
transcript/<aweme_id>/
  transcript.txt
  segments.json
  qwen3_asr.json
  transcript.cleaned.md
  transcript.cleaned.json
```

The database records `source_kind` (`subtitle`/`asr`), `variant`
(`raw`/`cleaned`), and `parent_transcript_id` for the cleaned record. Existing
rows are backfilled as `source_kind=asr, variant=raw`; their text and files are
not changed. Cleaned text and JSON are registered as distinct asset kinds.

`clean_transcript_work` is idempotent on raw transcript ID, raw content
fingerprint, cleaner provider, model, and prompt version. Every attempt writes
a processing audit containing input/output hashes, model, prompt version,
validation status, and error details.

All cleaned files are written to attempt-specific temporary paths and atomically
published. The publication path must use a per-work lock or an atomic
compare-and-publish protocol so a concurrent publisher cannot pass a guard and
then replace a newer semantic/fallback bundle (the known `publish_guard` TOCTOU
risk). Custom publishers that do not support the guard keyword must remain
compatible through an adapter or capability check.

## Ollama cleaner contract

The cleaner calls the local `/api/chat` endpoint with deterministic settings:
temperature zero, thinking disabled, configured timeout and keep-alive, and a
JSON response format. The prompt says to correct recognition errors and
terminology without summarizing or adding facts.

The response must preserve:

- segment count and order;
- segment IDs;
- exact `start` and `end` values;
- language and source references.

Only segment `text` may change. Long transcripts are processed in overlapping
segment batches and reassembled by ID. Hard validation rejects malformed JSON,
missing fields, timing changes, extra segments, empty output, or explanatory
Markdown around the JSON. Configurable drift heuristics reject extreme text
expansion/contraction or repetitive output. Timeout, unavailable model,
transport failure, cancellation, and lease loss leave raw available and mark
the cleaning attempt failed.

## Evidence-preserving note generation

Note generation receives the full cleaned transcript when available, not a
lossy preview. It becomes a two-pass workflow:

1. `evidence_extract` processes transcript chunks and persists
   `evidence_map.json` containing claims, arguments, examples, cases,
   analogies, counterexamples, limitations, applicability conditions, ordered
   steps, parameters, numbers, names, terminology, quotations, segment IDs,
   timestamps, and uncertainty markers.
2. `note_compose` creates the final note from the evidence map and transcript.

The default note must include the core conclusion, complete outline, concept
explanations, examples/cases, limitations and exceptions, ordered actionable
steps, timeline/evidence references, reusable quotations, takeaways, and
unconfirmed/possibly misrecognized content. Length is dynamic and evidence
complete: the system may produce a longer note rather than omit detail.

Coverage validation must flag missing steps, conditions, examples, numbers, or
named entities. Unsupported additions are labeled as inference and cannot be
presented as video facts. The previously observed overly concise output for the
"姜学长" video becomes a regression fixture, with explicit checks for its
examples, conditions, and steps.

## Configuration and operations

Proposed settings:

- `DKL_TRANSCRIPTION_SUBTITLE_ENABLED` (default `true`)
- `DKL_TRANSCRIPTION_SUBTITLE_ROOT`
- `DKL_TRANSCRIPTION_CLEANING_ENABLED` (default `true`)
- `DKL_TRANSCRIPTION_CLEANER_MODEL` (default `qwen3:4b`)
- `DKL_TRANSCRIPTION_CLEANER_TIMEOUT_SECONDS`
- `DKL_TRANSCRIPTION_CLEANER_BATCH_CHARS`
- `DKL_TRANSCRIPTION_CLEANER_MAX_DRIFT_RATIO`

The CLI will expose an explicit `clean-transcripts` command with
`--aweme-id`, `--limit`, and `--dry-run`. Historical transcripts are never
queued by startup or by the normal incremental path. Any backfill must proceed
through dry-run, a small authorized batch, audit inspection, and only then a
larger batch.

## Migration and rollout

1. Add provider contracts, local subtitle normalization, and quality tests while
   retaining the Qwen-only behavior as a safe feature flag path.
2. Add transcript variant persistence and `clean_transcript_work` with raw
   immutability and audit coverage.
3. Run one newly transcribed work through Ollama and verify validation,
   fallback, backup, and API/UI provenance display.
4. Enable automatic cleaning for new transcriptions.
5. Offer historical cleanup only through explicit dry-run and small-batch
   commands.

No step changes audio, cover, metadata, existing raw transcript files, or
pending historical queues.

## Verification

Unit tests cover subtitle parsing/scoring, provider selection, cleaner response
validation, batch reassembly, and note evidence extraction. Integration tests
cover subtitle-to-ASR fallback, job chaining, idempotency, cancellation,
heartbeat/stale recovery, GPU lease loss, raw immutability, atomic publication,
and note fallback. The existing backend/frontend suites, Ruff, compileall,
build, and `git diff --check` remain required gates.

