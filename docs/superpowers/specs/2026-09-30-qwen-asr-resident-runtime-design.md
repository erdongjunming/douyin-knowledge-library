# Qwen3-ASR Resident Runtime Design

**Goal:** Keep one local Qwen3-ASR model process alive per ASR worker so multiple transcription jobs reuse the loaded model while preserving the existing queue, GPU lease, audit, cancellation, and raw transcript publication contracts.

## First-principles constraints

- The raw subtitle or ASR transcript remains the factual source and is published only after validation.
- The runtime owns model loading and inference only. It does not write the database, stable transcript files, or audit records.
- The existing `TranscriptionService` remains the publication and rollback boundary.
- The first implementation handles one request at a time. Batch inference, ForcedAligner, and vLLM are separate follow-up experiments.
- The resident process is loopback/local-process only and is supervised by the ASR worker. It is never exposed as a network service.

## Architecture

`QwenAsrProvider` keeps a single runner for the lifetime of `WorkerRuntime`. The resident runner converts source audio to the existing temporary 16 kHz mono WAV, starts a configured Qwen server process lazily, waits for a readiness record, sends one JSON request, and returns one JSON response. The server loads `Qwen3-ASR-0.6B` once and handles sequential requests through stdin/stdout JSON lines.

Cancellation terminates the resident process when the current request cannot be cancelled inside the model call. Since the production ASR pool remains single-concurrency, this cannot interrupt another request. The next request starts a fresh process and reloads the model. Lease ownership and final publication remain in `TranscriptionService`.

## Readiness and failure

The server emits `ready` only after imports, model loading, and a lightweight model initialization step succeed. A malformed response, process exit, timeout, or cancellation is a failed attempt; the stable raw transcript remains untouched. The runner can be closed synchronously during worker shutdown and the next run can restart it.

## Scope of this slice

This slice adds the resident runtime, a server entry point, provider lifetime management, diagnostics for the selected runtime, and unit/integration tests using a synthetic server. It does not run a production backlog, modify `D:\\DouyinKnowledgeLibrary`, change ASR model size, enable batch inference, or alter historical notes.
