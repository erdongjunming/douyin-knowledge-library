# Semantic Notes Operations

This runbook describes the local-first note workflow for the Douyin Knowledge
Library. It assumes the source assets already exist under
`D:\DouyinKnowledgeLibrary` and does not download or rewrite them.

## Process layout

The API owns HTTP requests, schema migrations, queue reconciliation, and backup
maintenance. Two independent worker pools claim the durable SQLite queue:

```text
API (8002) -> SQLite queue
                 |-> general worker: sync, download, note jobs
                 `-> ASR worker: transcribe_work jobs
```

The ASR worker launches FFmpeg and supervises a resident local Qwen process. The
resident process loads `Qwen/Qwen3-ASR-0.6B` once and handles sequential JSON-line
requests; `DKL_TRANSCRIPTION_RUNTIME=subprocess` selects the compatibility path that
starts Qwen for each job. A cancelled job is fenced by its heartbeat and the active
resident process is terminated when the model call cannot be interrupted. Local semantic note
generation uses the same `gpu:0` lease, so ASR and Ollama inference do not run
concurrently.

The resident process is local to the ASR worker and is never exposed as an HTTP service.
It emits a readiness record only after imports and model loading succeed. The worker keeps
the existing temporary 16 kHz mono WAV conversion and `TranscriptionService` remains the
only component allowed to validate, audit, and publish raw transcript files. A runtime
failure therefore leaves the previous verified transcript untouched.

Start the complete local stack with `start-local.cmd`. For manual operation:

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn douyin_library.main:app --host 127.0.0.1 --port 8002
.\.venv\Scripts\python.exe -m douyin_library.worker_main --pool general --concurrency 2
.\.venv\Scripts\python.exe -m douyin_library.worker_main --pool asr --concurrency 1
```

## Pause and resume a worker pool

From `backend`, request that an ASR pool finish its current job and stop claiming
new work:

```powershell
.\.venv\Scripts\python.exe -m douyin_library.worker_main --pool asr --pause
.\.venv\Scripts\python.exe -m douyin_library.worker_main --pool asr --status
.\.venv\Scripts\python.exe -m douyin_library.worker_main --pool asr --resume
```

Use `--pool general` for the independent general pool. These commands print JSON
and do not start workers or initialize/change the job database. The control file
is `runtime/worker-pools/<pool>.json` under `DKL_STORAGE_ROOT` (normally
`D:\DouyinKnowledgeLibrary`). It is atomically replaced and survives worker
restarts; a missing file allows claims, while an unreadable or invalid file
blocks them until a successful pause/resume command replaces it.

Each polling worker reads the control immediately before claiming a job. Already
claimed work, including its heartbeat and GPU lease, completes normally. Paused
workers remain alive and resume at their next configured poll. `pause_requested`
describes the saved request, not confirmation that running work has drained;
check current job/lease status separately before restarting processes. Explicit
one-shot commands are not polling pools and are unaffected.

Workers started before this control was installed require a controlled restart
to load it. Writing the pause file does not retrofit an older running process.
Do not use launcher shutdown to drain active work: it can terminate process trees.

## Provider policy

The default `auto` chain is:

```text
Configured external provider -> Ollama backup -> deterministic rule fallback
```

Use `local` to prohibit external calls, `external` to explicitly opt into the
configured external endpoint, or `rule` for deterministic/offline operation.
For historical upgrades, the rule step is recorded as skipped: it cannot replace
an old note or count as a successful semantic upgrade. Ollama is not installed
or pulled by the application. Prepare this optional backup explicitly:

```powershell
ollama serve
ollama pull qwen3:4b
```

Local requests have no external API charge. They still consume local GPU/CPU,
memory, time, and electricity. An external endpoint is used only when its URL,
model, and key are all present and the selected policy allows it. The key is
read only to build the request header; it is excluded from prompts, jobs,
notes, and audit records. Monetary cost is `unknown` unless the provider
reports it or both configured per-million-token rates are available.

Check readiness without starting a generation:

```powershell
Invoke-RestMethod http://127.0.0.1:8002/api/notes/provider-status
```

The status response follows the selected policy. `available=true` means a
semantic attempt can be made. External configuration is not a connectivity or
model capability check: `external.ready=null` and `status=not_probed` reflect
that no inference request was made. When external is the selected provider,
top-level `ready` is also null. A ready Ollama backup may still be available.
Existing `local` jobs remain local; new web upgrades submit `auto`.

## Workflow references

The selected references are implementation guidance, not installed services or
new dependencies. Both still require an inference provider; installing a GitHub
project does not supply a usable model endpoint.

- [Open Notebook transformation source](https://github.com/lfnovo/open-notebook/blob/2d2df8a3cbb098776e56ca5ee77b9832f848228e/open_notebook/graphs/transformation.py)
  uses retained source text, a configurable transformation prompt/model, a
  separately saved insight, and classified errors. Its transformation graph is
  a single model node; it does not itself provide this project's completeness
  and guarded-publication gates.
- [LangExtract extraction entry point](https://github.com/google/langextract/blob/bd5d1ebb51db51e9f6d054c026489d1d8127b005/langextract/extraction.py)
  supports a supplied model, chunk batches and multiple extraction passes.
  Its [extraction data contract](https://github.com/google/langextract/blob/bd5d1ebb51db51e9f6d054c026489d1d8127b005/langextract/core/data.py)
  records source character intervals and alignment status. It is an extraction
  library, not a complete legacy-note writer.

The current adaptation reuses the existing structured pipeline and segment
evidence: raw input -> evidence atoms -> outline -> teaching sections -> source
review -> depth review/repair -> publication guard. Both the external adapter
and Ollama run this same pipeline. No LangExtract character-alignment module or
Open Notebook database/server is embedded. Long inputs retain full evidence;
an irreducible context request fails explicitly instead of truncating source
text. No provider settings or production library data are changed by the
implementation tests.

## Transcript source policy

When `DKL_TRANSCRIPTION_SUBTITLE_ENABLED=true`, the transcription worker evaluates the
retained local `subtitle/<aweme_id>/` files first. A quality-checked SRT, VTT, ASS, SSA, or
TXT draft becomes the raw transcript and skips Qwen ASR. An absent, placeholder-only, or
invalid subtitle invokes Qwen ASR against the retained audio. Set
`DKL_TRANSCRIPTION_CLEANING_ENABLED=true` to enqueue the optional `qwen3:4b` cleaner;
cleaned text is a derived child variant linked to the raw fingerprint. Cleaner timeout,
model failure, cancellation, lease loss, or schema/drift rejection records a reason and
keeps the raw subtitle/ASR files and database row unchanged. Note evidence always
uses raw transcript/segments. A cleaned child is loaded only as optional reading
assistance when its parent link and fingerprint validate.

Normal startup does not enqueue historical `clean_transcript_work` jobs. To clean legacy
raw transcripts, inspect the candidates first, then explicitly queue a bounded run:

```powershell
cd backend
.\.venv\Scripts\python.exe -m douyin_library.cli clean-transcripts `
  --dry-run --limit 3 --provider local --model qwen3:4b
.\.venv\Scripts\python.exe -m douyin_library.cli clean-transcripts `
  --aweme-id 123 --provider local --model qwen3:4b
```

The cleaner dry run is read-only and creates no jobs, provider calls, files, or source
changes. Use repeated `--aweme-id` options instead of `--limit` for an exact set.

## Generate incrementally

Always run a dry run first. It only scans `transcript.txt` and note state:

```powershell
cd backend
.\.venv\Scripts\python.exe -m douyin_library.cli generate-notes --dry-run --all
```

The dry run is read-only: it creates no queue jobs, provider calls, note files, or source
asset changes. Generation reuses retained metadata/transcript artifacts and never
downloads videos or overwrites source audio, covers, metadata, subtitles, or raw
transcripts.

Then use a small, explicit sample:

```powershell
.\.venv\Scripts\python.exe -m douyin_library.cli generate-notes `
  --limit 3 --provider local
```

Other controls:

```powershell
# One or more exact works.
.\.venv\Scripts\python.exe -m douyin_library.cli generate-notes `
  --aweme-id 123 --aweme-id 456

# Only eligible legacy, fallback or unknown bundles, with the primary provider first.
.\.venv\Scripts\python.exe -m douyin_library.cli generate-notes `
  --legacy-only --provider auto --limit 3 --dry-run

# Intentional rebuild. Never combine with --legacy-only.
.\.venv\Scripts\python.exe -m douyin_library.cli generate-notes `
  --force --limit 3 --provider local
```

The web endpoint `POST /api/notes/generate` exposes the same controls and has a
dry-run mode that creates no jobs. Existing notes are skipped by default. A
semantic candidate is published only after extraction, synthesis, independent
source review, an optional single repair pass, and deterministic coverage
validation. This evidence-complete gate requires every source segment to remain
traceable in the review and timeline. A failed semantic candidate is audited and the current bundle is
preserved. The rule fallback includes every source segment and is explicitly
marked `fallback`.

## Teaching-note quality and historical upgrades

Every semantic note uses the fixed teaching chapters `主题观点`, `核心观点`,
`方法与步骤`, `案例与结果`, `前提、边界与风险`, `行动建议`, `逐段证据`, and
`证据与局限`. The reader also keeps a separate `主题复盘`, `方法与步骤`,
`案例与反例`, `前提、边界与风险`, and `完整逐段时间线` view. A claim in any
chapter must point to one or more evidence atoms, each with a source segment and
timestamp. Missing categories are rendered explicitly as `视频未提供这一类信息。`;
they must not be silently filled with plausible prose.

`quality_gate=passed` means source coverage, ordered steps, examples/results,
conditions, numbers, named entities, uncertainty markers, promotions, and the
independent review all pass. Teaching-capable providers additionally require
`depth_passed=true`, `input_sufficient=true`, and `semantic_complete=true`.
`semantic_complete` is the upgrade gate, not merely a display label. If the
transcript has synthetic-segment warnings, is sparse, or is otherwise insufficient,
the system may publish a readable *new* candidate with `quality_gate=failed` so
the user can inspect the source timeline, but it cannot replace an existing
note during a historical upgrade. Text and visual coverage are separate:
unreviewed visuals are reported as a limitation and do not by themselves fail
the text gate.

Historical upgrades are intentionally conservative:

1. `--legacy-only` selects only bundles classified as `legacy`, `fallback`, or
   `unknown`. The selected policy is preserved through API, worker and service.
   Under `auto`, a completely configured external endpoint is tried first, then
   Ollama on configuration, transport, output or quality failure. `local`
   prohibits external calls; `external` tries only that semantic provider.
   Cancellation stops the chain. The rule step is never executed for upgrades.
2. The API and CLI cap one upgrade request at three works. Use repeated,
   explicit `--aweme-id` values for a reviewed batch.
3. A dry run is zero-write and reports eligible, already queued, depth-failed,
   and input-insufficient counts. A failed or cancelled attempt is recorded under
   `attempts/`; the previous `learning_note.md`, generated bundle, and source
   asset hashes remain unchanged. Retry the same IDs after fixing the provider or
   input; do not use `--force` with `--legacy-only`.
4. Failed semantic attempts include a reason in the audit. A text gate pass
   with `semantic_complete=false` is still a failed upgrade attempt. If no
   semantic provider succeeds, report
   `legacy_upgrade_no_semantic_provider_succeeded` and preserve the old bundle;
   batch and worker status must show failure, not success.

The note generator reuses retained `transcript.txt`, `segments.json`, metadata,
and may load a validated `transcript.cleaned.*` auxiliary child when its raw
fingerprint and segment identity match. Cleaning never replaces raw evidence.
A missing, stale, malformed, or failed cleaned variant records the reason. Note
generation never downloads a video, rewrites raw audio/subtitles/transcripts, or
copies original media into the note bundle.

Do not bypass Douyin/Chrome human verification, automate CAPTCHA clicks, export
cookies or tokens, or place signed media URLs in logs. If a visual supplement is
needed, use the already authorized browser workflow and label it as supplemental
evidence; subtitle/ASR remains the factual spine.

## Output and rollback

For each work, the current bundle is under:

```text
D:\DouyinKnowledgeLibrary\dy-note\<aweme_id>\
```

The published root includes the learning note, short Markdown/JSON brief,
budget, score, page metadata, coverage report, evidence map, generation audit, generation
manifest, and `assets\asset_manifest.json`. Failed attempts are compactly
recorded under `attempts\<UTC timestamp>\`; a forced replacement archives the
previous generated bundle under `versions\` before committing the new manifest.

The asset manifest points back to the original audio, cover, metadata, and ASR
files. Those source files are never replaced by note generation.

## Audit and database recovery

Every generation records the complete sanitized input, versioned prompts, raw
provider responses, parsed structures, provider decisions, errors, and start/
finish timestamps. Large request/response/stdout/stderr fields are written to:

```text
D:\DouyinKnowledgeLibrary\audit\processing\<year>\<month>\
```

SQLite stores a summary envelope with byte count, SHA-256, and path. The loader
rejects paths outside the audit root and detects tampering. To migrate older
inline rows, stop workers first and run:

```powershell
cd backend
.\.venv\Scripts\python.exe -m douyin_library.cli compact-audits --vacuum
```

Backups are verified SQLite snapshots under
`D:\DouyinKnowledgeLibrary\data\backups`. They are taken before a required
Alembic migration and at the configured interval (24 hours by default); the
latest seven are retained. Keep a backup before manually repairing a database.

## Verification checklist

### Resumable local historical upgrade

The repository runner pins historical upgrades to loopback Ollama and the
installed `qwen3:4b` model. It reuses the frozen inventory under `audit/` and
processes at most three existing non-semantic notes per batch:

```powershell
.\backend\.venv\Scripts\python.exe -u scripts/upgrade_legacy_notes.py --dry-run
.\backend\.venv\Scripts\python.exe -u scripts/upgrade_legacy_notes.py --only-batch 54
.\backend\.venv\Scripts\python.exe -u scripts/upgrade_legacy_notes.py
```

Batch indexes are zero-based. Only successful records with matching work IDs,
unchanged sources, and currently readable semantic bundles are skipped on resume.
Failed batches remain retryable. The storage-level `legacy-upgrade.lock` prevents
two runners from operating concurrently. The lock covers inventory updates too.

Resume rechecks the first checkpoint's source hashes and old-note hashes, including
the archived originals. A damaged completed bundle or missing original stops the
run instead of silently skipping the batch. Mixed low-information batches require
each generated note to remain complete and each refused note to remain unchanged.
Provider readiness is checked before every new batch. If local Ollama is stopped
or `qwen3:4b` is missing, the runner exits without creating failures for the rest
of the inventory; restore the existing local service and resume the same command.

`legacy-upgrade-batches.checkpoints.jsonl` retains source and old-note hashes
before each batch. `legacy-upgrade-batches.jsonl` records outcomes, source hashes,
current quality, readability, and old-note preservation or matching version
archive. An integrity mismatch stops the run immediately. A zero failure count
alone cannot mark a batch complete. No source download or ASR step is invoked.

Every 30 seconds stdout reports the active batch, process ID, and GPU phase.
Standalone generation renews its own GPU lease through unload and publication;
renewal failure cancels the request and preserves the current note. Existing
worker jobs retain their worker-managed heartbeats.

The local launcher starts Ollama before the API and worker pools, then waits for the exact
configured model. The historical runner has the same readiness gate and can start a
stopped loopback Ollama service without downloading a model. This keeps model startup from
being mistaken for a provider failure and prevents a worker from consuming the GPU before
the upgrade runner is ready.

The progress API includes IDs from `audit/legacy-upgrade-inventory*.json` and
batch checkpoints, so successful standalone upgrades remain in the completion
denominator. Completion still requires the published bundle's semantic gates.
Standalone running status requires a matching current batch audit and an
unexpired runner lease; stale audits do not count as running. Queue retries and
standalone attempts are resolved by their start times. Keep these inventory and
checkpoint files when retaining upgrade history.

Creating `audit/legacy-upgrade.stop` requests a stop at the next batch boundary.
Remove that operator-created marker before resuming. Exit codes are `0` for all
selected batches passing, `1` for generation failures, `2` for a requested stop,
`3` for an integrity mismatch, and `4` for an unavailable local model. Keep
stdout/stderr in timestamped files when
launching a detached runner, and wait for an audit result before claiming success.

An incomplete draft-claim review retries only the missing claims against their
source evidence. Only explicit reviewed verdicts are merged from the first call;
contradictions remain failures, and any claims still unreviewed remain unsupported.
This avoids repeatedly sending a long list that the model has already omitted from.

### Acceptance checks

1. Confirm `/api/health` is `ok` or `degraded` for a deliberate storage error.
2. Confirm `/api/notes/provider-status` reports the expected provider and model.
3. Run a dry run and record `total`, `completed`, and `selected`.
4. Generate one to three explicit IDs and inspect every required output file.
5. Check `generation_manifest.json`, `coverage_report.json`, and the audit
   descriptor before starting a larger run.
6. Re-run the dry run. A successful incremental run should report those IDs as
   completed and select no longer-needed work.
7. Compare hashes of the source audio, cover, metadata, transcript, segments,
   and Qwen JSON before and after generation.

If neither the configured primary nor Ollama is usable, an upgrade preserves
the existing bundle. `--provider rule` is available for new offline notes, but
cannot complete a historical semantic upgrade. A real endpoint still needs
separate authorization and validation with the intended data; synthetic test
success does not establish live provider readiness.
