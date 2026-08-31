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

The ASR worker launches FFmpeg and Qwen as children. A cancelled job is fenced
by its heartbeat and the child process tree is terminated. Local semantic note
generation uses the same `gpu:0` lease, so ASR and Ollama inference do not run
concurrently.

Start the complete local stack with `start-local.cmd`. For manual operation:

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn douyin_library.main:app --host 127.0.0.1 --port 8002
.\.venv\Scripts\python.exe -m douyin_library.worker_main --pool general --concurrency 2
.\.venv\Scripts\python.exe -m douyin_library.worker_main --pool asr --concurrency 1
```

## Provider policy

The default `auto` chain is:

```text
Ollama -> configured external provider -> deterministic rule fallback
```

Use `local` to prohibit external calls, `external` to explicitly opt into the
configured external endpoint, or `rule` for deterministic/offline operation.
Ollama is not installed or pulled by the application. Prepare it explicitly:

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

## Transcript source policy

When `DKL_TRANSCRIPTION_SUBTITLE_ENABLED=true`, the transcription worker evaluates the
retained local `subtitle/<aweme_id>/` files first. A quality-checked SRT, VTT, ASS, SSA, or
TXT draft becomes the raw transcript and skips Qwen ASR. An absent, placeholder-only, or
invalid subtitle invokes Qwen ASR against the retained audio. Set
`DKL_TRANSCRIPTION_CLEANING_ENABLED=true` to enqueue the optional `qwen3:4b` cleaner;
cleaned text is a derived child variant linked to the raw fingerprint. Cleaner timeout,
model failure, cancellation, lease loss, or schema/drift rejection records a reason and
keeps the raw subtitle/ASR files and database row unchanged. Note input uses cleaned text
only when that parent link and fingerprint validate; otherwise it explicitly reports a
raw-source fallback.

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

# Only legacy rule bundles (semantic bundles are not selected).
.\.venv\Scripts\python.exe -m douyin_library.cli generate-notes `
  --legacy-only --provider local --all

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

If Ollama is unavailable, do not repeatedly retry a long batch. Either start the
service and install the model explicitly, or use `--provider rule` while keeping
the fallback warning visible for later upgrade.
