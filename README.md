# Douyin Knowledge Library

Local-first workspace for collecting authorized Douyin content, retaining audio and
metadata, transcribing speech, and turning the transcript into evidence-linked learning
notes. Runtime data defaults to `D:\DouyinKnowledgeLibrary`; the application does not
silently move the library to the system drive.

## Prerequisites

### Backend (Python >= 3.12)

```powershell
cd backend
uv venv          # or: python -m venv .venv
uv pip install -e .   # pulls fastapi, sqlalchemy, alembic, keyring, ... plus the douyin-downloader submodule
```

`douyin-downloader` is vendored as the git submodule `upstream/jiji` (editable
path dependency). After a fresh clone run `git submodule update --init`, then
`uv sync` (or `uv pip install -e .`) from `backend`. Dev tooling:
`uv sync --group dev` (pytest, ruff).

**ffmpeg / ffprobe** are required for audio handling. Resolution order:
`DKL_FFMPEG_PATH` / `DKL_FFPROBE_PATH`, then system `PATH`, then
`D:\DouyinKnowledgeLibrary\tools\ffmpeg\*\bin\`.

### Frontend (Node.js, Vite 7 + React 19)

```powershell
cd frontend
npm install
npm run dev     # vite on 127.0.0.1:5175
```

### Local models

- **Ollama** with `qwen3:4b` installed (`ollama pull qwen3:4b`). Never
  auto-installed or auto-pulled; the launcher only waits for it.
- **Qwen3-ASR-0.6B** for transcription. The ASR worker spawns
  `scripts/qwen_asr_server.py` with a dedicated Python environment pointed at
  by `DKL_QWEN_PYTHON_PATH`. That environment needs the `qwen-asr` package and
  the model weights cached locally — see Known issues below for the
  `HF_ENDPOINT` requirement.

### Optional

- **Douyin cookie** (only for creator-scoped collection): stored in the
  Windows keyring via the UI, Settings → 账号凭据. Validated against Douyin
  before saving. Use a throwaway account, not your business one.
- **External note provider**: set `DKL_NOTE_PROVIDER_POLICY=external` plus
  base URL / model / API key (see Configuration).

## Configuration

All settings are environment variables with the `DKL_` prefix (case-insensitive
snake_case → UPPER_CASE). The commonly used ones:

| Variable | Default | Purpose |
|---|---|---|
| `DKL_STORAGE_ROOT` | `D:\DouyinKnowledgeLibrary` | Runtime data root (audio, transcripts, notes, DB) |
| `DKL_NOTE_OLLAMA_MODEL` | `qwen3:4b` | Local note model; overriding does not trigger a download |
| `DKL_NOTE_PROVIDER_POLICY` | `auto` | `auto` / `local` / `external` / `rule` |
| `DKL_NOTE_EXTERNAL_BASE_URL` / `_MODEL` / `_API_KEY` | unset | External provider; keys are never written to logs or notes |
| `DKL_TRANSCRIPTION_RUNTIME` | `resident` | `resident` (shared Qwen server) or `subprocess` (one process per job) |
| `DKL_QWEN_PYTHON_PATH` | auto | Python env used to run the ASR server |
| `DKL_QWEN_SERVER_SCRIPT_PATH` | `scripts\qwen_asr_server.py` | Reviewed copy of the ASR server script |
| `DKL_QWEN_OFFLINE` | `true` | Offline model loading; jobs never download weights |
| `DKL_TRANSCRIPTION_SUBTITLE_ENABLED` | `true` | Prefer retained subtitles over ASR |
| `DKL_TRANSCRIPTION_CLEANING_ENABLED` | `false` | Optional qwen3:4b transcript cleaning pass |
| `DKL_FFMPEG_PATH` / `DKL_FFPROBE_PATH` | unset | Explicit tool paths |
| `DKL_EMBEDDED_WORKERS` | `false` | Run workers in-process (tests, small setups) |
| `DKL_STOP_BACKEND_ON_EXIT` | `1` | `0` leaves the backend running when the launcher exits |

The launcher (`scripts\dev.ps1`) imports provider configuration from Windows
user environment variables via `scripts\local-runtime.ps1` without printing
the API key.

## Start locally

Double-click `start-local.cmd`, or run:

```powershell
.\start-local.cmd
```

The launcher first starts (or reuses) local Ollama and waits for the configured
`qwen3:4b` model, then starts four independently managed processes: FastAPI on `8002`, a
general job worker, an ASR worker, and Vite on `5175`. It keeps child logs under `.runtime\logs`
and terminates the complete process trees when the launcher exits. Open
`http://127.0.0.1:5175/` after the readiness message. To create or repair the desktop
shortcut, run `install-desktop-shortcut.cmd`.

For manual process control, start the pools from `backend`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn douyin_library.main:app --host 127.0.0.1 --port 8002
.\.venv\Scripts\python.exe -m douyin_library.worker_main --pool general --concurrency 2
.\.venv\Scripts\python.exe -m douyin_library.worker_main --pool asr --concurrency 1
```

The API defaults to `DKL_EMBEDDED_WORKERS=false`; tests and small single-process setups
can explicitly set it to `true`. General and ASR workers claim the same durable SQLite
queue, so they can be restarted independently.

## Local transcription

Chinese audio uses the shared `Qwen/Qwen3-ASR-0.6B` environment. The ASR worker converts
retained M4A audio to a temporary 16 kHz mono WAV, sends it to the default resident Qwen
runtime, and removes the temporary file. The resident runtime loads the model once per
ASR worker and accepts sequential local JSON-line requests; it is supervised by the worker
and is never exposed as a network service. Retained transcripts stay under
`D:\DouyinKnowledgeLibrary\transcript\<aweme_id>\`.

The runtime can be switched back to the one-process-per-job compatibility path with:

```powershell
$env:DKL_TRANSCRIPTION_RUNTIME = "subprocess"
```

Resident startup requires the configured Qwen Python environment and
`scripts\qwen_asr_server.py`. Use `DKL_QWEN_SERVER_SCRIPT_PATH` to point to a reviewed
copy, `DKL_QWEN_RUNTIME_START_TIMEOUT_SECONDS` to change the model-load deadline, and
`DKL_QWEN_OFFLINE=false` only when an explicitly network-enabled model preparation step
is intended. Runtime inference defaults to offline model loading and never downloads
weights as part of a transcription job.
The runtime reports readiness only after the model process loads successfully. A timeout,
process exit, malformed response, or cancellation keeps the stable raw transcript intact;
the next job can start a fresh resident process.

Transcription is subtitle-first when `DKL_TRANSCRIPTION_SUBTITLE_ENABLED=true`: a usable
local SRT/VTT/ASS/SSA/TXT subtitle is retained as the raw source and Qwen ASR is not
invoked. Missing, empty, or low-quality subtitles fall back to Qwen ASR. Optional
`DKL_TRANSCRIPTION_CLEANING_ENABLED=true` creates a derived cleaned variant with
`qwen3:4b`; a disabled, unavailable, cancelled, or invalid cleaner leaves the raw source
untouched and note generation records the raw fallback.

Normal startup does not enqueue historical `clean_transcript_work` jobs. Review legacy
cleaning candidates with a zero-write dry run, then explicitly select a bounded set:

```powershell
cd backend
.\.venv\Scripts\python.exe -m douyin_library.cli clean-transcripts `
  --dry-run --limit 3 --provider local --model qwen3:4b
.\.venv\Scripts\python.exe -m douyin_library.cli clean-transcripts `
  --dry-run --aweme-id <aweme_id> --provider local --model qwen3:4b
```

The dry run creates no jobs, provider calls, files, or source changes. Run without
`--dry-run` only after reviewing the result; repeat `--aweme-id` for an exact set.

Run one existing audio asset through the database, FTS, file, and audit pipeline:

```powershell
cd backend
.\.venv\Scripts\python.exe -m douyin_library.cli transcribe-one --aweme-id <aweme_id>
```

Cancellation is cooperative at the job layer and forceful at the process layer:
heartbeats notice remote cancellation, then terminate FFmpeg/Qwen and clean the temporary
WAV. A GPU lease (`gpu:0`) prevents ASR and local note inference from running together;
the lease expires or renews across worker processes.

## Note providers and cost

For new notes, `auto` tries an explicitly configured external provider, then Ollama,
then the deterministic rule fallback. `local` omits the external
provider, `external` opts into it, and `rule` is deterministic-only. Ollama is never
auto-installed or auto-pulled. Start and prepare it explicitly:

```powershell
ollama serve
ollama pull qwen3:4b
```

Ollama requests have zero external API charge, but use local GPU/CPU, memory, time, and
electricity. An external provider is contacted only when its base URL, model, and API key
are all configured (and the selected policy permits it). External cost is reported as
unknown unless the provider reports a cost or explicit per-million-token rates are set.
Keys are never written to prompts, jobs, notes, or audit payloads.

The full historical upgrade is a separate resumable operation so ordinary application
startup cannot compete with it silently. Start it after the local model is ready with:

```powershell
.\scripts\resume_legacy_upgrade.ps1
```

The runner starts Ollama itself when it is stopped, waits for `qwen3:4b`, and exits with a
provider-unavailable record if the model is not installed. It reuses the fixed inventory,
rechecks source and old-note hashes before skipping a batch, and keeps the old note when a
semantic or provider attempt fails.

## Incremental learning notes

The note pipeline reuses only retained `metadata`, `transcript.txt`, `segments.json`, and
`qwen3_asr.json`, plus `transcript.cleaned.md`/`transcript.cleaned.json` when the cleaned
variant validates its parent raw fingerprint and segment identities. Otherwise it uses
the raw source and records that fallback; it never downloads media or reruns ASR. It extracts evidence atoms,
synthesizes both a topic review and a complete chronological timeline, independently
checks source coverage, performs at most one repair pass, and publishes only a passed
evidence-complete semantic bundle. If semantic providers are unavailable, the rule fallback preserves every
source segment and is marked as `fallback`; it cannot overwrite a stronger existing note.
The dry run only reports eligible IDs and creates no jobs or files. Note generation never
downloads a video or overwrites source audio, cover, metadata, subtitle, or raw transcript
files; failed attempts are isolated under the note `attempts` directory.

Teaching notes always expose the fixed chapters `主题观点`, `核心观点`, `方法与步骤`,
`案例与结果`, `前提、边界与风险`, `行动建议`, `逐段证据`, and `证据与局限`, plus
the reader views `主题复盘` and `完整逐段时间线`. Every substantive statement is
linked to an evidence atom and timestamp. The renderer preserves concrete details
such as ordered steps, examples and results, numbers, names, conditions, ASR
uncertainty, and promotional content; when the source does not contain a category,
it says so explicitly instead of inventing one.

For an existing legacy/fallback note, `semantic_complete=true` is required before
replacement. Historical upgrades are Ollama-only and limited to three queued works
per request. A failed, cancelled, depth-incomplete, or input-insufficient upgrade
keeps the old note and all raw asset hashes intact. Use a zero-write dry run first,
then retry selected IDs after correcting the provider or transcript:

```powershell
cd backend
.\.venv\Scripts\python.exe -m douyin_library.cli generate-notes `
  --legacy-only --provider local --dry-run --limit 3
.\.venv\Scripts\python.exe -m douyin_library.cli generate-notes `
  --legacy-only --provider local --aweme-id <aweme_id>
```

The API reports pending/active/failed upgrade counts, including depth and input
shortfalls. `--force` cannot be combined with `--legacy-only`. Do not bypass human
verification, click CAPTCHA automatically, export cookies/tokens, or put signed
video URLs in logs. Browser AI or Doubao may only supplement visual context; the
subtitle/ASR transcript and evidence map remain the factual source.

Always inspect the candidate count first:

```powershell
cd backend
.\.venv\Scripts\python.exe -m douyin_library.cli generate-notes --dry-run --all
```

Useful controls:

```powershell
# Generate at most 3 missing notes using the local-first policy.
.\.venv\Scripts\python.exe -m douyin_library.cli generate-notes --limit 3
# Generate selected IDs (repeat the option for more than one).
.\.venv\Scripts\python.exe -m douyin_library.cli generate-notes --aweme-id <aweme_id>
# Upgrade only explicitly identified legacy rule bundles.
.\.venv\Scripts\python.exe -m douyin_library.cli generate-notes --legacy-only --provider local
# Rebuild a bundle intentionally; --force and --legacy-only cannot be combined.
.\.venv\Scripts\python.exe -m douyin_library.cli generate-notes --force --limit 3
# Rebuild an explicitly selected fallback through the configured cloud provider.
# Load Windows user provider settings when the current shell predates them.
. ..\scripts\local-runtime.ps1
Import-ProviderUserEnvironment
.\.venv\Scripts\python.exe -m douyin_library.cli generate-notes `
  --force --provider external --aweme-id <aweme_id> --dry-run
# Remove --dry-run to execute; repeat --aweme-id for each verified target.
```

Incremental runs skip a video when `dy-note\<aweme_id>\learning_note.md` exists.
Each published directory contains `learning_note.md`, `douyin_ai_brief.md`,
`douyin_ai_brief.json`, `note_budget.json`, `note_score.json`, `page_metadata.json`,
`coverage_report.json`, `evidence_map.json`, `generation_audit.json`, `generation_manifest.json`, and
`assets\asset_manifest.json`. Failed or cancelled attempts are retained under
`dy-note\<aweme_id>\attempts\`; previous bundles are archived under `versions\` before a
forced replacement.

## Audit and backup

Every synchronization, ASR, and note generation records sanitized input, prompts, raw
responses, parsed results, errors, and start/end timestamps in `processing_audit`. Large
fields are externalized under `D:\DouyinKnowledgeLibrary\audit\processing`; SQLite keeps
only an integrity-checked summary, byte count, SHA-256, and path. Compact legacy rows with:

```powershell
.\.venv\Scripts\python.exe -m douyin_library.cli compact-audits --vacuum
```

Backups are verified and serialized under
`D:\DouyinKnowledgeLibrary\data\backups` before required migrations and periodically
(24 hours by default), with the newest seven retained.

## Web library and troubleshooting

The web library paginates works in groups of 100 and provides search plus an extraction
failure filter. Clicking a work opens an in-page reader containing the full note; the
reader offers both `主题复盘` and `完整逐段时间线` sections without hiding source content.
Provider readiness and note progress are visible in the UI, including skipped, failed,
legacy-upgrade, and recent-work states.

If the page does not open, check `.runtime\logs`, confirm `8002` and `5175` are free, and
run `start-local.cmd` from the project directory. If notes show a fallback warning, check
`GET http://127.0.0.1:8002/api/notes/provider-status`, start Ollama, and install the model
explicitly; no source assets need to be downloaded again.

The launcher imports missing provider configuration from the Windows user environment
without printing the API key, and holds a project mutex to prevent simultaneous launches.
If project workers already exist, startup stops with `project_workers_already_running`;
reuse those services or stop their owning launcher first. Explicit failed-job retries use
`POST /api/jobs/<id>/retry`; an existing active task for the same work returns HTTP 409.

## Example notes

[`examples/notes/`](examples/notes/) contains five representative learning notes
picked from a local library of 2,283, spanning AI, investing, philosophy, growth,
and wealth topics. They are generated locally (Ollama qwen3:4b) from ASR
transcripts and are included only to demonstrate pipeline output quality;
copyright of the original videos remains with their authors (links in each note
header). `examples/notes/ai-context-engineering/legacy_v1.md` shows the same note
as produced before the semantic pipeline upgrade, for before/after comparison.

## Known issues and gotchas

- **Network to github.com**: on mainland-China networks the github.com TLS
  handshake is commonly reset (SNI blocking) while `api.github.com` works.
  `git push`/`fetch` needs a working proxy — e.g.
  `git config http.https://github.com.proxy http://127.0.0.1:7897` — or the
  push fails with connection-closed errors. Browser access to github.com has
  the same symptom.
- **ASR model loading needs `HF_ENDPOINT=https://hf-mirror.com` on every
  load**, not just the first. Without it the HuggingFace cache misses and the
  load re-attempts huggingface.co and dies on the walled host. Persist it at
  the user level (or pass `DKL_QWEN_OFFLINE=true` only after weights are fully
  cached) and restart the ASR worker. A single successful load under an ad-hoc
  environment does not prove the worker will see it.
- **`qwen-asr` install side effects**: installing `qwen-asr` upgrades
  `transformers` in that environment. If the same env is shared with other
  pipelines (e.g. voice-generation stacks), pin `transformers` back first when
  they break.
- **GPU scheduling**: ASR transcription and local note inference share the
  `gpu:0` lease and never run together. Batch note generation and transcription
  should be staggered deliberately; running both at once just serializes on
  the lease.
- **No automatic retry for failed transcription jobs**: failed jobs sit until
  resumed. Use `POST /api/jobs/bulk-resume {"job_ids": [...]}` for batches or
  `POST /api/jobs/<id>/retry` for a single job. Note there is no
  `GET /api/jobs/<id>` route — list jobs with `GET /api/jobs?limit=N`.
- **Semantic chapter classifier is faithful but blunt**: 方法与步骤
  (methods & steps) occasionally labels a video full of concrete methods as
  "not provided". Cross-check that chapter against the full timeline before
  trusting a coverage report.
- **Windows long paths**: test leftovers under `backend/.pytest-tmp*` can
  exceed MAX_PATH and make `git add` fail with "Filename too long". They are
  gitignored; delete the directories if the index ever chokes.
- **This repository intentionally contains no runtime data**: the local
  library (`D:\DouyinKnowledgeLibrary`, tens of GB of audio/transcripts/notes)
  is not version-controlled and not uploaded. Only code, docs and the five
  curated note samples under `examples/notes/` are published.
