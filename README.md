# Douyin Knowledge Library

Local-first workspace for collecting authorized Douyin content, extracting audio, transcribing speech, and organizing knowledge notes.

Runtime data defaults to `D:\DouyinKnowledgeLibrary`. The application must not silently fall back to the system drive when that location is unavailable.

## Local transcription

Chinese audio is transcribed with the shared `Qwen/Qwen3-ASR-0.6B` environment. The
FastAPI process launches ASR as a separate subprocess, converts retained M4A audio to
a temporary 16 kHz mono WAV under the work's D-drive transcript directory, and removes
the WAV after inference.

Retained transcript files are stored under:

```text
D:\DouyinKnowledgeLibrary\transcript\<aweme_id>\
```

Run one existing audio asset through the complete database, FTS, file and audit pipeline:

```powershell
cd backend
.\.venv\Scripts\python.exe -m douyin_library.cli transcribe-one --aweme-id <aweme_id>
```

Every synchronization and ASR operation writes a `processing_audit` record containing
the complete sanitized request and response, stdout/stderr, timestamps and file hashes.
Cookie, token and signed URL values are redacted before persistence.
