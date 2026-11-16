import hashlib
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from douyin_library.config import Settings
from douyin_library.db import create_schema
from douyin_library.jobs.runtime import build_worker_runtime
from douyin_library.models import Asset, Transcript, Work
from douyin_library.transcription.providers.qwen import QwenAsrProvider
from douyin_library.transcription.providers.qwen_resident import QwenResidentRunner
from douyin_library.transcription.service import SubprocessResult


async def _converter(
    audio_path: Path,
    wav_path: Path,
    cancel_event,
) -> SubprocessResult:
    del audio_path, cancel_event
    wav_path.write_bytes(b"wav")
    return SubprocessResult(returncode=0, stdout="", stderr="")


@pytest.mark.asyncio
async def test_resident_runner_publishes_two_transcripts_with_one_model_process(
    tmp_path: Path,
) -> None:
    starts = tmp_path / "starts.txt"
    server = tmp_path / "server.py"
    server.write_text(
        f"""
import json
from pathlib import Path
Path({str(starts)!r}).write_text('start\\n', encoding='utf-8')
print(json.dumps({{'event': 'ready'}}), flush=True)
for line in __import__('sys').stdin:
    request = json.loads(line)
    Path(request['out']).write_text(json.dumps({{
        'model': 'Qwen/Qwen3-ASR-0.6B',
        'backend': 'qwen3-asr',
        'language': 'Chinese',
        'segments': [{{'start': 0.0, 'end': 1.0, 'text': '常驻转写。'}}],
        'text': '常驻转写。',
    }}, ensure_ascii=False), encoding='utf-8')
    print(
        json.dumps({{'ok': True, 'request_id': request['request_id'], 'returncode': 0}}),
        flush=True,
    )
""",
        encoding="utf-8",
    )
    storage = tmp_path / "storage"
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    audio_bytes = b"same-audio"
    audio_paths: list[Path] = []
    work_ids: list[int] = []
    with Session(engine) as session:
        for aweme_id in ("resident-1", "resident-2"):
            audio_path = storage / "audio" / f"{aweme_id}.m4a"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(audio_bytes)
            work = Work(aweme_id=aweme_id)
            session.add(work)
            session.flush()
            session.add(
                Asset(
                    work_id=work.id,
                    kind="audio",
                    path=str(audio_path),
                    bytes=len(audio_bytes),
                    sha256=hashlib.sha256(audio_bytes).hexdigest(),
                )
            )
            audio_paths.append(audio_path)
            work_ids.append(work.id)
        session.commit()

    runner = QwenResidentRunner(
        python_path=Path(sys.executable),
        server_script_path=server,
        model="Qwen/Qwen3-ASR-0.6B",
        language="Chinese",
        chunk_seconds=60,
        startup_timeout_seconds=5,
        audio_converter=_converter,
    )
    runtime = build_worker_runtime(
        engine,
        Settings(storage_root=storage),
        transcriber_factory=lambda settings: QwenAsrProvider(runner),
    )
    try:
        jobs = [
            runtime.queue.create(kind="transcribe_work", work_id=work_id)
            for work_id in work_ids
        ]
        assert await runtime.handlers.run_one() is True
        assert await runtime.handlers.run_one() is True
        assert all(runtime.queue.get(job.id).status == "succeeded" for job in jobs)
        texts = runtime.session.scalars(
            select(Transcript.text).order_by(Transcript.work_id)
        ).all()
        assert texts == ["常驻转写。", "常驻转写。"]
        assert starts.read_text(encoding="utf-8").splitlines() == ["start"]
    finally:
        runtime.close()
        engine.dispose()
