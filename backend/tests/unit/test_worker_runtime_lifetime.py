from pathlib import Path

from sqlalchemy import create_engine

from douyin_library.config import Settings
from douyin_library.db import create_schema
from douyin_library.jobs.runtime import build_worker_runtime


def test_worker_runtime_constructs_one_transcriber_and_closes_it(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    created: list[object] = []

    class FakeTranscriber:
        def close(self) -> None:
            created.append("closed")

    def factory(settings: Settings) -> FakeTranscriber:
        del settings
        transcriber = FakeTranscriber()
        created.append(transcriber)
        return transcriber

    runtime = build_worker_runtime(
        engine,
        Settings(storage_root=tmp_path / "storage"),
        transcriber_factory=factory,
    )

    assert len(created) == 1
    runtime.close()
    assert created[1:] == ["closed"]
    engine.dispose()
