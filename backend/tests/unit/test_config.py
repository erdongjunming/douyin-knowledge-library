from pathlib import Path

from douyin_library.config import Settings


def test_default_storage_root_is_d_drive(monkeypatch) -> None:
    monkeypatch.delenv("DKL_STORAGE_ROOT", raising=False)

    settings = Settings()

    assert settings.storage_root == Path(r"D:\DouyinKnowledgeLibrary")

