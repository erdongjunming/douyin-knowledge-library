from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import create_engine

from douyin_library.config import Settings
from douyin_library.db import rebuild_work_fts


def _database_url(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m douyin_library.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    rebuild_parser = subparsers.add_parser("rebuild-index")
    rebuild_parser.add_argument("--database", type=Path)
    args = parser.parse_args(argv)

    if args.command == "rebuild-index":
        database_path = args.database or Settings().storage_root / "data" / "library.db"
        if not database_path.is_file():
            parser.error(f"database does not exist: {database_path}")
        count = rebuild_work_fts(create_engine(_database_url(database_path)))
        print(f"Rebuilt full-text index for {count} works")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
