import hashlib
import json
from datetime import UTC, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from douyin_library.db import create_schema
from douyin_library.media.storage import StorageLayout
from douyin_library.models import Asset, Job, Work
from douyin_library.transcription.inventory import (
    CandidateSelection,
    build_inventory,
    select_raw_asr_candidates,
)


def test_inventory_is_sorted_and_read_only(tmp_path):
    engine = create_engine("sqlite://")
    create_schema(engine)
    layout = StorageLayout(tmp_path)
    with Session(engine) as session:
        works = [Work(aweme_id="200"), Work(aweme_id="100")]
        session.add_all(works)
        session.flush()
        payload = b"audio"
        path = tmp_path / "audio.m4a"
        path.write_bytes(payload)
        session.add(
            Asset(
                work_id=works[0].id,
                kind="audio",
                path=str(path),
                bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                probe_json=json.dumps({"streams": [{"codec_name": "aac"}]}),
                verified_at=datetime.now(UTC),
            )
        )
        session.commit()
        before = session.query(Job).count()
        report = build_inventory(session, layout)
        assert tuple(i.aweme_id for i in report.items) == ("100", "200")
        assert isinstance(report.items, tuple)
        assert session.query(Job).count() == before


def test_candidate_selection_filters_before_limit(tmp_path):
    engine = create_engine("sqlite://")
    create_schema(engine)
    layout = StorageLayout(tmp_path)
    with Session(engine) as session:
        for aweme_id in ("1", "2", "3"):
            w = Work(aweme_id=aweme_id)
            session.add(w)
            session.flush()
            p = tmp_path / f"{aweme_id}.m4a"
            p.write_bytes(b"x")
            session.add(
                Asset(
                    work_id=w.id,
                    kind="audio",
                    path=str(p),
                    bytes=1,
                    sha256=hashlib.sha256(b"x").hexdigest(),
                    probe_json='{"streams":[{}]}',
                    verified_at=datetime.now(UTC),
                )
            )
        session.commit()
        report = build_inventory(session, layout)
    result = select_raw_asr_candidates(report, aweme_ids=[" 2 ", "3"], limit=1)
    assert isinstance(result, CandidateSelection)
    assert result.selected_ids == ("2",)
    assert tuple(result) == ("2",)
