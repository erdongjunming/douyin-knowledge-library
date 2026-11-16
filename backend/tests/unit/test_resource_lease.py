from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session

from douyin_library.db import create_schema
from douyin_library.jobs.leases import ResourceLeaseStore
from douyin_library.models import ResourceLease


def _store(tmp_path: Path) -> tuple[object, Session]:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    create_schema(engine)
    return engine, Session(engine)


def test_active_resource_can_only_have_one_owner(tmp_path: Path) -> None:
    _, session = _store(tmp_path)
    try:
        leases = ResourceLeaseStore(session)
        first = leases.try_acquire("gpu:0", "worker-a", 1, ttl=timedelta(minutes=1))
        second = leases.try_acquire("gpu:0", "worker-b", 2, ttl=timedelta(minutes=1))

        assert first is not None
        assert second is None
        assert leases.is_owned(first)
    finally:
        session.close()


def test_expired_lease_can_be_taken_but_old_token_cannot_release_it(tmp_path: Path) -> None:
    _, session = _store(tmp_path)
    try:
        leases = ResourceLeaseStore(session)
        first = leases.try_acquire("gpu:0", "worker-a", 1, ttl=timedelta(seconds=1))
        assert first is not None
        session.execute(
            update(ResourceLease)
            .where(ResourceLease.resource_name == "gpu:0")
            .values(expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
        session.commit()

        second = leases.try_acquire("gpu:0", "worker-b", 2, ttl=timedelta(minutes=1))
        assert second is not None
        assert leases.release(first) is False
        assert leases.is_owned(second) is True
        assert second.lease_token != first.lease_token
    finally:
        session.close()


def test_renew_requires_exact_owner_and_extends_expiry(tmp_path: Path) -> None:
    _, session = _store(tmp_path)
    try:
        leases = ResourceLeaseStore(session)
        handle = leases.try_acquire("gpu:0", "worker-a", 1, ttl=timedelta(seconds=5))
        assert handle is not None
        before = handle.expires_at

        assert leases.renew(handle, ttl=timedelta(minutes=1)) is True
        assert leases.is_owned(handle)
        renewed = session.scalar(
            select(ResourceLease).where(ResourceLease.resource_name == "gpu:0")
        )
        assert renewed is not None
        assert renewed.expires_at.replace(tzinfo=UTC) > before
        assert leases.renew(handle, worker_id="worker-b", ttl=timedelta(minutes=1)) is False
    finally:
        session.close()


def test_release_and_purge_expired_are_token_fenced(tmp_path: Path) -> None:
    _, session = _store(tmp_path)
    try:
        leases = ResourceLeaseStore(session)
        handle = leases.try_acquire("gpu:0", "worker-a", 1, ttl=timedelta(seconds=1))
        assert handle is not None
        session.execute(
            update(ResourceLease)
            .where(ResourceLease.resource_name == "gpu:0")
            .values(expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
        session.commit()
        assert leases.purge_expired() == 1
        assert leases.is_owned(handle) is False
        assert leases.release(handle) is False
    finally:
        session.close()
