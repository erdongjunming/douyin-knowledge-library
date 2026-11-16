"""Cross-process resource leases backed by SQLite.

The ASR and local note workers share a GPU but run in separate processes. A
row keyed by ``resource_name`` gives both workers an atomic compare-and-swap
operation and a fencing token. Lease methods intentionally perform a short
transaction and never sleep while holding a database transaction; callers can
poll :meth:`ResourceLeaseStore.try_acquire` between attempts.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from numbers import Real
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session

from douyin_library.models import ResourceLease

DEFAULT_LEASE_TTL = timedelta(seconds=30)
_UNSET = object()


@dataclass(frozen=True, slots=True)
class ResourceLeaseHandle:
    """Immutable snapshot returned after a successful acquisition.

    A snapshot is deliberately returned instead of a live ORM instance. If an
    expired row is replaced, an older caller retains its original token and
    therefore cannot accidentally release the replacement.
    """

    resource_name: str
    lease_token: str
    owner_worker_id: str
    owner_job_id: int | None
    acquired_at: datetime
    heartbeat_at: datetime
    expires_at: datetime

    @property
    def token(self) -> str:
        return self.lease_token

    @property
    def worker_id(self) -> str:
        return self.owner_worker_id

    @property
    def job_id(self) -> int | None:
        return self.owner_job_id


LeaseHandle = ResourceLeaseHandle


class ResourceLeaseStore:
    """Atomic CRUD operations for :class:`ResourceLease`.

    ``bind`` may be an existing SQLAlchemy ``Session`` (the application path)
    or an ``Engine``/``Connection``. For the latter each operation creates and
    closes its own short-lived session, suitable for independent workers.
    """

    def __init__(
        self,
        bind: Session | Engine | Connection,
        *,
        default_ttl: timedelta | Real = DEFAULT_LEASE_TTL,
        clock: Callable[[], datetime] | None = None,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        if isinstance(bind, Session):
            self.session: Session | None = bind
            self.bind: Engine | Connection | None = None
        else:
            self.session = None
            self.bind = bind
        self.default_ttl = _coerce_ttl(default_ttl)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(32))

    @contextmanager
    def _session_scope(self) -> Iterator[Session]:
        if self.session is not None:
            yield self.session
            return
        if self.bind is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("resource_lease_store_not_bound")
        with Session(self.bind) as session:
            yield session

    def try_acquire(
        self,
        resource_name: str,
        worker_id: str | None = None,
        owner_job_id: int | None = None,
        *,
        ttl: timedelta | Real | None = None,
        owner_worker_id: str | None = None,
        job_id: int | None = None,
    ) -> ResourceLeaseHandle | None:
        """Atomically acquire a resource if it is free or expired."""

        resource_name = _require_resource_name(resource_name)
        worker_id = _coalesce_worker(worker_id, owner_worker_id)
        worker_id = _require_worker_id(worker_id)
        owner_job_id = _coalesce_job(owner_job_id, job_id)
        lease_ttl = _coerce_ttl(self.default_ttl if ttl is None else ttl)
        now = _aware_utc(self._clock())
        expires_at = now + lease_ttl

        # A token collision is vanishingly unlikely with token_urlsafe, but a
        # bounded retry keeps a custom/test token factory from poisoning a
        # worker when a different resource already owns that token.
        for _ in range(3):
            token = str(self._token_factory())
            if not token or len(token) > 64:
                raise ValueError("lease_token_must_be_1_to_64_characters")
            with self._session_scope() as session:
                statement = sqlite_insert(ResourceLease).values(
                    resource_name=resource_name,
                    lease_token=token,
                    owner_worker_id=worker_id,
                    owner_job_id=owner_job_id,
                    acquired_at=now,
                    heartbeat_at=now,
                    expires_at=expires_at,
                )
                statement = statement.on_conflict_do_update(
                    index_elements=[ResourceLease.resource_name],
                    set_={
                        "lease_token": statement.excluded.lease_token,
                        "owner_worker_id": statement.excluded.owner_worker_id,
                        "owner_job_id": statement.excluded.owner_job_id,
                        "acquired_at": statement.excluded.acquired_at,
                        "heartbeat_at": statement.excluded.heartbeat_at,
                        "expires_at": statement.excluded.expires_at,
                    },
                    where=ResourceLease.expires_at <= statement.excluded.acquired_at,
                )
                try:
                    with session.no_autoflush:
                        result = session.execute(statement)
                        acquired = result.rowcount != 0
                        row = None
                        if acquired:
                            # Select before commit while SQLite still holds the
                            # write lock, then copy values into an immutable
                            # fencing handle.
                            row = session.scalar(
                                select(ResourceLease).where(
                                    ResourceLease.resource_name == resource_name,
                                    ResourceLease.lease_token == token,
                                )
                            )
                            acquired = row is not None
                        session.commit()
                except Exception as exc:
                    session.rollback()
                    if _is_unique_token_collision(exc):
                        continue
                    raise
                if not acquired or row is None:
                    return None
                return _snapshot(row)
        raise RuntimeError("resource_lease_token_collision")

    def renew(
        self,
        lease_or_resource: ResourceLeaseHandle | ResourceLease | str,
        lease_token: str | None = None,
        worker_id: str | None = None,
        owner_job_id: int | None | object = _UNSET,
        *,
        ttl: timedelta | Real | None = None,
        token: str | None = None,
        owner_worker_id: str | None = None,
        job_id: int | None | object = _UNSET,
    ) -> bool:
        """Extend a lease when all fencing and owner fields still match."""

        resource, actual_token, actual_worker, actual_job = _identity(
            lease_or_resource,
            lease_token=lease_token,
            worker_id=worker_id,
            owner_job_id=owner_job_id,
            token=token,
            owner_worker_id=owner_worker_id,
            job_id=job_id,
        )
        lease_ttl = _coerce_ttl(self.default_ttl if ttl is None else ttl)
        now = _aware_utc(self._clock())
        with self._session_scope() as session:
            statement = (
                update(ResourceLease)
                .where(
                    ResourceLease.resource_name == resource,
                    ResourceLease.lease_token == actual_token,
                    ResourceLease.owner_worker_id == actual_worker,
                    _job_predicate(actual_job),
                    ResourceLease.expires_at > now,
                )
                .values(heartbeat_at=now, expires_at=now + lease_ttl)
            )
            try:
                result = session.execute(statement)
                session.commit()
            except Exception:
                session.rollback()
                raise
        return bool(result.rowcount)

    def release(
        self,
        lease_or_resource: ResourceLeaseHandle | ResourceLease | str,
        lease_token: str | None = None,
        worker_id: str | None = None,
        owner_job_id: int | None | object = _UNSET,
        *,
        token: str | None = None,
        owner_worker_id: str | None = None,
        job_id: int | None | object = _UNSET,
    ) -> bool:
        """Release exactly the lease represented by a fencing handle."""

        resource, actual_token, actual_worker, actual_job = _identity(
            lease_or_resource,
            lease_token=lease_token,
            worker_id=worker_id,
            owner_job_id=owner_job_id,
            token=token,
            owner_worker_id=owner_worker_id,
            job_id=job_id,
        )
        with self._session_scope() as session:
            statement = delete(ResourceLease).where(
                ResourceLease.resource_name == resource,
                ResourceLease.lease_token == actual_token,
                ResourceLease.owner_worker_id == actual_worker,
                _job_predicate(actual_job),
            )
            try:
                result = session.execute(statement)
                session.commit()
            except Exception:
                session.rollback()
                raise
        return bool(result.rowcount)

    def is_owned(
        self,
        lease_or_resource: ResourceLeaseHandle | ResourceLease | str,
        lease_token: str | None = None,
        worker_id: str | None = None,
        owner_job_id: int | None | object = _UNSET,
        *,
        token: str | None = None,
        owner_worker_id: str | None = None,
        job_id: int | None | object = _UNSET,
    ) -> bool:
        """Return whether a matching, unexpired lease is currently owned."""

        resource, actual_token, actual_worker, actual_job = _identity(
            lease_or_resource,
            lease_token=lease_token,
            worker_id=worker_id,
            owner_job_id=owner_job_id,
            token=token,
            owner_worker_id=owner_worker_id,
            job_id=job_id,
        )
        now = _aware_utc(self._clock())
        with self._session_scope() as session:
            statement = select(ResourceLease.resource_name).where(
                ResourceLease.resource_name == resource,
                ResourceLease.lease_token == actual_token,
                ResourceLease.owner_worker_id == actual_worker,
                _job_predicate(actual_job),
                ResourceLease.expires_at > now,
            )
            try:
                found = session.scalar(statement) is not None
                session.commit()
            except Exception:
                session.rollback()
                raise
        return found

    def purge_expired(self, resource_name: str | None = None) -> int:
        """Delete only expired leases and return the number removed."""

        now = _aware_utc(self._clock())
        with self._session_scope() as session:
            statement = delete(ResourceLease).where(ResourceLease.expires_at <= now)
            if resource_name is not None:
                statement = statement.where(
                    ResourceLease.resource_name == _require_resource_name(resource_name)
                )
            try:
                result = session.execute(statement)
                session.commit()
            except Exception:
                session.rollback()
                raise
        return max(int(result.rowcount or 0), 0)

    def for_owner(
        self,
        worker_id: str,
        owner_job_id: int | None | object = _UNSET,
        *,
        job_id: int | None | object = _UNSET,
    ) -> list[ResourceLeaseHandle]:
        """Return snapshots for a worker, optionally narrowed to one job."""

        worker_id = _require_worker_id(worker_id)
        actual_job = _coalesce_optional_job(owner_job_id, job_id)
        with self._session_scope() as session:
            statement = select(ResourceLease).where(
                ResourceLease.owner_worker_id == worker_id,
                _job_predicate(actual_job),
            )
            try:
                rows = [_snapshot(row) for row in session.scalars(statement)]
                session.commit()
            except Exception:
                session.rollback()
                raise
        return rows

    def renew_for_owner(
        self,
        worker_id: str,
        owner_job_id: int | None | object = _UNSET,
        *,
        ttl: timedelta | Real | None = None,
        job_id: int | None | object = _UNSET,
    ) -> int:
        """Renew all currently valid leases owned by a worker/job pair."""

        worker_id = _require_worker_id(worker_id)
        actual_job = _coalesce_optional_job(owner_job_id, job_id)
        lease_ttl = _coerce_ttl(self.default_ttl if ttl is None else ttl)
        now = _aware_utc(self._clock())
        with self._session_scope() as session:
            statement = (
                update(ResourceLease)
                .where(
                    ResourceLease.owner_worker_id == worker_id,
                    _job_predicate(actual_job),
                    ResourceLease.expires_at > now,
                )
                .values(heartbeat_at=now, expires_at=now + lease_ttl)
            )
            try:
                result = session.execute(statement)
                session.commit()
            except Exception:
                session.rollback()
                raise
        return max(int(result.rowcount or 0), 0)


def _snapshot(row: ResourceLease) -> ResourceLeaseHandle:
    return ResourceLeaseHandle(
        resource_name=row.resource_name,
        lease_token=row.lease_token,
        owner_worker_id=row.owner_worker_id,
        owner_job_id=row.owner_job_id,
        acquired_at=_aware_utc(row.acquired_at),
        heartbeat_at=_aware_utc(row.heartbeat_at),
        expires_at=_aware_utc(row.expires_at),
    )


def _identity(
    lease_or_resource: ResourceLeaseHandle | ResourceLease | str,
    *,
    lease_token: str | None,
    worker_id: str | None,
    owner_job_id: int | None | object,
    token: str | None,
    owner_worker_id: str | None,
    job_id: int | None | object,
) -> tuple[str, str, str, int | None]:
    if isinstance(lease_or_resource, str):
        resource = lease_or_resource
        default_token = default_worker = None
        default_job: int | None = None
    else:
        resource = getattr(lease_or_resource, "resource_name", None)
        default_token = getattr(lease_or_resource, "lease_token", None)
        default_worker = getattr(lease_or_resource, "owner_worker_id", None)
        default_job = getattr(lease_or_resource, "owner_job_id", None)
    actual_token = token if token is not None else lease_token
    if actual_token is None:
        actual_token = default_token
    actual_worker = owner_worker_id if owner_worker_id is not None else worker_id
    if actual_worker is None:
        actual_worker = default_worker
    actual_job = default_job if owner_job_id is _UNSET else owner_job_id
    if job_id is not _UNSET:
        actual_job = job_id
    if resource is None:
        raise ValueError("resource_name_required")
    if actual_token is None:
        raise ValueError("lease_token_required")
    if actual_worker is None:
        raise ValueError("owner_worker_id_required")
    return (
        _require_resource_name(str(resource)),
        str(actual_token),
        _require_worker_id(str(actual_worker)),
        _normalise_job(actual_job),
    )


def _job_predicate(owner_job_id: int | None | object) -> Any:
    if owner_job_id is _UNSET:
        return True
    if owner_job_id is None:
        return ResourceLease.owner_job_id.is_(None)
    return ResourceLease.owner_job_id == owner_job_id


def _coalesce_worker(value: str | None, alias: str | None) -> str | None:
    if value is not None and alias is not None and value != alias:
        raise ValueError("worker_id_alias_conflict")
    return value if value is not None else alias


def _coalesce_job(value: int | None, alias: int | None) -> int | None:
    if value is not None and alias is not None and value != alias:
        raise ValueError("job_id_alias_conflict")
    return value if value is not None else alias


def _coalesce_optional_job(
    value: int | None | object, alias: int | None | object
) -> int | None | object:
    if value is not _UNSET and alias is not _UNSET and value != alias:
        raise ValueError("job_id_alias_conflict")
    if value is not _UNSET:
        return _normalise_job(value)
    if alias is not _UNSET:
        return _normalise_job(alias)
    return _UNSET


def _normalise_job(value: int | None | object) -> int | None:
    if value is _UNSET:
        return None
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("owner_job_id_must_be_integer_or_null")
    return value


def _require_resource_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("resource_name_required")
    if len(value) > 80:
        raise ValueError("resource_name_too_long")
    return value


def _require_worker_id(value: str | None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("owner_worker_id_required")
    if len(value) > 120:
        raise ValueError("owner_worker_id_too_long")
    return value


def _coerce_ttl(value: timedelta | Real) -> timedelta:
    if isinstance(value, timedelta):
        result = value
    elif isinstance(value, Real) and not isinstance(value, bool):
        result = timedelta(seconds=float(value))
    else:
        raise TypeError("lease_ttl_must_be_timedelta_or_seconds")
    if result <= timedelta(0):
        raise ValueError("lease_ttl_must_be_positive")
    return result


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _is_unique_token_collision(exc: Exception) -> bool:
    text = str(exc).lower()
    return "unique constraint failed" in text and "lease_token" in text
