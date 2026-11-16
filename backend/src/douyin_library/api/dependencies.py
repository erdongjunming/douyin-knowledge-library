from __future__ import annotations

from collections.abc import Iterator

from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from douyin_library.jobs.control import JobControl


def get_runtime_session(request: Request) -> Iterator[Session]:
    runtime = getattr(request.app.state, "runtime", None)
    engine = getattr(runtime, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="storage_not_initialized",
        )
    with Session(engine) as session:
        yield session


def get_job_control(request: Request) -> JobControl | None:
    """Return the in-process cancellation bridge when one is available.

    Production normally runs workers in separate processes, so the API cannot
    have a local ``JobControl`` object.  Durable cancellation is still handled
    by the job row; a local bridge is only an optional fast-path signal.
    """

    control = getattr(request.app.state, "job_control", None)
    return control if isinstance(control, JobControl) else None
