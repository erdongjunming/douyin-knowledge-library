from __future__ import annotations

from collections.abc import Iterator

from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session


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
