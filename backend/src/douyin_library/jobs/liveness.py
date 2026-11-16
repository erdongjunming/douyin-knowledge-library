"""Runtime evidence for independent pools, separate from durable claim control."""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HEARTBEAT_INTERVAL_SECONDS = 5.0
HEARTBEAT_STALE_SECONDS = 20.0
_SCHEMA = "douyin-worker-pool-heartbeat-v1"


class PoolHeartbeat:
    def __init__(self, root: Path, pool: str) -> None:
        self.pool = pool
        self.path = (
            root / "runtime" / "worker-pools" / f"{pool}.{uuid.uuid4().hex}.heartbeat.json"
        )

    def publish(self, state: str = "running") -> None:
        payload = {
            "schema": _SCHEMA, "pool": self.pool, "pid": os.getpid(), "state": state,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    async def run(self) -> None:
        while True:
            self.publish()
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)


def pool_liveness(root: Path, pool: str) -> dict[str, Any]:
    now = datetime.now(UTC)
    alive = 0
    failed = False
    for path in (root / "runtime" / "worker-pools").glob(f"{pool}.*.heartbeat.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != _SCHEMA or payload.get("pool") != pool:
                continue
            updated = datetime.fromisoformat(payload["updated_at"])
            if updated.tzinfo is None:
                continue
            age = (now - updated).total_seconds()
            if not 0 <= age <= HEARTBEAT_STALE_SECONDS:
                continue
            if payload.get("state") == "failed":
                failed = True
            elif payload.get("state") == "running" and _pid_running(payload.get("pid")):
                alive += 1
        except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return {
        "status": "alive" if alive else ("failed" if failed else "unavailable"),
        "processes": alive,
    }


def _pid_running(pid: Any) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) is not a portable Windows liveness probe.
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            # A fresh heartbeat remains evidence when process query permission
            # is denied; an invalid/nonexistent PID is never alive.
            return ctypes.get_last_error() == 5
        try:
            code = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
