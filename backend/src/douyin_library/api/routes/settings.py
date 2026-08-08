from typing import Annotated, Protocol

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from douyin_library.config import Settings, diagnostics
from douyin_library.security.cookie_store import CookieStore
from douyin_library.upstream.jiji_gateway import JijiGateway

router = APIRouter(prefix="/api/settings", tags=["settings"])


class CookieValidationRequest(BaseModel):
    cookie: str | None = None


class CookieValidationResponse(BaseModel):
    valid: bool
    account_nickname: str | None
    expires_or_unknown: str
    risk_controlled: bool


class CookieValidator(Protocol):
    async def validate(self, cookie_header: str | None) -> dict[str, object]: ...


class StoredCookieValidator:
    def __init__(self, store: CookieStore | None = None) -> None:
        self.store = store or CookieStore()

    async def validate(self, cookie_header: str | None) -> dict[str, object]:
        cookie = cookie_header or self.store.read()
        if not cookie:
            return {
                "valid": False,
                "account_nickname": None,
                "expires_or_unknown": "unknown",
                "risk_controlled": False,
            }
        gateway = JijiGateway.from_cookie_header(cookie)
        try:
            return await gateway.validate_account()
        finally:
            await gateway.close()


def get_cookie_validator() -> CookieValidator:
    return StoredCookieValidator()


@router.get("/diagnostics")
async def get_diagnostics() -> dict[str, dict[str, str]]:
    return diagnostics(Settings())


@router.post(
    "/douyin-cookie/validate",
    response_model=CookieValidationResponse,
)
async def validate_douyin_cookie(
    validator: Annotated[CookieValidator, Depends(get_cookie_validator)],
    payload: CookieValidationRequest | None = None,
) -> dict[str, object]:
    return await validator.validate(payload.cookie if payload else None)
