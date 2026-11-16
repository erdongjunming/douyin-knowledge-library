import logging
from typing import Annotated, Protocol

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from douyin_library.config import Settings, diagnostics
from douyin_library.security.cookie_store import CookieStore, CookieStoreUnavailableError
from douyin_library.upstream.jiji_gateway import JijiGateway

router = APIRouter(prefix="/api/settings", tags=["settings"])
logger = logging.getLogger(__name__)


class CookieValidationRequest(BaseModel):
    cookie: str | None = None


class CookieSaveRequest(BaseModel):
    cookie: str = Field(min_length=1)


class CookieValidationResponse(BaseModel):
    valid: bool
    account_nickname: str | None
    expires_or_unknown: str
    risk_controlled: bool


class CookieStatusResponse(BaseModel):
    configured: bool


class CookieSaveResponse(CookieValidationResponse):
    configured: bool


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


def get_cookie_store() -> CookieStore:
    return CookieStore()


@router.get("/diagnostics")
async def get_diagnostics() -> dict[str, object]:
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


@router.get(
    "/douyin-cookie",
    response_model=CookieStatusResponse,
)
async def get_douyin_cookie_status(
    store: Annotated[CookieStore, Depends(get_cookie_store)],
) -> dict[str, bool]:
    try:
        return {"configured": bool(store.read())}
    except CookieStoreUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="douyin_cookie_store_unavailable",
        ) from exc


@router.put(
    "/douyin-cookie",
    response_model=CookieSaveResponse,
)
async def save_douyin_cookie(
    payload: CookieSaveRequest,
    validator: Annotated[CookieValidator, Depends(get_cookie_validator)],
    store: Annotated[CookieStore, Depends(get_cookie_store)],
) -> dict[str, object]:
    result = await validator.validate(payload.cookie)
    if not result.get("valid"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="douyin_cookie_invalid",
        )
    try:
        store.save(payload.cookie)
    except CookieStoreUnavailableError as exc:
        logger.warning("Douyin credential store unavailable: %s", type(exc.__cause__).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="douyin_cookie_store_unavailable",
        ) from exc
    return {"configured": True, **result}


@router.delete(
    "/douyin-cookie",
    response_model=CookieStatusResponse,
)
async def delete_douyin_cookie(
    store: Annotated[CookieStore, Depends(get_cookie_store)],
) -> dict[str, bool]:
    try:
        store.delete()
    except CookieStoreUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="douyin_cookie_store_unavailable",
        ) from exc
    return {"configured": False}
