"""API route handlers for Browser Automation with TinyFish (Phase 9 & 10)."""

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, status

from app.services.automation.tinyfish.adapter import (
    TinyFishAdapter,
    TinyFishAuthenticationError,
    TinyFishConnectionError,
    TinyFishError,
    TinyFishNotConfiguredError,
    TinyFishTimeoutError,
)
from app.services.automation.tinyfish.models import TinyFishTaskRequest, TinyFishTaskResponse

router = APIRouter(prefix="/automation/tinyfish", tags=["Browser Automation (TinyFish)"])


def get_tinyfish_adapter() -> TinyFishAdapter:
    """Dependency provider for TinyFishAdapter."""
    return TinyFishAdapter()


@router.post(
    "/execute",
    response_model=TinyFishTaskResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute Bounded Browser Automation Task",
    description=(
        "Executes a validated browser automation task (navigate, extract, click, screenshot, evaluate) "
        "via TinyFish. Browser automation is isolated from model inference pathways."
    ),
)
async def execute_browser_task(
    request: TinyFishTaskRequest,
    adapter: TinyFishAdapter = Depends(get_tinyfish_adapter),
) -> TinyFishTaskResponse:
    try:
        return await adapter.execute_task(request)
    except TinyFishNotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=exc.message,
        ) from exc
    except TinyFishAuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=exc.message,
        ) from exc
    except TinyFishTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=exc.message,
        ) from exc
    except TinyFishConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=exc.message,
        ) from exc
    except TinyFishError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=exc.message,
        ) from exc


@router.get(
    "/status",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="TinyFish Service Configuration Status",
)
def get_tinyfish_status(
    adapter: TinyFishAdapter = Depends(get_tinyfish_adapter),
) -> Dict[str, Any]:
    return adapter.get_status()
