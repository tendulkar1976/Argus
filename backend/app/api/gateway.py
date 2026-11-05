"""Gateway API route handlers for prompt optimization and frontier model dispatch (Phase 4.2)."""

from fastapi import APIRouter, HTTPException, status

from app.schemas.gateway import (
    GatewayDispatchRequest,
    GatewayDispatchResponse,
)
from app.services.gateway_service import (
    GatewayDispatchDisabledError,
    GatewayService,
)
from app.services.ollama_service import OllamaServiceError
from app.services.providers.base import (
    AuthenticationError,
    ConfigurationError,
    MissingCredentialsError,
    ProviderAPIError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    UnsupportedModelError,
)

router = APIRouter(prefix="/gateway", tags=["Gateway Dispatch"])


@router.post(
    "/dispatch",
    response_model=GatewayDispatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Dispatch Prompt through Argus to Frontier Model",
    description=(
        "Optimizes a client prompt using local inference (Ollama), preserves semantic constraints, "
        "and dispatches the verified prompt to the configured frontier model (e.g. gpt-6.1-sol). "
        "Returns the model response alongside actual token usage, verified costs, and estimated token savings."
    ),
)
async def dispatch_prompt(request: GatewayDispatchRequest) -> GatewayDispatchResponse:
    """Execute end-to-end prompt optimization and frontier model dispatch."""
    try:
        return await GatewayService.dispatch(request)
    except GatewayDispatchDisabledError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
    except MissingCredentialsError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
    except RateLimitError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
    except ProviderTimeoutError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
    except UnsupportedModelError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
    except ProviderConnectionError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
    except (ProviderAPIError, ConfigurationError, ProviderError) as exc:
        raise HTTPException(
            status_code=getattr(exc, "status_code", 502),
            detail=exc.message,
        ) from exc
    except OllamaServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=f"Local optimization failure before dispatch: {exc.message}",
        ) from exc
