"""API route handlers for Unified Pipeline Execution, Telemetry, and Models (Phase 10)."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.schemas.execution import UnifiedExecutionRequest, UnifiedExecutionResponse
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
from app.services.providers.registry import ProviderRegistry
from app.services.routing.models import ModelCapability
from app.services.routing.registry import ModelCapabilityRegistry
from app.services.telemetry.models import TelemetryQueryFilter, TelemetryRecord, TelemetrySummary
from app.services.telemetry.service import TelemetryService
from app.services.unified_execution_service import UnifiedExecutionService

router = APIRouter(prefix="", tags=["Unified Execution & Telemetry"])


def get_execution_service() -> UnifiedExecutionService:
    """Dependency provider for UnifiedExecutionService."""
    return UnifiedExecutionService()


def get_telemetry_service() -> TelemetryService:
    """Dependency provider for TelemetryService."""
    return TelemetryService.get_instance()


@router.post(
    "/execute",
    response_model=UnifiedExecutionResponse,
    status_code=status.HTTP_200_OK,
    summary="End-to-End Argus Model Execution",
    description=(
        "Executes a request through the full Argus stack: "
        "RABCO multi-turn context optimization (or single prompt), "
        "Intelligent Model Routing, Multi-Provider dispatch with controlled fallback, "
        "decimal-safe cost calculation, and persistent telemetry recording."
    ),
)
async def execute_request(
    request: UnifiedExecutionRequest,
    service: UnifiedExecutionService = Depends(get_execution_service),
) -> UnifiedExecutionResponse:
    try:
        return await service.execute(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except MissingCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=exc.message,
        ) from exc
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=exc.message,
        ) from exc
    except RateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=exc.message,
        ) from exc
    except UnsupportedModelError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.message,
        ) from exc
    except ProviderTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=exc.message,
        ) from exc
    except ProviderConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=exc.message,
        ) from exc
    except ProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=exc.message,
        ) from exc


@router.get(
    "/telemetry/records",
    response_model=List[TelemetryRecord],
    status_code=status.HTTP_200_OK,
    summary="Query Telemetry Audit Records",
)
def get_telemetry_records(
    provider: Optional[str] = Query(None, description="Filter by provider"),
    model: Optional[str] = Query(None, description="Filter by model"),
    task_category: Optional[str] = Query(None, description="Filter by task category"),
    status: Optional[str] = Query(None, description="Filter by status (success/error/fallback)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    service: TelemetryService = Depends(get_telemetry_service),
) -> List[TelemetryRecord]:
    filters = TelemetryQueryFilter(
        provider=provider,
        model=model,
        task_category=task_category,
        status=status,
        limit=limit,
        offset=offset,
    )
    return service.query(filters)


@router.get(
    "/telemetry/summary",
    response_model=TelemetrySummary,
    status_code=status.HTTP_200_OK,
    summary="Get Aggregated Telemetry Analytics",
)
def get_telemetry_summary(
    hours: Optional[int] = Query(None, ge=1, description="Optional trailing hours window"),
    service: TelemetryService = Depends(get_telemetry_service),
) -> TelemetrySummary:
    return service.get_summary(hours=hours)


@router.get(
    "/models",
    response_model=List[ModelCapability],
    status_code=status.HTTP_200_OK,
    summary="List Registered Models & Capabilities",
)
def list_registered_models() -> List[ModelCapability]:
    return ModelCapabilityRegistry.list_models()


@router.get(
    "/providers/status",
    response_model=Dict[str, Dict[str, Any]],
    status_code=status.HTTP_200_OK,
    summary="Check Multi-Provider Configuration Statuses",
)
def get_providers_status() -> Dict[str, Dict[str, Any]]:
    return ProviderRegistry.get_all_statuses()
