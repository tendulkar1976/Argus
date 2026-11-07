"""API route handlers for prompt optimization and Ollama status."""

from fastapi import APIRouter, HTTPException, status
from app.schemas.optimizer import (
    OptimizeRequest,
    OptimizeResponse,
    OllamaStatusResponse,
)
from app.schemas.comparison import (
    DualExecutionRequest,
    DualExecutionResponse,
)
from app.services.ollama_service import (
    OllamaService,
    OllamaServiceError,
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaResponseError,
)
from app.services.providers.base import ProviderError
from app.services.dual_execution_service import DualExecutionService

router = APIRouter(tags=["Prompt Optimization"])


@router.post(
    "/optimize",
    response_model=OptimizeResponse,
    status_code=status.HTTP_200_OK,
    summary="Optimize Prompt",
    description="Rewrites a user prompt using the local Ollama LLM to reduce input tokens while preserving intent and constraints.",
)
async def optimize_prompt(request: OptimizeRequest) -> OptimizeResponse:
    """Optimize prompt with local Ollama model."""
    try:
        return await OllamaService.optimize_prompt(request)
    except (OllamaConnectionError, OllamaTimeoutError, OllamaResponseError, OllamaServiceError) as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.post(
    "/compare",
    response_model=DualExecutionResponse,
    status_code=status.HTTP_200_OK,
    summary="A/B Dual Execution & Explainable Comparison",
    description=(
        "Optimizes an input prompt, dispatches both original and optimized versions "
        "to the exact same configured frontier LLM, captures independent execution metrics, "
        "generates transformation explanations, and evaluates response quality across 9 dimensions."
    ),
)
async def compare_prompts(request: DualExecutionRequest) -> DualExecutionResponse:
    """Execute original vs optimized prompt A/B comparison."""
    try:
        return await DualExecutionService.execute(request)
    except (OllamaConnectionError, OllamaTimeoutError, OllamaResponseError, OllamaServiceError) as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
    except ProviderError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dual execution comparison failed: {str(exc)}",
        ) from exc


@router.get(
    "/ollama/status",
    response_model=OllamaStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Ollama Service Status",
    description="Inspects the local Ollama service connection and verifies model availability.",
)
async def check_ollama_status() -> OllamaStatusResponse:
    """Verify local Ollama service health and model installation."""
    return await OllamaService.check_status()
