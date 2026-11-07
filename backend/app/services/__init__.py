"""Service layer for Argus business and domain logic."""

from .health_service import HealthService
from .ollama_service import (
    OllamaService,
    OllamaServiceError,
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaResponseError,
)
from .task_analyzer import TaskAnalyzer, TaskAnalysisResult
from .strategy_selector import StrategySelector
from .prompt_rewriter import PromptRewriter
from .candidate_validator import CandidateValidator, ValidationResult
from .cost_calculator import CostCalculator, CostCalculationResult, SavingsEstimationResult
from .gateway_service import GatewayService, GatewayDispatchDisabledError
from .transformation_explainer import TransformationExplainer
from .response_comparator import ResponseComparator

__all__ = [
    "HealthService",
    "OllamaService",
    "OllamaServiceError",
    "OllamaConnectionError",
    "OllamaTimeoutError",
    "OllamaResponseError",
    "TaskAnalyzer",
    "TaskAnalysisResult",
    "StrategySelector",
    "PromptRewriter",
    "CandidateValidator",
    "ValidationResult",
    "CostCalculator",
    "CostCalculationResult",
    "SavingsEstimationResult",
    "GatewayService",
    "GatewayDispatchDisabledError",
    "UnifiedExecutionService",
    "TransformationExplainer",
    "ResponseComparator",
    "DualExecutionService",
]


def __getattr__(name: str):
    """Lazy import for services to prevent circular dependency cycles during schema loading."""
    if name == "UnifiedExecutionService":
        from .unified_execution_service import UnifiedExecutionService

        return UnifiedExecutionService
    elif name == "DualExecutionService":
        from .dual_execution_service import DualExecutionService

        return DualExecutionService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

