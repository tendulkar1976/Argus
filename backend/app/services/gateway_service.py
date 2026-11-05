"""Gateway Service orchestrating prompt optimization, upstream dispatch, and telemetry (Phase 4.2).

Enforces:
- Upstream dispatch disabled by default guardrail (ENABLE_UPSTREAM_DISPATCH).
- Single-pass optimization ensuring prompts are never double-optimized.
- Safe fallback: if optimization validation rejects an unsafe candidate, original prompt is used.
- Honest telemetry: separates upstream API usage from heuristic savings estimates.
- Provider isolation: dispatches through BaseLLMProvider interface.
"""

from typing import Optional
from app.core.config import settings
from app.schemas.optimizer import OptimizeRequest
from app.schemas.gateway import (
    GatewayDispatchRequest,
    GatewayDispatchResponse,
    GatewaySavingsTelemetry,
    GatewayUsageTelemetry,
)
from app.services.cost_calculator import CostCalculator
from app.services.ollama_service import OllamaService
from app.services.providers.openai_provider import OpenAIProvider, openai_provider


class GatewayDispatchDisabledError(Exception):
    """Raised when upstream dispatch is invoked while ENABLE_UPSTREAM_DISPATCH is False."""

    def __init__(
        self,
        message: str = (
            "Upstream model dispatch is disabled by default. "
            "Set ENABLE_UPSTREAM_DISPATCH=true in the backend environment to enable."
        ),
    ):
        super().__init__(message)
        self.message = message
        self.status_code = 403


class GatewayService:
    """Orchestrates prompt optimization, safety checks, provider dispatch, and cost telemetry."""

    @classmethod
    async def dispatch(
        cls,
        request: GatewayDispatchRequest,
        provider_override: Optional[OpenAIProvider] = None,
        internal_bypass_optimization: bool = False,
    ) -> GatewayDispatchResponse:
        """Process and dispatch a prompt to the configured frontier model.

        1. Validates upstream dispatch enablement.
        2. Applies prompt optimization pipeline exactly once (unless internal programmatic bypass is set).
        3. Preserves original prompt if optimization fallback occurs.
        4. Dispatches the effective prompt to the upstream provider.
        5. Extracts actual token usage and computes verified costs.
        6. Computes heuristic savings estimates with mandatory disclosure.
        """
        # Step 1: Upstream dispatch enablement guardrail
        if not settings.ENABLE_UPSTREAM_DISPATCH:
            raise GatewayDispatchDisabledError()

        # Step 2: Optimization pipeline (preventing double optimization)
        effective_prompt: str
        was_optimized: bool
        opt_summary = None
        opt_latency_ms = 0

        if internal_bypass_optimization:
            # Internal programmatic bypass only (never accessible from public HTTP body)
            effective_prompt = request.prompt
            was_optimized = False
            baseline_estimation = OllamaService.estimate_tokens(effective_prompt)
            baseline_tokens = baseline_estimation.estimated_tokens
            optimized_tokens = baseline_tokens
        else:
            # Standard path: optimize and validate exactly once
            opt_req = OptimizeRequest(prompt=request.prompt, mode=request.mode)
            opt_resp = await OllamaService.optimize_prompt(opt_req)


            effective_prompt = opt_resp.optimized_prompt
            was_optimized = not opt_resp.is_preserved
            opt_latency_ms = opt_resp.execution_time_ms
            baseline_tokens = opt_resp.original_metrics.estimated_tokens
            optimized_tokens = opt_resp.optimized_metrics.estimated_tokens

            opt_summary = {
                "task_category": opt_resp.task_category,
                "strategy": opt_resp.strategy,
                "applied_strategy": opt_resp.applied_strategy,
                "outcome": opt_resp.outcome,
                "validation_status": opt_resp.validation_status,
                "preserved_constraints": opt_resp.preserved_constraints,
                "is_preserved": opt_resp.is_preserved,
            }

        # Step 3: Resolve provider and dispatch prompt
        target_model = (request.target_model or settings.OPENAI_MODEL).strip()

        if provider_override is not None:
            active_provider = provider_override
        elif target_model == openai_provider.model:
            active_provider = openai_provider
        else:
            active_provider = OpenAIProvider(model=target_model)

        exec_result = await active_provider.generate(
            prompt=effective_prompt,
            system_prompt=request.system_prompt,
        )

        # Step 4: Calculate verified costs from actual usage
        cost_result = CostCalculator.calculate_cost(
            model_id=target_model,
            actual_input_tokens=exec_result.input_tokens,
            actual_output_tokens=exec_result.output_tokens,
            cached_tokens=exec_result.cached_tokens,
            reasoning_tokens=exec_result.reasoning_tokens,
        )

        # Step 5: Calculate heuristic savings estimates
        savings_result = CostCalculator.estimate_savings(
            model_id=target_model,
            baseline_estimated_tokens=baseline_tokens,
            optimized_estimated_tokens=optimized_tokens,
        )

        # Step 6: Construct and return typed response
        usage_telemetry = GatewayUsageTelemetry(
            actual_input_tokens=exec_result.input_tokens,
            actual_output_tokens=exec_result.output_tokens,
            actual_total_tokens=exec_result.total_tokens,
            cached_tokens=exec_result.cached_tokens,
            reasoning_tokens=exec_result.reasoning_tokens,
            actual_cost_usd=cost_result.actual_cost_usd,
            actual_input_cost_usd=cost_result.actual_input_cost_usd,
            actual_output_cost_usd=cost_result.actual_output_cost_usd,
            pricing_status=cost_result.pricing_status,
            currency=cost_result.currency,
            pricing_tier=cost_result.pricing_tier,
            pricing_source_url=cost_result.pricing_source_url,
            last_verified_date=cost_result.last_verified_date,
        )

        savings_telemetry = GatewaySavingsTelemetry(
            baseline_estimated_tokens=savings_result.baseline_estimated_tokens,
            optimized_estimated_tokens=savings_result.optimized_estimated_tokens,
            estimated_token_savings=savings_result.estimated_token_savings,
            estimated_cost_savings_usd=savings_result.estimated_cost_savings_usd,
            savings_disclaimer=savings_result.savings_disclaimer,
        )

        return GatewayDispatchResponse(
            model_response=exec_result.content,
            effective_prompt=effective_prompt,
            was_optimized=was_optimized,
            provider=exec_result.provider_name,
            model=exec_result.model,
            execution_latency_ms=int(exec_result.latency_seconds * 1000),
            optimization_latency_ms=opt_latency_ms,
            usage=usage_telemetry,
            estimated_savings=savings_telemetry,
            optimization_summary=opt_summary,
        )
