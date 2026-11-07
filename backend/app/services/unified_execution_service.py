"""Unified Execution Service orchestrating RABCO, Intelligent Routing, and Providers (Phase 10).

Connects:
1. Input validation & correlation tracking.
2. Multi-turn RABCO context optimization with strict safety bypass guarantees.
3. Intelligent model routing and complexity assessment.
4. Provider adapter dispatch with graceful fallback.
5. Unified telemetry recording and cost tracking.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Type
from app.core.config import settings
from app.research.rabco.orchestrator import OrchestrationDecision, RABCOOrchestrator
from app.schemas.execution import UnifiedExecutionRequest, UnifiedExecutionResponse
from app.schemas.rabco import ConversationMessage
from app.services.providers.base import BaseLLMProvider, ProviderError, ProviderExecutionResult
from app.services.providers.registry import ProviderRegistry
from app.services.routing.engine import ModelRoutingEngine
from app.services.routing.models import RoutingDecision, RoutingPolicy
from app.services.telemetry.service import TelemetryService

logger = logging.getLogger(__name__)


class UnifiedExecutionService:
    """Orchestrates end-to-end execution across the full Argus stack."""

    def __init__(
        self,
        routing_engine: Optional[ModelRoutingEngine] = None,
        rabco_orchestrator: Optional[RABCOOrchestrator] = None,
        telemetry_service: Optional[TelemetryService] = None,
        provider_registry: Type[ProviderRegistry] = ProviderRegistry,
    ) -> None:
        self.routing_engine = routing_engine or ModelRoutingEngine()
        self.rabco_orchestrator = rabco_orchestrator or RABCOOrchestrator()
        self.telemetry_service = telemetry_service or TelemetryService.get_instance()
        self.provider_registry = provider_registry

    async def execute(self, request: UnifiedExecutionRequest) -> UnifiedExecutionResponse:
        """Execute request through RABCO optimization, routing, and provider dispatch."""
        req_id = request.get_effective_request_id()
        start_time = time.perf_counter()

        # Step 1: Input validation
        if not request.prompt and not request.messages:
            raise ValueError("Invalid request: either 'prompt' or 'messages' must be provided.")

        effective_prompt: str = ""
        rabco_diagnostics: Optional[Dict[str, Any]] = None
        rabco_processing_ms: Optional[int] = None

        # Step 2: Context Optimization via RABCO (multi-turn)
        if request.messages and request.enable_rabco:
            rabco_start = time.perf_counter()
            rabco_result = self.rabco_orchestrator.orchestrate(
                messages=tuple(request.messages),
                raise_on_failure=False,
            )
            rabco_processing_ms = int((time.perf_counter() - rabco_start) * 1000)

            rabco_diagnostics = {
                "decision": rabco_result.decision.value,
                "bypassed": rabco_result.bypassed,
                "bypass_reason": rabco_result.bypass_reason,
                "failed_stage": rabco_result.failed_stage.value if rabco_result.failed_stage else None,
                "token_reduction_percent": rabco_result.phase5_result.token_reduction_percent,
                "original_tokens": rabco_result.phase5_result.original_token_estimate,
                "optimized_tokens": rabco_result.phase5_result.optimized_token_estimate,
                "final_preservation_status": rabco_result.phase5_result.final_preservation_status,
            }

            # Assemble effective prompt text from final context messages
            final_messages = rabco_result.final_context_messages
            text_parts = [f"[{msg.role.value.upper()}]: {msg.content}" for msg in final_messages]
            effective_prompt = "\n".join(text_parts)
            if request.prompt:
                effective_prompt = f"{effective_prompt}\n[USER]: {request.prompt}"

        elif request.messages and not request.enable_rabco:
            text_parts = [f"[{msg.role.value.upper()}]: {msg.content}" for msg in request.messages]
            effective_prompt = "\n".join(text_parts)
            if request.prompt:
                effective_prompt = f"{effective_prompt}\n[USER]: {request.prompt}"
        else:
            effective_prompt = request.prompt or ""

        # Step 3: Model Routing Selection
        task_category = "general"
        routing_diagnostics: Optional[Dict[str, Any]] = None
        fallback_chain: List[str] = []

        if request.model:
            # Explicit user override bypasses dynamic routing
            selected_model = request.model.strip()
            if request.provider:
                selected_provider = request.provider.strip().lower()
            else:
                prov_adapter = self.provider_registry.get_provider_for_model(selected_model)
                selected_provider = prov_adapter.provider_name

            routing_diagnostics = {
                "manual_override": True,
                "selected_model": selected_model,
                "selected_provider": selected_provider,
            }
        else:
            # Dynamic intelligent routing
            configured_status = self.provider_registry.list_configured_providers()
            policy = RoutingPolicy(strategy=request.strategy)
            routing_decision = self.routing_engine.route(
                prompt=effective_prompt,
                context_messages=request.messages,
                constraints=request.constraints,
                policy=policy,
                configured_providers=configured_status,
            )
            selected_model = routing_decision.selected_model
            selected_provider = routing_decision.selected_provider
            task_category = routing_decision.task_category.value
            fallback_chain = list(routing_decision.fallback_chain)
            routing_diagnostics = {
                "task_category": task_category,
                "complexity_score": routing_decision.complexity.score,
                "applied_strategy": routing_decision.applied_strategy,
                "selection_rationale": routing_decision.selection_rationale,
                "effective_weights": routing_decision.effective_weights,
                "fallback_chain": fallback_chain,
            }

        # Step 4: Provider Adapter Dispatch with Controlled Fallback
        exec_result: Optional[ProviderExecutionResult] = None
        fallback_used = False
        fallback_from: Optional[str] = None
        attempts = 0
        last_error: Optional[Exception] = None

        candidate_queue = [selected_model] + [m for m in fallback_chain if m != selected_model]

        for target_model in candidate_queue:
            attempts += 1
            try:
                provider_inst = self.provider_registry.get_provider_for_model(target_model)
                exec_result = await provider_inst.generate(
                    prompt=effective_prompt,
                    system_prompt=request.system_prompt,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
                if target_model != selected_model:
                    fallback_used = True
                    fallback_from = selected_model
                    selected_model = target_model
                    selected_provider = provider_inst.provider_name
                break
            except ProviderError as err:
                last_error = err
                logger.warning(
                    "Model dispatch attempt %d failed for model '%s': %s",
                    attempts,
                    target_model,
                    str(err),
                )
                if not fallback_chain:
                    break
            except Exception as err:
                last_error = err
                logger.error(
                    "Unexpected dispatch exception for model '%s': %s",
                    target_model,
                    str(err),
                )
                break

        if exec_result is None:
            # Record failed execution in telemetry before raising
            total_duration_ms = int((time.perf_counter() - start_time) * 1000)
            self.telemetry_service.record_execution(
                request_id=req_id,
                provider=selected_provider,
                model=selected_model,
                latency_ms=total_duration_ms,
                status="error",
                task_category=task_category,
                routing_decision=routing_diagnostics,
                error_category=type(last_error).__name__ if last_error else "ProviderError",
                retry_count=attempts - 1,
                fallback_used=fallback_used,
                fallback_model_from=fallback_from,
                rabco_processing_time_ms=rabco_processing_ms,
            )
            if last_error:
                raise last_error
            raise ProviderError("All candidate model dispatches failed.")

        # Step 5: Telemetry and Cost Calculation
        total_duration_ms = int((time.perf_counter() - start_time) * 1000)
        telemetry_rec = self.telemetry_service.record_execution(
            request_id=req_id,
            provider=selected_provider,
            model=selected_model,
            latency_ms=total_duration_ms,
            status="success",
            task_category=task_category,
            routing_decision=routing_diagnostics,
            input_tokens=exec_result.input_tokens,
            output_tokens=exec_result.output_tokens,
            cached_tokens=exec_result.cached_tokens,
            total_tokens=exec_result.total_tokens,
            reasoning_tokens=exec_result.reasoning_tokens,
            retry_count=attempts - 1,
            fallback_used=fallback_used,
            fallback_model_from=fallback_from,
            rabco_processing_time_ms=rabco_processing_ms,
        )

        cost_usd = telemetry_rec.estimated_cost_usd if telemetry_rec else None
        cost_status = telemetry_rec.cost_status if telemetry_rec else "unrecorded"

        usage_dict = {
            "input_tokens": exec_result.input_tokens,
            "output_tokens": exec_result.output_tokens,
            "total_tokens": exec_result.total_tokens,
            "cached_tokens": exec_result.cached_tokens,
            "reasoning_tokens": exec_result.reasoning_tokens,
        }

        # Step 6: Assemble Unified Response
        return UnifiedExecutionResponse(
            request_id=req_id,
            model_response=exec_result.content,
            effective_prompt=effective_prompt,
            selected_model=selected_model,
            selected_provider=selected_provider,
            fallback_used=fallback_used,
            fallback_model_from=fallback_from,
            execution_latency_ms=total_duration_ms,
            routing=routing_diagnostics,
            rabco=rabco_diagnostics,
            telemetry={"telemetry_recorded": telemetry_rec is not None},
            usage=usage_dict,
            cost_usd=cost_usd,
            cost_status=cost_status,
        )
