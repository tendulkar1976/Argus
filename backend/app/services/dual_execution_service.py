"""Dual Prompt Execution and Explainable Comparison Service (Phase 11.1).

Coordinates:
1. Stage 1: Prompt optimization via OllamaService.
2. Stage 2: Transformation breakdown via TransformationExplainer.
3. Stage 3: Independent dual dispatch of Original Prompt (Execution A)
   and Optimized Prompt (Execution B) to the exact same main LLM.
4. Stage 4: Strict equivalence auditing (model, provider, and parameters).
5. Stage 5: Resource accounting and comparative performance metrics.
6. Stage 6: Multi-dimensional response comparison via ResponseComparator.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import time
import uuid
from typing import Optional, Tuple

from app.core.pricing_registry import get_model_pricing
from app.schemas.comparison import (
    ComparisonPerformanceMetrics,
    DualExecutionRequest,
    DualExecutionResponse,
    ExecutionMetricDetails,
    SingleExecutionResult,
)
from app.schemas.optimizer import OptimizeRequest, OptimizeResponse
from app.services.cost_calculator import CostCalculator
from app.services.ollama_service import OllamaService
from app.services.providers.base import BaseLLMProvider, ProviderError, ProviderExecutionResult
from app.services.providers.registry import ProviderRegistry
from app.services.response_comparator import ResponseComparator
from app.services.transformation_explainer import TransformationExplainer

logger = logging.getLogger(__name__)


class DualExecutionService:
    """Manages parallel A/B prompt dispatch, response evaluation, and metrics."""

    @classmethod
    async def execute(cls, request: DualExecutionRequest) -> DualExecutionResponse:
        """Execute end-to-end dual optimization, execution, and evaluation."""
        req_id = f"ab_{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(timezone.utc).isoformat()
        target_model = request.target_model.strip()

        # Step 1: Optimize prompt via Argus local inference pipeline
        opt_req = OptimizeRequest(
            prompt=request.prompt,
            mode=request.mode,  # type: ignore[arg-type]
            target_model=target_model,
        )
        opt_res: OptimizeResponse = await OllamaService.optimize_prompt(opt_req)

        # Step 2: Generate explainable transformation records
        transformations = TransformationExplainer.explain(
            original_prompt=request.prompt,
            optimized_prompt=opt_res.optimized_prompt,
            optimization_metadata=opt_res,
        )

        # Step 3: Resolve provider for configured target model
        provider = ProviderRegistry.get_provider_for_model(target_model)
        target_provider = provider.provider_name

        # Step 4: Parallel Dual Execution (Execution A: Original, Execution B: Optimized)
        # Both executions receive identical system prompts, temperatures, and constraints
        task_a = cls._dispatch_single(
            prompt=request.prompt,
            model=target_model,
            provider=provider,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        task_b = cls._dispatch_single(
            prompt=opt_res.optimized_prompt,
            model=target_model,
            provider=provider,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # Await both executions concurrently
        result_a, result_b = await asyncio.gather(task_a, task_b)

        # Step 5: Equivalence Audit
        is_equivalent = (
            result_a.metrics.model == result_b.metrics.model
            and result_a.metrics.provider == result_b.metrics.provider
            and not (result_a.fallback_used or result_b.fallback_used)
        )
        equivalence_notes = None
        if not is_equivalent:
            equivalence_notes = (
                f"Executions used differing parameters or fallbacks: "
                f"A=({result_a.metrics.provider}:{result_a.metrics.model}), "
                f"B=({result_b.metrics.provider}:{result_b.metrics.model})"
            )

        # Step 6: Comparative Performance & Token Accounting
        performance = cls._calculate_performance(
            opt_res=opt_res,
            exec_a=result_a,
            exec_b=result_b,
        )

        # Step 7: Response Quality Comparison
        comparison = None
        if (
            request.run_comparison
            and result_a.metrics.status == "success"
            and result_b.metrics.status == "success"
            and result_a.response_text
            and result_b.response_text
        ):
            try:
                comparison = await ResponseComparator.compare(
                    original_prompt=request.prompt,
                    optimized_prompt=opt_res.optimized_prompt,
                    response_original=result_a.response_text,
                    response_optimized=result_b.response_text,
                    evaluator_model=target_model,
                    system_instruction=request.system_prompt,
                )
            except Exception as exc:
                logger.error("Failed to run response comparison: %s", str(exc))

        return DualExecutionResponse(
            request_id=req_id,
            timestamp=now_iso,
            target_model=target_model,
            target_provider=target_provider,
            is_equivalent_conditions=is_equivalent,
            equivalence_notes=equivalence_notes,
            optimization=opt_res,
            transformations=transformations,
            original_execution=result_a,
            optimized_execution=result_b,
            comparison=comparison,
            performance=performance,
        )

    @classmethod
    async def _dispatch_single(
        cls,
        prompt: str,
        model: str,
        provider: BaseLLMProvider,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> SingleExecutionResult:
        """Execute a single prompt branch with full error isolation and metric extraction."""
        start_time = time.perf_counter()
        pricing = get_model_pricing(model)

        try:
            exec_res: ProviderExecutionResult = await provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            duration_ms = int((time.perf_counter() - start_time) * 1000)

            # Calculate cost
            cost_usd = None
            cost_status = "unavailable"
            if pricing and exec_res.input_tokens is not None and exec_res.output_tokens is not None:
                calc = CostCalculator.calculate_cost(
                    model=model,
                    input_tokens=exec_res.input_tokens,
                    output_tokens=exec_res.output_tokens,
                    cached_tokens=exec_res.cached_tokens or 0,
                )
                cost_usd = float(calc.total_cost_usd)
                cost_status = "actual" if pricing.input_per_million > 0 else "free"
            elif pricing:
                cost_status = "estimated"

            metrics = ExecutionMetricDetails(
                input_tokens=exec_res.input_tokens,
                output_tokens=exec_res.output_tokens,
                total_tokens=exec_res.total_tokens,
                cached_tokens=exec_res.cached_tokens,
                latency_ms=duration_ms,
                cost_usd=cost_usd,
                cost_status=cost_status,
                provider=exec_res.provider_name,
                model=exec_res.model,
                status="success",
            )

            return SingleExecutionResult(
                prompt=prompt,
                response_text=exec_res.content,
                metrics=metrics,
                fallback_used=False,
            )

        except ProviderError as exc:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            metrics = ExecutionMetricDetails(
                latency_ms=duration_ms,
                cost_usd=None,
                cost_status="unavailable",
                provider=provider.provider_name,
                model=model,
                status="error",
                error_message=exc.message,
            )
            return SingleExecutionResult(
                prompt=prompt,
                response_text="",
                metrics=metrics,
            )

        except Exception as exc:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            metrics = ExecutionMetricDetails(
                latency_ms=duration_ms,
                cost_usd=None,
                cost_status="unavailable",
                provider=provider.provider_name,
                model=model,
                status="error",
                error_message=str(exc),
            )
            return SingleExecutionResult(
                prompt=prompt,
                response_text="",
                metrics=metrics,
            )

    @classmethod
    def _calculate_performance(
        cls,
        opt_res: OptimizeResponse,
        exec_a: SingleExecutionResult,
        exec_b: SingleExecutionResult,
    ) -> ComparisonPerformanceMetrics:
        """Compute relative variance in tokens, costs, and latencies between executions."""
        t_orig = (
            exec_a.metrics.input_tokens
            if exec_a.metrics.input_tokens is not None
            else opt_res.original_metrics.estimated_tokens
        )
        t_opt = (
            exec_b.metrics.input_tokens
            if exec_b.metrics.input_tokens is not None
            else opt_res.optimized_metrics.estimated_tokens
        )

        prompt_saved = t_orig - t_opt
        reduction_pct = ((t_orig - t_opt) / t_orig * 100.0) if t_orig > 0 else 0.0

        resp_orig = exec_a.metrics.output_tokens
        resp_opt = exec_b.metrics.output_tokens
        resp_diff = (resp_opt - resp_orig) if (resp_opt is not None and resp_orig is not None) else None

        tot_orig = exec_a.metrics.total_tokens
        tot_opt = exec_b.metrics.total_tokens
        tot_diff = (tot_opt - tot_orig) if (tot_opt is not None and tot_orig is not None) else None

        cost_orig = exec_a.metrics.cost_usd
        cost_opt = exec_b.metrics.cost_usd
        cost_saved = None
        cost_saved_pct = None
        if cost_orig is not None and cost_opt is not None:
            cost_saved = cost_orig - cost_opt
            cost_saved_pct = (cost_saved / cost_orig * 100.0) if cost_orig > 0 else 0.0

        latency_diff = exec_b.metrics.latency_ms - exec_a.metrics.latency_ms

        return ComparisonPerformanceMetrics(
            prompt_tokens_original=t_orig,
            prompt_tokens_optimized=t_opt,
            prompt_tokens_saved=prompt_saved,
            prompt_token_reduction_percent=round(reduction_pct, 2),
            response_tokens_original=resp_orig,
            response_tokens_optimized=resp_opt,
            response_token_difference=resp_diff,
            total_tokens_original=tot_orig,
            total_tokens_optimized=tot_opt,
            total_token_difference=tot_diff,
            cost_original_usd=round(cost_orig, 6) if cost_orig is not None else None,
            cost_optimized_usd=round(cost_opt, 6) if cost_opt is not None else None,
            cost_saved_usd=round(cost_saved, 6) if cost_saved is not None else None,
            cost_savings_percent=round(cost_saved_pct, 2) if cost_saved_pct is not None else None,
            latency_original_ms=exec_a.metrics.latency_ms,
            latency_optimized_ms=exec_b.metrics.latency_ms,
            latency_difference_ms=latency_diff,
        )
