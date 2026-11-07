"""Intelligent Model Routing Engine for Argus (Phase 6).

Coordinates:
- Task domain classification.
- Multi-indicator prompt complexity evaluation.
- Context window bounds verification.
- Provider eligibility and user constraint filtering.
- Multi-criteria model scoring (Quality, Cost, Latency).
- Fallback chain formation without loops.
- Diagnostic reasoning and rejection logging.
"""

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from app.core.config import settings
from app.services.routing.classifier import ComplexityEstimator, TaskClassifier
from app.services.routing.models import (
    CandidateEvaluation,
    ComplexityScore,
    ModelCapability,
    RoutingConstraints,
    RoutingDecision,
    RoutingPolicy,
    RoutingTaskCategory,
)
from app.services.routing.registry import ModelCapabilityRegistry
from app.services.routing.scorer import ModelScorer


class ModelRoutingEngine:
    """Intelligent router selecting the optimal LLM for each incoming request."""

    def __init__(self, default_policy: Optional[RoutingPolicy] = None) -> None:
        if default_policy is not None:
            self._default_policy = default_policy
        else:
            self._default_policy = RoutingPolicy(
                strategy=getattr(settings, "ROUTING_DEFAULT_STRATEGY", "balanced"),
                weight_quality=getattr(settings, "ROUTING_WEIGHT_QUALITY", 0.40),
                weight_cost=getattr(settings, "ROUTING_WEIGHT_COST", 0.35),
                weight_latency=getattr(settings, "ROUTING_WEIGHT_LATENCY", 0.25),
                fallback_chain=getattr(
                    settings,
                    "ROUTING_FALLBACK_CHAIN",
                    ["gemini-2.0-flash", "deepseek-chat", "qwen2.5:1.5b"],
                ),
            )

    @staticmethod
    def _is_provider_available(provider_name: str, configured_providers: Optional[Dict[str, bool]]) -> bool:
        """Check if a provider is configured and available."""
        if configured_providers is not None and provider_name in configured_providers:
            return configured_providers[provider_name]

        # Check environment configuration for available credentialed services:
        prov = provider_name.lower()
        if prov == "gemini":
            key = getattr(settings, "GEMINI_API_KEY", None)
            return bool(key and not key.startswith("your-") and key != "changeme")
        elif prov == "deepseek":
            key = getattr(settings, "DEEPSEEK_API_KEY", None)
            return bool(key and not key.startswith("your-") and key != "changeme")
        elif prov == "ollama":
            # Local Ollama is configured through local endpoint
            return True
        return False

    def route(
        self,
        prompt: str,
        context_messages: Optional[Sequence[Any]] = None,
        constraints: Optional[RoutingConstraints] = None,
        policy: Optional[RoutingPolicy] = None,
        configured_providers: Optional[Dict[str, bool]] = None,
    ) -> RoutingDecision:
        """Evaluate prompt requirements and return the optimal model routing decision."""
        start_time = time.perf_counter()
        effective_policy = policy or self._default_policy
        effective_constraints = constraints or RoutingConstraints()

        # Step 1: Task classification & complexity estimation
        task_category = TaskClassifier.classify(prompt)
        complexity = ComplexityEstimator.estimate(prompt)

        # Estimate total tokens
        input_tokens = complexity.token_count_estimate
        if context_messages:
            for msg in context_messages:
                content = getattr(msg, "content", "")
                if content:
                    input_tokens += max(1, len(content) // 4)
        estimated_total_tokens = input_tokens + 500  # expected completion buffer

        # Step 2: Evaluate all candidate models in registry
        all_models = ModelCapabilityRegistry.list_models()
        evaluations: List[CandidateEvaluation] = []
        rejections: Dict[str, str] = {}
        eligible_candidates: List[Tuple[ModelCapability, CandidateEvaluation]] = []

        for model in all_models:
            eligible = True
            ineligibility_reason: Optional[str] = None

            # Provider availability check
            if not self._is_provider_available(model.provider_name, configured_providers):
                eligible = False
                ineligibility_reason = f"Provider '{model.provider_name}' is not configured or disabled"

            # Context window check
            elif estimated_total_tokens > model.context_window:
                eligible = False
                ineligibility_reason = (
                    f"Required context ({estimated_total_tokens} tokens) exceeds "
                    f"model limit ({model.context_window} tokens)"
                )

            # User constraints checks
            elif (
                effective_constraints.allowed_providers
                and model.provider_name not in effective_constraints.allowed_providers
            ):
                eligible = False
                ineligibility_reason = f"Provider '{model.provider_name}' not in allowed_providers list"

            elif (
                effective_constraints.disallowed_models
                and model.model_id in effective_constraints.disallowed_models
            ):
                eligible = False
                ineligibility_reason = f"Model '{model.model_id}' is explicitly disallowed"

            elif effective_constraints.requires_coding and model.coding_score < 0.75:
                eligible = False
                ineligibility_reason = f"Model coding capability ({model.coding_score}) below required threshold (0.75)"

            elif effective_constraints.requires_reasoning and model.reasoning_score < 0.85:
                eligible = False
                ineligibility_reason = (
                    f"Model reasoning capability ({model.reasoning_score}) below required threshold (0.85)"
                )

            elif effective_constraints.requires_tools and not model.supports_tools:
                eligible = False
                ineligibility_reason = "Model does not support function calling / tools"

            elif (
                effective_constraints.min_context_window
                and model.context_window < effective_constraints.min_context_window
            ):
                eligible = False
                ineligibility_reason = (
                    f"Model context capacity ({model.context_window}) is below requested "
                    f"minimum ({effective_constraints.min_context_window})"
                )

            # Evaluate candidate
            evaluation = ModelScorer.evaluate_candidate(
                model=model,
                category=task_category,
                complexity=complexity,
                estimated_input_tokens=input_tokens,
                policy=effective_policy,
                eligible=eligible,
                ineligibility_reason=ineligibility_reason,
            )
            evaluations.append(evaluation)

            if eligible:
                # Max cost filter if requested
                if (
                    effective_constraints.max_cost_usd is not None
                    and evaluation.estimated_input_cost_usd is not None
                    and evaluation.estimated_input_cost_usd > effective_constraints.max_cost_usd
                ):
                    rejections[model.model_id] = (
                        f"Estimated cost (${evaluation.estimated_input_cost_usd:.5f}) exceeds "
                        f"maximum allowed (${effective_constraints.max_cost_usd:.5f})"
                    )
                    continue

                # Max latency filter if requested
                if (
                    effective_constraints.max_latency_ms is not None
                    and evaluation.estimated_latency_ms is not None
                    and evaluation.estimated_latency_ms > effective_constraints.max_latency_ms
                ):
                    rejections[model.model_id] = (
                        f"Estimated latency ({evaluation.estimated_latency_ms}ms) exceeds "
                        f"maximum allowed ({effective_constraints.max_latency_ms}ms)"
                    )
                    continue

                eligible_candidates.append((model, evaluation))
            else:
                rejections[model.model_id] = ineligibility_reason or "Ineligible"

        # Step 3: Selection and fallback ranking
        effective_weights = ModelScorer.resolve_effective_weights(effective_policy)

        if eligible_candidates:
            # Sort descending by composite score
            eligible_candidates.sort(key=lambda pair: pair[1].composite_score, reverse=True)
            top_model, top_eval = eligible_candidates[0]
            selected_model = top_model.model_id
            selected_provider = top_model.provider_name

            # Build fallback chain from remaining eligible candidates
            fallback_chain = [pair[0].model_id for pair in eligible_candidates[1:4]]

            selection_rationale = (
                f"Selected '{selected_model}' ({selected_provider}) with composite score {top_eval.composite_score:.3f} "
                f"(quality={top_eval.quality_score:.2f}, cost={top_eval.cost_score:.2f}, latency={top_eval.latency_score:.2f}) "
                f"for task '{task_category.value}' at complexity {complexity.score:.2f} using '{effective_policy.strategy}' strategy."
            )
        else:
            # Safe emergency fallback
            fallback_model_id = getattr(settings, "ROUTING_DEFAULT_MODEL", "gemini-2.0-flash")
            selected_model = fallback_model_id
            cap = ModelCapabilityRegistry.get_model(fallback_model_id)
            selected_provider = cap.provider_name if cap else "gemini"
            fallback_chain = [m for m in effective_policy.fallback_chain if m != selected_model]
            selection_rationale = (
                f"No candidates satisfied all eligibility criteria. Defaulting to safe fallback '{selected_model}'. "
                f"Filter rejections: {len(rejections)} models rejected."
            )

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        return RoutingDecision(
            selected_model=selected_model,
            selected_provider=selected_provider,
            task_category=task_category,
            complexity=complexity,
            applied_strategy=effective_policy.strategy,
            effective_weights=effective_weights,
            candidate_evaluations=evaluations,
            fallback_chain=fallback_chain,
            selection_rationale=selection_rationale,
            rejections=rejections,
            routing_latency_ms=duration_ms,
        )
