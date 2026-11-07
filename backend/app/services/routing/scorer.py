"""Model scoring and ranking engine for Intelligent Model Routing (Phase 6).

Evaluates:
- Quality/capability fit based on domain and complexity.
- Cost efficiency derived from verified pricing registry.
- Latency score derived from model latency tier and typical latency.
- Composite multi-objective ranking based on configured policy weights.
"""

from decimal import Decimal
from typing import Dict, Optional, Tuple
from app.core.pricing_registry import get_model_pricing
from app.services.routing.models import (
    CandidateEvaluation,
    ComplexityScore,
    CostTier,
    LatencyTier,
    ModelCapability,
    RoutingPolicy,
    RoutingTaskCategory,
)


class ModelScorer:
    """Computes normalized multidimensional scores for candidate models."""

    @classmethod
    def compute_quality_score(
        cls,
        model: ModelCapability,
        category: RoutingTaskCategory,
        complexity: ComplexityScore,
    ) -> float:
        """Calculate quality alignment score between model capabilities and task requirements."""
        if category in {RoutingTaskCategory.CODING, RoutingTaskCategory.DATA_ANALYSIS}:
            base_score = model.coding_score
        elif category in {RoutingTaskCategory.REASONING, RoutingTaskCategory.MATHEMATICS}:
            base_score = model.reasoning_score
        else:
            base_score = model.general_quality_score

        # Complexity scaling:
        # High complexity (>0.6) rewards models with higher reasoning capabilities
        # Low complexity (<0.3) can be well satisfied by budget/local models
        if complexity.score > 0.6:
            scaling = 0.8 + 0.2 * model.reasoning_score
            quality = base_score * scaling
        elif complexity.score < 0.3:
            # For simple tasks, any adequate model scores well
            quality = min(1.0, base_score + 0.15)
        else:
            quality = base_score

        return max(0.0, min(1.0, round(quality, 4)))

    @classmethod
    def compute_cost_score(
        cls,
        model: ModelCapability,
        estimated_input_tokens: int,
        estimated_output_tokens: int = 500,
    ) -> Tuple[float, Optional[float]]:
        """Calculate cost efficiency score (1.0 = free/cheapest, 0.0 = highly expensive)."""
        pricing = get_model_pricing(model.model_id)
        if pricing is None:
            # If not in pricing registry, score by cost tier
            tier_scores = {
                CostTier.FREE: 1.0,
                CostTier.BUDGET: 0.85,
                CostTier.STANDARD: 0.50,
                CostTier.PREMIUM: 0.20,
            }
            return tier_scores.get(model.cost_tier, 0.50), None

        input_price = pricing.input_per_million
        output_price = pricing.output_per_million

        cost_decimal = (
            (Decimal(estimated_input_tokens) / Decimal(1_000_000)) * input_price
            + (Decimal(estimated_output_tokens) / Decimal(1_000_000)) * output_price
        )
        cost_usd = float(cost_decimal)

        if cost_usd == 0.0:
            return 1.0, 0.0

        # Score decays with cost: $0.001 -> ~0.91, $0.01 -> ~0.50, $0.10 -> ~0.09
        cost_score = 1.0 / (1.0 + cost_usd * 100.0)
        return max(0.05, min(1.0, round(cost_score, 4))), round(cost_usd, 6)

    @classmethod
    def compute_latency_score(cls, model: ModelCapability) -> Tuple[float, int]:
        """Calculate latency score (1.0 = fastest/lowest latency, 0.0 = slow)."""
        tier_scores = {
            LatencyTier.ULTRA_FAST: 1.0,
            LatencyTier.FAST: 0.85,
            LatencyTier.MODERATE: 0.60,
            LatencyTier.SLOW: 0.30,
        }
        score = tier_scores.get(model.latency_tier, 0.50)
        return score, model.typical_latency_ms

    @classmethod
    def resolve_effective_weights(cls, policy: RoutingPolicy) -> Dict[str, float]:
        """Determine normalized scoring weights based on active strategy."""
        strat = (policy.strategy or "balanced").strip().lower()
        if strat == "cost_optimized":
            return {"quality": 0.25, "cost": 0.60, "latency": 0.15}
        elif strat == "performance":
            return {"quality": 0.35, "cost": 0.10, "latency": 0.55}
        elif strat == "quality":
            return {"quality": 0.70, "cost": 0.15, "latency": 0.15}
        else:
            # Balanced: normalize policy weights
            total = policy.weight_quality + policy.weight_cost + policy.weight_latency
            if total <= 0:
                total = 1.0
            return {
                "quality": policy.weight_quality / total,
                "cost": policy.weight_cost / total,
                "latency": policy.weight_latency / total,
            }

    @classmethod
    def evaluate_candidate(
        cls,
        model: ModelCapability,
        category: RoutingTaskCategory,
        complexity: ComplexityScore,
        estimated_input_tokens: int,
        policy: RoutingPolicy,
        eligible: bool,
        ineligibility_reason: Optional[str] = None,
    ) -> CandidateEvaluation:
        """Compute full evaluation metrics for a candidate model."""
        if not eligible:
            return CandidateEvaluation(
                model_id=model.model_id,
                provider_name=model.provider_name,
                eligible=False,
                ineligibility_reason=ineligibility_reason or "Ineligible based on constraints or provider status",
                quality_score=0.0,
                cost_score=0.0,
                latency_score=0.0,
                composite_score=0.0,
            )

        quality = cls.compute_quality_score(model, category, complexity)
        cost_score, estimated_cost_usd = cls.compute_cost_score(model, estimated_input_tokens)
        latency_score, estimated_latency_ms = cls.compute_latency_score(model)

        weights = cls.resolve_effective_weights(policy)
        composite = (
            weights["quality"] * quality
            + weights["cost"] * cost_score
            + weights["latency"] * latency_score
        )

        return CandidateEvaluation(
            model_id=model.model_id,
            provider_name=model.provider_name,
            eligible=True,
            ineligibility_reason=None,
            quality_score=quality,
            cost_score=cost_score,
            latency_score=latency_score,
            composite_score=round(composite, 4),
            estimated_input_cost_usd=estimated_cost_usd,
            estimated_latency_ms=estimated_latency_ms,
        )
