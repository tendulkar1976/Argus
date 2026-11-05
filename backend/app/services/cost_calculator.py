"""Cost Calculation Service for Frontier Model Dispatches (Phase 4.2).

Enforces strict accounting rules:
- Pricing data is consumed from app.core.pricing_registry (never hardcoded in logic).
- Calculations use decimal-safe arithmetic (Python Decimal) to eliminate float precision errors.
- Cached input tokens are deducted from standard input tokens to prevent double-counting.
- Reasoning tokens are part of completion_tokens and are not double-added.
- Missing or unsupported pricing returns an explicit status rather than fabricating zero cost.
- Heuristic token savings are strictly distinguished from actual API usage and billing.
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Optional
from pydantic import BaseModel, Field

from app.core.pricing_registry import ModelPricingRecord, get_model_pricing


class CostCalculationResult(BaseModel):
    """Result of cost calculation for a model dispatch."""

    pricing_status: str = Field(
        description="Outcome of pricing resolution: 'calculated', 'pricing_unavailable', 'usage_unavailable', or 'unsupported_context_bounds'"
    )
    currency: str = Field(default="USD", description="Currency unit")
    actual_cost_usd: Optional[float] = Field(
        default=None,
        description="Exact cost calculated from verified model rates and upstream usage",
    )
    actual_input_cost_usd: Optional[float] = Field(
        default=None,
        description="Portion of cost attributable to input tokens (including cached tokens)",
    )
    actual_output_cost_usd: Optional[float] = Field(
        default=None,
        description="Portion of cost attributable to output generation tokens",
    )
    pricing_tier: Optional[str] = Field(default=None, description="Applied pricing tier")
    pricing_source_url: Optional[str] = Field(default=None, description="Source URL for verified rates")
    last_verified_date: Optional[str] = Field(default=None, description="Date rates were verified")


class SavingsEstimationResult(BaseModel):
    """Honest separation of heuristic optimization savings from verified billing."""

    baseline_estimated_tokens: int = Field(description="Heuristic estimate of original input prompt tokens")
    optimized_estimated_tokens: int = Field(description="Heuristic estimate of optimized input prompt tokens")
    estimated_token_savings: int = Field(description="Estimated reduction in input tokens (non-negative)")
    estimated_cost_savings_usd: Optional[float] = Field(
        default=None,
        description="Projected input token dollar savings based on verified standard input rate",
    )
    savings_disclaimer: str = Field(
        default=(
            "Estimated token and cost savings are heuristic approximations based on character-to-token ratios. "
            "Actual billable tokens and savings depend on upstream provider tokenizer alignment and local runtime cost."
        ),
        description="Mandatory disclosure explaining heuristic nature of estimated savings",
    )


class CostCalculator:
    """Service providing decimal-safe cost calculations and savings projections."""

    @classmethod
    def calculate_cost(
        cls,
        model_id: str,
        actual_input_tokens: Optional[int],
        actual_output_tokens: Optional[int],
        cached_tokens: Optional[int] = 0,
        reasoning_tokens: Optional[int] = 0,
        pricing_tier: str = "standard",
        pricing_override: Optional[ModelPricingRecord] = None,
    ) -> CostCalculationResult:
        """Calculate total upstream API cost from actual usage and verified rates.

        Guarantees:
        - Never substitutes zero cost for unknown pricing or missing usage.
        - Accurately discounts cached input tokens.
        - Does not double-bill reasoning tokens on top of completion tokens.
        """
        pricing = pricing_override or get_model_pricing(model_id, pricing_tier=pricing_tier)
        if pricing is None:
            return CostCalculationResult(
                pricing_status="pricing_unavailable",
                currency="USD",
                actual_cost_usd=None,
            )

        if actual_input_tokens is None or actual_output_tokens is None:
            return CostCalculationResult(
                pricing_status="usage_unavailable",
                currency=pricing.currency,
                actual_cost_usd=None,
                pricing_tier=pricing.pricing_tier,
                pricing_source_url=pricing.source_url,
                last_verified_date=pricing.last_verified_date,
            )

        # Context limit verification
        if actual_input_tokens > pricing.context_window_tokens:
            return CostCalculationResult(
                pricing_status="unsupported_context_bounds",
                currency=pricing.currency,
                actual_cost_usd=None,
                pricing_tier=pricing.pricing_tier,
                pricing_source_url=pricing.source_url,
                last_verified_date=pricing.last_verified_date,
            )

        one_million = Decimal("1000000")
        total_input = Decimal(actual_input_tokens)
        cached = Decimal(cached_tokens or 0)
        # Ensure cached tokens do not exceed reported input tokens
        cached = min(cached, total_input)
        uncached_input = max(Decimal(0), total_input - cached)

        cached_rate = (
            pricing.cached_input_per_million
            if pricing.cached_input_per_million is not None
            else pricing.input_per_million
        )

        input_cost = (
            (uncached_input * pricing.input_per_million / one_million)
            + (cached * cached_rate / one_million)
        )

        # Note on reasoning tokens:
        # In OpenAI's Chat Completions API, completion_tokens includes reasoning_tokens.
        # Reasoning tokens are billed at the standard output rate for that model tier.
        # Adding reasoning_tokens to completion_tokens would double-count them.
        total_output = Decimal(actual_output_tokens)
        output_cost = total_output * pricing.output_per_million / one_million

        total_cost = input_cost + output_cost

        # Round to 6 decimal places for dollar precision
        quantize_format = Decimal("0.000001")
        total_cost_rounded = float(total_cost.quantize(quantize_format, rounding=ROUND_HALF_UP))
        input_cost_rounded = float(input_cost.quantize(quantize_format, rounding=ROUND_HALF_UP))
        output_cost_rounded = float(output_cost.quantize(quantize_format, rounding=ROUND_HALF_UP))

        return CostCalculationResult(
            pricing_status="calculated",
            currency=pricing.currency,
            actual_cost_usd=total_cost_rounded,
            actual_input_cost_usd=input_cost_rounded,
            actual_output_cost_usd=output_cost_rounded,
            pricing_tier=pricing.pricing_tier,
            pricing_source_url=pricing.source_url,
            last_verified_date=pricing.last_verified_date,
        )

    @classmethod
    def estimate_savings(
        cls,
        model_id: str,
        baseline_estimated_tokens: int,
        optimized_estimated_tokens: int,
        pricing_tier: str = "standard",
        pricing_override: Optional[ModelPricingRecord] = None,
    ) -> SavingsEstimationResult:
        """Project potential cost savings by comparing like-for-like heuristic prompt token estimates.

        Never compares heuristic tokens to actual upstream tokens or claims guaranteed dollar savings.
        """
        token_savings = max(0, baseline_estimated_tokens - optimized_estimated_tokens)

        pricing = pricing_override or get_model_pricing(model_id, pricing_tier=pricing_tier)
        cost_savings: Optional[float] = None
        if pricing is not None and token_savings > 0:
            one_million = Decimal("1000000")
            raw_savings = Decimal(token_savings) * pricing.input_per_million / one_million
            cost_savings = float(raw_savings.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))

        return SavingsEstimationResult(
            baseline_estimated_tokens=baseline_estimated_tokens,
            optimized_estimated_tokens=optimized_estimated_tokens,
            estimated_token_savings=token_savings,
            estimated_cost_savings_usd=cost_savings,
        )
