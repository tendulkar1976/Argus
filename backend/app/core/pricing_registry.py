"""Centralized Model Pricing Registry for Frontier Providers (Phase 4.2).

Enforces strict zero-hardcoding and data provenance standards:
- All pricing figures are verified against official provider documentation.
- Pricing data is kept completely decoupled from calculation logic.
- Provenance metadata (tier, source URL, last verification date, context limits) is recorded.
- No pricing tiers, rates, or billing rules are ever invented or assumed.
"""

from decimal import Decimal
from typing import Dict, Optional
from pydantic import BaseModel, Field


class ModelPricingRecord(BaseModel):
    """Immutable, auditable pricing and token boundary specification for a model."""

    model_id: str = Field(description="Unique model identifier matching provider API")
    provider: str = Field(description="Provider family name (e.g. openai)")
    pricing_tier: str = Field(default="standard", description="Pricing tier (standard, flex, batch, fast)")
    currency: str = Field(default="USD", description="Currency code")
    input_per_million: Decimal = Field(description="Cost in currency per 1,000,000 standard input tokens")
    output_per_million: Decimal = Field(description="Cost in currency per 1,000,000 standard output tokens")
    cached_input_per_million: Optional[Decimal] = Field(
        default=None,
        description="Cost in currency per 1,000,000 cached input read tokens, if supported",
    )
    cache_write_per_million: Optional[Decimal] = Field(
        default=None,
        description="Cost in currency per 1,000,000 cache write tokens, if supported",
    )
    context_window_tokens: int = Field(description="Maximum total context window capacity")
    max_output_tokens: int = Field(description="Maximum generation output capacity")
    source_url: str = Field(description="Official URL where this pricing was verified")
    last_verified_date: str = Field(description="ISO-8601 date when pricing was confirmed (YYYY-MM-DD)")

    model_config = {
        "arbitrary_types_allowed": True,
        "frozen": True,
    }


# Centralized, verified pricing table.
# Verified against official OpenAI documentation as of 2026-10-30.
VERIFIED_PRICING_REGISTRY: Dict[str, Dict[str, ModelPricingRecord]] = {
    "gpt-6.1-sol": {
        "standard": ModelPricingRecord(
            model_id="gpt-6.1-sol",
            provider="openai",
            pricing_tier="standard",
            currency="USD",
            input_per_million=Decimal("2.00"),
            output_per_million=Decimal("10.00"),
            cached_input_per_million=Decimal("0.10"),
            cache_write_per_million=Decimal("2.50"),
            context_window_tokens=1_048_576,  # 1.05M tokens
            max_output_tokens=131_072,        # 128K tokens
            source_url="https://platform.openai.com/docs/pricing",
            last_verified_date="2026-10-30",
        )
    }
}


def get_model_pricing(
    model_id: str,
    pricing_tier: str = "standard",
) -> Optional[ModelPricingRecord]:
    """Retrieve verified pricing record for a model and tier.

    Returns None if the model or tier has not been officially verified.
    """
    model_key = (model_id or "").strip().lower()
    tier_key = (pricing_tier or "standard").strip().lower()

    tiers = VERIFIED_PRICING_REGISTRY.get(model_key)
    if not tiers:
        return None

    return tiers.get(tier_key)
