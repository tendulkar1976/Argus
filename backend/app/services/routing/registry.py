"""Centralized Model Capability Registry for Intelligent Routing (Phase 6).

Maintains verified capability specifications for supported models across providers with available credentials:
1. Google Gemini
2. DeepSeek
3. Ollama (local endpoint)

TinyFish is strictly isolated as a browser automation capability and is excluded from LLM routing.
Uncredentialed providers (OpenAI, OpenRouter, Groq) are excluded.
"""

from typing import Dict, List, Optional
from app.services.routing.models import CostTier, LatencyTier, ModelCapability


class ModelCapabilityRegistry:
    """Registry maintaining verified capability profiles for model routing."""

    _MODELS: Dict[str, ModelCapability] = {
        # Google Gemini Models (Credentialed Service 1)
        "gemini-2.0-flash": ModelCapability(
            model_id="gemini-2.0-flash",
            provider_name="gemini",
            context_window=1_048_576,
            max_output_tokens=8_192,
            coding_score=0.87,
            reasoning_score=0.86,
            general_quality_score=0.88,
            latency_tier=LatencyTier.FAST,
            typical_latency_ms=750,
            cost_tier=CostTier.BUDGET,
            supports_system_prompt=True,
            supports_tools=True,
        ),
        "gemini-1.5-pro": ModelCapability(
            model_id="gemini-1.5-pro",
            provider_name="gemini",
            context_window=2_097_152,
            max_output_tokens=8_192,
            coding_score=0.90,
            reasoning_score=0.91,
            general_quality_score=0.93,
            latency_tier=LatencyTier.MODERATE,
            typical_latency_ms=2100,
            cost_tier=CostTier.STANDARD,
            supports_system_prompt=True,
            supports_tools=True,
        ),
        "gemini-1.5-flash": ModelCapability(
            model_id="gemini-1.5-flash",
            provider_name="gemini",
            context_window=1_048_576,
            max_output_tokens=8_192,
            coding_score=0.80,
            reasoning_score=0.79,
            general_quality_score=0.84,
            latency_tier=LatencyTier.FAST,
            typical_latency_ms=650,
            cost_tier=CostTier.BUDGET,
            supports_system_prompt=True,
            supports_tools=True,
        ),

        # DeepSeek Models (Credentialed Service 2)
        "deepseek-chat": ModelCapability(
            model_id="deepseek-chat",
            provider_name="deepseek",
            context_window=65_536,
            max_output_tokens=8_192,
            coding_score=0.89,
            reasoning_score=0.88,
            general_quality_score=0.89,
            latency_tier=LatencyTier.MODERATE,
            typical_latency_ms=1600,
            cost_tier=CostTier.BUDGET,
            supports_system_prompt=True,
            supports_tools=True,
        ),
        "deepseek-reasoner": ModelCapability(
            model_id="deepseek-reasoner",
            provider_name="deepseek",
            context_window=65_536,
            max_output_tokens=8_192,
            coding_score=0.96,
            reasoning_score=0.98,
            general_quality_score=0.92,
            latency_tier=LatencyTier.SLOW,
            typical_latency_ms=8000,
            cost_tier=CostTier.STANDARD,
            supports_system_prompt=True,
            supports_tools=False,
        ),

        # Ollama Local Models (Local GPU/CPU, Zero External API Cost)
        "qwen2.5:1.5b": ModelCapability(
            model_id="qwen2.5:1.5b",
            provider_name="ollama",
            context_window=32_768,
            max_output_tokens=8_192,
            coding_score=0.68,
            reasoning_score=0.65,
            general_quality_score=0.72,
            latency_tier=LatencyTier.FAST,
            typical_latency_ms=900,
            cost_tier=CostTier.FREE,
            supports_system_prompt=True,
            supports_tools=False,
        ),
        "llama3.2:3b": ModelCapability(
            model_id="llama3.2:3b",
            provider_name="ollama",
            context_window=131_072,
            max_output_tokens=8_192,
            coding_score=0.72,
            reasoning_score=0.70,
            general_quality_score=0.76,
            latency_tier=LatencyTier.FAST,
            typical_latency_ms=1100,
            cost_tier=CostTier.FREE,
            supports_system_prompt=True,
            supports_tools=False,
        ),
    }

    @classmethod
    def get_model(cls, model_id: str) -> Optional[ModelCapability]:
        """Look up capability profile for a model identifier."""
        return cls._MODELS.get((model_id or "").strip().lower())

    @classmethod
    def list_models(cls) -> List[ModelCapability]:
        """Return list of all registered models."""
        return list(cls._MODELS.values())

    @classmethod
    def get_models_for_provider(cls, provider_name: str) -> List[ModelCapability]:
        """Return models belonging to a specific provider family."""
        target = provider_name.strip().lower()
        return [m for m in cls._MODELS.values() if m.provider_name.lower() == target]
