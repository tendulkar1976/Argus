"""Centralized Provider Registry & Factory for Multi-Provider Architecture (Phase 7).

Manages adapter lifecycle, model-to-provider resolution, availability auditing,
and safe diagnostic status reporting.
"""

from typing import Any, Dict, Optional, Type
from app.services.providers.base import BaseLLMProvider, UnsupportedModelError
from app.services.providers.deepseek_provider import DeepSeekProvider
from app.services.providers.gemini_provider import GeminiProvider
from app.services.providers.ollama_provider import OllamaProvider
from app.services.routing.registry import ModelCapabilityRegistry


class ProviderRegistry:
    """Factory and registry managing multi-provider instances and dispatch resolution."""

    # Exactly available services with credentials / local endpoints:
    _PROVIDER_CLASSES: Dict[str, Type[BaseLLMProvider]] = {
        "gemini": GeminiProvider,
        "deepseek": DeepSeekProvider,
        "ollama": OllamaProvider,
    }

    @classmethod
    def get_provider(
        cls,
        provider_name: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseLLMProvider:
        """Instantiate a provider adapter by provider name."""
        name_key = provider_name.strip().lower()
        provider_cls = cls._PROVIDER_CLASSES.get(name_key)
        if not provider_cls:
            raise UnsupportedModelError(
                f"Provider '{provider_name}' is unsupported or credentials have not been configured. "
                f"Available active providers: gemini, deepseek, ollama."
            )

        if model:
            return provider_cls(model=model, **kwargs)
        return provider_cls(**kwargs)

    @classmethod
    def get_provider_for_model(cls, model_id: str, **kwargs: Any) -> BaseLLMProvider:
        """Resolve model capability registry to instantiate appropriate provider."""
        cap = ModelCapabilityRegistry.get_model(model_id)
        if cap is not None:
            return cls.get_provider(cap.provider_name, model=model_id, **kwargs)

        # Fallback heuristic if model is not registered:
        m_lower = model_id.lower()
        if "gemini" in m_lower:
            return cls.get_provider("gemini", model=model_id, **kwargs)
        elif "deepseek" in m_lower:
            return cls.get_provider("deepseek", model=model_id, **kwargs)
        elif "qwen" in m_lower or "ollama" in m_lower:
            return cls.get_provider("ollama", model=model_id, **kwargs)
        else:
            raise UnsupportedModelError(
                f"Model '{model_id}' is unsupported or credentials have not been configured. "
                f"Available active providers: gemini, deepseek, ollama."
            )

    @classmethod
    def list_configured_providers(cls) -> Dict[str, bool]:
        """Audit configuration status across all supported provider families."""
        statuses: Dict[str, bool] = {}
        for name, p_cls in cls._PROVIDER_CLASSES.items():
            try:
                instance = p_cls()
                statuses[name] = instance.is_configured()
            except Exception:
                statuses[name] = False
        return statuses

    @classmethod
    def get_all_statuses(cls) -> Dict[str, Dict[str, Any]]:
        """Return diagnostic status summary for all providers with masked secrets."""
        diagnostics: Dict[str, Dict[str, Any]] = {}
        for name, p_cls in cls._PROVIDER_CLASSES.items():
            try:
                instance = p_cls()
                diagnostics[name] = instance.get_status()
            except Exception as exc:
                diagnostics[name] = {
                    "provider": name,
                    "configured": False,
                    "error": str(exc),
                }
        return diagnostics
