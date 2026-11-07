"""Base interface for frontier LLM providers in Argus.

Enforces provider isolation so multiple providers (OpenAI, Anthropic, Gemini, etc.)
can be swapped or added without modifying core middleware logic.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ProviderError(Exception):
    """Base exception for all LLM provider errors."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class MissingCredentialsError(ProviderError):
    """Raised when required provider API keys or credentials are not configured."""

    def __init__(self, message: str = "Provider credentials are not configured."):
        super().__init__(message, status_code=503)


class ConfigurationError(ProviderError):
    """Raised when provider configuration is invalid or missing."""

    def __init__(self, message: str = "Provider configuration error."):
        super().__init__(message, status_code=500)


class AuthenticationError(ProviderError):
    """Raised when upstream authentication fails (e.g. invalid API key)."""

    def __init__(self, message: str = "Upstream provider authentication failed."):
        super().__init__(message, status_code=401)


class RateLimitError(ProviderError):
    """Raised when upstream rate limit or quota is exceeded."""

    def __init__(self, message: str = "Upstream provider rate limit exceeded."):
        super().__init__(message, status_code=429)


class ProviderTimeoutError(ProviderError):
    """Raised when upstream provider request times out."""

    def __init__(self, message: str = "Upstream provider request timed out."):
        super().__init__(message, status_code=504)


class ProviderConnectionError(ProviderError):
    """Raised when connection to upstream provider fails."""

    def __init__(self, message: str = "Failed to connect to upstream provider."):
        super().__init__(message, status_code=502)


class UnsupportedModelError(ProviderError):
    """Raised when the requested model is not supported by the provider."""

    def __init__(self, message: str = "Requested model is unsupported or not found."):
        super().__init__(message, status_code=400)


class ProviderAPIError(ProviderError):
    """Raised for general non-retryable upstream API errors."""

    def __init__(self, message: str = "Upstream provider API error.", status_code: int = 502):
        super().__init__(message, status_code=status_code)


class ProviderExecutionResult(BaseModel):
    """Typed internal execution result returned by a provider implementation."""

    provider_name: str = Field(description="Name of the provider executing the request")
    model: str = Field(description="Actual model identifier used for generation")
    content: str = Field(description="Generated textual content")
    input_tokens: Optional[int] = Field(default=None, description="Actual prompt input tokens")
    output_tokens: Optional[int] = Field(default=None, description="Actual generated output tokens")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens consumed")
    cached_tokens: Optional[int] = Field(default=None, description="Tokens read from prompt cache")
    reasoning_tokens: Optional[int] = Field(default=None, description="Tokens consumed during reasoning")
    latency_seconds: float = Field(default=0.0, description="End-to-end provider execution duration in seconds")
    raw_finish_reason: Optional[str] = Field(default=None, description="Upstream finish reason (e.g. stop, length)")


class BaseLLMProvider(ABC):
    """Abstract interface defining required provider behaviors."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Machine-readable name of the provider (e.g. 'openai')."""
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Target model identifier (e.g. 'gpt-6.1-sol')."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if required credentials and client can be initialized."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return non-sensitive status summary for health/diagnostics.

        Must NEVER expose secret API keys or credentials.
        """
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderExecutionResult:
        """Execute text generation on the upstream provider.

        Must return a ProviderExecutionResult or raise a subclass of ProviderError.
        """
        pass

    async def check_availability(self) -> bool:
        """Check if provider is online and available. Defaults to is_configured()."""
        return self.is_configured()

    def get_capabilities(self, model_id: Optional[str] = None) -> Optional[Any]:
        """Return registered capability specification for target model, if available."""
        from app.services.routing.registry import ModelCapabilityRegistry
        target = model_id or self.model
        return ModelCapabilityRegistry.get_model(target)

