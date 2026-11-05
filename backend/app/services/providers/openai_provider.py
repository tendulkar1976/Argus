"""OpenAI Provider implementation for Argus.

Uses the official OpenAI Python SDK (openai >= 1.50.0).
Provides isolated client management, credential validation, safe error handling,
and upstream prompt execution with token telemetry.
"""

import logging
import time
from typing import Any, Dict, List, Optional
import openai
from openai import AsyncOpenAI

from app.core.config import settings
from app.services.providers.base import (
    AuthenticationError,
    BaseLLMProvider,
    ConfigurationError,
    MissingCredentialsError,
    ProviderAPIError,
    ProviderConnectionError,
    ProviderExecutionResult,
    ProviderTimeoutError,
    RateLimitError,
    UnsupportedModelError,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """Encapsulates OpenAI SDK integration, client lifecycle, and text generation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        """Initialize provider with environment settings or explicit overrides."""
        self._api_key = api_key if api_key is not None else settings.OPENAI_API_KEY
        raw_base_url = base_url if base_url is not None else getattr(settings, "OPENAI_BASE_URL", None)
        self._base_url = raw_base_url.strip() if raw_base_url and raw_base_url.strip() else None
        self._model = (model or settings.OPENAI_MODEL or "gpt-6.1-sol").strip()
        self._timeout = timeout if timeout is not None else settings.OPENAI_TIMEOUT_SECONDS
        self._max_retries = max_retries if max_retries is not None else settings.OPENAI_MAX_RETRIES
        self._reasoning_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else getattr(settings, "OPENAI_REASONING_EFFORT", "medium")
        )

        self._client: Optional[AsyncOpenAI] = None


    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> Optional[str]:
        return self._base_url

    KNOWN_PLACEHOLDERS = frozenset({
        "your-openai-api-key-here",
        "your_openai_api_key_here",
        "your-api-key-here",
        "your_api_key_here",
        "your-api-key",
        "your_api_key",
        "your-key-here",
        "your_key_here",
        "changeme",
        "placeholder",
        "insert-key-here",
        "insert_key_here",
        "replace-with-your-key",
        "sk-...",
    })

    @classmethod
    def is_placeholder_key(cls, key: str) -> bool:
        """Check if an API key string matches known placeholder patterns.

        Does not reject legitimate keys:
        - Rejection is strictly limited to exact known placeholder strings,
          strings enclosed in template brackets (e.g. '<your-api-key>'),
          or strings starting with 'your-' / 'your_'.
        - Legitimate keys (e.g. starting with 'sk-') are preserved.
        """
        cleaned = key.strip().lower()
        if not cleaned:
            return True
        if cleaned in cls.KNOWN_PLACEHOLDERS:
            return True
        if cleaned.startswith("<") and cleaned.endswith(">"):
            return True
        if cleaned.startswith("your-") or cleaned.startswith("your_"):
            return True
        return False

    REASONING_EFFORT_SUPPORTED_MODELS = frozenset({
        "gpt-6.1-sol",
        "o1",
        "o1-mini",
        "o1-preview",
        "o3-mini",
    })

    @classmethod
    def supports_reasoning_effort(cls, model_id: str) -> bool:
        """Check whether a model explicitly supports the OpenAI reasoning_effort parameter.

        Restricts parameter passing strictly to confirmed OpenAI reasoning models:
        - gpt-6.1-sol
        - o1 family (o1, o1-mini, o1-preview)
        - o3 family (o3-mini)
        Prevents unsupported parameter errors when routing through OpenRouter or standard chat models.
        """
        clean = (model_id or "").strip().lower()
        if not clean:
            return False
        if clean in cls.REASONING_EFFORT_SUPPORTED_MODELS:
            return True
        if clean.startswith(("o1-", "o3-")):
            return True
        return False

    def is_configured(self) -> bool:
        """Return True if an API key is present, non-empty, and not a known placeholder."""
        if not self._api_key or not self._api_key.strip():
            return False
        return not self.is_placeholder_key(self._api_key)

    def get_client(self) -> AsyncOpenAI:
        """Retrieve or initialize the AsyncOpenAI client instance.

        Raises:
            MissingCredentialsError: If OPENAI_API_KEY is not configured, empty, or set to a placeholder.
            ConfigurationError: If client initialization fails.
        """
        if not self.is_configured():
            raise MissingCredentialsError(
                "OpenAI API key is missing or set to an unconfigured placeholder. "
                "Set a valid OPENAI_API_KEY in the backend environment."
            )

        if self._client is None:
            try:
                client_kwargs: Dict[str, Any] = {
                    "api_key": self._api_key.strip(),
                    "timeout": self._timeout,
                    "max_retries": self._max_retries,
                }
                if self._base_url:
                    client_kwargs["base_url"] = self._base_url

                self._client = AsyncOpenAI(**client_kwargs)
            except Exception as exc:
                logger.error("Failed to initialize OpenAI client: %s", type(exc).__name__)
                raise ConfigurationError(f"Failed to initialize OpenAI client: {exc}") from exc

        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderExecutionResult:
        """Dispatch prompt to configured OpenAI model and return typed execution telemetry.

        Guarantees:
        - Never logs or exposes raw API keys, bearer tokens, or full prompt contents.
        - Translates upstream SDK exceptions into safe, typed ProviderError categories.
        - Accurately captures actual token usage, cached tokens, and reasoning tokens.
        - Distinguishes missing usage from zero usage.
        """
        client = self.get_client()

        messages: List[Dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})

        # Build execution payload
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "timeout": self._timeout,
        }

        # Apply reasoning effort if configured and supported
        if self._reasoning_effort and self.supports_reasoning_effort(self.model):
            call_kwargs["reasoning_effort"] = self._reasoning_effort

        # Merge any caller-provided overrides
        for k, v in kwargs.items():
            if v is not None:
                call_kwargs[k] = v

        logger.debug(
            "Dispatching prompt to OpenAI (model=%s, prompt_len=%d)",
            self.model,
            len(prompt),
        )

        start_time = time.perf_counter()
        try:
            response = await client.chat.completions.create(**call_kwargs)
        except openai.AuthenticationError as exc:
            logger.error("OpenAI authentication failed: %s", type(exc).__name__)
            raise AuthenticationError("OpenAI authentication failed. Verify configured OPENAI_API_KEY.") from exc
        except openai.PermissionDeniedError as exc:
            logger.error("OpenAI permission denied: %s", type(exc).__name__)
            raise AuthenticationError("OpenAI permission denied for model or resource.") from exc
        except openai.NotFoundError as exc:
            logger.error("OpenAI model not found: %s", self.model)
            raise UnsupportedModelError(f"Model '{self.model}' is not supported or not found on OpenAI.") from exc
        except openai.RateLimitError as exc:
            logger.warning("OpenAI rate limit encountered.")
            raise RateLimitError("OpenAI rate limit or quota exceeded.") from exc
        except openai.APITimeoutError as exc:
            logger.warning("OpenAI request timed out after %.1fs", self._timeout)
            raise ProviderTimeoutError(f"OpenAI request timed out after {self._timeout}s.") from exc
        except openai.APIConnectionError as exc:
            logger.error("OpenAI connection error: %s", type(exc).__name__)
            raise ProviderConnectionError("Failed to connect to OpenAI API.") from exc
        except openai.BadRequestError as exc:
            logger.error("OpenAI bad request: %s", type(exc).__name__)
            raise ProviderAPIError(f"OpenAI bad request error: {exc.message}", status_code=400) from exc
        except (openai.InternalServerError, openai.APIStatusError) as exc:
            status_code = getattr(exc, "status_code", 502)
            logger.error("OpenAI upstream error (%s)", status_code)
            raise ProviderAPIError(f"OpenAI returned upstream error ({status_code}).", status_code=502) from exc
        except Exception as exc:
            logger.error("Unexpected error during OpenAI dispatch: %s", type(exc).__name__)
            raise ProviderAPIError(f"Unexpected error communicating with OpenAI: {type(exc).__name__}", status_code=502) from exc

        latency_seconds = time.perf_counter() - start_time

        # Extract textual content safely
        content = ""
        raw_finish_reason = None
        if response.choices:
            choice = response.choices[0]
            raw_finish_reason = getattr(choice, "finish_reason", None)
            if choice.message and choice.message.content:
                content = choice.message.content

        # Extract usage data, keeping missing data distinct from zero
        input_tokens = None
        output_tokens = None
        total_tokens = None
        cached_tokens = None
        reasoning_tokens = None

        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)

            # Cached token details
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details:
                cached_tokens = getattr(prompt_details, "cached_tokens", None)

            # Reasoning token details
            completion_details = getattr(usage, "completion_tokens_details", None)
            if completion_details:
                reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)

        return ProviderExecutionResult(
            provider_name=self.provider_name,
            model=self.model,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            latency_seconds=latency_seconds,
            raw_finish_reason=raw_finish_reason,
        )

    def get_status(self) -> Dict[str, Any]:
        """Return safe metadata without revealing secrets or raw API keys."""
        return {
            "provider": self.provider_name,
            "model": self.model,
            "base_url": self._base_url,
            "configured": self.is_configured(),
            "timeout_seconds": self._timeout,
            "max_retries": self._max_retries,
            "reasoning_effort": self._reasoning_effort,
        }

    def __repr__(self) -> str:
        """Safe representation ensuring API keys are never leaked to logs or tracebacks."""
        base_url_info = f", base_url={self._base_url!r}" if self._base_url else ""
        return (
            f"OpenAIProvider(provider={self.provider_name!r}, "
            f"model={self.model!r}{base_url_info}, configured={self.is_configured()})"
        )


# Global singleton instance for injection
openai_provider = OpenAIProvider()
