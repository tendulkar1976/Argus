"""DeepSeek Provider implementation for Argus (Phase 7).

Uses OpenAI-compatible AsyncOpenAI SDK configured with DeepSeek endpoint.
Supports:
- Standard chat models (deepseek-chat)
- Deep reasoning models (deepseek-reasoner / R1) with reasoning tokens telemetry
- Prompt caching telemetry (cache hit vs miss)
- Secure credential handling and normalized exceptions
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


class DeepSeekProvider(BaseLLMProvider):
    """Encapsulates DeepSeek API integration via OpenAI-compatible SDK."""

    KNOWN_PLACEHOLDERS = frozenset({
        "your-deepseek-api-key-here",
        "your_deepseek_api_key_here",
        "your-api-key-here",
        "your_api_key_here",
        "changeme",
        "placeholder",
    })

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 2,
    ) -> None:
        self._api_key = api_key if api_key is not None else getattr(settings, "DEEPSEEK_API_KEY", None)
        self._base_url = (
            base_url
            if base_url is not None
            else getattr(settings, "DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        ).rstrip("/")
        self._model = (model or getattr(settings, "DEEPSEEK_DEFAULT_MODEL", "deepseek-chat")).strip()
        self._timeout = timeout if timeout is not None else getattr(settings, "DEEPSEEK_TIMEOUT_SECONDS", 60.0)
        self._max_retries = max_retries
        self._client: Optional[AsyncOpenAI] = None

    @property
    def provider_name(self) -> str:
        return "deepseek"

    @property
    def model(self) -> str:
        return self._model

    @classmethod
    def is_placeholder_key(cls, key: str) -> bool:
        cleaned = (key or "").strip().lower()
        if not cleaned:
            return True
        if cleaned in cls.KNOWN_PLACEHOLDERS:
            return True
        if cleaned.startswith("your-") or cleaned.startswith("your_"):
            return True
        return False

    def is_configured(self) -> bool:
        if not self._api_key:
            return False
        return not self.is_placeholder_key(self._api_key)

    def _get_client(self) -> AsyncOpenAI:
        if not self.is_configured():
            raise MissingCredentialsError(
                "DeepSeek API key is not configured. Set DEEPSEEK_API_KEY in the environment."
            )
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        return self._client

    def get_status(self) -> Dict[str, Any]:
        has_key = self.is_configured()
        masked_key = None
        if has_key and self._api_key:
            trimmed = self._api_key.strip()
            if len(trimmed) > 8:
                masked_key = f"{trimmed[:4]}...{trimmed[-4:]}"
            else:
                masked_key = "***"

        return {
            "provider": self.provider_name,
            "configured": has_key,
            "model": self._model,
            "base_url": self._base_url,
            "masked_key": masked_key,
            "timeout_seconds": self._timeout,
        }

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderExecutionResult:
        client = self._get_client()

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        request_kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "timeout": self._timeout,
        }
        if "temperature" in kwargs:
            request_kwargs["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            request_kwargs["max_tokens"] = kwargs["max_tokens"]

        start_time = time.perf_counter()
        try:
            response = await client.chat.completions.create(**request_kwargs)
        except openai.AuthenticationError as exc:
            raise AuthenticationError(f"DeepSeek authentication failed: {str(exc)}") from exc
        except openai.RateLimitError as exc:
            raise RateLimitError(f"DeepSeek rate limit exceeded: {str(exc)}") from exc
        except openai.APITimeoutError as exc:
            raise ProviderTimeoutError(f"DeepSeek request timed out: {str(exc)}") from exc
        except openai.APIConnectionError as exc:
            raise ProviderConnectionError(f"Failed to connect to DeepSeek: {str(exc)}") from exc
        except openai.NotFoundError as exc:
            raise UnsupportedModelError(f"Model '{self._model}' not found in DeepSeek: {str(exc)}") from exc
        except openai.BadRequestError as exc:
            raise ProviderAPIError(f"Bad request to DeepSeek: {str(exc)}", status_code=400) from exc
        except openai.APIError as exc:
            raise ProviderAPIError(f"DeepSeek API error: {str(exc)}", status_code=getattr(exc, "status_code", 502)) from exc
        except Exception as exc:
            raise ProviderAPIError(f"Unexpected error calling DeepSeek: {str(exc)}") from exc

        duration = time.perf_counter() - start_time

        try:
            choice = response.choices[0]
            message_obj = choice.message
            content = message_obj.content or ""
            finish_reason = choice.finish_reason

            # Extract reasoning tokens and prompt cache tokens
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else None
            output_tokens = usage.completion_tokens if usage else None
            total_tokens = usage.total_tokens if usage else None

            cached_tokens = None
            if usage and hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", None)

            reasoning_tokens = None
            if usage and hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", None)

            return ProviderExecutionResult(
                provider_name=self.provider_name,
                model=self._model,
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
                latency_seconds=duration,
                raw_finish_reason=str(finish_reason) if finish_reason else None,
            )
        except (IndexError, AttributeError) as exc:
            raise ProviderAPIError(f"Malformed response payload from DeepSeek: {str(exc)}") from exc
