"""Ollama Local LLM Provider implementation for Argus (Phase 7).

Integrates local Ollama instances (e.g. qwen2.5:1.5b, llama3.2:3b) as standard
downstream generation models under the BaseLLMProvider interface.
Guarantees:
- Zero external API costs.
- Native token counts (prompt_eval_count, eval_count).
- Full local offline inference.
"""

import logging
import time
from typing import Any, Dict, List, Optional
import httpx

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


class OllamaProvider(BaseLLMProvider):
    """Encapsulates Ollama local HTTP API (/api/chat) for general model generation."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._base_url = (
            base_url
            if base_url is not None
            else getattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        ).rstrip("/")
        self._model = (model or getattr(settings, "OLLAMA_MODEL", "qwen2.5:1.5b")).strip()
        self._timeout = timeout if timeout is not None else getattr(settings, "OLLAMA_LLM_TIMEOUT_SECONDS", 60.0)

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    def is_configured(self) -> bool:
        # Local Ollama requires a valid base URL; no API key needed
        return bool(self._base_url)

    async def check_availability(self) -> bool:
        """Ping Ollama status endpoint to verify daemon is running."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self._base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    def get_status(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "configured": self.is_configured(),
            "model": self._model,
            "base_url": self._base_url,
            "timeout_seconds": self._timeout,
        }

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderExecutionResult:
        endpoint = f"{self._base_url}/api/chat"

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }

        options: Dict[str, Any] = {}
        if "temperature" in kwargs:
            options["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            options["top_p"] = kwargs["top_p"]
        if options:
            payload["options"] = options

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(endpoint, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Ollama request timed out after {self._timeout}s: {str(exc)}"
            ) from exc
        except httpx.NetworkError as exc:
            raise ProviderConnectionError(
                f"Failed to connect to local Ollama daemon at {self._base_url}: {str(exc)}"
            ) from exc
        except Exception as exc:
            raise ProviderAPIError(f"Unexpected error communicating with Ollama: {str(exc)}") from exc

        duration = time.perf_counter() - start_time

        if response.status_code != 200:
            if response.status_code == 404:
                raise UnsupportedModelError(
                    f"Model '{self._model}' not found in Ollama. Pull it using 'ollama pull {self._model}'."
                )
            raise ProviderAPIError(
                f"Ollama returned error status {response.status_code}: {response.text}",
                status_code=response.status_code,
            )

        try:
            data = response.json()
            message = data.get("message", {})
            content = message.get("content", "")

            # Extract token telemetry
            prompt_tokens = data.get("prompt_eval_count")
            output_tokens = data.get("eval_count")
            total_tokens = None
            if prompt_tokens is not None and output_tokens is not None:
                total_tokens = prompt_tokens + output_tokens

            return ProviderExecutionResult(
                provider_name=self.provider_name,
                model=self._model,
                content=content,
                input_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_tokens=0,
                reasoning_tokens=None,
                latency_seconds=duration,
                raw_finish_reason=data.get("done_reason") or "stop",
            )
        except Exception as exc:
            raise ProviderAPIError(f"Failed to parse Ollama response: {str(exc)}") from exc
