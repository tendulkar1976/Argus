"""Google Gemini Provider implementation for Argus (Phase 7).

Uses Google Generative Language REST API via HTTPX async client.
Provides isolated client management, credential validation, safe error handling,
token telemetry extraction, and model status reporting.
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


class GeminiProvider(BaseLLMProvider):
    """Encapsulates Google Gemini REST API integration and text generation."""

    KNOWN_PLACEHOLDERS = frozenset({
        "your-gemini-api-key-here",
        "your_gemini_api_key_here",
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
    ) -> None:
        self._api_key = api_key if api_key is not None else getattr(settings, "GEMINI_API_KEY", None)
        self._base_url = (
            base_url
            if base_url is not None
            else getattr(settings, "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
        ).rstrip("/")
        self._model = (model or getattr(settings, "GEMINI_DEFAULT_MODEL", "gemini-2.0-flash")).strip()
        self._timeout = timeout if timeout is not None else getattr(settings, "GEMINI_TIMEOUT_SECONDS", 60.0)

    @property
    def provider_name(self) -> str:
        return "gemini"

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
        if not self.is_configured():
            raise MissingCredentialsError(
                "Gemini API key is not configured. Set GEMINI_API_KEY in the environment."
            )

        endpoint = f"{self._base_url}/models/{self._model}:generateContent"
        params = {"key": self._api_key}

        # Build payload
        contents: List[Dict[str, Any]] = [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
        payload: Dict[str, Any] = {"contents": contents}

        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }

        generation_config: Dict[str, Any] = {}
        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            generation_config["maxOutputTokens"] = kwargs["max_tokens"]
        if generation_config:
            payload["generationConfig"] = generation_config

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(endpoint, params=params, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Gemini request timed out after {self._timeout}s: {str(exc)}"
            ) from exc
        except httpx.NetworkError as exc:
            raise ProviderConnectionError(
                f"Failed to connect to Google Gemini endpoint: {str(exc)}"
            ) from exc
        except Exception as exc:
            raise ProviderAPIError(f"Unexpected error connecting to Gemini: {str(exc)}") from exc

        duration = time.perf_counter() - start_time

        # Handle HTTP error status codes
        if response.status_code != 200:
            error_msg = f"Gemini API returned status {response.status_code}"
            try:
                err_data = response.json().get("error", {})
                detail = err_data.get("message", response.text)
                error_msg = f"{error_msg}: {detail}"
            except Exception:
                error_msg = f"{error_msg}: {response.text}"

            if response.status_code in {401, 403}:
                raise AuthenticationError(error_msg)
            elif response.status_code == 429:
                raise RateLimitError(error_msg)
            elif response.status_code == 404:
                raise UnsupportedModelError(f"Model '{self._model}' not found in Gemini: {error_msg}")
            elif response.status_code == 400:
                raise ProviderAPIError(f"Bad request to Gemini: {error_msg}", status_code=400)
            elif response.status_code >= 500:
                raise ProviderConnectionError(f"Gemini server error: {error_msg}")
            else:
                raise ProviderAPIError(error_msg, status_code=response.status_code)

        # Parse success payload
        try:
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise ProviderAPIError("Gemini returned empty candidate response")

            first_candidate = candidates[0]
            content_parts = first_candidate.get("content", {}).get("parts", [])
            text_chunks = [p.get("text", "") for p in content_parts if "text" in p]
            content = "".join(text_chunks)
            finish_reason = first_candidate.get("finishReason")

            usage_metadata = data.get("usageMetadata", {})
            input_tokens = usage_metadata.get("promptTokenCount")
            output_tokens = usage_metadata.get("candidatesTokenCount")
            total_tokens = usage_metadata.get("totalTokenCount")
            cached_tokens = usage_metadata.get("cachedContentTokenCount")

            return ProviderExecutionResult(
                provider_name=self.provider_name,
                model=self._model,
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                reasoning_tokens=None,
                latency_seconds=duration,
                raw_finish_reason=str(finish_reason) if finish_reason else None,
            )
        except (KeyError, ValueError) as exc:
            raise ProviderAPIError(f"Failed to parse Gemini response payload: {str(exc)}") from exc
