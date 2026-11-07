"""TinyFish Browser Automation Adapter (Phase 9).

Provides a dedicated integration boundary for browser automation:
- Enforces strict credential protection.
- Separates browser operations from LLM generation.
- Handles task execution, timeouts, and status normalization.
"""

import logging
import time
from typing import Any, Dict, Optional
import httpx

from app.core.config import settings
from app.services.automation.tinyfish.models import (
    TinyFishActionType,
    TinyFishTaskRequest,
    TinyFishTaskResponse,
    TinyFishTaskStatus,
)

logger = logging.getLogger(__name__)


class TinyFishError(Exception):
    """Base exception for all TinyFish automation errors."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class TinyFishNotConfiguredError(TinyFishError):
    """Raised when TinyFish automation is invoked without proper configuration or enablement."""

    def __init__(
        self,
        message: str = (
            "TinyFish browser automation is not enabled or credentials are missing. "
            "Set TINYFISH_ENABLED=true and TINYFISH_API_KEY in the environment."
        ),
    ):
        super().__init__(message, status_code=503)


class TinyFishAuthenticationError(TinyFishError):
    """Raised when upstream TinyFish API authentication fails."""

    def __init__(self, message: str = "TinyFish upstream authentication failed."):
        super().__init__(message, status_code=401)


class TinyFishTimeoutError(TinyFishError):
    """Raised when browser automation task exceeds configured timeout."""

    def __init__(self, message: str = "TinyFish automation task timed out."):
        super().__init__(message, status_code=504)


class TinyFishConnectionError(TinyFishError):
    """Raised when connection to TinyFish API fails."""

    def __init__(self, message: str = "Failed to connect to TinyFish automation service."):
        super().__init__(message, status_code=502)


class TinyFishExecutionError(TinyFishError):
    """Raised when browser execution fails on the target webpage."""

    def __init__(self, message: str = "Browser execution error occurred.", status_code: int = 502):
        super().__init__(message, status_code=status_code)


class TinyFishAdapter:
    """Adapter managing communication with the TinyFish Browser Automation API."""

    KNOWN_PLACEHOLDERS = frozenset({
        "your-tinyfish-api-key-here",
        "your_tinyfish_api_key_here",
        "your-api-key-here",
        "your_api_key_here",
        "placeholder",
        "changeme",
    })

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self._enabled = enabled if enabled is not None else getattr(settings, "TINYFISH_ENABLED", False)
        self._api_key = api_key if api_key is not None else getattr(settings, "TINYFISH_API_KEY", None)
        self._base_url = (
            base_url
            if base_url is not None
            else getattr(settings, "TINYFISH_BASE_URL", "https://api.tinyfish.io/v1")
        ).rstrip("/")
        self._timeout = timeout if timeout is not None else getattr(settings, "TINYFISH_TIMEOUT_SECONDS", 120.0)

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
        if not self._enabled:
            return False
        if not self._api_key:
            return False
        return not self.is_placeholder_key(self._api_key)

    def get_status(self) -> Dict[str, Any]:
        has_key = self.is_configured()
        masked_key = None
        if self._api_key and not self.is_placeholder_key(self._api_key):
            trimmed = self._api_key.strip()
            masked_key = f"{trimmed[:4]}...{trimmed[-4:]}" if len(trimmed) > 8 else "***"

        return {
            "service": "tinyfish",
            "type": "browser_automation",
            "enabled": self._enabled,
            "configured": has_key,
            "base_url": self._base_url,
            "masked_key": masked_key,
            "timeout_seconds": self._timeout,
        }

    async def execute_task(self, request: TinyFishTaskRequest) -> TinyFishTaskResponse:
        """Dispatch a validated browser automation task to the TinyFish API."""
        if not self.is_configured():
            raise TinyFishNotConfiguredError()

        timeout_sec = request.timeout_seconds or self._timeout
        endpoint = f"{self._base_url}/tasks"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "task_id": request.task_id,
            "action": request.action.value,
            "url": request.url,
            "parameters": request.parameters,
            "capture_screenshot": request.capture_screenshot,
            "timeout": timeout_sec,
        }

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout_sec) as client:
                response = await client.post(endpoint, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return TinyFishTaskResponse(
                task_id=request.task_id,
                action=request.action,
                url=request.url,
                status=TinyFishTaskStatus.TIMED_OUT,
                result_data=None,
                screenshot_base64=None,
                error_message=f"Browser automation timed out after {timeout_sec}s: {str(exc)}",
                execution_time_ms=duration_ms,
                integration_mode="live_api",
            )
        except httpx.NetworkError as exc:
            raise TinyFishConnectionError(
                f"Failed to connect to TinyFish service at {self._base_url}: {str(exc)}"
            ) from exc
        except Exception as exc:
            raise TinyFishExecutionError(f"Unexpected error executing TinyFish task: {str(exc)}") from exc

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        # Handle HTTP error status codes
        if response.status_code != 200:
            error_detail = response.text
            try:
                error_detail = response.json().get("message", response.text)
            except Exception:
                pass

            if response.status_code in {401, 403}:
                raise TinyFishAuthenticationError(f"TinyFish auth error ({response.status_code}): {error_detail}")
            elif response.status_code == 408 or response.status_code == 504:
                return TinyFishTaskResponse(
                    task_id=request.task_id,
                    action=request.action,
                    url=request.url,
                    status=TinyFishTaskStatus.TIMED_OUT,
                    error_message=f"TinyFish server reported timeout: {error_detail}",
                    execution_time_ms=duration_ms,
                    integration_mode="live_api",
                )
            else:
                return TinyFishTaskResponse(
                    task_id=request.task_id,
                    action=request.action,
                    url=request.url,
                    status=TinyFishTaskStatus.FAILED,
                    error_message=f"TinyFish error ({response.status_code}): {error_detail}",
                    execution_time_ms=duration_ms,
                    integration_mode="live_api",
                )

        # Parse success body
        try:
            data = response.json()
            raw_status = data.get("status", "completed").lower()
            status_enum = TinyFishTaskStatus.COMPLETED
            if raw_status in {"failed", "error"}:
                status_enum = TinyFishTaskStatus.FAILED
            elif raw_status in {"timeout", "timed_out"}:
                status_enum = TinyFishTaskStatus.TIMED_OUT

            return TinyFishTaskResponse(
                task_id=request.task_id,
                action=request.action,
                url=request.url,
                status=status_enum,
                result_data=data.get("result"),
                screenshot_base64=data.get("screenshot"),
                error_message=data.get("error"),
                execution_time_ms=duration_ms,
                integration_mode="live_api",
            )
        except Exception as exc:
            raise TinyFishExecutionError(f"Failed to parse TinyFish response payload: {str(exc)}") from exc
