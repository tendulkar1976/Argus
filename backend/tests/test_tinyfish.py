"""Unit tests for Phase 9 TinyFish Browser Automation Integration.

Covers:
1. Configuration and placeholder validation.
2. Target URL protocol validation.
3. Unconfigured execution guardrail (TinyFishNotConfiguredError).
4. Task dispatch and normalized response mapping.
5. Error handling (authentication, timeout, connection failures).
"""

from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from pydantic import ValidationError

from app.services.automation.tinyfish.adapter import (
    TinyFishAdapter,
    TinyFishAuthenticationError,
    TinyFishConnectionError,
    TinyFishNotConfiguredError,
)
from app.services.automation.tinyfish.models import (
    TinyFishActionType,
    TinyFishTaskRequest,
    TinyFishTaskResponse,
    TinyFishTaskStatus,
)


# ============================================================================
# 1. Configuration & Validation Tests
# ============================================================================

def test_tinyfish_unconfigured_by_default():
    adapter = TinyFishAdapter(enabled=False)
    assert not adapter.is_configured()

    status = adapter.get_status()
    assert status["configured"] is False
    assert status["enabled"] is False


def test_tinyfish_rejects_placeholder_credentials():
    adapter = TinyFishAdapter(enabled=True, api_key="your-tinyfish-api-key-here")
    assert not adapter.is_configured()


def test_tinyfish_url_validation():
    # Valid URLs
    req1 = TinyFishTaskRequest(
        task_id="t1",
        action=TinyFishActionType.NAVIGATE,
        url="https://example.com",
    )
    assert req1.url == "https://example.com"

    # Invalid URL scheme
    with pytest.raises(ValidationError):
        TinyFishTaskRequest(
            task_id="t2",
            action=TinyFishActionType.NAVIGATE,
            url="ftp://invalid.com",
        )


# ============================================================================
# 2. Execution & Error Handling Tests
# ============================================================================

@pytest.mark.asyncio
async def test_tinyfish_raises_when_not_configured():
    adapter = TinyFishAdapter(enabled=False)
    req = TinyFishTaskRequest(
        task_id="task_001",
        action=TinyFishActionType.NAVIGATE,
        url="https://example.com",
    )
    with pytest.raises(TinyFishNotConfiguredError):
        await adapter.execute_task(req)


@pytest.mark.asyncio
async def test_tinyfish_successful_task_execution():
    adapter = TinyFishAdapter(enabled=True, api_key="tf_live_valid_key_123")

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "completed",
        "result": {"title": "Example Domain", "headings": ["Example"]},
        "screenshot": None,
    }

    req = TinyFishTaskRequest(
        task_id="task_extract_1",
        action=TinyFishActionType.EXTRACT,
        url="https://example.com",
        parameters={"selector": "h1"},
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        result = await adapter.execute_task(req)

    assert isinstance(result, TinyFishTaskResponse)
    assert result.status == TinyFishTaskStatus.COMPLETED
    assert result.result_data == {"title": "Example Domain", "headings": ["Example"]}
    assert result.execution_time_ms >= 0


@pytest.mark.asyncio
async def test_tinyfish_auth_error_handling():
    adapter = TinyFishAdapter(enabled=True, api_key="invalid_key")

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"

    req = TinyFishTaskRequest(
        task_id="task_auth_err",
        action=TinyFishActionType.NAVIGATE,
        url="https://example.com",
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        with pytest.raises(TinyFishAuthenticationError):
            await adapter.execute_task(req)


@pytest.mark.asyncio
async def test_tinyfish_timeout_handling():
    adapter = TinyFishAdapter(enabled=True, api_key="valid_key")

    req = TinyFishTaskRequest(
        task_id="task_timeout",
        action=TinyFishActionType.AUTOMATE_WORKFLOW,
        url="https://slow-site.com",
        timeout_seconds=5.0,
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Read timed out")
        result = await adapter.execute_task(req)

    assert result.status == TinyFishTaskStatus.TIMED_OUT
    assert "timed out" in result.error_message.lower()
