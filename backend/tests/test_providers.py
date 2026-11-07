"""Unit tests for Phase 7 Multi-Provider Integration.

Covers available services with credentials:
1. Google Gemini
2. DeepSeek
3. Ollama (local)

Verifies:
- Credential configuration and placeholder rejection.
- Secret key masking in status diagnostics.
- Response normalization and token extraction across Gemini, DeepSeek, and Ollama.
- Error normalization into unified ProviderError hierarchy.
- ProviderRegistry resolution and rejection of uncredentialed providers.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from app.services.providers.base import (
    AuthenticationError,
    MissingCredentialsError,
    ProviderConnectionError,
    ProviderExecutionResult,
    ProviderTimeoutError,
    RateLimitError,
    UnsupportedModelError,
)
from app.services.providers.deepseek_provider import DeepSeekProvider
from app.services.providers.gemini_provider import GeminiProvider
from app.services.providers.ollama_provider import OllamaProvider
from app.services.providers.registry import ProviderRegistry


# ============================================================================
# 1. Credential & Placeholder Validation Tests
# ============================================================================

def test_placeholder_detection():
    assert GeminiProvider.is_placeholder_key("your-gemini-api-key-here")
    assert GeminiProvider.is_placeholder_key("your_api_key")
    assert not GeminiProvider.is_placeholder_key("AQ.valid-gemini-key-123456")

    assert DeepSeekProvider.is_placeholder_key("your-deepseek-api-key-here")
    assert not DeepSeekProvider.is_placeholder_key("sk-deepseek-valid-key-999")


def test_status_masks_credentials():
    gemini = GeminiProvider(api_key="AQ.mock-test-gemini-key-for-testing-only-12345")
    status = gemini.get_status()
    assert status["configured"] is True
    # Ensure raw full key is NEVER printed in status
    assert "AQ.mock-test-gemini-key-for-testing-only-12345" not in str(status)
    assert status["masked_key"] is not None
    assert "..." in status["masked_key"]


# ============================================================================
# 2. Google Gemini Provider Tests
# ============================================================================

@pytest.mark.asyncio
async def test_gemini_generate_success():
    provider = GeminiProvider(api_key="valid-test-key")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello from Gemini 2.0 Flash!"}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 15,
            "candidatesTokenCount": 25,
            "totalTokenCount": 40,
            "cachedContentTokenCount": 5,
        },
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await provider.generate(prompt="Say hello", system_prompt="You are helpful")

    assert isinstance(result, ProviderExecutionResult)
    assert result.provider_name == "gemini"
    assert result.content == "Hello from Gemini 2.0 Flash!"
    assert result.input_tokens == 15
    assert result.output_tokens == 25
    assert result.total_tokens == 40
    assert result.cached_tokens == 5


@pytest.mark.asyncio
async def test_gemini_error_normalization():
    provider = GeminiProvider(api_key="valid-test-key")

    # 401 Authentication error
    mock_401 = MagicMock(spec=httpx.Response)
    mock_401.status_code = 401
    mock_401.text = "API key not valid"
    mock_401.json.return_value = {"error": {"message": "API key not valid"}}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_401
        with pytest.raises(AuthenticationError):
            await provider.generate(prompt="Test")

    # 429 Rate limit error
    mock_429 = MagicMock(spec=httpx.Response)
    mock_429.status_code = 429
    mock_429.text = "Quota exceeded"
    mock_429.json.return_value = {"error": {"message": "Quota exceeded"}}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_429
        with pytest.raises(RateLimitError):
            await provider.generate(prompt="Test")


# ============================================================================
# 3. DeepSeek Provider Tests
# ============================================================================

@pytest.mark.asyncio
async def test_deepseek_generate_with_reasoning_tokens():
    provider = DeepSeekProvider(api_key="valid-deepseek-key", model="deepseek-reasoner")

    mock_chat_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Calculated answer"
    mock_choice.finish_reason = "stop"
    mock_chat_completion.choices = [mock_choice]

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 50
    mock_usage.completion_tokens = 120
    mock_usage.total_tokens = 170

    mock_usage.prompt_tokens_details = MagicMock()
    mock_usage.prompt_tokens_details.cached_tokens = 20

    mock_usage.completion_tokens_details = MagicMock()
    mock_usage.completion_tokens_details.reasoning_tokens = 80

    mock_chat_completion.usage = mock_usage

    with patch.object(provider, "_get_client") as mock_client_getter:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_chat_completion)
        mock_client_getter.return_value = mock_client

        result = await provider.generate(prompt="Solve equation")

    assert result.provider_name == "deepseek"
    assert result.content == "Calculated answer"
    assert result.input_tokens == 50
    assert result.output_tokens == 120
    assert result.cached_tokens == 20
    assert result.reasoning_tokens == 80


# ============================================================================
# 4. Ollama Provider Tests
# ============================================================================

@pytest.mark.asyncio
async def test_ollama_generate_success():
    provider = OllamaProvider(model="qwen2.5:1.5b")

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "qwen2.5:1.5b",
        "message": {"role": "assistant", "content": "Local Qwen output"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 18,
        "eval_count": 32,
    }

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await provider.generate(prompt="Hello local model")

    assert result.provider_name == "ollama"
    assert result.content == "Local Qwen output"
    assert result.input_tokens == 18
    assert result.output_tokens == 32
    assert result.total_tokens == 50


# ============================================================================
# 5. Provider Registry Resolution Tests
# ============================================================================

def test_provider_registry_resolution():
    gemini_prov = ProviderRegistry.get_provider_for_model("gemini-2.0-flash")
    assert isinstance(gemini_prov, GeminiProvider)

    deepseek_prov = ProviderRegistry.get_provider_for_model("deepseek-chat")
    assert isinstance(deepseek_prov, DeepSeekProvider)

    ollama_prov = ProviderRegistry.get_provider_for_model("qwen2.5:1.5b")
    assert isinstance(ollama_prov, OllamaProvider)

    # Uncredentialed providers are rejected
    with pytest.raises(UnsupportedModelError):
        ProviderRegistry.get_provider_for_model("uncredentialed-unknown-model")
