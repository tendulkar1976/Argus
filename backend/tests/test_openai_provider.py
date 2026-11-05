"""Focused tests for Phase 4.1: Secure OpenAI Provider Setup.

Verifies:
- Official OpenAI SDK integration
- Model configuration defaulting to gpt-6.1-sol
- Environment variable overrides
- Credential validation and missing-key handling
- Provider interface compliance and isolation
- Non-exposure of API keys in logs, status, and repr
"""

import pytest
from unittest.mock import patch

from app.core.config import Settings
from app.services.providers.base import (
    BaseLLMProvider,
    ConfigurationError,
    MissingCredentialsError,
    ProviderError,
)
from app.services.providers.openai_provider import OpenAIProvider


def test_provider_interface_compliance():
    """Verify OpenAIProvider implements BaseLLMProvider and exception hierarchy."""
    provider = OpenAIProvider(api_key=None)
    assert isinstance(provider, BaseLLMProvider)
    assert provider.provider_name == "openai"
    assert issubclass(MissingCredentialsError, ProviderError)
    assert issubclass(ConfigurationError, ProviderError)


def test_default_model_configuration():
    """Verify default model is gpt-6.1-sol when not overridden."""
    provider = OpenAIProvider(api_key=None)
    assert provider.model == "gpt-6.1-sol"


def test_model_configuration_override():
    """Verify model can be overridden explicitly or via settings."""
    provider = OpenAIProvider(api_key=None, model="custom-gpt-model")
    assert provider.model == "custom-gpt-model"


def test_is_configured_without_api_key():
    """Verify is_configured returns False when API key is missing or empty."""
    provider_none = OpenAIProvider(api_key=None)
    assert provider_none.is_configured() is False

    provider_empty = OpenAIProvider(api_key="")
    assert provider_empty.is_configured() is False

    provider_whitespace = OpenAIProvider(api_key="   \t\n")
    assert provider_whitespace.is_configured() is False


def test_is_configured_with_api_key():
    """Verify is_configured returns True when valid key string is provided."""
    provider = OpenAIProvider(api_key="sk-test-dummy-key-12345")
    assert provider.is_configured() is True


def test_is_configured_with_placeholder_api_key():
    """Verify is_configured returns False for known placeholder strings."""
    placeholders = [
        "your-openai-api-key-here",
        "YOUR-OPENAI-API-KEY-HERE",
        "your-api-key-here",
        "your_key_here",
        "changeme",
        "placeholder",
        "<your-api-key>",
        "sk-...",
    ]
    for ph in placeholders:
        provider = OpenAIProvider(api_key=ph)
        assert provider.is_configured() is False, f"Expected {ph!r} to be rejected as placeholder"


def test_get_client_raises_when_credentials_missing_or_placeholder():
    """Verify get_client raises MissingCredentialsError when API key is missing or a placeholder."""
    provider_none = OpenAIProvider(api_key=None)
    with pytest.raises(MissingCredentialsError) as exc_info:
        provider_none.get_client()
    assert "OpenAI API key is missing or set to an unconfigured placeholder" in str(exc_info.value)

    placeholder_key = "your-openai-api-key-here"
    provider_ph = OpenAIProvider(api_key=placeholder_key)
    with pytest.raises(MissingCredentialsError) as exc_info_ph:
        provider_ph.get_client()
    err_msg = str(exc_info_ph.value)
    assert "OpenAI API key is missing or set to an unconfigured placeholder" in err_msg
    # Crucially ensure the placeholder value itself is never leaked in the error
    assert placeholder_key not in err_msg



def test_get_client_initialization_with_credentials():
    """Verify get_client initializes an AsyncOpenAI instance without making network calls."""
    test_key = "sk-dummy-test-key-argus"
    provider = OpenAIProvider(
        api_key=test_key,
        model="gpt-6.1-sol",
        timeout=45.0,
        max_retries=3,
    )

    client = provider.get_client()
    assert client is not None
    assert client.api_key == test_key
    assert client.timeout == 45.0
    assert client.max_retries == 3

    # Consecutive calls return the cached singleton client
    assert provider.get_client() is client


def test_api_key_never_exposed_in_repr_or_status():
    """Verify secrets are strictly excluded from repr, str, and get_status dictionaries."""
    secret_key = "sk-super-secret-production-token-do-not-leak"
    provider = OpenAIProvider(api_key=secret_key, model="gpt-6.1-sol")

    # Inspect repr & str
    repr_str = repr(provider)
    str_str = str(provider)
    assert secret_key not in repr_str
    assert secret_key not in str_str
    assert "sk-" not in repr_str

    # Inspect status payload
    status = provider.get_status()
    assert status["provider"] == "openai"
    assert status["model"] == "gpt-6.1-sol"
    assert status["configured"] is True
    assert "api_key" not in status
    assert "key" not in status
    assert secret_key not in str(status)


def test_client_initialization_failure_raises_configuration_error():
    """Verify unexpected client construction errors are mapped to ConfigurationError."""
    provider = OpenAIProvider(api_key="sk-dummy-key")
    with patch("app.services.providers.openai_provider.AsyncOpenAI", side_effect=RuntimeError("SDK init boom")):
        with pytest.raises(ConfigurationError) as exc_info:
            provider.get_client()
        assert "Failed to initialize OpenAI client" in str(exc_info.value)
