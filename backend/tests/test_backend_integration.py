"""Unit tests for Phase 10 Complete Backend Integration.

Covers available services with credentials:
1. Google Gemini
2. DeepSeek
3. Ollama (local)

Verifies:
1. End-to-end execution of UnifiedExecutionService with prompt input.
2. Multi-turn execution with ConversationMessage context and RABCO optimization.
3. Explicit model override bypassing dynamic routing.
4. Provider failure and controlled fallback dispatch between credentialed providers.
5. Exhausted candidate failure propagation.
6. FastAPI API endpoint contracts (/api/execute, /api/telemetry/records, /api/models, /api/providers/status).
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.research.rabco.orchestrator import OrchestrationDecision, RABCOOrchestrator, RABCOOrchestrationResult
from app.schemas.execution import UnifiedExecutionRequest, UnifiedExecutionResponse
from app.schemas.rabco import (
    ConversationMessage,
    EvidenceCriticality,
    EvidenceType,
    EvidenceUnit,
    MessageRole,
    SourceSpan,
    ValidationStatus,
)
from app.services.providers.base import (
    ProviderError,
    ProviderExecutionResult,
    RateLimitError,
)
from app.services.providers.registry import ProviderRegistry
from app.services.routing.engine import ModelRoutingEngine
from app.services.telemetry.service import TelemetryService
from app.services.telemetry.storage import InMemoryTelemetryStorage
from app.services.unified_execution_service import UnifiedExecutionService


# ============================================================================
# Test Fixtures & Helpers
# ============================================================================

def make_test_message(msg_id: str, content: str, role: MessageRole = MessageRole.USER) -> ConversationMessage:
    return ConversationMessage(
        message_id=msg_id,
        role=role,
        content=content,
        sequence_index=0,
        timestamp=datetime.now(timezone.utc),
    )


# ============================================================================
# 1. Service Level End-to-End Tests
# ============================================================================

@pytest.mark.asyncio
async def test_unified_service_prompt_execution():
    storage = InMemoryTelemetryStorage()
    telemetry = TelemetryService(storage=storage)
    service = UnifiedExecutionService(telemetry_service=telemetry)

    mock_provider = MagicMock()
    mock_provider.provider_name = "gemini"
    mock_provider.generate = AsyncMock(
        return_value=ProviderExecutionResult(
            provider_name="gemini",
            model="gemini-2.0-flash",
            content="Generated answer for binary search complexity: O(log n).",
            input_tokens=25,
            output_tokens=30,
            total_tokens=55,
            latency_seconds=0.45,
        )
    )

    with patch.object(ProviderRegistry, "get_provider_for_model", return_value=mock_provider):
        req = UnifiedExecutionRequest(
            prompt="What is the time complexity of binary search?",
            strategy="balanced",
        )
        resp = await service.execute(req)

    assert isinstance(resp, UnifiedExecutionResponse)
    assert resp.model_response == "Generated answer for binary search complexity: O(log n)."
    assert resp.selected_provider == "gemini"
    assert not resp.fallback_used
    assert resp.execution_latency_ms >= 0
    assert resp.usage["total_tokens"] == 55

    # Verify telemetry was recorded in storage
    recent = storage.get_by_request_id(resp.request_id)
    assert recent is not None
    assert recent.status == "success"


@pytest.mark.asyncio
async def test_unified_service_rabco_messages_execution():
    storage = InMemoryTelemetryStorage()
    telemetry = TelemetryService(storage=storage)

    msg1 = make_test_message("msg_0", "System rule: AES-256 encryption is mandatory.", MessageRole.SYSTEM)
    msg2 = make_test_message("msg_1", "Deploy the microservice to us-central1.", MessageRole.USER)

    mock_provider = MagicMock()
    mock_provider.provider_name = "gemini"
    mock_provider.generate = AsyncMock(
        return_value=ProviderExecutionResult(
            provider_name="gemini",
            model="gemini-2.0-flash",
            content="Microservice deployment plan configured with AES-256 in us-central1.",
            input_tokens=40,
            output_tokens=60,
            total_tokens=100,
            latency_seconds=0.6,
        )
    )

    service = UnifiedExecutionService(telemetry_service=telemetry)

    with patch.object(ProviderRegistry, "get_provider_for_model", return_value=mock_provider):
        req = UnifiedExecutionRequest(
            messages=[msg1, msg2],
            model="gemini-2.0-flash",
            enable_rabco=True,
        )
        resp = await service.execute(req)

    assert resp.selected_model == "gemini-2.0-flash"
    assert resp.rabco is not None
    assert "decision" in resp.rabco
    assert resp.model_response.startswith("Microservice deployment plan")


@pytest.mark.asyncio
async def test_unified_service_controlled_fallback():
    """When the primary model candidate fails with RateLimitError, dispatch falls back to secondary model."""
    storage = InMemoryTelemetryStorage()
    telemetry = TelemetryService(storage=storage)
    service = UnifiedExecutionService(telemetry_service=telemetry)

    # Primary provider fails with rate limit
    mock_primary = MagicMock()
    mock_primary.provider_name = "gemini"
    mock_primary.generate = AsyncMock(side_effect=RateLimitError("Rate limit exceeded"))

    # Fallback secondary provider succeeds
    mock_fallback = MagicMock()
    mock_fallback.provider_name = "deepseek"
    mock_fallback.generate = AsyncMock(
        return_value=ProviderExecutionResult(
            provider_name="deepseek",
            model="deepseek-chat",
            content="Fallback answer from DeepSeek",
            input_tokens=20,
            output_tokens=25,
            total_tokens=45,
            latency_seconds=0.85,
        )
    )

    def mock_get_provider(model_id: str, **kwargs):
        if "gemini" in model_id:
            return mock_primary
        return mock_fallback

    with patch.object(ProviderRegistry, "get_provider_for_model", side_effect=mock_get_provider):
        # Force routing fallback chain between available providers
        with patch.object(
            ModelRoutingEngine,
            "route",
            return_value=MagicMock(
                selected_model="gemini-2.0-flash",
                selected_provider="gemini",
                task_category=MagicMock(value="general"),
                complexity=MagicMock(score=0.2),
                applied_strategy="balanced",
                selection_rationale="Primary selected",
                effective_weights={},
                fallback_chain=["deepseek-chat"],
            ),
        ):
            req = UnifiedExecutionRequest(prompt="Tell me a joke.")
            resp = await service.execute(req)

    assert resp.fallback_used is True
    assert resp.fallback_model_from == "gemini-2.0-flash"
    assert resp.selected_model == "deepseek-chat"
    assert resp.model_response == "Fallback answer from DeepSeek"


@pytest.mark.asyncio
async def test_unified_service_all_candidates_failed_raises():
    """If all candidate models in queue fail, error is re-raised and logged to telemetry."""
    storage = InMemoryTelemetryStorage()
    telemetry = TelemetryService(storage=storage)
    service = UnifiedExecutionService(telemetry_service=telemetry)

    mock_failing_prov = MagicMock()
    mock_failing_prov.provider_name = "gemini"
    mock_failing_prov.generate = AsyncMock(side_effect=ProviderError("Fatal API failure"))

    with patch.object(ProviderRegistry, "get_provider_for_model", return_value=mock_failing_prov):
        req = UnifiedExecutionRequest(model="gemini-2.0-flash", prompt="Test failing prompt")
        with pytest.raises(ProviderError):
            await service.execute(req)

    # Telemetry should record the error
    summary = storage.get_summary()
    assert summary.failed_requests == 1


# ============================================================================
# 2. FastAPI HTTP Endpoint Tests
# ============================================================================

@pytest.mark.asyncio
async def test_api_execute_endpoint():
    mock_resp = ProviderExecutionResult(
        provider_name="ollama",
        model="qwen2.5:1.5b",
        content="FastAPI test response",
        input_tokens=10,
        output_tokens=15,
        total_tokens=25,
        latency_seconds=0.3,
    )

    mock_prov = MagicMock()
    mock_prov.provider_name = "ollama"
    mock_prov.generate = AsyncMock(return_value=mock_resp)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with patch.object(ProviderRegistry, "get_provider_for_model", return_value=mock_prov):
            response = await client.post(
                "/api/execute",
                json={
                    "prompt": "Test API prompt execution",
                    "model": "qwen2.5:1.5b",
                },
            )

    assert response.status_code == 200
    data = response.json()
    assert data["model_response"] == "FastAPI test response"
    assert data["selected_model"] == "qwen2.5:1.5b"
    assert data["selected_provider"] == "ollama"


@pytest.mark.asyncio
async def test_api_telemetry_and_models_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Check /api/models (only available models: gemini, deepseek, ollama)
        models_res = await client.get("/api/models")
        assert models_res.status_code == 200
        models_data = models_res.json()
        assert len(models_data) >= 5
        model_ids = [m["model_id"] for m in models_data]
        assert "gemini-2.0-flash" in model_ids
        assert "deepseek-chat" in model_ids
        assert "qwen2.5:1.5b" in model_ids
        # Uncredentialed models are not present
        assert "gpt-4o" not in model_ids
        assert "llama-3.3-70b-versatile" not in model_ids

        # Check /api/providers/status
        prov_res = await client.get("/api/providers/status")
        assert prov_res.status_code == 200
        prov_data = prov_res.json()
        assert "gemini" in prov_data
        assert "deepseek" in prov_data
        assert "ollama" in prov_data
        assert "openai" not in prov_data
        assert "groq" not in prov_data

        # Check /api/telemetry/summary
        telemetry_res = await client.get("/api/telemetry/summary")
        assert telemetry_res.status_code == 200
        telemetry_data = telemetry_res.json()
        assert "total_requests" in telemetry_data
