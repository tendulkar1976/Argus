"""Unit tests for Phase 8 Cost and Telemetry Engine.

Covers available services with credentials:
1. Google Gemini
2. DeepSeek
3. Ollama (local)

Verifies:
1. Multi-provider cost calculation and decimal safety.
2. Local model zero-cost handling (calculated_local).
3. Distinguishing provider-reported cost from estimated cost.
4. Handling missing pricing metadata or missing token usage.
5. In-memory and SQLite persistence backends.
6. Summary analytics aggregation.
7. Telemetry persistence error isolation.
"""

from unittest.mock import MagicMock, patch
import pytest

from app.services.telemetry.models import TelemetryQueryFilter, TelemetryRecord
from app.services.telemetry.service import TelemetryService
from app.services.telemetry.storage import InMemoryTelemetryStorage, SQLiteTelemetryStorage


# ============================================================================
# 1. Cost Accounting & Status Tests
# ============================================================================

def test_telemetry_cost_calculation_gemini():
    storage = InMemoryTelemetryStorage()
    service = TelemetryService(storage=storage)

    # gemini-1.5-pro: 1,000 input tokens ($1.25/M) + 500 output tokens ($5.00/M)
    # input: 0.00125, output: 0.0025 => total: 0.00375
    rec = service.record_execution(
        request_id="req_001",
        provider="gemini",
        model="gemini-1.5-pro",
        latency_ms=1200,
        status="success",
        input_tokens=1000,
        output_tokens=500,
    )

    assert rec is not None
    assert rec.estimated_cost_usd == 0.00375
    assert rec.cost_status == "calculated"
    assert rec.currency == "USD"


def test_telemetry_local_ollama_zero_cost():
    storage = InMemoryTelemetryStorage()
    service = TelemetryService(storage=storage)

    rec = service.record_execution(
        request_id="req_local",
        provider="ollama",
        model="qwen2.5:1.5b",
        latency_ms=850,
        status="success",
        input_tokens=500,
        output_tokens=250,
    )

    assert rec is not None
    assert rec.estimated_cost_usd == 0.0
    assert rec.cost_status == "calculated_local"


def test_telemetry_unregistered_model_pricing_unavailable():
    storage = InMemoryTelemetryStorage()
    service = TelemetryService(storage=storage)

    rec = service.record_execution(
        request_id="req_unknown",
        provider="custom_provider",
        model="unverified-custom-model-999",
        latency_ms=500,
        status="success",
        input_tokens=500,
        output_tokens=100,
    )

    assert rec is not None
    assert rec.estimated_cost_usd is None
    assert rec.cost_status == "pricing_unavailable"


def test_telemetry_distinguishes_provider_reported_cost():
    storage = InMemoryTelemetryStorage()
    service = TelemetryService(storage=storage)

    rec = service.record_execution(
        request_id="req_reported",
        provider="gemini",
        model="gemini-2.0-flash",
        latency_ms=600,
        status="success",
        input_tokens=1000,
        output_tokens=200,
        provider_reported_cost_usd=0.00015,
    )

    assert rec.provider_reported_cost_usd == 0.00015
    assert rec.estimated_cost_usd is not None


# ============================================================================
# 2. Persistence & Storage Backends
# ============================================================================

def test_sqlite_storage_roundtrip(tmp_path):
    db_file = str(tmp_path / "test_telemetry.db")
    storage = SQLiteTelemetryStorage(db_path=db_file)
    service = TelemetryService(storage=storage)

    rec = service.record_execution(
        request_id="req_sqlite_1",
        provider="gemini",
        model="gemini-2.0-flash",
        latency_ms=450,
        status="success",
        task_category="coding",
        input_tokens=2000,
        output_tokens=400,
    )

    fetched = service.get_by_request_id("req_sqlite_1")
    assert fetched is not None
    assert fetched.request_id == "req_sqlite_1"
    assert fetched.provider == "gemini"
    assert fetched.model == "gemini-2.0-flash"
    assert fetched.input_tokens == 2000


def test_telemetry_query_and_summary():
    storage = InMemoryTelemetryStorage()
    service = TelemetryService(storage=storage)

    service.record_execution(
        request_id="req_a",
        provider="gemini",
        model="gemini-2.0-flash",
        latency_ms=1000,
        status="success",
        task_category="reasoning",
        input_tokens=1000,
        output_tokens=500,
    )
    service.record_execution(
        request_id="req_b",
        provider="deepseek",
        model="deepseek-chat",
        latency_ms=800,
        status="success",
        task_category="coding",
        input_tokens=500,
        output_tokens=200,
    )
    service.record_execution(
        request_id="req_c",
        provider="ollama",
        model="qwen2.5:1.5b",
        latency_ms=500,
        status="error",
        error_category="connection_error",
        input_tokens=0,
        output_tokens=0,
    )

    # Query filtered by provider
    filtered = service.query(TelemetryQueryFilter(provider="deepseek"))
    assert len(filtered) == 1
    assert filtered[0].request_id == "req_b"

    # Aggregated summary
    summary = service.get_summary()
    assert summary.total_requests == 3
    assert summary.successful_requests == 2
    assert summary.failed_requests == 1
    assert summary.requests_by_provider["gemini"] == 1
    assert summary.requests_by_provider["deepseek"] == 1
    assert summary.requests_by_provider["ollama"] == 1


# ============================================================================
# 3. Error Isolation Policy Tests
# ============================================================================

def test_telemetry_persistence_failure_does_not_raise():
    """If underlying storage throws an exception, record_execution traps it safely."""
    mock_storage = MagicMock()
    mock_storage.save.side_effect = RuntimeError("Disk full")

    service = TelemetryService(storage=mock_storage)

    # Must NOT raise RuntimeError
    rec = service.record_execution(
        request_id="req_fail_safe",
        provider="gemini",
        model="gemini-2.0-flash",
        latency_ms=800,
        status="success",
    )
    assert rec is not None
    assert rec.request_id == "req_fail_safe"
