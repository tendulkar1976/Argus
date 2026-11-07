"""Data models and schemas for the Cost & Telemetry Engine (Phase 8).

Defines typed telemetry records, aggregated metrics, and query filters.
Guarantees:
- Clear separation between provider-reported cost, Argus-calculated cost, and unknown cost.
- Comprehensive request lifecycle tracking (routing, latency, optimization, error categorization).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class TelemetryRecord(BaseModel):
    """Immutable audit record representing execution telemetry for a single model request."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(description="Unique request trace identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="ISO-8601 UTC timestamp of execution",
    )
    provider: str = Field(description="Target provider family (e.g. gemini, deepseek, ollama)")
    model: str = Field(description="Target model identifier used for generation")
    task_category: str = Field(default="general", description="Classified task domain")
    routing_decision: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Summary of routing decision and alternative models evaluated",
    )
    input_tokens: Optional[int] = Field(default=None, ge=0, description="Actual input prompt tokens")
    output_tokens: Optional[int] = Field(default=None, ge=0, description="Actual output completion tokens")
    cached_tokens: Optional[int] = Field(default=None, ge=0, description="Tokens read from prompt cache")
    total_tokens: Optional[int] = Field(default=None, ge=0, description="Total billed/consumed tokens")
    estimated_cost_usd: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Argus decimal-safe calculated cost based on verified rates",
    )
    provider_reported_cost_usd: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Billed cost explicitly reported by provider API, if available",
    )
    cost_status: str = Field(
        description="Cost resolution: 'calculated', 'calculated_local', 'provider_reported', 'pricing_unavailable', or 'usage_unavailable'",
    )
    currency: str = Field(default="USD", description="Currency unit")
    latency_ms: int = Field(ge=0, description="Total provider execution latency in milliseconds")
    time_to_first_token_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Streaming time to first token if measured",
    )
    status: str = Field(description="Execution status: 'success', 'error', 'fallback'")
    error_category: Optional[str] = Field(
        default=None,
        description="Normalized category if request failed (e.g. timeout, rate_limit, auth)",
    )
    retry_count: int = Field(default=0, ge=0, description="Number of retries attempted")
    fallback_used: bool = Field(default=False, description="Whether execution fell back to a secondary model")
    fallback_model_from: Optional[str] = Field(
        default=None,
        description="Primary model that failed before fallback succeeded",
    )
    rabco_processing_time_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Duration of RABCO context optimization pipeline in ms",
    )
    prompt_optimization_metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Token savings and compression metrics if prompt was optimized",
    )


class TelemetrySummary(BaseModel):
    """Aggregated usage and financial metrics across a query window."""

    model_config = ConfigDict(frozen=True)

    total_requests: int = Field(ge=0)
    successful_requests: int = Field(ge=0)
    failed_requests: int = Field(ge=0)
    fallback_requests: int = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_cached_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    total_estimated_cost_usd: float = Field(ge=0.0)
    average_latency_ms: float = Field(ge=0.0)
    requests_by_provider: Dict[str, int] = Field(default_factory=dict)
    requests_by_model: Dict[str, int] = Field(default_factory=dict)
    requests_by_task: Dict[str, int] = Field(default_factory=dict)
    cost_by_provider: Dict[str, float] = Field(default_factory=dict)


class TelemetryQueryFilter(BaseModel):
    """Query filter parameters for retrieving telemetry records."""

    provider: Optional[str] = None
    model: Optional[str] = None
    task_category: Optional[str] = None
    status: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
