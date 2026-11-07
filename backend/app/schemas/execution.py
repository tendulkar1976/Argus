"""Request and response schemas for the Unified Execution Pipeline (Phase 10)."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.rabco import ConversationMessage
from app.services.routing.models import RoutingConstraints


class UnifiedExecutionRequest(BaseModel):
    """Request schema for end-to-end Argus pipeline execution."""

    model_config = ConfigDict(extra="forbid")

    request_id: Optional[str] = Field(
        default=None,
        description="Optional client-provided correlation ID (auto-generated if omitted)",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Single prompt text for single-turn optimization and dispatch",
    )
    messages: Optional[List[ConversationMessage]] = Field(
        default=None,
        description="Multi-turn conversation history for RABCO context optimization",
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional explicit model identifier override, bypassing routing selection",
    )
    provider: Optional[str] = Field(
        default=None,
        description="Optional explicit provider identifier override",
    )
    strategy: str = Field(
        default="balanced",
        description="Routing policy strategy ('balanced', 'cost_optimized', 'performance', 'quality')",
    )
    constraints: Optional[RoutingConstraints] = Field(
        default=None,
        description="Optional routing constraints (e.g. max_cost, max_latency, allowed_providers)",
    )
    enable_rabco: bool = Field(
        default=True,
        description="Enable multi-turn RABCO context optimization when messages are provided",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system instruction for the downstream model",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum generation output tokens",
    )

    @field_validator("prompt", mode="before")
    @classmethod
    def validate_prompt_or_messages(cls, v: Optional[str], info: Any) -> Optional[str]:
        # Handled in root/model validator if needed
        return v

    def get_effective_request_id(self) -> str:
        return self.request_id or f"argus_{uuid.uuid4().hex[:12]}"


class UnifiedExecutionResponse(BaseModel):
    """Normalized response schema for the unified Argus pipeline."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(description="Unique request trace identifier")
    model_response: str = Field(description="Final generated textual response from the model")
    effective_prompt: str = Field(description="Prompt text delivered to the downstream model")
    selected_model: str = Field(description="Model used for generation")
    selected_provider: str = Field(description="Provider family that executed the request")
    fallback_used: bool = Field(default=False, description="Whether fallback model was invoked")
    fallback_model_from: Optional[str] = Field(default=None, description="Primary model that failed if fallback was used")
    execution_latency_ms: int = Field(ge=0, description="Total execution duration in milliseconds")
    routing: Optional[Dict[str, Any]] = Field(default=None, description="Routing diagnostics")
    rabco: Optional[Dict[str, Any]] = Field(default=None, description="RABCO context optimization diagnostics")
    telemetry: Optional[Dict[str, Any]] = Field(default=None, description="Recorded telemetry metadata")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Extracted token usage metrics")
    cost_usd: Optional[float] = Field(default=None, description="Calculated or provider-reported cost in USD")
    cost_status: str = Field(description="Cost calculation status")
