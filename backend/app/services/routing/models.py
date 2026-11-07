"""Domain models and schemas for Intelligent Model Routing (Phase 6).

Enforces:
- Structured task categorization and complexity evaluation.
- Explicit capability definitions for model selection.
- Configurable multi-criteria scoring policies.
- Detailed routing diagnostics explaining selections and rejections.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class RoutingTaskCategory(str, Enum):
    """Categorical classification of incoming prompt tasks."""

    CODING = "coding"
    REASONING = "reasoning"
    MATHEMATICS = "mathematics"
    DATA_ANALYSIS = "data_analysis"
    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    CREATIVE_WRITING = "creative_writing"
    GENERAL_DIALOGUE = "general_dialogue"
    SYSTEM_ORCHESTRATION = "system_orchestration"


class LatencyTier(str, Enum):
    """Latency categorization tier for models."""

    ULTRA_FAST = "ultra_fast"  # Sub-second streaming (< 500ms)
    FAST = "fast"              # e.g. Gemini Flash (~500ms - 1.5s)
    MODERATE = "moderate"      # e.g. GPT-4o, DeepSeek-Chat (~1.5s - 4s)
    SLOW = "slow"              # e.g. o1, o3-mini, deepseek-reasoner (~4s - 20s+)


class CostTier(str, Enum):
    """Cost tier categorization."""

    FREE = "free"              # e.g. local Ollama ($0.00)
    BUDGET = "budget"          # < $0.50 / 1M tokens
    STANDARD = "standard"      # $0.50 - $5.00 / 1M tokens
    PREMIUM = "premium"        # > $5.00 / 1M tokens


class ComplexityScore(BaseModel):
    """Multi-indicator complexity assessment for a prompt."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0.0, le=1.0, description="Overall complexity index (0.0=trivial to 1.0=frontier-demanding)")
    token_count_estimate: int = Field(ge=0, description="Estimated input token count")
    constraint_count: int = Field(ge=0, description="Number of detected explicit constraints and prohibitions")
    has_code_syntax: bool = Field(default=False, description="Whether prompt contains code blocks or syntax")
    has_multi_step_logic: bool = Field(default=False, description="Whether multi-step instructions are required")
    reasoning_density: float = Field(ge=0.0, le=1.0, description="Density of reasoning markers and logic queries")
    justification: str = Field(description="Explanation of complexity calculation")


class ModelCapability(BaseModel):
    """Verified capability profile of a registered model."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(description="Unique model identifier matching provider API")
    provider_name: str = Field(description="Provider family name (e.g. gemini, deepseek, ollama)")
    context_window: int = Field(gt=0, description="Maximum input context tokens")
    max_output_tokens: int = Field(gt=0, description="Maximum output tokens")
    coding_score: float = Field(ge=0.0, le=1.0, description="Capability rating for code generation and debugging")
    reasoning_score: float = Field(ge=0.0, le=1.0, description="Capability rating for multi-hop and complex reasoning")
    general_quality_score: float = Field(ge=0.0, le=1.0, description="General knowledge and writing quality")
    latency_tier: LatencyTier = Field(description="Typical latency category")
    typical_latency_ms: int = Field(ge=0, description="Configured estimate of round-trip latency in ms")
    cost_tier: CostTier = Field(description="General cost classification")
    supports_system_prompt: bool = Field(default=True, description="Whether provider supports system role messages")
    supports_tools: bool = Field(default=False, description="Whether function calling is supported")


class RoutingConstraints(BaseModel):
    """User-specified or caller-enforced routing constraints."""

    model_config = ConfigDict(frozen=True)

    max_cost_usd: Optional[float] = Field(default=None, description="Upper bound on estimated request cost")
    max_latency_ms: Optional[int] = Field(default=None, description="Maximum acceptable typical latency in ms")
    preferred_providers: Optional[List[str]] = Field(default=None, description="Prioritize models from these providers")
    allowed_providers: Optional[List[str]] = Field(default=None, description="Hard whitelist of allowable providers")
    disallowed_models: Optional[List[str]] = Field(default=None, description="Blacklisted model identifiers")
    requires_coding: bool = Field(default=False, description="Enforce minimum coding proficiency")
    requires_reasoning: bool = Field(default=False, description="Enforce deep reasoning proficiency")
    requires_tools: bool = Field(default=False, description="Require function calling support")
    min_context_window: Optional[int] = Field(default=None, description="Minimum required context capacity")


class RoutingPolicy(BaseModel):
    """Configurable scoring weights and routing strategy."""

    model_config = ConfigDict(frozen=True)

    strategy: str = Field(
        default="balanced",
        description="Active strategy profile: 'balanced', 'cost_optimized', 'performance', or 'quality'",
    )
    weight_quality: float = Field(default=0.40, ge=0.0, le=1.0)
    weight_cost: float = Field(default=0.35, ge=0.0, le=1.0)
    weight_latency: float = Field(default=0.25, ge=0.0, le=1.0)
    fallback_chain: List[str] = Field(
        default_factory=lambda: ["gemini-2.0-flash", "deepseek-chat", "qwen2.5:1.5b"]
    )
    allow_fallback: bool = Field(default=True, description="Enable fallback when primary candidate fails")
    max_fallback_attempts: int = Field(default=3, ge=1, le=5)


class CandidateEvaluation(BaseModel):
    """Diagnostic evaluation record for a single candidate model."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    provider_name: str
    eligible: bool
    ineligibility_reason: Optional[str] = None
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    cost_score: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    composite_score: float = Field(default=0.0, ge=0.0, le=1.0)
    estimated_input_cost_usd: Optional[float] = None
    estimated_latency_ms: Optional[int] = None


class RoutingDecision(BaseModel):
    """Complete diagnostic routing decision produced by the ModelRoutingEngine."""

    model_config = ConfigDict(frozen=True)

    selected_model: str = Field(description="Target model identifier selected for execution")
    selected_provider: str = Field(description="Provider owning the selected model")
    task_category: RoutingTaskCategory = Field(description="Detected task domain")
    complexity: ComplexityScore = Field(description="Complexity evaluation of the request")
    applied_strategy: str = Field(description="Strategy name applied to balance scores")
    effective_weights: Dict[str, float] = Field(description="Actual weights used for composite scoring")
    candidate_evaluations: List[CandidateEvaluation] = Field(description="Detailed scoring for all evaluated models")
    fallback_chain: List[str] = Field(description="Ordered list of secondary models if primary fails")
    selection_rationale: str = Field(description="Explicit human-readable explanation of why model was chosen")
    rejections: Dict[str, str] = Field(default_factory=dict, description="Reasons for model ineligibility or rejection")
    routing_latency_ms: int = Field(ge=0, description="Time taken to evaluate routing decision in milliseconds")
