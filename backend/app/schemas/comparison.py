"""Pydantic schemas for Dual Prompt Execution and Explainable Comparison (Phase 11.1).

Defines contracts for:
- Individual execution telemetry and response metrics.
- Transformation reasoning records explaining prompt modifications.
- Dimension-by-dimension response quality comparisons.
- Performance and cost accounting between original and optimized executions.
- Dual execution request and response envelopes.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.optimizer import OptimizeResponse


ComparisonWinner = Literal["original", "optimized", "equivalent", "inconclusive"]
OverallVerdict = Literal["optimized_better", "original_better", "similar", "inconclusive"]
TransformationCategory = Literal[
    "redundancy_removal",
    "instruction_clarification",
    "constraint_preservation",
    "context_restructuring",
    "ambiguity_reduction",
    "output_format_specification",
    "unnecessary_wording_removal",
    "general_optimization",
]


class ExecutionMetricDetails(BaseModel):
    """Detailed telemetry and resource metrics for a single LLM execution."""

    model_config = ConfigDict(frozen=True)

    input_tokens: Optional[int] = Field(default=None, description="Actual input prompt tokens reported by provider")
    output_tokens: Optional[int] = Field(default=None, description="Actual generated output tokens reported by provider")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens consumed")
    cached_tokens: Optional[int] = Field(default=None, description="Tokens read from prompt cache")
    latency_ms: int = Field(ge=0, description="End-to-end execution latency in milliseconds")
    cost_usd: Optional[float] = Field(default=None, description="Calculated or provider-reported cost in USD")
    cost_status: str = Field(default="estimated", description="Cost provenance: 'actual', 'estimated', 'free', 'unavailable'")
    provider: str = Field(description="Provider family name (e.g. 'gemini', 'deepseek', 'ollama')")
    model: str = Field(description="Exact model identifier used for inference")
    status: str = Field(default="success", description="Execution outcome: 'success', 'error', 'timeout'")
    error_message: Optional[str] = Field(default=None, description="Error detail if execution failed")


class SingleExecutionResult(BaseModel):
    """Captures prompt delivery and response output for one execution branch."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(description="Exact prompt text dispatched to the LLM")
    response_text: str = Field(default="", description="Generated textual output from the LLM")
    metrics: ExecutionMetricDetails = Field(description="Telemetry and resource accounting")
    fallback_used: bool = Field(default=False, description="Whether fallback model was triggered")
    fallback_from: Optional[str] = Field(default=None, description="Primary model if fallback occurred")


class TransformationRecord(BaseModel):
    """Captures an individual, observable transformation applied to the prompt."""

    model_config = ConfigDict(frozen=True)

    original_segment: str = Field(description="Original prompt text segment before modification")
    optimized_segment: str = Field(description="Resulting text segment after optimization")
    category: TransformationCategory = Field(description="Categorical nature of the transformation")
    reason: str = Field(description="Specific rationale for making this modification")
    expected_benefit: str = Field(description="Anticipated execution or cognitive benefit")
    intent_preserved: bool = Field(default=True, description="Whether original semantic intent was preserved")
    potential_tradeoff_or_risk: Optional[str] = Field(
        default=None,
        description="Any potential trade-off, information loss, or downstream risk",
    )


class DimensionEvaluation(BaseModel):
    """Evaluation score and evidence for a single comparison dimension."""

    model_config = ConfigDict(frozen=True)

    dimension: str = Field(description="Name of the evaluated quality dimension")
    dimension_label: str = Field(description="Human-readable title of the dimension")
    original_assessment: str = Field(description="Qualitative assessment of Response A (Original)")
    optimized_assessment: str = Field(description="Qualitative assessment of Response B (Optimized)")
    difference: str = Field(description="Key observable differences between the two outputs")
    winner: ComparisonWinner = Field(description="Which response performed better on this dimension")
    explanation: str = Field(description="Concise, evidence-based evaluation rationale")


class ResponseComparison(BaseModel):
    """Structured, multi-dimensional comparison of the two generated responses."""

    model_config = ConfigDict(frozen=True)

    dimensions: List[DimensionEvaluation] = Field(
        default_factory=list,
        description="Evaluated quality dimensions",
    )
    overall_verdict: OverallVerdict = Field(
        default="similar",
        description="Synthesized comparison verdict",
    )
    summary_rationale: str = Field(description="Summary explanation supporting the overall verdict")
    original_strengths: List[str] = Field(default_factory=list, description="Observed strengths of the original response")
    original_weaknesses: List[str] = Field(default_factory=list, description="Observed weaknesses of the original response")
    optimized_strengths: List[str] = Field(default_factory=list, description="Observed strengths of the optimized response")
    optimized_weaknesses: List[str] = Field(default_factory=list, description="Observed weaknesses of the optimized response")
    evaluator_metrics: Optional[ExecutionMetricDetails] = Field(
        default=None,
        description="Isolated telemetry and cost for the comparison evaluation step",
    )


class ComparisonPerformanceMetrics(BaseModel):
    """Comparative performance, token footprint, and economic variance."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens_original: int = Field(ge=0, description="Input tokens for original prompt")
    prompt_tokens_optimized: int = Field(ge=0, description="Input tokens for optimized prompt")
    prompt_tokens_saved: int = Field(description="Net input tokens saved (orig - opt)")
    prompt_token_reduction_percent: float = Field(description="Percentage reduction in prompt tokens")

    response_tokens_original: Optional[int] = Field(default=None, description="Output tokens generated by original prompt")
    response_tokens_optimized: Optional[int] = Field(default=None, description="Output tokens generated by optimized prompt")
    response_token_difference: Optional[int] = Field(default=None, description="Difference in output tokens (opt - orig)")

    total_tokens_original: Optional[int] = Field(default=None, description="Total tokens consumed by original execution")
    total_tokens_optimized: Optional[int] = Field(default=None, description="Total tokens consumed by optimized execution")
    total_token_difference: Optional[int] = Field(default=None, description="Net token difference across full execution")

    cost_original_usd: Optional[float] = Field(default=None, description="Cost of original prompt execution in USD")
    cost_optimized_usd: Optional[float] = Field(default=None, description="Cost of optimized prompt execution in USD")
    cost_saved_usd: Optional[float] = Field(default=None, description="Net cost saved in USD (orig - opt)")
    cost_savings_percent: Optional[float] = Field(default=None, description="Percentage cost saved")

    latency_original_ms: int = Field(ge=0, description="Latency of original execution in ms")
    latency_optimized_ms: int = Field(ge=0, description="Latency of optimized execution in ms")
    latency_difference_ms: int = Field(description="Difference in latency (opt - orig)")


class DualExecutionRequest(BaseModel):
    """Request payload to perform dual prompt optimization, A/B execution, and comparison."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1, description="Raw input prompt to optimize and test")
    mode: str = Field(default="balanced", description="Optimization profile: 'balanced', 'concise', 'structure'")
    target_model: str = Field(default="gemini-2.0-flash", description="Target frontier model identifier")
    system_prompt: Optional[str] = Field(default=None, description="Optional common system instruction for both executions")
    temperature: Optional[float] = Field(default=0.2, ge=0.0, le=2.0, description="Common sampling temperature")
    max_tokens: Optional[int] = Field(default=None, gt=0, description="Common maximum generation tokens")
    run_comparison: bool = Field(default=True, description="Whether to run the LLM-based response quality comparison")


class DualExecutionResponse(BaseModel):
    """Full A/B execution and explainable comparison envelope."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(default_factory=lambda: f"ab_{uuid.uuid4().hex[:12]}")
    timestamp: str = Field(description="ISO-8601 execution timestamp")
    target_model: str = Field(description="Model used for dual execution")
    target_provider: str = Field(description="Provider used for dual execution")
    is_equivalent_conditions: bool = Field(
        default=True,
        description="Whether both executions operated under identical model, provider, and configuration conditions",
    )
    equivalence_notes: Optional[str] = Field(
        default=None,
        description="Notes explaining any differences in execution conditions (e.g. fallback)",
    )
    optimization: OptimizeResponse = Field(description="Prompt optimization results and task metadata")
    transformations: List[TransformationRecord] = Field(
        default_factory=list,
        description="Granular explanation of what Argus changed",
    )
    original_execution: SingleExecutionResult = Field(description="Execution A results (Original Prompt)")
    optimized_execution: SingleExecutionResult = Field(description="Execution B results (Optimized Prompt)")
    comparison: Optional[ResponseComparison] = Field(
        default=None,
        description="Detailed response quality and dimensional comparison",
    )
    performance: ComparisonPerformanceMetrics = Field(description="Token, cost, and latency comparisons")
