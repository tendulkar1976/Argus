"""RABCO Foundation Schemas: Immutable Conversation and Evidence Models.

Project: Argus — Risk-Aware, Counterfactual-Bounded Context Optimization (RABCO).
Phase: 1 — Foundation Schemas & Provenance Contracts.

Key Architectural Guarantees:
1. Immutability: Conversation messages and extracted evidence models are frozen (ConfigDict(frozen=True)).
2. Strict Provenance: Every evidence unit references exact source spans (message_id + character offsets).
3. Zero-Based Character Spans: Offsets are zero-based, end-exclusive [start_offset, end_offset).
4. Derived Summaries: SummaryUnits explicitly declare source evidence/message references.
5. Independent Research Domain: Completely decoupled from single-prompt optimizer and gateway logic.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ============================================================================
# Enums
# ============================================================================

class MessageRole(str, Enum):
    """Permitted conversational roles for multi-turn dialogue context."""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class EvidenceType(str, Enum):
    """Semantic taxonomy of atomic evidence units extracted from conversation turns."""

    FACT = "fact"
    INSTRUCTION = "instruction"
    CONSTRAINT = "constraint"
    PREFERENCE = "preference"
    GOAL = "goal"
    DECISION = "decision"
    CORRECTION = "correction"
    CONDITION = "condition"
    EXCEPTION = "exception"
    NUMERICAL_VALUE = "numerical_value"
    HISTORICAL_STATE = "historical_state"
    INTERMEDIATE_RESULT = "intermediate_result"
    OTHER = "other"


class EvidenceCriticality(str, Enum):
    """Categorical criticality tier for risk-aware retention auditing."""

    MANDATORY = "mandatory"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class RelationshipType(str, Enum):
    """Directed relational link types between atomic evidence units."""

    DEPENDS_ON = "depends_on"
    CORRECTS = "corrects"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    APPLIES_TO = "applies_to"
    CONDITIONAL_ON = "conditional_on"
    EXCEPTION_TO = "exception_to"
    REFERS_TO = "refers_to"
    SAME_ENTITY = "same_entity"
    RELATED_TO = "related_to"


class ValidationStatus(str, Enum):
    """Verification lifecycle state for evidence units and generated summaries."""

    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    REQUIRES_REVIEW = "requires_review"


class CompressionMode(str, Enum):
    """Context optimization operational profile."""

    EXTRACTIVE = "extractive"
    GENERATIVE = "generative"
    HYBRID = "hybrid"


# ============================================================================
# Core Conversation & Provenance Models
# ============================================================================

class ConversationMessage(BaseModel):
    """Immutable single turn within a multi-turn conversation history.

    Guarantees:
    - Immutable (frozen=True) to ensure the original conversation record cannot be mutated in place.
    - Preserves exact raw message content without pre-processing or normalization.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=False,
    )

    message_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the conversation turn (e.g. 'msg_001', UUID)",
    )
    role: MessageRole = Field(
        ...,
        description="Role of the message creator (system, developer, user, assistant, tool)",
    )
    content: str = Field(
        ...,
        description="Raw, unmutated text content of the message",
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Time the message was recorded (must be timezone-aware if provided)",
    )
    sequence_index: int = Field(
        ...,
        ge=0,
        description="Zero-based chronological position of the message in the conversation thread",
    )
    name: Optional[str] = Field(
        default=None,
        description="Optional participant or tool function name",
    )

    @field_validator("message_id")
    @classmethod
    def validate_message_id_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message_id cannot be whitespace-only.")
        return v

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_timezone_aware(cls, v: Optional[datetime]) -> Optional[datetime]:
        if v is not None and (v.tzinfo is None or v.tzinfo.utcoffset(v) is None):
            raise ValueError("timestamp must be timezone-aware (e.g., UTC).")
        return v


class SourceSpan(BaseModel):
    """Exact character coordinates identifying where an evidence unit originates within a message.

    Uses standard Python slicing conventions:
    - start_offset is zero-based and inclusive.
    - end_offset is zero-based and exclusive.
    - Span length is end_offset - start_offset.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    message_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the ConversationMessage containing this span",
    )
    start_offset: int = Field(
        ...,
        ge=0,
        description="Zero-based inclusive start character index in message content",
    )
    end_offset: int = Field(
        ...,
        gt=0,
        description="Zero-based exclusive end character index in message content",
    )

    @field_validator("message_id")
    @classmethod
    def validate_message_id_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message_id cannot be whitespace-only.")
        return v

    @model_validator(mode="after")
    def validate_span_bounds(self) -> "SourceSpan":
        if self.end_offset <= self.start_offset:
            raise ValueError(
                f"end_offset ({self.end_offset}) must be strictly greater than "
                f"start_offset ({self.start_offset})."
            )
        return self


class EvidenceUnit(BaseModel):
    """Atomic, provenance-anchored semantic unit extracted from a conversation span.

    Guarantees:
    - Immutable (frozen=True) to prevent post-extraction tampering.
    - Contains exact source span coordinates.
    - Self-validates that original_text length matches span length (cross-message content
      verification is deferred to the future service layer).
    - Immutable tuple collections for dependencies and contradictions.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    evidence_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the evidence unit (e.g. 'ev_001')",
    )
    source_span: SourceSpan = Field(
        ...,
        description="Exact message identifier and character offsets where this evidence originated",
    )
    evidence_type: EvidenceType = Field(
        ...,
        description="Semantic category of the evidence",
    )
    original_text: str = Field(
        ...,
        min_length=1,
        description="Exact verbatim substring extracted from the source message",
    )
    criticality: EvidenceCriticality = Field(
        default=EvidenceCriticality.UNKNOWN,
        description="Criticality classification for risk budgeting",
    )
    subject: Optional[str] = Field(
        default=None,
        description="Optional semantic subject of the statement or constraint",
    )
    relation: Optional[str] = Field(
        default=None,
        description="Optional semantic predicate or operational relation",
    )
    object_value: Optional[str] = Field(
        default=None,
        description="Optional semantic object or target value",
    )
    valid_from: Optional[datetime] = Field(
        default=None,
        description="Optional temporal start of validity (must be timezone-aware if present)",
    )
    valid_until: Optional[datetime] = Field(
        default=None,
        description="Optional temporal expiration of validity (must be timezone-aware if present)",
    )
    dependency_ids: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Identifiers of antecedent evidence units upon which this unit depends",
    )
    contradiction_ids: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Identifiers of evidence units that conflict with or contradict this unit",
    )
    extraction_method: str = Field(
        ...,
        min_length=1,
        description="Provenance of extraction technique (e.g., 'regex_rule', 'heuristic', 'slm_qwen2.5')",
    )
    validation_status: ValidationStatus = Field(
        default=ValidationStatus.PENDING,
        description="Current verification status of this evidence unit",
    )

    @field_validator("evidence_id", "extraction_method")
    @classmethod
    def validate_non_blank_strings(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be whitespace-only.")
        return v

    @field_validator("valid_from", "valid_until")
    @classmethod
    def validate_temporal_timezone_aware(cls, v: Optional[datetime]) -> Optional[datetime]:
        if v is not None and (v.tzinfo is None or v.tzinfo.utcoffset(v) is None):
            raise ValueError("Temporal validity timestamp must be timezone-aware.")
        return v

    @model_validator(mode="after")
    def validate_span_correspondence_and_temporal_order(self) -> "EvidenceUnit":
        expected_len = self.source_span.end_offset - self.source_span.start_offset
        actual_len = len(self.original_text)
        if actual_len != expected_len:
            raise ValueError(
                f"original_text length ({actual_len}) does not match source_span length ({expected_len}). "
                "Evidence text must correspond exactly to the character span."
            )

        if self.valid_from is not None and self.valid_until is not None:
            if self.valid_until < self.valid_from:
                raise ValueError(
                    f"valid_until ({self.valid_until}) cannot precede valid_from ({self.valid_from})."
                )

        return self


class EvidenceRelationship(BaseModel):
    """Directed relational link between two atomic evidence units."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    relationship_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for this relationship edge",
    )
    source_evidence_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the origin evidence unit",
    )
    target_evidence_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the destination evidence unit",
    )
    relationship_type: RelationshipType = Field(
        ...,
        description="Directed link classification (e.g. depends_on, supersedes, contradicts)",
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional contextual explanation of the relationship",
    )

    @field_validator("relationship_id", "source_evidence_id", "target_evidence_id")
    @classmethod
    def validate_ids_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Identifier cannot be whitespace-only.")
        return v

    @model_validator(mode="after")
    def validate_not_self_referential(self) -> "EvidenceRelationship":
        if self.source_evidence_id == self.target_evidence_id:
            raise ValueError(
                f"Self-referential relationship is invalid: source_evidence_id and "
                f"target_evidence_id are both '{self.source_evidence_id}'."
            )
        return self


class SummaryUnit(BaseModel):
    """Derived multi-span synthesis representing consolidated context.

    Guarantees:
    - Immutable (frozen=True).
    - Distinctly marked as derived information (never original evidence).
    - Requires at least one source evidence reference.
    - Explicit provenance tracking for generating provider and model if synthesized via LLM.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    summary_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for this derived summary unit",
    )
    summary_text: str = Field(
        ...,
        min_length=1,
        description="Synthesized summary text",
    )
    source_evidence_ids: Tuple[str, ...] = Field(
        ...,
        min_length=1,
        description="Tuple of evidence unit IDs from which this summary was derived (minimum 1 required)",
    )
    source_message_ids: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Tuple of source message IDs covered by this summary",
    )
    generation_method: str = Field(
        ...,
        min_length=1,
        description="Technique used for generation (e.g. 'extractive_condensation', 'selective_slm')",
    )
    provider_name: Optional[str] = Field(
        default=None,
        description="Name of provider if generated via an external or local model (e.g. 'ollama')",
    )
    model_identifier: Optional[str] = Field(
        default=None,
        description="Exact model identifier used if generated via LLM (e.g. 'qwen2.5:1.5b')",
    )
    validation_status: ValidationStatus = Field(
        default=ValidationStatus.PENDING,
        description="Current verification status of this summary",
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="Creation timestamp (must be timezone-aware if present)",
    )

    @field_validator("summary_id", "summary_text", "generation_method")
    @classmethod
    def validate_non_blank_strings(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be whitespace-only.")
        return v

    @field_validator("created_at")
    @classmethod
    def validate_created_at_timezone_aware(cls, v: Optional[datetime]) -> Optional[datetime]:
        if v is not None and (v.tzinfo is None or v.tzinfo.utcoffset(v) is None):
            raise ValueError("created_at timestamp must be timezone-aware.")
        return v

    @field_validator("source_evidence_ids")
    @classmethod
    def validate_at_least_one_evidence_source(cls, v: Tuple[str, ...]) -> Tuple[str, ...]:
        if not v:
            raise ValueError("A summary must reference at least one source evidence unit.")
        for item in v:
            if not item.strip():
                raise ValueError("source_evidence_ids cannot contain blank identifiers.")
        return v


# ============================================================================
# Request & Response Contracts
# ============================================================================

class RABCOOptimizeRequest(BaseModel):
    """Incoming request to optimize a multi-turn conversation history under risk and token constraints."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    messages: Tuple[ConversationMessage, ...] = Field(
        ...,
        min_length=1,
        description="Immutable sequence of conversation turns to optimize (minimum 1 turn required)",
    )
    current_query: str = Field(
        ...,
        min_length=1,
        description="The active user turn or prompt requiring optimal context",
    )
    target_token_budget: int = Field(
        ...,
        gt=0,
        description="Strict upper bound on total estimated tokens for optimized context (must be positive)",
    )
    risk_tolerance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Allowable risk threshold for context omission (0.0=conservative/strict, 1.0=aggressive)",
    )
    compression_mode: CompressionMode = Field(
        default=CompressionMode.HYBRID,
        description="Optimization strategy profile ('extractive', 'generative', 'hybrid')",
    )
    enable_counterfactual_audit: bool = Field(
        default=False,
        description="Whether to perform active counterfactual query probing during risk assessment",
    )

    @field_validator("current_query")
    @classmethod
    def validate_query_not_blank(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("current_query cannot be empty or whitespace-only.")
        return trimmed

    @field_validator("messages")
    @classmethod
    def validate_messages_not_empty(cls, v: Tuple[ConversationMessage, ...]) -> Tuple[ConversationMessage, ...]:
        if not v:
            raise ValueError("messages sequence cannot be empty.")
        return v


class RABCOOptimizeResponse(BaseModel):
    """Response containing the optimized conversational context and audit telemetry.

    Note on Token Accounting:
    - original_token_count and optimized_token_count represent locally estimated token footprints
      derived via character or tokenizer heuristics. They are distinct from actual billable
      tokens reported by downstream provider APIs.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    optimized_messages: Tuple[ConversationMessage, ...] = Field(
        ...,
        description="Optimized context window of conversation turns ready for downstream LLM dispatch",
    )
    extracted_evidence_count: int = Field(
        ...,
        ge=0,
        description="Total number of atomic evidence units extracted from conversation history",
    )
    retained_evidence_count: int = Field(
        ...,
        ge=0,
        description="Number of evidence units retained in the optimized context view",
    )
    summary_count: int = Field(
        ...,
        ge=0,
        description="Number of derived SummaryUnits generated to consolidate turns",
    )
    original_token_count: int = Field(
        ...,
        ge=0,
        description="Locally estimated token footprint of unoptimized input messages",
    )
    optimized_token_count: int = Field(
        ...,
        ge=0,
        description="Locally estimated token footprint of optimized messages view",
    )
    token_reduction_percent: float = Field(
        ...,
        description="Percentage reduction in estimated token footprint",
    )
    optimization_bypassed: bool = Field(
        ...,
        description="Whether optimization was bypassed (e.g. due to safety risk or small context)",
    )
    bypass_reason: Optional[str] = Field(
        default=None,
        description="Explanation if optimization was bypassed",
    )
    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Duration of RABCO pipeline execution in milliseconds",
    )
