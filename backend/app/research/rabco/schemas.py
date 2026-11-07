"""RABCO Phase 5A: Typed Schemas and Data Contracts.

Project: Argus — Risk-Aware, Counterfactual-Bounded Context Optimization (RABCO).
Phase: 5A — Typed Schemas and Data Contracts.

Architectural Guarantees:
1. Immutability: All schemas inherit BaseModel with ConfigDict(frozen=True, extra="forbid").
2. Evidence & Message Provenance: Probes, audit records, and recovery structures explicitly
   track originating source message IDs and evidence unit IDs.
3. Decoupled Auditor Interface: Auditor-facing probe data (AuditorFacingProbe) is strictly
   separated from expected evidence mappings (ProbeEvidenceBinding) to prevent circular answer-key leakage.
4. Discrete Preservation Taxonomy: Explicitly represents PRESERVED, PARTIALLY_PRESERVED, MISSING,
   AMBIGUOUS, and UNVERIFIABLE without conflating uncertainty with preservation.
5. Uncalibrated Risk Indicators: Risk metrics are transparent observable counts and sample ratios.
   They are explicitly distinguished from calibrated statistical probabilities.
6. Recovery & Invariant Reporting: Accurately represents recovery attempts, restored evidence IDs,
   budget conflict details, and fail-safe bypass rationales.
7. Centralized Configuration Sourcing: Operational defaults are sourced from app.core.config.settings
   without scattered hardcoding. All algorithm thresholds are marked as provisional research baselines.
"""

from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from app.core.config import settings
from app.schemas.rabco import (
    ConversationMessage,
    EvidenceCriticality,
    EvidenceRelationship,
    EvidenceType,
    EvidenceUnit,
    RelationshipType,
    SummaryUnit,
    ValidationStatus,
)


def _freeze_value(v: Any) -> Any:
    """Recursively freeze mappings, sequences, and sets to ensure deep immutability."""
    if isinstance(v, MappingProxyType):
        return MappingProxyType({k: _freeze_value(val) for k, val in v.items()})
    if isinstance(v, Mapping):
        return MappingProxyType({k: _freeze_value(val) for k, val in v.items()})
    if isinstance(v, (list, tuple)):
        return tuple(_freeze_value(item) for item in v)
    if isinstance(v, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in v)
    return v


def _freeze_mapping(v: Any) -> MappingProxyType:
    """Defensively copy and recursively wrap a mapping in an immutable MappingProxyType.

    Architectural Guarantees:
    - Deep immutability: Nested mappings and sequences are recursively frozen.
    - In-place mutation of the returned proxy is rejected with TypeError.
    - Defensive deep copying prevents external modifications to the original input mapping
      from retroactively altering the model's stored internal state.
    - Preserves standard Pydantic serialization compatibility via @field_serializer.
    """
    if v is None:
        return MappingProxyType({})
    if isinstance(v, Mapping):
        return MappingProxyType({k: _freeze_value(val) for k, val in v.items()})
    raise ValueError(f"Value must be a mapping, got {type(v).__name__}")


def _unfreeze_value(v: Any) -> Any:
    """Recursively convert MappingProxyType, frozensets, and tuples back to serializable Python structures.

    Architectural Guarantees:
    - model_dump() returns standard serializable Python dictionaries and lists.
    - model_dump_json() serializes to valid JSON without mappingproxy type errors.
    """
    if isinstance(v, (MappingProxyType, Mapping)):
        return {str(k): _unfreeze_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_unfreeze_value(item) for item in v]
    if isinstance(v, (set, frozenset)):
        return [_unfreeze_value(item) for item in v]
    return v


# ============================================================================
# Enums
# ============================================================================

class EvidencePreservationStatus(str, Enum):
    """Discrete categorization of evidence recoverability under counterfactual probing."""

    PRESERVED = "preserved"
    PARTIALLY_PRESERVED = "partially_preserved"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    UNVERIFIABLE = "unverifiable"


class ProbeCategory(str, Enum):
    """Categorical classification of counterfactual follow-up probes."""

    NUMERICAL_VALUE = "numerical_value"
    CORRECTION = "correction"
    CONSTRAINT = "constraint"
    EXCEPTION = "exception"
    DEPENDENCY = "dependency"
    CONTRADICTION = "contradiction"
    HISTORICAL_STATE = "historical_state"
    PREFERENCE_UPDATE = "preference_update"
    GENERAL = "general"


class ProbeSetType(str, Enum):
    """Partitioning taxonomy for counterfactual probes to prevent experimental data leakage."""

    DEVELOPMENT = "development"
    AUDIT = "audit"
    HELD_OUT = "held_out"


# ============================================================================
# Decoupled Probe Models (Auditor-Facing vs Expected Evidence Mapping)
# ============================================================================

class AuditorFacingProbe(BaseModel):
    """Auditor-visible probe representation stripped of expected evidence answer keys.

    Architectural Principle:
    The independent auditor evaluates whether a question is answerable from compressed context
    WITHOUT receiving the expected evidence IDs as a circular evaluation hint.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    probe_id: str = Field(..., min_length=1, description="Unique identifier for the probe")
    question: str = Field(..., min_length=1, description="Follow-up question to be evaluated")
    probe_category: ProbeCategory = Field(..., description="Targeted functional category")
    source_message_ids: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Message turn identifiers relevant to the question's origin",
    )
    probe_set: ProbeSetType = Field(
        default=ProbeSetType.AUDIT,
        description="Partition tier: audit, development, or held-out",
    )
    metadata: Mapping[str, Any] = Field(
        default_factory=dict,
        description="Non-ground-truth diagnostic annotations (e.g. template ID)",
    )

    @field_validator("probe_id", "question")
    @classmethod
    def validate_non_blank_strings(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be whitespace-only.")
        return v

    @field_validator("metadata", mode="after")
    @classmethod
    def validate_metadata_immutable(cls, v: Any) -> Mapping[str, Any]:
        return _freeze_mapping(v)

    @field_serializer("metadata", when_used="always")
    def serialize_metadata(self, v: Mapping[str, Any]) -> Dict[str, Any]:
        return _unfreeze_value(v)


class ProbeEvidenceBinding(BaseModel):
    """Ground-truth mapping between a probe and the evidence required to answer it.

    Kept separate from AuditorFacingProbe to prevent circular prompt evaluation leakage.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    probe_id: str = Field(..., min_length=1, description="Identifier of the associated probe")
    required_evidence_ids: Tuple[str, ...] = Field(
        ...,
        min_length=1,
        description="Declared ground-truth evidence unit IDs needed to answer this question",
    )
    related_relationship_ids: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Relational edge IDs relevant to the question's relational structure",
    )
    target_evidence_text: Optional[str] = Field(
        default=None,
        description="Verbatim anchor text from original evidence used to synthesize the probe",
    )

    @field_validator("probe_id")
    @classmethod
    def validate_probe_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("probe_id cannot be blank.")
        return v

    @field_validator("required_evidence_ids")
    @classmethod
    def validate_required_evidence_ids(cls, v: Tuple[str, ...]) -> Tuple[str, ...]:
        if not v:
            raise ValueError("At least one required evidence ID must be specified.")
        for item in v:
            if not item.strip():
                raise ValueError("Evidence IDs cannot be blank.")
        return v


class CounterfactualProbe(BaseModel):
    """Complete composite counterfactual probe uniting question data and ground-truth bindings.

    Provides explicit helper methods to extract auditor-facing views without answer keys.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    probe_id: str = Field(..., min_length=1, description="Unique deterministic identifier for this probe")
    question: str = Field(..., min_length=1, description="Plausible follow-up question")
    probe_category: ProbeCategory = Field(..., description="Targeted evidence classification")
    required_evidence_ids: Tuple[str, ...] = Field(
        ...,
        min_length=1,
        description="Source evidence unit IDs needed to answer this question (ground-truth reference)",
    )
    related_relationship_ids: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Relationship edge IDs relevant to the question's relational structure",
    )
    generation_method: str = Field(
        ...,
        min_length=1,
        description="Provenance of generation (e.g. 'deterministic_template', 'selective_llm_ollama')",
    )
    source_message_ids: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Originating message turn IDs for evidence provenance",
    )
    validation_status: ValidationStatus = Field(
        default=ValidationStatus.VALIDATED,
        description="Verification lifecycle status of this probe",
    )
    probe_set: ProbeSetType = Field(
        default=ProbeSetType.AUDIT,
        description="Partition type: audit (used in pipeline), development, or held-out (reserved)",
    )
    target_evidence_text: Optional[str] = Field(
        default=None,
        description="Verbatim anchor text from original evidence used to synthesize question",
    )
    metadata: Mapping[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostic metadata regarding generation cues",
    )

    @field_validator("probe_id", "question", "generation_method")
    @classmethod
    def validate_non_blank_strings(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be whitespace-only.")
        return v

    @field_validator("metadata", mode="after")
    @classmethod
    def validate_metadata_immutable(cls, v: Any) -> Mapping[str, Any]:
        return _freeze_mapping(v)

    @field_serializer("metadata", when_used="always")
    def serialize_metadata(self, v: Mapping[str, Any]) -> Dict[str, Any]:
        return _unfreeze_value(v)

    @field_validator("required_evidence_ids")
    @classmethod
    def validate_required_evidence_ids(cls, v: Tuple[str, ...]) -> Tuple[str, ...]:
        if not v:
            raise ValueError("At least one required evidence ID must be specified.")
        for item in v:
            if not item.strip():
                raise ValueError("Evidence IDs cannot be blank.")
        return v

    def to_auditor_view(self) -> AuditorFacingProbe:
        """Derive an auditor-facing probe view stripped of expected evidence answer keys."""
        return AuditorFacingProbe(
            probe_id=self.probe_id,
            question=self.question,
            probe_category=self.probe_category,
            source_message_ids=self.source_message_ids,
            probe_set=self.probe_set,
            metadata=self.metadata,
        )

    def to_evidence_binding(self) -> ProbeEvidenceBinding:
        """Derive the expected evidence ground-truth mapping for this probe."""
        return ProbeEvidenceBinding(
            probe_id=self.probe_id,
            required_evidence_ids=self.required_evidence_ids,
            related_relationship_ids=self.related_relationship_ids,
            target_evidence_text=self.target_evidence_text,
        )


class ProbeGenerationResult(BaseModel):
    """Result of probe generation containing generated probes, bindings, and telemetry.

    Generation Statistics (granular breakdown):
    - deterministic_count: Probes produced by relationship-aware and type-aware templates.
    - llm_candidate_count: Raw candidates received from the LLM response.
    - llm_accepted_count: LLM candidates that passed grounding and structural validation.
    - rejected_candidate_count: Total LLM candidates rejected (structural + evidence + duplicate).
    - duplicate_rejected_count: Candidates rejected specifically due to duplicate question normalization.
    - invalid_evidence_rejected_count: Candidates rejected due to missing or unverifiable evidence IDs.
    - total_accepted_count: Final accepted probe count (deterministic + llm_accepted_count).

    Research Integrity Limitation:
    Generated probes represent a finite sample from the probe category space.
    A passing preservation score indicates preservation only across the generated probe set,
    not across all possible future questions from the original conversation.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    probes: Tuple[CounterfactualProbe, ...] = Field(
        ...,
        description="Validated counterfactual probes ready for preservation auditing",
    )
    deterministic_count: int = Field(..., ge=0, description="Count of template-generated probes (type-based and relationship-based)")
    llm_candidate_count: int = Field(..., ge=0, description="Count of raw LLM candidate probes received before validation")
    llm_accepted_count: int = Field(
        default=0,
        ge=0,
        description="Count of LLM candidates accepted after grounding and structural validation",
    )
    rejected_candidate_count: int = Field(..., ge=0, description="Total LLM candidate probes rejected (structural + evidence + duplicate)")
    duplicate_rejected_count: int = Field(
        default=0,
        ge=0,
        description="Count of candidates rejected due to normalized question deduplication",
    )
    invalid_evidence_rejected_count: int = Field(
        default=0,
        ge=0,
        description="Count of candidates rejected due to missing or unverifiable evidence ID references",
    )
    total_accepted_count: int = Field(
        default=0,
        ge=0,
        description="Final total probes accepted (deterministic_count + llm_accepted_count)",
    )
    generation_method_summary: str = Field(
        ...,
        min_length=1,
        description="Summary of generators used (e.g. 'hybrid_deterministic_plus_ollama')",
    )


# ============================================================================
# Preservation Audit Models
# ============================================================================

class ProbeAuditResult(BaseModel):
    """Independent preservation evaluation for a single counterfactual probe."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    probe_id: str = Field(..., min_length=1, description="Identifier of the evaluated probe")
    question: str = Field(..., min_length=1, description="Probe question text")
    preservation_status: EvidencePreservationStatus = Field(
        ...,
        description="Outcome classification: preserved, partially_preserved, missing, ambiguous, unverifiable",
    )
    required_evidence_ids: Tuple[str, ...] = Field(
        ...,
        description="Declared ground-truth evidence units needed for this probe",
    )
    missing_evidence_ids: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Evidence unit IDs unrecoverable from the evaluated compressed context",
    )
    partially_preserved_ids: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Evidence unit IDs whose content is partially represented in summaries",
    )
    auditor_type: str = Field(
        ...,
        min_length=1,
        description="Layer that produced this audit (e.g. 'deterministic_provenance', 'semantic_slm')",
    )
    explanation: str = Field(..., min_length=1, description="Detailed rationale of the preservation assessment")
    uncertainty_flag: bool = Field(
        default=False,
        description="Flagged True if evaluation involved ambiguity rather than exact verification",
    )

    @field_validator("probe_id", "question", "auditor_type", "explanation")
    @classmethod
    def validate_non_blank_strings(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be whitespace-only.")
        return v


class RelationshipAuditDetail(BaseModel):
    """Detailed audit outcome for an individual EvidenceRelationship edge."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    relationship_id: str = Field(..., min_length=1, description="Identifier of the evaluated relationship")
    relationship_type: RelationshipType = Field(..., description="Relational edge type")
    source_evidence_id: str = Field(..., min_length=1, description="Source evidence ID")
    target_evidence_id: str = Field(..., min_length=1, description="Target evidence ID")
    source_preserved: bool = Field(..., description="Whether source evidence is preserved")
    target_preserved: bool = Field(..., description="Whether target evidence is preserved")
    is_applicable: bool = Field(..., description="Whether this relationship applies in the evaluated context")
    is_preserved: bool = Field(..., description="Whether relationship semantics were successfully preserved")
    failure_category: Optional[str] = Field(default=None, description="Category of relationship failure, if any")
    failure_explanation: Optional[str] = Field(default=None, description="Detailed explanation of failure, if any")

    @field_validator("relationship_id", "source_evidence_id", "target_evidence_id")
    @classmethod
    def validate_non_blank_strings(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be whitespace-only.")
        return v


class RiskAssessment(BaseModel):
    """Transparent multi-indicator preservation vector without uncalibrated probabilities.

    Scientific Principle:
    These indicators represent discrete sample ratios and observable violation counts.
    They are explicitly distinct from calibrated statistical probabilities.

    Safety Invariants vs. Configurable Tolerance:
    - mandatory_evidence_loss: STRICT SAFETY INVARIANT. Must be 0. Any loss of mandatory evidence
      (missing, partially preserved, ambiguous, or unverifiable) violates core safety requirements
      and unconditionally mandates selective recovery or original-context bypass.
    - non_mandatory_probe_failure_rate: Separately configurable tolerance threshold evaluated
      against policy settings. Zero failure on non-mandatory probes is NOT a universal safety requirement.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    total_probes_audited: int = Field(..., ge=0, description="Number of validated audit probes evaluated")
    total_probes_generated: int = Field(
        default=0,
        ge=0,
        description="Total number of candidate probes generated before audit sampling",
    )
    probe_failure_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall ratio of failed/missing/ambiguous probes to total audited probes",
    )
    non_mandatory_probe_failure_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ratio of failed/missing/ambiguous non-mandatory probes to total non-mandatory probes",
    )
    probe_coverage: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio of fully preserved probes to total evaluated or generated probes",
    )
    mandatory_evidence_loss: int = Field(
        ...,
        ge=0,
        description=(
            "Count of mandatory evidence units missing, partially preserved, ambiguous, or unverifiable. "
            "STRICT SAFETY INVARIANT: Any value > 0 indicates critical evidence loss "
            "and unconditionally mandates selective recovery or original-context bypass."
        ),
    )
    dependency_preservation: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Ratio of preserved modeled dependencies to total applicable dependencies, or None if unavailable",
    )
    contradiction_preservation: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Ratio of preserved contradiction pairs, or None if unavailable",
    )
    summary_integrity_pass_count: int = Field(
        ...,
        ge=0,
        description="Count of summaries passing grounded validation in compressed view",
    )
    summary_integrity_uncertain_count: int = Field(
        ...,
        ge=0,
        description="Count of summaries with unverified or ambiguous semantic support",
    )
    recovery_effectiveness: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Ratio of previously missing checks successfully resolved after recovery, or None if recovery was not performed",
    )
    bypass_recommended: bool = Field(
        ...,
        description="Whether risk indicators recommend original-context bypass",
    )
    bypass_reason: Optional[str] = Field(
        default=None,
        description="Explicit explanation if bypass is recommended",
    )
    composite_risk_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Provisional heuristic composite risk score normalized across available dimensions, "
            "or None if no dimensions are available. Research heuristic only; not a calibrated probability."
        ),
    )
    relationship_audit_details: Tuple[RelationshipAuditDetail, ...] = Field(
        default_factory=tuple,
        description="Typed, immutable audit records for all evaluated relationship edges",
    )
    indicator_calculation_methods: Mapping[str, Any] = Field(
        default_factory=dict,
        description="Documentation of exact formula and methodology for each indicator",
    )
    probability_calibration_disclaimer: str = Field(
        default="Indicators represent observable counts and sample ratios. They are not calibrated probabilities.",
        description="Scientific disclaimer distinguishing observable metrics from true probabilities",
    )

    @field_validator("indicator_calculation_methods", mode="after")
    @classmethod
    def validate_methods_immutable(cls, v: Any) -> Mapping[str, Any]:
        return _freeze_mapping(v)

    @field_serializer("indicator_calculation_methods", when_used="always")
    def serialize_methods(self, v: Mapping[str, Any]) -> Dict[str, Any]:
        return _unfreeze_value(v)


# ============================================================================
# Selective Evidence Recovery Models
# ============================================================================

class RecoveryRecord(BaseModel):
    """Audit log of a single selective restoration cycle."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    attempt_number: int = Field(..., ge=1, description="Sequential recovery attempt index")
    restored_evidence_ids: Tuple[str, ...] = Field(
        ...,
        description="Original evidence units restored from immutable source history",
    )
    restored_relationship_ids: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Relational edges re-activated by restoration",
    )
    missing_evidence_ids_targeted: Tuple[str, ...] = Field(
        ...,
        description="Failed probe evidence IDs targeted by this recovery cycle",
    )
    token_estimate_before: int = Field(..., ge=0, description="Estimated token footprint prior to restoration")
    token_estimate_after: int = Field(..., ge=0, description="Estimated token footprint after restoration")
    restoration_reason: str = Field(..., min_length=1, description="Audit rationale explaining why units were restored")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Time of recovery execution (timezone-aware UTC)",
    )

    @field_validator("restoration_reason")
    @classmethod
    def validate_non_blank_reason(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("restoration_reason cannot be blank.")
        return v


class RecoveryResult(BaseModel):
    """Terminal outcome of the bounded selective recovery loop."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    final_selected_evidence: Tuple[EvidenceUnit, ...] = Field(
        ...,
        description="Final set of verbatim original evidence units retained",
    )
    final_validated_summaries: Tuple[SummaryUnit, ...] = Field(
        ...,
        description="Final set of validated SummaryUnits retained",
    )
    recovery_attempts: Tuple[RecoveryRecord, ...] = Field(
        default_factory=tuple,
        description="Chronological log of recovery iterations executed",
    )
    total_recovery_attempts: int = Field(..., ge=0, description="Count of recovery cycles executed")
    recovery_limit_reached: bool = Field(..., description="Whether max recovery attempts was exhausted")
    final_preservation_status: str = Field(
        ...,
        min_length=1,
        description="Terminal status (e.g. 'audit_passed', 'recovery_exhausted', 'bypassed')",
    )
    bypass_required: bool = Field(..., description="Whether context must revert to immutable original messages")
    bypass_reason: Optional[str] = Field(default=None, description="Diagnostic reason if bypass is required")


# ============================================================================
# Policies & Final Deliverable Contract
# ============================================================================

class RABCOAuditPolicy(BaseModel):
    """Configurable audit parameters for Phase 5 risk assessment.

    Centralized Sourcing & Independence Principles:
    - All defaults are sourced from app.core.config.settings with zero scattered hardcoding.
    - Model Independence vs. Runtime Reuse: Using the same local Ollama model runtime provides
      prompt-level independence only, NOT physical model independence. Physical independence
      requires configuring a distinct secondary model endpoint or weight set via
      RABCO_AUDITOR_OLLAMA_MODEL / RABCO_AUDITOR_OLLAMA_BASE_URL.
    - Failure Tolerance Separation: Zero tolerance for mandatory evidence loss is an inviolable
      safety invariant. Non-mandatory probe failure tolerance is separately configurable via
      max_acceptable_non_mandatory_failure_rate. Zero tolerance on non-mandatory probes is NOT
      a universal safety requirement; the default 0.0 is an experimental research starting baseline.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    enable_deterministic_audit: bool = Field(
        default=True,
        description="Whether to perform structural and provenance checks against source evidence",
    )
    enable_semantic_audit: bool = Field(
        default_factory=lambda: settings.RABCO_ENABLE_SEMANTIC_AUDIT,
        description="Whether to query independent local SLM for subtle semantic preservation checks",
    )
    auditor_ollama_model: str = Field(
        default_factory=lambda: settings.RABCO_AUDITOR_OLLAMA_MODEL or settings.OLLAMA_MODEL,
        description=(
            "Local Ollama model identifier used for independent semantic auditing. "
            "Defaults to settings.RABCO_AUDITOR_OLLAMA_MODEL or settings.OLLAMA_MODEL. "
            "RESEARCH NOTE: Sharing the default model runtime provides prompt-level independence "
            "only, NOT physical model independence. Physical independence requires configuring a "
            "distinct secondary model endpoint or weight set."
        ),
    )
    auditor_ollama_base_url: str = Field(
        default_factory=lambda: settings.RABCO_AUDITOR_OLLAMA_BASE_URL or settings.OLLAMA_BASE_URL,
        description="Base URL for independent auditor model inference. Defaults to settings.RABCO_AUDITOR_OLLAMA_BASE_URL or settings.OLLAMA_BASE_URL",
    )
    auditor_timeout_seconds: float = Field(
        default=settings.OLLAMA_TIMEOUT_SECONDS,
        description="Timeout in seconds for independent semantic auditing requests",
    )
    auditor_temperature: float = Field(
        default_factory=lambda: settings.RABCO_AUDITOR_TEMPERATURE,
        ge=0.0,
        le=1.0,
        description=(
            "Inference temperature for independent auditor model (0.0 to 1.0). "
            "Sourced from settings.RABCO_AUDITOR_TEMPERATURE (provisional baseline default: 0.0 "
            "for deterministic evaluation)."
        ),
    )
    chars_per_token_heuristic: float = Field(
        default=settings.CHARS_PER_TOKEN_HEURISTIC,
        gt=0.0,
        description="Character ratio heuristic for token estimation",
    )
    max_acceptable_non_mandatory_failure_rate: float = Field(
        default_factory=lambda: settings.RABCO_MAX_NON_MANDATORY_FAILURE_RATE,
        ge=0.0,
        le=1.0,
        description=(
            "Provisional tolerance threshold for non-mandatory probe failures (0.0 to 1.0). "
            "Sourced from settings.RABCO_MAX_NON_MANDATORY_FAILURE_RATE. Default 0.0 is an "
            "experimental starting baseline; zero tolerance for all probe failures is NOT a "
            "universal safety requirement. Mandatory evidence loss is enforced separately as "
            "an inviolable safety invariant."
        ),
    )
    max_acceptable_failure_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional alias for max_acceptable_non_mandatory_failure_rate. If explicitly provided, "
            "it sets the non-mandatory failure tolerance."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_failure_rate_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "max_acceptable_failure_rate" in data and data["max_acceptable_failure_rate"] is not None:
                if "max_acceptable_non_mandatory_failure_rate" not in data:
                    data["max_acceptable_non_mandatory_failure_rate"] = data["max_acceptable_failure_rate"]
        return data


class RABCORecoveryPolicy(BaseModel):
    """Configurable recovery parameters for Phase 5 bounded restoration.

    Centralized Sourcing:
    - max_recovery_attempts is sourced from settings.RABCO_MAX_RECOVERY_ATTEMPTS.
      The provisional baseline default (2) is an experimental starting value,
      not an empirically verified optimum.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    max_recovery_attempts: int = Field(
        default_factory=lambda: settings.RABCO_MAX_RECOVERY_ATTEMPTS,
        ge=1,
        le=5,
        description=(
            "Strict upper limit on sequential recovery iterations. "
            "Sourced from settings.RABCO_MAX_RECOVERY_ATTEMPTS. "
            "Provisional baseline default: 2 (experimental starting value, not an empirically verified optimum)."
        ),
    )
    enable_relational_closure: bool = Field(
        default=True,
        description="Whether to pull prerequisite dependencies and conditions into the restoration set",
    )
    bypass_on_failure: bool = Field(
        default=True,
        description="Whether to revert to immutable original messages if recovery cannot establish safety",
    )


class RABCOPhase5Result(BaseModel):
    """Complete structured deliverable of Phase 5 counterfactual risk auditing and recovery."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    final_context_messages: Tuple[ConversationMessage, ...] = Field(
        ...,
        description="Optimized context messages ready for downstream consumption, or original messages if bypassed",
    )
    original_messages_ref: Tuple[ConversationMessage, ...] = Field(
        ...,
        description="Immutable reference to the uncompressed original conversation turns",
    )
    probes: Tuple[CounterfactualProbe, ...] = Field(
        ...,
        description="Counterfactual probes generated and validated for preservation testing",
    )
    audit_results: Tuple[ProbeAuditResult, ...] = Field(
        ...,
        description="Independent audit results per counterfactual probe",
    )
    missing_evidence_ids: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Evidence unit IDs identified as missing or degraded during auditing",
    )
    restored_evidence_ids: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Original evidence unit IDs restored during selective recovery cycles",
    )
    recovery_attempt_count: int = Field(
        default=0,
        ge=0,
        description="Number of bounded recovery attempts performed",
    )
    recovery_records: Tuple[RecoveryRecord, ...] = Field(
        default_factory=tuple,
        description="Detailed chronological audit log of each recovery iteration",
    )
    final_preservation_status: str = Field(
        ...,
        min_length=1,
        description="Final preservation state (e.g. 'verified_preserved', 'recovered_preserved', 'bypassed_safe')",
    )
    budget_status: str = Field(
        default="unconstrained",
        description="Budget evaluation: 'feasible', 'budget_conflict', or 'unconstrained'",
    )
    budget_conflict_details: Optional[str] = Field(
        default=None,
        description="Diagnostic details when mandatory context exceeds target budget",
    )
    risk_assessment: RiskAssessment = Field(
        ...,
        description="Multi-indicator preservation vector documenting risk metrics",
    )
    original_token_estimate: int = Field(
        ...,
        ge=0,
        description="Estimated token footprint of uncompressed input context",
    )
    optimized_token_estimate: int = Field(
        ...,
        ge=0,
        description="Estimated token footprint of final optimized context messages",
    )
    token_reduction_percent: float = Field(
        ...,
        description="Percentage reduction in token footprint achieved by optimized context",
    )
    token_estimation_method: str = Field(
        ...,
        min_length=1,
        description="Method used to calculate token footprints",
    )
    bypassed: bool = Field(
        ...,
        description="Whether compression was bypassed in favor of verbatim original conversation history",
    )
    bypass_reason: Optional[str] = Field(
        default=None,
        description="Explicit rationale explaining why bypass was triggered",
    )
    auditor_configuration: Mapping[str, Any] = Field(
        default_factory=dict,
        description="Model and policy settings used during independent preservation auditing",
    )
    processing_time_ms: int = Field(
        default=0,
        ge=0,
        description="Total duration of Phase 5 execution in milliseconds",
    )

    @field_validator("auditor_configuration", mode="after")
    @classmethod
    def validate_auditor_config_immutable(cls, v: Any) -> Mapping[str, Any]:
        return _freeze_mapping(v)

    @field_serializer("auditor_configuration", when_used="always")
    def serialize_auditor_config(self, v: Mapping[str, Any]) -> Dict[str, Any]:
        return _unfreeze_value(v)
