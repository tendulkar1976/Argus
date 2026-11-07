"""RABCO: Risk-Aware, Counterfactual-Bounded Context Optimization research package."""

# Phase 2: Semantic Evidence Extraction Engine
from .evidence_extractor import EvidenceExtractor, ProvenanceVerificationError

# Phase 3: Evidence Relationship Mapping
from .relationship_mapper import (
    RelationshipMapper,
    RelationshipRule,
    RelationshipRuleRegistry,
    RelationshipThresholds,
    RelationshipValidationError,
    create_default_registry,
)

# Phase 4: Hybrid Context Compression
from .hybrid_compressor import (
    CompressionPolicy,
    CompressionResult,
    HybridCompressor,
    OmittedEvidence,
    SummaryValidationOutcome,
)

# Phase 5B: Counterfactual Probe Generator
from .counterfactual_generator import CounterfactualProbeGenerator

# Phase 5C: Risk Model
from .risk_model import RiskModel, RiskScoringWeights

# Phase 5D: Preservation Auditor
from .preservation_auditor import PreservationAuditor

# Phase 5E: Selective Evidence Recovery
from .selective_recovery import (
    RecoveryDiagnostics,
    SelectiveRecoveryEngine,
    SelectiveRecoveryOutcome,
)

# Phase 5F: End-to-End Orchestration Layer
from .orchestrator import (
    OrchestrationDecision,
    PipelineStage,
    RABCOOrchestrationResult,
    RABCOOrchestrator,
    RABCOPipelineError,
)

# Phase 5A: Counterfactual Probing, Risk Auditing, and Recovery Contracts
from .schemas import (
    AuditorFacingProbe,
    CounterfactualProbe,
    EvidencePreservationStatus,
    ProbeAuditResult,
    ProbeCategory,
    ProbeEvidenceBinding,
    ProbeGenerationResult,
    ProbeSetType,
    RABCOAuditPolicy,
    RABCOPhase5Result,
    RABCORecoveryPolicy,
    RecoveryRecord,
    RecoveryResult,
    RelationshipAuditDetail,
    RiskAssessment,
)

__all__ = [
    # Phase 2
    "EvidenceExtractor",
    "ProvenanceVerificationError",
    # Phase 3
    "RelationshipMapper",
    "RelationshipRule",
    "RelationshipRuleRegistry",
    "RelationshipThresholds",
    "RelationshipValidationError",
    "create_default_registry",
    # Phase 4
    "CompressionPolicy",
    "CompressionResult",
    "HybridCompressor",
    "OmittedEvidence",
    "SummaryValidationOutcome",
    # Phase 5B
    "CounterfactualProbeGenerator",
    # Phase 5C
    "RiskModel",
    "RiskScoringWeights",
    # Phase 5D
    "PreservationAuditor",
    # Phase 5E
    "RecoveryDiagnostics",
    "SelectiveRecoveryEngine",
    "SelectiveRecoveryOutcome",
    # Phase 5F
    "OrchestrationDecision",
    "PipelineStage",
    "RABCOOrchestrationResult",
    "RABCOOrchestrator",
    "RABCOPipelineError",
    # Phase 5A Schemas & Contracts
    "AuditorFacingProbe",
    "CounterfactualProbe",
    "EvidencePreservationStatus",
    "ProbeAuditResult",
    "ProbeCategory",
    "ProbeEvidenceBinding",
    "ProbeGenerationResult",
    "ProbeSetType",
    "RABCOAuditPolicy",
    "RABCOPhase5Result",
    "RABCORecoveryPolicy",
    "RecoveryRecord",
    "RecoveryResult",
    "RelationshipAuditDetail",
    "RiskAssessment",
]
