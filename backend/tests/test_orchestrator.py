"""Focused test suite for RABCO Phase 5F End-to-End Orchestrator.

Covers:
1. Successful execution through all phases (safe_to_continue, 0 recovery attempts).
2. Initial unsafe assessment followed by successful selective recovery (recovery_completed).
3. Recovery exhaustion triggering fail-safe bypass (recovery_unsuccessful / bypass_required).
4. Strict mandatory evidence safety invariant (mandatory loss unconditionally mandates bypass).
5. Audit failure handling (failed probes trigger recovery or bypass).
6. Stage-level failure isolation (compression failure caught, EXECUTION_FAILED, fail-safe bypass).
7. Probe generation failure isolation (caught gracefully with stage attribution).
8. Propagation of complete Phase 5 deliverable contracts and diagnostics.
9. Respect for configured recovery attempt limits.
10. End-to-end integration with Phase 2 through Phase 5E components from raw conversation messages.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence
from unittest.mock import MagicMock, patch
import pytest

from app.schemas.rabco import (
    ConversationMessage,
    EvidenceCriticality,
    EvidenceRelationship,
    EvidenceType,
    EvidenceUnit,
    MessageRole,
    RelationshipType,
    SourceSpan,
    SummaryUnit,
    ValidationStatus,
)
from app.research.rabco.hybrid_compressor import (
    CompressionPolicy,
    CompressionResult,
    HybridCompressor,
    OmittedEvidence,
    SummaryValidationOutcome,
)
from app.research.rabco.counterfactual_generator import CounterfactualProbeGenerator
from app.research.rabco.preservation_auditor import PreservationAuditor
from app.research.rabco.risk_model import RiskModel
from app.research.rabco.selective_recovery import (
    RecoveryDiagnostics,
    SelectiveRecoveryEngine,
    SelectiveRecoveryOutcome,
)
from app.research.rabco.schemas import (
    CounterfactualProbe,
    EvidencePreservationStatus,
    ProbeAuditResult,
    ProbeCategory,
    ProbeEvidenceBinding,
    ProbeSetType,
    RABCOAuditPolicy,
    RABCOPhase5Result,
    RABCORecoveryPolicy,
    RecoveryRecord,
    RecoveryResult,
    RelationshipAuditDetail,
    RiskAssessment,
)
from app.research.rabco.orchestrator import (
    OrchestrationDecision,
    PipelineStage,
    RABCOOrchestrationResult,
    RABCOOrchestrator,
    RABCOPipelineError,
)


# ============================================================================
# Test Fixture Helpers
# ============================================================================

def make_message(
    msg_id: str,
    content: str,
    role: MessageRole = MessageRole.USER,
    seq: int = 0,
) -> ConversationMessage:
    """Helper to build a valid immutable ConversationMessage."""
    return ConversationMessage(
        message_id=msg_id,
        role=role,
        content=content,
        sequence_index=seq,
        timestamp=datetime.now(timezone.utc),
    )


def make_evidence(
    evidence_id: str,
    text: str = "Verbatim evidence text",
    criticality: EvidenceCriticality = EvidenceCriticality.HIGH,
    evidence_type: EvidenceType = EvidenceType.FACT,
    message_id: str = "msg_001",
    start_offset: int = 0,
) -> EvidenceUnit:
    """Helper to build a valid EvidenceUnit satisfying span invariants."""
    return EvidenceUnit(
        evidence_id=evidence_id,
        source_span=SourceSpan(
            message_id=message_id,
            start_offset=start_offset,
            end_offset=start_offset + len(text),
        ),
        evidence_type=evidence_type,
        original_text=text,
        criticality=criticality,
        extraction_method="heuristic_extractor",
        validation_status=ValidationStatus.VALIDATED,
    )


def make_omitted_evidence(
    evidence_id: str,
    reason: str = "budget_pruning",
    source_message_id: str = "msg_001",
    criticality: EvidenceCriticality = EvidenceCriticality.HIGH,
    evidence_type: EvidenceType = EvidenceType.FACT,
    original_text: str = "Verbatim evidence text",
) -> OmittedEvidence:
    """Helper to build a valid OmittedEvidence record."""
    return OmittedEvidence(
        evidence_id=evidence_id,
        reason=reason,
        source_message_id=source_message_id,
        criticality=criticality,
        evidence_type=evidence_type,
        original_text=original_text,
    )


def make_probe(
    probe_id: str,
    required_evidence_ids: Sequence[str],
    question: str = "Is this critical evidence preserved in context?",
    probe_category: ProbeCategory = ProbeCategory.GENERAL,
    probe_set: ProbeSetType = ProbeSetType.AUDIT,
) -> CounterfactualProbe:
    """Helper to build a valid CounterfactualProbe."""
    return CounterfactualProbe(
        probe_id=probe_id,
        question=question,
        probe_category=probe_category,
        required_evidence_ids=tuple(required_evidence_ids),
        generation_method="deterministic_template",
        validation_status=ValidationStatus.VALIDATED,
        probe_set=probe_set,
    )


# ============================================================================
# 1. Successful Pipeline Execution Tests
# ============================================================================

def test_orchestration_successful_without_recovery():
    """Initial compression preserves all evidence cleanly; pipeline returns SAFE_TO_CONTINUE."""
    msg1 = make_message("msg_1", "The primary service runs on port 8080.", MessageRole.USER, 0)
    ev1 = make_evidence("ev_port", text="port 8080", message_id="msg_1", start_offset=27)

    comp = CompressionResult(
        selected_evidence=(ev1,),
        validated_summaries=(),
        omitted_evidence=(),
        validation_outcomes=(),
        original_token_estimate=20,
        compressed_token_estimate=10,
        token_reduction_percent=50.0,
        token_estimation_method="heuristic",
    )
    probe = make_probe("probe_port", ["ev_port"])

    orchestrator = RABCOOrchestrator()
    result = orchestrator.orchestrate(
        messages=(msg1,),
        evidence_units=(ev1,),
        compression_result=comp,
        probes=(probe,),
    )

    assert result.decision == OrchestrationDecision.SAFE_TO_CONTINUE
    assert not result.bypassed
    assert result.bypass_reason is None
    assert result.phase5_result.final_preservation_status == "verified_preserved"
    assert result.failed_stage is None
    assert result.error_message is None
    assert len(result.final_context_messages) > 0


# ============================================================================
# 2. Unsafe Initial Assessment Followed by Successful Recovery Tests
# ============================================================================

def test_orchestration_recovery_completed():
    """Initial compression omits critical evidence; selective recovery restores it and re-audit passes."""
    msg1 = make_message("msg_1", "TLS encryption certificate expires in 60 days.", MessageRole.USER, 0)
    ev1 = make_evidence("ev_tls", text="expires in 60 days", message_id="msg_1", start_offset=27)

    # Initially omitted
    comp = CompressionResult(
        selected_evidence=(),
        validated_summaries=(),
        omitted_evidence=(make_omitted_evidence("ev_tls", original_text=ev1.original_text),),
        validation_outcomes=(),
        original_token_estimate=25,
        compressed_token_estimate=0,
        token_reduction_percent=100.0,
        token_estimation_method="heuristic",
    )
    probe = make_probe("probe_tls", ["ev_tls"])

    orchestrator = RABCOOrchestrator()
    result = orchestrator.orchestrate(
        messages=(msg1,),
        evidence_units=(ev1,),
        compression_result=comp,
        probes=(probe,),
    )

    assert result.decision == OrchestrationDecision.RECOVERY_COMPLETED
    assert not result.bypassed
    assert result.phase5_result.final_preservation_status == "recovered_preserved"
    assert "ev_tls" in result.phase5_result.restored_evidence_ids
    assert result.recovery_outcome is not None
    assert result.recovery_outcome.total_recovery_attempts == 1


# ============================================================================
# 3. Recovery Exhaustion and Fail-Safe Bypass Tests
# ============================================================================

def test_orchestration_recovery_exhaustion_triggers_bypass():
    """Probe references unresolvable evidence; recovery exhausts attempts and pipeline reverts to original messages."""
    msg1 = make_message("msg_1", "System host is cluster-alpha.", MessageRole.USER, 0)
    ev_real = make_evidence("ev_real", text="cluster-alpha", message_id="msg_1", start_offset=15)

    comp = CompressionResult(
        selected_evidence=(ev_real,),
        validated_summaries=(),
        omitted_evidence=(),
        validation_outcomes=(),
        original_token_estimate=20,
        compressed_token_estimate=10,
        token_reduction_percent=50.0,
        token_estimation_method="heuristic",
    )

    # Probe references an unresolvable ID not present in evidence_units
    probe_ghost = make_probe("probe_ghost", ["ev_ghost_999"])

    rec_policy = RABCORecoveryPolicy(max_recovery_attempts=2, bypass_on_failure=True)
    orchestrator = RABCOOrchestrator(recovery_policy=rec_policy)
    result = orchestrator.orchestrate(
        messages=(msg1,),
        evidence_units=(ev_real,),
        compression_result=comp,
        probes=(probe_ghost,),
        recovery_policy=rec_policy,
    )

    assert result.bypassed
    assert result.decision in {OrchestrationDecision.RECOVERY_UNSUCCESSFUL, OrchestrationDecision.BYPASS_REQUIRED}
    assert result.bypass_reason is not None
    # On bypass, final_context_messages must strictly match immutable original messages
    assert result.final_context_messages == (msg1,)
    assert result.phase5_result.original_messages_ref == (msg1,)


# ============================================================================
# 4. Mandatory Evidence Safety Invariant Tests
# ============================================================================

def test_orchestration_mandatory_evidence_loss_unconditionally_bypasses():
    """Mandatory evidence unit omitted from context unconditionally mandates fail-safe original context bypass."""
    msg_sys = make_message("msg_sys", "NEVER expose API secret keys in client logs.", MessageRole.SYSTEM, 0)
    ev_mand = make_evidence(
        "ev_secret_mand",
        text="NEVER expose API secret keys",
        criticality=EvidenceCriticality.MANDATORY,
        evidence_type=EvidenceType.CONSTRAINT,
        message_id="msg_sys",
        start_offset=0,
    )

    # Omitted from compression view
    comp = CompressionResult(
        selected_evidence=(),
        validated_summaries=(),
        omitted_evidence=(make_omitted_evidence("ev_secret_mand", criticality=EvidenceCriticality.MANDATORY),),
        validation_outcomes=(),
        original_token_estimate=30,
        compressed_token_estimate=0,
        token_reduction_percent=100.0,
        token_estimation_method="heuristic",
    )

    # Even if recovery policy had bypass_on_failure=False, mandatory loss forces bypass!
    rec_policy = RABCORecoveryPolicy(max_recovery_attempts=1, bypass_on_failure=False)
    orchestrator = RABCOOrchestrator(recovery_policy=rec_policy)

    # Simulate recovery failure by passing empty evidence pool so it cannot be restored
    result = orchestrator.orchestrate(
        messages=(msg_sys,),
        evidence_units=(),  # ev_secret_mand cannot be restored
        compression_result=comp,
        recovery_policy=rec_policy,
    )

    assert result.bypassed
    assert "mandatory" in result.bypass_reason.lower()
    assert result.final_context_messages == (msg_sys,)


# ============================================================================
# 5. Pipeline Stage Failure Isolation Tests
# ============================================================================

def test_orchestration_compression_failure_isolation():
    """Exception in compression stage is caught, attributes failed stage, and engages fail-safe bypass."""
    msg1 = make_message("msg_1", "Deploying service to eu-west1.", MessageRole.USER, 0)

    # Mock compressor that raises an unexpected exception
    mock_compressor = MagicMock(spec=HybridCompressor)
    mock_compressor.compress.side_effect = RuntimeError("Compression engine memory overflow")

    orchestrator = RABCOOrchestrator(compressor=mock_compressor)
    result = orchestrator.orchestrate(
        messages=(msg1,),
        raise_on_failure=False,
    )

    assert result.decision == OrchestrationDecision.EXECUTION_FAILED
    assert result.bypassed
    assert result.failed_stage == PipelineStage.COMPRESSION
    assert "Compression engine memory overflow" in result.error_message
    assert result.final_context_messages == (msg1,)


def test_orchestration_probe_generation_failure_isolation():
    """Exception in probe generation stage is isolated and attributes stage correctly."""
    msg1 = make_message("msg_1", "Configuration parameter X=100.", MessageRole.USER, 0)
    ev1 = make_evidence("ev_cfg", text="X=100", message_id="msg_1", start_offset=24)

    mock_generator = MagicMock(spec=CounterfactualProbeGenerator)
    mock_generator.generate_probes.side_effect = ValueError("Corrupt probe template mapping")

    orchestrator = RABCOOrchestrator(probe_generator=mock_generator)
    result = orchestrator.orchestrate(
        messages=(msg1,),
        evidence_units=(ev1,),
        raise_on_failure=False,
    )

    assert result.decision == OrchestrationDecision.EXECUTION_FAILED
    assert result.bypassed
    assert result.failed_stage == PipelineStage.PROBE_GENERATION
    assert "Corrupt probe template mapping" in result.error_message


def test_orchestration_raise_on_failure_flag():
    """When raise_on_failure=True, stage exceptions are re-raised as RABCOPipelineError."""
    msg1 = make_message("msg_1", "Test message", MessageRole.USER, 0)

    mock_compressor = MagicMock(spec=HybridCompressor)
    mock_compressor.compress.side_effect = RuntimeError("Fatal compressor panic")

    orchestrator = RABCOOrchestrator(compressor=mock_compressor)
    with pytest.raises(RABCOPipelineError) as exc_info:
        orchestrator.orchestrate(
            messages=(msg1,),
            raise_on_failure=True,
        )

    assert exc_info.value.stage == PipelineStage.COMPRESSION
    assert "Fatal compressor panic" in exc_info.value.message


# ============================================================================
# 6. Propagation of Deliverable Contract & Telemetry Tests
# ============================================================================

def test_orchestration_deliverable_contract_and_telemetry():
    """Orchestrator produces valid schemas.RABCOPhase5Result with complete telemetry."""
    msg1 = make_message("msg_1", "Server timeout is 30s.", MessageRole.USER, 0)
    ev1 = make_evidence("ev_time", text="timeout is 30s", message_id="msg_1", start_offset=7)

    comp = CompressionResult(
        selected_evidence=(ev1,),
        validated_summaries=(),
        omitted_evidence=(),
        validation_outcomes=(),
        original_token_estimate=15,
        compressed_token_estimate=10,
        token_reduction_percent=33.33,
        token_estimation_method="heuristic",
    )
    probe = make_probe("probe_time", ["ev_time"])

    orchestrator = RABCOOrchestrator()
    result = orchestrator.orchestrate(
        messages=(msg1,),
        evidence_units=(ev1,),
        compression_result=comp,
        probes=(probe,),
    )

    p5_res = result.to_phase5_result()
    assert isinstance(p5_res, RABCOPhase5Result)
    assert p5_res.original_messages_ref == (msg1,)
    assert p5_res.risk_assessment.mandatory_evidence_loss == 0
    assert p5_res.token_reduction_percent == 33.33
    assert p5_res.processing_time_ms >= 0
    assert "auditor_model" in p5_res.auditor_configuration
    assert len(result.execution_trace) > 0


# ============================================================================
# 7. Empty Conversation Handling Tests
# ============================================================================

def test_orchestration_empty_messages_exits_safely():
    """Empty conversation input executes without exceptions and returns safe status."""
    orchestrator = RABCOOrchestrator()
    result = orchestrator.orchestrate(messages=())

    assert result.decision == OrchestrationDecision.SAFE_TO_CONTINUE
    assert not result.bypassed
    assert result.final_context_messages == ()
    assert result.phase5_result.final_preservation_status == "empty_context_safe"


# ============================================================================
# 8. Full End-to-End Extraction-to-Decision Tests
# ============================================================================

def test_orchestration_full_pipeline_from_raw_messages():
    """Complete pipeline runs from raw ConversationMessages through extraction, mapping, compression, and auditing."""
    msg_system = make_message(
        "msg_0",
        "System constraint: Encryption algorithm must be AES-256.",
        MessageRole.SYSTEM,
        0,
    )
    msg_user = make_message(
        "msg_1",
        "Please deploy the database replica to region us-central1.",
        MessageRole.USER,
        1,
    )

    orchestrator = RABCOOrchestrator()
    result = orchestrator.orchestrate(
        messages=(msg_system, msg_user),
    )

    # Pipeline completes with a valid decision
    assert result.decision in {
        OrchestrationDecision.SAFE_TO_CONTINUE,
        OrchestrationDecision.RECOVERY_COMPLETED,
        OrchestrationDecision.BYPASS_REQUIRED,
    }
    assert result.failed_stage is None
    assert len(result.phase5_result.final_context_messages) > 0
    assert result.phase5_result.original_messages_ref == (msg_system, msg_user)
