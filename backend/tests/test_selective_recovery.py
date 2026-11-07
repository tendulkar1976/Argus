"""Focused test suite for RABCO Phase 5E Selective Evidence Recovery.

Covers:
1. Missing evidence selection (omitted evidence targeted from failed probes).
2. Partial evidence recovery (upgrades summary-only evidence to verbatim extractive retention).
3. Mandatory evidence prioritization (mandatory units restored first).
4. Relationship-aware recovery (DEPENDS_ON and CONTRADICTS relational closure).
5. Duplicate evidence elimination (multiple probes targeting the same unit restored once).
6. Evidence provenance preservation (verbatim text, source span, and criticality preserved).
7. No unnecessary recovery (0 attempts executed when context is already preserved).
8. Recovery attempt limits (bounded iterations strictly respect max_recovery_attempts).
9. Failed recovery handling (unresolvable evidence IDs not present in source history).
10. Successful re-audit verification (re-audit transitions probe from MISSING to PRESERVED).
11. Unresolved evidence reporting (unrecoverable units documented in diagnostics).
12. Phase 5D & Phase 5C interface compatibility (end-to-end integration).
"""

from __future__ import annotations

from typing import Sequence
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
    OmittedEvidence,
    SummaryValidationOutcome,
)
from app.research.rabco.schemas import (
    CounterfactualProbe,
    EvidencePreservationStatus,
    ProbeAuditResult,
    ProbeCategory,
    ProbeEvidenceBinding,
    ProbeSetType,
    RABCOAuditPolicy,
    RABCORecoveryPolicy,
    RecoveryRecord,
    RecoveryResult,
    RelationshipAuditDetail,
    RiskAssessment,
)
from app.research.rabco.preservation_auditor import PreservationAuditor
from app.research.rabco.risk_model import RiskModel
from app.research.rabco.selective_recovery import (
    RecoveryDiagnostics,
    SelectiveRecoveryEngine,
    SelectiveRecoveryOutcome,
)


# ============================================================================
# Test Fixture Helpers
# ============================================================================

def make_evidence(
    evidence_id: str,
    text: str = "Verbatim evidence text",
    criticality: EvidenceCriticality = EvidenceCriticality.HIGH,
    evidence_type: EvidenceType = EvidenceType.FACT,
    message_id: str = "msg_001",
) -> EvidenceUnit:
    """Helper to build a valid EvidenceUnit satisfying character span invariants."""
    return EvidenceUnit(
        evidence_id=evidence_id,
        source_span=SourceSpan(
            message_id=message_id,
            start_offset=0,
            end_offset=len(text),
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


def make_summary(
    summary_id: str,
    text: str,
    source_evidence_ids: Sequence[str],
) -> SummaryUnit:
    """Helper to build a valid SummaryUnit."""
    return SummaryUnit(
        summary_id=summary_id,
        summary_text=text,
        source_evidence_ids=tuple(source_evidence_ids),
        generation_method="test_summary",
    )


def make_probe(
    probe_id: str,
    required_evidence_ids: Sequence[str],
    related_relationship_ids: Sequence[str] = (),
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
        related_relationship_ids=tuple(related_relationship_ids),
        generation_method="deterministic_template",
        validation_status=ValidationStatus.VALIDATED,
        probe_set=probe_set,
    )


def make_audit_result(
    probe_id: str,
    status: EvidencePreservationStatus,
    required_ids: Sequence[str],
    missing_ids: Sequence[str] = (),
    partial_ids: Sequence[str] = (),
) -> ProbeAuditResult:
    """Helper to build a valid ProbeAuditResult."""
    return ProbeAuditResult(
        probe_id=probe_id,
        question=f"Test probe question for {probe_id}?",
        preservation_status=status,
        required_evidence_ids=tuple(required_ids),
        missing_evidence_ids=tuple(missing_ids),
        partially_preserved_ids=tuple(partial_ids),
        auditor_type="deterministic_provenance",
        explanation=f"Audit outcome: {status.value}",
        uncertainty_flag=(status in {EvidencePreservationStatus.AMBIGUOUS, EvidencePreservationStatus.UNVERIFIABLE}),
    )


# ============================================================================
# 1. Missing Evidence Selection Tests
# ============================================================================

def test_missing_evidence_selection():
    """Missing evidence from failed probe is correctly identified and selectively recovered."""
    ev1 = make_evidence("ev_001", text="Primary database host is db.argus.local")
    ev2 = make_evidence("ev_002", text="Backup retention window is 30 days")

    # Initial state: ev1 is extractive, ev2 was omitted
    comp = CompressionResult(
        selected_evidence=(ev1,),
        validated_summaries=(),
        omitted_evidence=(make_omitted_evidence("ev_002", original_text=ev2.original_text),),
        validation_outcomes=(),
        original_token_estimate=50,
        compressed_token_estimate=25,
        token_reduction_percent=50.0,
        token_estimation_method="heuristic",
    )

    probe = make_probe("probe_backup", ["ev_002"])
    audit = make_audit_result("probe_backup", EvidencePreservationStatus.MISSING, ["ev_002"], missing_ids=["ev_002"])

    engine = SelectiveRecoveryEngine()
    outcome = engine.recover(
        evidence_units=(ev1, ev2),
        compression_result=comp,
        audit_results=(audit,),
        probes=(probe,),
    )

    assert "ev_002" in outcome.diagnostics.selected_evidence_ids
    assert "ev_002" in outcome.diagnostics.successfully_recovered_ids
    assert outcome.total_recovery_attempts == 1
    assert outcome.recovery_result.final_preservation_status == "recovered_preserved"
    assert not outcome.bypass_required

    # Final selected evidence now contains both ev1 and ev2
    final_eids = {u.evidence_id for u in outcome.final_selected_evidence}
    assert final_eids == {"ev_001", "ev_002"}

    # Omitted evidence in updated CompressionResult no longer has ev_002
    assert len(outcome.updated_compression_result.omitted_evidence) == 0


# ============================================================================
# 2. Partial Evidence Recovery Tests
# ============================================================================

def test_partial_evidence_recovery():
    """Evidence surviving only in summary narrative is upgraded to verbatim extractive retention."""
    ev1 = make_evidence("ev_num", text="Rate limit is 500 requests per minute", evidence_type=EvidenceType.NUMERICAL_VALUE)
    summary = make_summary("sum_01", "Rate limits are applied per minute.", ["ev_num"])

    # Initial state: ev_num is summarized but not extractive
    comp = CompressionResult(
        selected_evidence=(),
        validated_summaries=(summary,),
        omitted_evidence=(),
        validation_outcomes=(),
        original_token_estimate=30,
        compressed_token_estimate=15,
        token_reduction_percent=50.0,
        token_estimation_method="heuristic",
    )

    probe = make_probe("probe_rate", ["ev_num"])
    audit = make_audit_result("probe_rate", EvidencePreservationStatus.PARTIALLY_PRESERVED, ["ev_num"], partial_ids=["ev_num"])

    engine = SelectiveRecoveryEngine()
    outcome = engine.recover(
        evidence_units=(ev1,),
        compression_result=comp,
        audit_results=(audit,),
        probes=(probe,),
    )

    assert "ev_num" in outcome.diagnostics.selected_evidence_ids
    assert "ev_num" in outcome.diagnostics.successfully_recovered_ids
    assert outcome.total_recovery_attempts == 1
    assert outcome.recovery_result.final_preservation_status == "recovered_preserved"
    assert any(u.evidence_id == "ev_num" for u in outcome.final_selected_evidence)


# ============================================================================
# 3. Mandatory Evidence Prioritization Tests
# ============================================================================

def test_mandatory_evidence_prioritization():
    """Mandatory evidence is prioritized first in multi-unit selective restoration."""
    ev_mand = make_evidence("ev_mand", text="Security token must be rotated hourly", criticality=EvidenceCriticality.MANDATORY)
    ev_low = make_evidence("ev_low", text="UI color theme is dark slate", criticality=EvidenceCriticality.LOW)

    comp = CompressionResult(
        selected_evidence=(),
        validated_summaries=(),
        omitted_evidence=(
            make_omitted_evidence("ev_mand", criticality=EvidenceCriticality.MANDATORY, original_text=ev_mand.original_text),
            make_omitted_evidence("ev_low", criticality=EvidenceCriticality.LOW, original_text=ev_low.original_text),
        ),
        validation_outcomes=(),
        original_token_estimate=40,
        compressed_token_estimate=0,
        token_reduction_percent=100.0,
        token_estimation_method="heuristic",
    )

    probe_mand = make_probe("probe_mand", ["ev_mand"])
    probe_low = make_probe("probe_low", ["ev_low"])
    audit_mand = make_audit_result("probe_mand", EvidencePreservationStatus.MISSING, ["ev_mand"], missing_ids=["ev_mand"])
    audit_low = make_audit_result("probe_low", EvidencePreservationStatus.MISSING, ["ev_low"], missing_ids=["ev_low"])

    engine = SelectiveRecoveryEngine()
    outcome = engine.recover(
        evidence_units=(ev_mand, ev_low),
        compression_result=comp,
        audit_results=(audit_mand, audit_low),
        probes=(probe_mand, probe_low),
    )

    # Both are restored, but mandatory unit is restored and verified
    assert "ev_mand" in outcome.diagnostics.successfully_recovered_ids
    assert "ev_low" in outcome.diagnostics.successfully_recovered_ids
    assert outcome.remaining_risk.mandatory_evidence_loss == 0
    assert not outcome.bypass_required


# ============================================================================
# 4. Relationship-Aware Recovery Tests
# ============================================================================

def test_relationship_aware_recovery_depends_on():
    """Relational closure pulls prerequisite dependency into restoration set when dependent is targeted."""
    ev_action = make_evidence("ev_action", text="Deploying database replica to us-central1")
    ev_prereq = make_evidence("ev_prereq", text="VPC network peering must be active first")

    rel = EvidenceRelationship(
        relationship_id="rel_dep_01",
        source_evidence_id="ev_action",
        target_evidence_id="ev_prereq",
        relationship_type=RelationshipType.DEPENDS_ON,
    )

    # Initial state: both omitted
    comp = CompressionResult(
        selected_evidence=(),
        validated_summaries=(),
        omitted_evidence=(
            make_omitted_evidence("ev_action", original_text=ev_action.original_text),
            make_omitted_evidence("ev_prereq", original_text=ev_prereq.original_text),
        ),
        validation_outcomes=(),
        original_token_estimate=50,
        compressed_token_estimate=0,
        token_reduction_percent=100.0,
        token_estimation_method="heuristic",
    )

    # Probe targets only the action
    probe = make_probe("probe_act", ["ev_action"], related_relationship_ids=["rel_dep_01"])
    audit = make_audit_result("probe_act", EvidencePreservationStatus.MISSING, ["ev_action"], missing_ids=["ev_action"])

    policy = RABCORecoveryPolicy(enable_relational_closure=True)
    engine = SelectiveRecoveryEngine(recovery_policy=policy)
    outcome = engine.recover(
        evidence_units=(ev_action, ev_prereq),
        compression_result=comp,
        audit_results=(audit,),
        probes=(probe,),
        relationships=(rel,),
    )

    # Prerequisite is automatically restored alongside dependent
    restored_eids = outcome.diagnostics.selected_evidence_ids
    assert "ev_action" in restored_eids
    assert "ev_prereq" in restored_eids
    assert "rel_dep_01" in outcome.updated_compression_result.retained_relationship_ids
    assert not outcome.bypass_required


def test_relationship_aware_recovery_contradicts():
    """Opposing contradiction pole is pulled in to prevent one-sided bias."""
    ev_rule = make_evidence("ev_rule", text="All external HTTP access is disallowed")
    ev_except = make_evidence("ev_conflict", text="Webhook endpoint accepts port 443 externally")

    rel = EvidenceRelationship(
        relationship_id="rel_contra_01",
        source_evidence_id="ev_rule",
        target_evidence_id="ev_conflict",
        relationship_type=RelationshipType.CONTRADICTS,
    )

    # Initial state: ev_rule is retained, but opposing pole ev_conflict is omitted
    comp = CompressionResult(
        selected_evidence=(ev_rule,),
        validated_summaries=(),
        omitted_evidence=(make_omitted_evidence("ev_conflict", original_text=ev_except.original_text),),
        validation_outcomes=(),
        original_token_estimate=50,
        compressed_token_estimate=25,
        token_reduction_percent=50.0,
        token_estimation_method="heuristic",
    )

    probe = make_probe("probe_conflict", ["ev_conflict"], related_relationship_ids=["rel_contra_01"])
    audit = make_audit_result("probe_conflict", EvidencePreservationStatus.MISSING, ["ev_conflict"], missing_ids=["ev_conflict"])

    engine = SelectiveRecoveryEngine()
    outcome = engine.recover(
        evidence_units=(ev_rule, ev_except),
        compression_result=comp,
        audit_results=(audit,),
        probes=(probe,),
        relationships=(rel,),
    )

    assert "ev_conflict" in outcome.diagnostics.selected_evidence_ids
    assert "rel_contra_01" in outcome.updated_compression_result.retained_relationship_ids
    assert outcome.recovery_result.final_preservation_status == "recovered_preserved"


# ============================================================================
# 5. Duplicate Evidence Elimination Tests
# ============================================================================

def test_duplicate_evidence_elimination():
    """Multiple probes referencing the same missing evidence ID trigger exactly one restoration."""
    ev = make_evidence("ev_shared", text="Shared secret key is K-991204")

    comp = CompressionResult(
        selected_evidence=(),
        validated_summaries=(),
        omitted_evidence=(make_omitted_evidence("ev_shared", original_text=ev.original_text),),
        validation_outcomes=(),
        original_token_estimate=30,
        compressed_token_estimate=0,
        token_reduction_percent=100.0,
        token_estimation_method="heuristic",
    )

    probe1 = make_probe("probe_1", ["ev_shared"])
    probe2 = make_probe("probe_2", ["ev_shared"])
    audit1 = make_audit_result("probe_1", EvidencePreservationStatus.MISSING, ["ev_shared"], missing_ids=["ev_shared"])
    audit2 = make_audit_result("probe_2", EvidencePreservationStatus.MISSING, ["ev_shared"], missing_ids=["ev_shared"])

    engine = SelectiveRecoveryEngine()
    outcome = engine.recover(
        evidence_units=(ev,),
        compression_result=comp,
        audit_results=(audit1, audit2),
        probes=(probe1, probe2),
    )

    # Exactly 1 restoration record, and exactly 1 unit in final_selected_evidence
    assert outcome.total_recovery_attempts == 1
    assert len(outcome.final_selected_evidence) == 1
    assert outcome.final_selected_evidence[0].evidence_id == "ev_shared"
    assert outcome.diagnostics.selected_evidence_ids == ("ev_shared",)


# ============================================================================
# 6. Evidence Provenance Preservation Tests
# ============================================================================

def test_evidence_provenance_preservation():
    """Restored evidence retains exact source spans, character offsets, and metadata without alteration."""
    text = "Disk encryption algorithm is AES-256-GCM authenticated"
    start_offset = 12
    end_offset = start_offset + len(text)
    ev = EvidenceUnit(
        evidence_id="ev_prov_01",
        source_span=SourceSpan(message_id="turn_042", start_offset=start_offset, end_offset=end_offset),
        evidence_type=EvidenceType.CONSTRAINT,
        original_text=text,
        criticality=EvidenceCriticality.MANDATORY,
        extraction_method="provenance_exact_extractor",
        validation_status=ValidationStatus.VALIDATED,
    )

    comp = CompressionResult(
        selected_evidence=(),
        validated_summaries=(),
        omitted_evidence=(make_omitted_evidence("ev_prov_01", original_text=ev.original_text),),
        validation_outcomes=(),
        original_token_estimate=40,
        compressed_token_estimate=0,
        token_reduction_percent=100.0,
        token_estimation_method="heuristic",
    )

    probe = make_probe("probe_prov", ["ev_prov_01"])
    audit = make_audit_result("probe_prov", EvidencePreservationStatus.MISSING, ["ev_prov_01"], missing_ids=["ev_prov_01"])

    engine = SelectiveRecoveryEngine()
    outcome = engine.recover(
        evidence_units=(ev,),
        compression_result=comp,
        audit_results=(audit,),
        probes=(probe,),
    )

    restored = outcome.final_selected_evidence[0]
    assert restored.evidence_id == "ev_prov_01"
    assert restored.source_span.message_id == "turn_042"
    assert restored.source_span.start_offset == start_offset
    assert restored.source_span.end_offset == end_offset
    assert restored.original_text == text
    assert restored.criticality == EvidenceCriticality.MANDATORY


# ============================================================================
# 7. No Unnecessary Recovery Tests
# ============================================================================

def test_no_unnecessary_recovery_when_context_preserved():
    """When context already satisfies probes and safety invariants, 0 recovery cycles are executed."""
    ev = make_evidence("ev_ok", text="Port number is 443")

    comp = CompressionResult(
        selected_evidence=(ev,),
        validated_summaries=(),
        omitted_evidence=(),
        validation_outcomes=(),
        original_token_estimate=20,
        compressed_token_estimate=20,
        token_reduction_percent=0.0,
        token_estimation_method="heuristic",
    )

    probe = make_probe("probe_ok", ["ev_ok"])
    audit = make_audit_result("probe_ok", EvidencePreservationStatus.PRESERVED, ["ev_ok"])

    engine = SelectiveRecoveryEngine()
    outcome = engine.recover(
        evidence_units=(ev,),
        compression_result=comp,
        audit_results=(audit,),
        probes=(probe,),
    )

    assert outcome.total_recovery_attempts == 0
    assert outcome.recovery_result.final_preservation_status == "no_recovery_needed"
    assert len(outcome.diagnostics.selected_evidence_ids) == 0
    assert not outcome.bypass_required


# ============================================================================
# 8. Recovery Attempt Limits Tests
# ============================================================================

def test_recovery_attempt_limits_enforced():
    """Recovery terminates when max_recovery_attempts is reached, triggering bypass on failure."""
    ev_mand = make_evidence("ev_mand_broken", text="Mandatory cryptographic salt", criticality=EvidenceCriticality.MANDATORY)

    # Initial state: ev_mand is omitted
    comp = CompressionResult(
        selected_evidence=(),
        validated_summaries=(),
        omitted_evidence=(make_omitted_evidence("ev_mand_broken", criticality=EvidenceCriticality.MANDATORY),),
        validation_outcomes=(),
        original_token_estimate=30,
        compressed_token_estimate=0,
        token_reduction_percent=100.0,
        token_estimation_method="heuristic",
    )

    # Probe references fake non-existent evidence that cannot be restored from source pool
    probe = make_probe("probe_unresolvable", ["ev_non_existent"])
    audit = make_audit_result("probe_unresolvable", EvidencePreservationStatus.MISSING, ["ev_non_existent"], missing_ids=["ev_non_existent"])

    policy = RABCORecoveryPolicy(max_recovery_attempts=2, bypass_on_failure=True)
    engine = SelectiveRecoveryEngine(recovery_policy=policy)
    outcome = engine.recover(
        evidence_units=(ev_mand,),  # ev_non_existent is NOT in evidence_units
        compression_result=comp,
        audit_results=(audit,),
        probes=(probe,),
    )

    # ev_mand was missing and was restored, but ev_non_existent probe still fails
    # Engine terminates bounded loop and mandates bypass
    assert outcome.recovery_result.bypass_required
    assert "recovery_exhausted" in outcome.recovery_result.final_preservation_status


# ============================================================================
# 9. Failed Recovery Handling & Unresolved Evidence Reporting Tests
# ============================================================================

def test_failed_recovery_handling_and_unresolved_reporting():
    """Unresolvable evidence IDs absent from source history are documented as unresolved."""
    ev_valid = make_evidence("ev_valid", text="Valid system username")

    comp = CompressionResult(
        selected_evidence=(ev_valid,),
        validated_summaries=(),
        omitted_evidence=(),
        validation_outcomes=(),
        original_token_estimate=20,
        compressed_token_estimate=20,
        token_reduction_percent=0.0,
        token_estimation_method="heuristic",
    )

    probe = make_probe("probe_hallucinated", ["ev_ghost_101"])
    audit = make_audit_result("probe_hallucinated", EvidencePreservationStatus.MISSING, ["ev_ghost_101"], missing_ids=["ev_ghost_101"])

    engine = SelectiveRecoveryEngine()
    outcome = engine.recover(
        evidence_units=(ev_valid,),  # ev_ghost_101 is not in authoritative pool
        compression_result=comp,
        audit_results=(audit,),
        probes=(probe,),
    )

    assert "ev_ghost_101" in outcome.diagnostics.unresolved_evidence_ids
    assert "ev_ghost_101" not in outcome.diagnostics.selected_evidence_ids


# ============================================================================
# 10. Successful Re-Audit Verification Tests
# ============================================================================

def test_successful_re_audit_transitions_probe():
    """Re-audit with PreservationAuditor verifies that probe transitions from MISSING to PRESERVED."""
    ev = make_evidence("ev_recovered", text="Cluster authentication token is BEARER_XYZ")

    comp = CompressionResult(
        selected_evidence=(),
        validated_summaries=(),
        omitted_evidence=(make_omitted_evidence("ev_recovered", original_text=ev.original_text),),
        validation_outcomes=(),
        original_token_estimate=30,
        compressed_token_estimate=0,
        token_reduction_percent=100.0,
        token_estimation_method="heuristic",
    )

    probe = make_probe("probe_auth", ["ev_recovered"])
    audit = make_audit_result("probe_auth", EvidencePreservationStatus.MISSING, ["ev_recovered"], missing_ids=["ev_recovered"])

    engine = SelectiveRecoveryEngine()
    outcome = engine.recover(
        evidence_units=(ev,),
        compression_result=comp,
        audit_results=(audit,),
        probes=(probe,),
    )

    # Post-recovery re-audit results should show probe is PRESERVED
    final_audits = outcome.re_audit_results
    assert len(final_audits) == 1
    assert final_audits[0].preservation_status == EvidencePreservationStatus.PRESERVED
    assert outcome.recovery_result.final_preservation_status == "recovered_preserved"
    assert not outcome.bypass_required


# ============================================================================
# 11. End-to-End Compatibility with Phase 5C RiskModel
# ============================================================================

def test_end_to_end_phase_5c_risk_model_compatibility():
    """Full lifecycle: Compressor -> Auditor -> SelectiveRecovery -> RiskModel."""
    ev1 = make_evidence("ev_comp_1", text="Server host is production-01", criticality=EvidenceCriticality.HIGH)
    ev2 = make_evidence("ev_comp_2", text="TLS certificates expire in 90 days", criticality=EvidenceCriticality.HIGH)

    # Initial compression omitted ev2
    comp = CompressionResult(
        selected_evidence=(ev1,),
        validated_summaries=(),
        omitted_evidence=(make_omitted_evidence("ev_comp_2", original_text=ev2.original_text),),
        validation_outcomes=(),
        original_token_estimate=50,
        compressed_token_estimate=25,
        token_reduction_percent=50.0,
        token_estimation_method="heuristic",
    )

    probes = (
        make_probe("probe_h1", ["ev_comp_1"]),
        make_probe("probe_h2", ["ev_comp_2"]),
    )

    # 1. Phase 5D initial audit
    auditor = PreservationAuditor()
    initial_audits = auditor.audit_probes(
        probes=probes,
        evidence_units=(ev1, ev2),
        compression_result=comp,
    )
    statuses = {a.probe_id: a.preservation_status for a in initial_audits}
    assert statuses["probe_h1"] == EvidencePreservationStatus.PRESERVED
    assert statuses["probe_h2"] == EvidencePreservationStatus.MISSING

    # 2. Phase 5E selective recovery
    recovery_engine = SelectiveRecoveryEngine(auditor=auditor)
    recovery_outcome = recovery_engine.recover(
        evidence_units=(ev1, ev2),
        compression_result=comp,
        audit_results=initial_audits,
        probes=probes,
    )

    assert recovery_outcome.recovery_result.final_preservation_status == "recovered_preserved"
    assert not recovery_outcome.bypass_required

    # 3. Phase 5C RiskModel assessment using recovery outcome
    risk_model = RiskModel()
    final_risk = risk_model.assess(
        audit_results=recovery_outcome.re_audit_results,
        evidence_units=(ev1, ev2),
        compression_result=recovery_outcome.updated_compression_result,
        recovery_result=recovery_outcome.recovery_result,
        probes=probes,
    )

    assert final_risk.probe_failure_rate == 0.0
    assert final_risk.mandatory_evidence_loss == 0
    assert not final_risk.bypass_recommended
    assert final_risk.bypass_reason is None
    assert final_risk.probe_coverage == 1.0
