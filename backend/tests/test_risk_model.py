"""Focused test suite for RABCO Phase 5C Risk Model.

Covers:
1. Mandatory evidence partially preserved triggers bypass.
2. Mandatory evidence represented only by an unverified summary triggers bypass.
3. Mandatory evidence missing without a corresponding probe triggers bypass.
4. Zero applicable relationships return None and do not penalize scores or trigger bypass.
5. Invalid aggregate scoring weights raise ValidationError.
6. Composite score renormalization when some dimensions are unavailable, and None when all unavailable.
7. Generated versus audited probe coverage calculation.
8. Relationship audit diagnostic serialization (model_dump and model_dump_json).
9. Directional relationship semantics: DEPENDS_ON, CONTRADICTS, CORRECTS, SUPERSEDES, EXCEPTION_TO, CONDITIONAL_ON.
10. Inviolable bypass invariants cannot be overridden by non-mandatory tolerances.
"""

from __future__ import annotations

import json
from typing import Sequence
import pytest
from pydantic import ValidationError

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
    ProbeSetType,
    RABCOAuditPolicy,
    RelationshipAuditDetail,
    RiskAssessment,
)
from app.research.rabco.risk_model import RiskModel, RiskScoringWeights


# ============================================================================
# Test Fixture Helpers
# ============================================================================

def make_evidence(
    evidence_id: str,
    text: str = "Evidence verbatim content",
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


def make_probe(
    probe_id: str,
    required_evidence_ids: Sequence[str],
    question: str = "Is this critical evidence preserved in context?",
    probe_category: ProbeCategory = ProbeCategory.GENERAL,
) -> CounterfactualProbe:
    """Helper to build a valid CounterfactualProbe."""
    return CounterfactualProbe(
        probe_id=probe_id,
        question=question,
        probe_category=probe_category,
        required_evidence_ids=tuple(required_evidence_ids),
        generation_method="deterministic_template",
        validation_status=ValidationStatus.VALIDATED,
        probe_set=ProbeSetType.AUDIT,
    )


def make_audit_result(
    probe_id: str,
    required_evidence_ids: Sequence[str],
    status: EvidencePreservationStatus,
    missing_evidence_ids: Sequence[str] = (),
    partially_preserved_ids: Sequence[str] = (),
) -> ProbeAuditResult:
    """Helper to build a valid ProbeAuditResult."""
    return ProbeAuditResult(
        probe_id=probe_id,
        question="Follow-up verification question?",
        preservation_status=status,
        required_evidence_ids=tuple(required_evidence_ids),
        missing_evidence_ids=tuple(missing_evidence_ids),
        partially_preserved_ids=tuple(partially_preserved_ids),
        auditor_type="deterministic_provenance",
        explanation=f"Preservation verification outcome: {status.value}",
    )


def make_omitted_evidence(
    evidence_id: str,
    reason: str = "budget_reduction",
    source_message_id: str = "msg_001",
    criticality: EvidenceCriticality = EvidenceCriticality.HIGH,
    evidence_type: EvidenceType = EvidenceType.FACT,
    original_text: str = "Evidence verbatim content",
) -> OmittedEvidence:
    """Helper to build a valid OmittedEvidence unit with all required schema fields."""
    return OmittedEvidence(
        evidence_id=evidence_id,
        reason=reason,
        source_message_id=source_message_id,
        criticality=criticality,
        evidence_type=evidence_type,
        original_text=original_text,
    )


# ============================================================================
# 1. Mandatory Evidence Safety Invariant Tests
# ============================================================================

def test_mandatory_evidence_partially_preserved_triggers_bypass():
    """Safety Invariant: PARTIALLY_PRESERVED mandatory evidence is a failure and forces bypass."""
    ev_mand = make_evidence("ev_mand_001", criticality=EvidenceCriticality.MANDATORY)
    audit = make_audit_result(
        probe_id="probe_001",
        required_evidence_ids=["ev_mand_001"],
        status=EvidencePreservationStatus.PARTIALLY_PRESERVED,
        partially_preserved_ids=["ev_mand_001"],
    )

    model = RiskModel()
    assessment = model.assess(
        audit_results=[audit],
        evidence_units=[ev_mand],
    )

    assert assessment.mandatory_evidence_loss == 1
    assert assessment.bypass_recommended is True
    assert assessment.bypass_reason is not None
    assert "STRICT SAFETY INVARIANT VIOLATION" in assessment.bypass_reason
    assert "ev_mand_001" in assessment.bypass_reason
    assert "partially_preserved" in assessment.bypass_reason


def test_mandatory_evidence_unverified_summary_triggers_bypass():
    """Mandatory evidence summarized without independent probe audit is PARTIALLY_PRESERVED, triggering bypass."""
    ev_mand = make_evidence("ev_mand_sum", criticality=EvidenceCriticality.MANDATORY)

    summary = SummaryUnit(
        summary_id="sum_001",
        summary_text="Mandatory info mentioned briefly",
        source_evidence_ids=("ev_mand_sum",),
        generation_method="extractive_condensation",
        validation_status=ValidationStatus.VALIDATED,
    )

    # Compressor selected no extractive evidence, but included summary
    compression_res = CompressionResult(
        selected_evidence=(),
        validated_summaries=(summary,),
        omitted_evidence=(),
        validation_outcomes=(),
        original_token_estimate=100,
        compressed_token_estimate=30,
        token_reduction_percent=70.0,
        token_estimation_method="heuristic",
    )

    model = RiskModel()
    assessment = model.assess(
        audit_results=[],
        evidence_units=[ev_mand],
        compression_result=compression_res,
    )

    # Without probe confirmation, summarized mandatory evidence is treated as PARTIALLY_PRESERVED
    assert assessment.mandatory_evidence_loss == 1
    assert assessment.bypass_recommended is True
    assert "ev_mand_sum" in assessment.bypass_reason


def test_mandatory_evidence_missing_without_probe_triggers_bypass():
    """Mandatory evidence omitted from compression without any evaluating probe triggers bypass."""
    ev_mand = make_evidence("ev_mand_omitted", criticality=EvidenceCriticality.MANDATORY)

    compression_res = CompressionResult(
        selected_evidence=(),
        validated_summaries=(),
        omitted_evidence=(
            make_omitted_evidence(
                evidence_id="ev_mand_omitted",
                reason="budget_reduction",
                criticality=EvidenceCriticality.MANDATORY,
                original_text=ev_mand.original_text,
            ),
        ),
        validation_outcomes=(),
        original_token_estimate=100,
        compressed_token_estimate=0,
        token_reduction_percent=100.0,
        token_estimation_method="heuristic",
    )

    model = RiskModel()
    assessment = model.assess(
        audit_results=[],
        evidence_units=[ev_mand],
        compression_result=compression_res,
    )

    assert assessment.mandatory_evidence_loss == 1
    assert assessment.bypass_recommended is True
    assert "ev_mand_omitted" in assessment.bypass_reason
    assert "missing" in assessment.bypass_reason


# ============================================================================
# 2. Relationship Indicators & Zero Applicable Cases
# ============================================================================

def test_zero_applicable_relationships_returns_none_and_no_penalty():
    """When zero applicable relationships exist, indicators return None rather than 1.0 or artificial penalties."""
    ev1 = make_evidence("ev_action", criticality=EvidenceCriticality.HIGH)
    ev2 = make_evidence("ev_condition", criticality=EvidenceCriticality.HIGH)

    rel_dep = EvidenceRelationship(
        relationship_id="rel_dep_01",
        source_evidence_id="ev_action",
        target_evidence_id="ev_condition",
        relationship_type=RelationshipType.DEPENDS_ON,
    )
    rel_contra = EvidenceRelationship(
        relationship_id="rel_contra_01",
        source_evidence_id="ev_action",
        target_evidence_id="ev_condition",
        relationship_type=RelationshipType.CONTRADICTS,
    )

    # Neither evidence unit is retained (both omitted)
    compression_res = CompressionResult(
        selected_evidence=(),
        validated_summaries=(),
        omitted_evidence=(
            make_omitted_evidence(
                evidence_id="ev_action",
                reason="omitted_for_test",
                criticality=EvidenceCriticality.HIGH,
                original_text=ev1.original_text,
            ),
            make_omitted_evidence(
                evidence_id="ev_condition",
                reason="omitted_for_test",
                criticality=EvidenceCriticality.HIGH,
                original_text=ev2.original_text,
            ),
        ),
        validation_outcomes=(),
        original_token_estimate=100,
        compressed_token_estimate=0,
        token_reduction_percent=100.0,
        token_estimation_method="heuristic",
    )

    policy = CompressionPolicy(
        preserve_all_dependencies=True,
        preserve_all_contradictions=True,
    )

    model = RiskModel()
    assessment = model.assess(
        audit_results=[],
        evidence_units=[ev1, ev2],
        relationships=[rel_dep, rel_contra],
        compression_result=compression_res,
        compression_policy=policy,
    )

    # Both must be None (unavailable) because 0 applicable relationships were present
    assert assessment.dependency_preservation is None
    assert assessment.contradiction_preservation is None

    # Zero applicable relationships must not trigger invariant breaches or penalties
    assert assessment.indicator_calculation_methods["broken_relationships"] == []
    assert len(assessment.relationship_audit_details) == 2
    assert assessment.relationship_audit_details[0].is_applicable is False
    assert assessment.relationship_audit_details[0].is_preserved is True
    assert assessment.relationship_audit_details[1].is_applicable is False
    assert assessment.relationship_audit_details[1].is_preserved is True


# ============================================================================
# 3. Weight Validation Tests
# ============================================================================

def test_invalid_aggregate_scoring_weights_raises_error():
    """RiskScoringWeights validates individual weights and rejects aggregate sums != 1.0."""
    # Sum exceeds 1.0
    with pytest.raises(ValueError, match="Aggregate RiskScoringWeights must sum to 1.0"):
        RiskScoringWeights(
            weight_probe_failure=0.40,
            weight_mandatory_loss=0.40,
            weight_dependency_failure=0.20,
            weight_contradiction_failure=0.20,
            weight_summary_uncertainty=0.10,
        )

    # Sum less than 1.0
    with pytest.raises(ValueError, match="Aggregate RiskScoringWeights must sum to 1.0"):
        RiskScoringWeights(
            weight_probe_failure=0.10,
            weight_mandatory_loss=0.10,
            weight_dependency_failure=0.10,
            weight_contradiction_failure=0.10,
            weight_summary_uncertainty=0.10,
        )

    # Individual weight negative
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        RiskScoringWeights(weight_probe_failure=-0.1)

    # Individual weight > 1.0
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        RiskScoringWeights(weight_mandatory_loss=1.5)


# ============================================================================
# 4. Composite Score Renormalization Tests
# ============================================================================

def test_composite_score_renormalization_when_dimensions_unavailable():
    """Composite scoring renormalizes weights across available dimensions and returns None when all unavailable."""
    ev_mand = make_evidence("ev_mand", criticality=EvidenceCriticality.MANDATORY)

    # Only probe failure and mandatory loss available (no relationships, no summaries)
    audit = make_audit_result(
        probe_id="p1",
        required_evidence_ids=["ev_mand"],
        status=EvidencePreservationStatus.PRESERVED,
    )

    model = RiskModel()
    assessment = model.assess(
        audit_results=[audit],
        evidence_units=[ev_mand],
    )

    # Configured weights: probe 0.30, mandatory 0.30 (sum = 0.60)
    # Renormalized: 0.30/0.60 = 0.50 each. Both indicators are 0.0 risk.
    assert assessment.composite_risk_score == 0.0
    telemetry = assessment.indicator_calculation_methods["provisional_composite_risk_score"]
    assert tuple(telemetry["available_dimensions"]) == ("probe_failure", "mandatory_loss")
    assert telemetry["renormalized_weights"]["probe_failure"] == 0.5
    assert telemetry["renormalized_weights"]["mandatory_loss"] == 0.5

    # Case where NO dimensions are available:
    # 0 probes, 0 mandatory units, 0 relationships, 0 summaries
    ev_high = make_evidence("ev_high", criticality=EvidenceCriticality.HIGH)
    assessment_empty = model.assess(
        audit_results=[],
        evidence_units=[ev_high],
    )
    assert assessment_empty.composite_risk_score is None
    telemetry_empty = assessment_empty.indicator_calculation_methods["provisional_composite_risk_score"]
    assert telemetry_empty["score"] is None
    assert tuple(telemetry_empty["available_dimensions"]) == ()
    assert dict(telemetry_empty["renormalized_weights"]) == {}


# ============================================================================
# 5. Generated vs. Audited Probe Coverage Tests
# ============================================================================

def test_generated_vs_audited_probe_coverage():
    """Probe coverage uses total candidate probes as denominator and does not claim unaudited probes as preserved."""
    ev = make_evidence("ev_common", criticality=EvidenceCriticality.HIGH)

    # 10 candidate probes generated
    generated_probes = [make_probe(f"probe_{i}", ["ev_common"]) for i in range(10)]

    # Only 4 probes were audited, and all 4 preserved
    audited_probes = [
        make_audit_result(f"probe_{i}", ["ev_common"], EvidencePreservationStatus.PRESERVED)
        for i in range(4)
    ]

    model = RiskModel()
    assessment = model.assess(
        audit_results=audited_probes,
        evidence_units=[ev],
        probes=generated_probes,
    )

    assert assessment.total_probes_audited == 4
    assert assessment.total_probes_generated == 10
    # True coverage across generated probe space: 4 / 10 = 0.40, NOT 1.0!
    assert assessment.probe_coverage == 0.4
    cov_telemetry = assessment.indicator_calculation_methods["probe_coverage"]
    assert cov_telemetry["unaudited_probe_count"] == 6
    assert len(cov_telemetry["unaudited_probe_ids"]) == 6

    # When probes is omitted, falls back to audited probes as denominator
    assessment_no_probes = model.assess(
        audit_results=audited_probes,
        evidence_units=[ev],
    )
    assert assessment_no_probes.total_probes_audited == 4
    assert assessment_no_probes.total_probes_generated == 0
    assert assessment_no_probes.probe_coverage == 1.0


# ============================================================================
# 6. Relationship Audit Diagnostic Serialization Tests
# ============================================================================

def test_relationship_audit_diagnostic_serialization():
    """RelationshipAuditDetail is typed, immutable, and serializes cleanly via model_dump / model_dump_json."""
    ev1 = make_evidence("ev1", criticality=EvidenceCriticality.HIGH)
    ev2 = make_evidence("ev2", criticality=EvidenceCriticality.HIGH)

    rel = EvidenceRelationship(
        relationship_id="rel_001",
        source_evidence_id="ev1",
        target_evidence_id="ev2",
        relationship_type=RelationshipType.DEPENDS_ON,
    )

    audit1 = make_audit_result("p1", ["ev1"], EvidencePreservationStatus.PRESERVED)
    audit2 = make_audit_result("p2", ["ev2"], EvidencePreservationStatus.MISSING, missing_evidence_ids=["ev2"])

    model = RiskModel()
    assessment = model.assess(
        audit_results=[audit1, audit2],
        evidence_units=[ev1, ev2],
        relationships=[rel],
    )

    assert len(assessment.relationship_audit_details) == 1
    detail = assessment.relationship_audit_details[0]
    assert isinstance(detail, RelationshipAuditDetail)
    assert detail.relationship_id == "rel_001"
    assert detail.source_preserved is True
    assert detail.target_preserved is False
    assert detail.is_applicable is True
    assert detail.is_preserved is False
    assert detail.failure_category == "prerequisite_missing"

    # Test Pydantic serialization
    # In Python mode, model_dump() preserves the immutable tuple collection type
    dumped = assessment.model_dump()
    assert isinstance(dumped["relationship_audit_details"], tuple)
    assert len(dumped["relationship_audit_details"]) == 1
    assert dumped["relationship_audit_details"][0]["relationship_id"] == "rel_001"
    assert dumped["relationship_audit_details"][0]["failure_category"] == "prerequisite_missing"

    # In JSON mode, model_dump(mode="json") serializes tuples as lists
    dumped_json_mode = assessment.model_dump(mode="json")
    assert isinstance(dumped_json_mode["relationship_audit_details"], list)
    assert dumped_json_mode["relationship_audit_details"][0]["relationship_id"] == "rel_001"

    # model_dump_json() outputs valid JSON array
    json_str = assessment.model_dump_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert isinstance(parsed["relationship_audit_details"], list)
    assert parsed["relationship_audit_details"][0]["relationship_id"] == "rel_001"


# ============================================================================
# 7. Directional Relationship Semantics Tests
# ============================================================================

def test_directional_relationship_semantics_depends_on():
    """DEPENDS_ON: If dependent is retained, prerequisite must be preserved."""
    ev_src = make_evidence("ev_dependent")
    ev_tgt = make_evidence("ev_prereq")
    rel = EvidenceRelationship(
        relationship_id="r_dep",
        source_evidence_id="ev_dependent",
        target_evidence_id="ev_prereq",
        relationship_type=RelationshipType.DEPENDS_ON,
    )

    model = RiskModel()
    policy = CompressionPolicy(preserve_all_dependencies=True)

    # Case A: Dependent retained, prerequisite missing -> breach
    a_dep_ok = make_audit_result("p1", ["ev_dependent"], EvidencePreservationStatus.PRESERVED)
    a_pre_miss = make_audit_result("p2", ["ev_prereq"], EvidencePreservationStatus.MISSING, missing_evidence_ids=["ev_prereq"])

    res_a = model.assess(
        audit_results=[a_dep_ok, a_pre_miss],
        evidence_units=[ev_src, ev_tgt],
        relationships=[rel],
        compression_policy=policy,
    )
    assert res_a.dependency_preservation == 0.0
    assert res_a.bypass_recommended is True
    assert "CORE INVARIANT BREACH (preserve_all_dependencies)" in res_a.bypass_reason

    # Case B: Prerequisite retained, dependent omitted -> dependency intact (non-applicable)
    a_dep_miss = make_audit_result("p3", ["ev_dependent"], EvidencePreservationStatus.MISSING, missing_evidence_ids=["ev_dependent"])
    a_pre_ok = make_audit_result("p4", ["ev_prereq"], EvidencePreservationStatus.PRESERVED)

    res_b = model.assess(
        audit_results=[a_dep_miss, a_pre_ok],
        evidence_units=[ev_src, ev_tgt],
        relationships=[rel],
        compression_policy=policy,
    )
    assert res_b.dependency_preservation is None  # 0 applicable dependencies
    assert res_b.relationship_audit_details[0].is_applicable is False
    assert res_b.relationship_audit_details[0].is_preserved is True


def test_directional_relationship_semantics_contradicts():
    """CONTRADICTS: Suppressing one side of an active contradiction violates relational balance."""
    ev1 = make_evidence("ev_contra_1")
    ev2 = make_evidence("ev_contra_2")
    rel = EvidenceRelationship(
        relationship_id="r_contra",
        source_evidence_id="ev_contra_1",
        target_evidence_id="ev_contra_2",
        relationship_type=RelationshipType.CONTRADICTS,
    )

    model = RiskModel()
    policy = CompressionPolicy(preserve_all_contradictions=True)

    # One side preserved, one side missing -> breach
    a1 = make_audit_result("p1", ["ev_contra_1"], EvidencePreservationStatus.PRESERVED)
    a2 = make_audit_result("p2", ["ev_contra_2"], EvidencePreservationStatus.MISSING, missing_evidence_ids=["ev_contra_2"])

    res = model.assess(
        audit_results=[a1, a2],
        evidence_units=[ev1, ev2],
        relationships=[rel],
        compression_policy=policy,
    )
    assert res.contradiction_preservation == 0.0
    assert res.bypass_recommended is True
    assert "CORE INVARIANT BREACH (preserve_all_contradictions)" in res.bypass_reason

    # Both sides preserved -> intact
    a2_ok = make_audit_result("p2", ["ev_contra_2"], EvidencePreservationStatus.PRESERVED)
    res_ok = model.assess(
        audit_results=[a1, a2_ok],
        evidence_units=[ev1, ev2],
        relationships=[rel],
        compression_policy=policy,
    )
    assert res_ok.contradiction_preservation == 1.0


def test_directional_relationship_semantics_corrects_and_supersedes():
    """CORRECTS & SUPERSEDES: Severing correction chain triggers invariant breach."""
    ev_corr = make_evidence("ev_correction", evidence_type=EvidenceType.CORRECTION)
    ev_old = make_evidence("ev_old_value")
    rel = EvidenceRelationship(
        relationship_id="r_corr",
        source_evidence_id="ev_correction",
        target_evidence_id="ev_old_value",
        relationship_type=RelationshipType.CORRECTS,
    )

    model = RiskModel()
    policy = CompressionPolicy(preserve_all_corrections=True)

    # Only correction preserved, old value missing
    a_corr = make_audit_result("p1", ["ev_correction"], EvidencePreservationStatus.PRESERVED)
    a_old = make_audit_result("p2", ["ev_old_value"], EvidencePreservationStatus.MISSING, missing_evidence_ids=["ev_old_value"])

    res = model.assess(
        audit_results=[a_corr, a_old],
        evidence_units=[ev_corr, ev_old],
        relationships=[rel],
        compression_policy=policy,
    )
    assert res.bypass_recommended is True
    assert "CORE INVARIANT BREACH (preserve_all_corrections)" in res.bypass_reason


def test_directional_relationship_semantics_exception_to():
    """EXCEPTION_TO: Retaining rule while discarding exception triggers invariant breach."""
    ev_exc = make_evidence("ev_exception", evidence_type=EvidenceType.EXCEPTION)
    ev_rule = make_evidence("ev_rule", evidence_type=EvidenceType.CONSTRAINT)
    rel = EvidenceRelationship(
        relationship_id="r_exc",
        source_evidence_id="ev_exception",
        target_evidence_id="ev_rule",
        relationship_type=RelationshipType.EXCEPTION_TO,
    )

    model = RiskModel()
    policy = CompressionPolicy(preserve_all_exceptions=True)

    # Rule preserved, exception missing -> breach
    a_exc = make_audit_result("p1", ["ev_exception"], EvidencePreservationStatus.MISSING, missing_evidence_ids=["ev_exception"])
    a_rule = make_audit_result("p2", ["ev_rule"], EvidencePreservationStatus.PRESERVED)

    res = model.assess(
        audit_results=[a_exc, a_rule],
        evidence_units=[ev_exc, ev_rule],
        relationships=[rel],
        compression_policy=policy,
    )
    assert res.bypass_recommended is True
    assert "CORE INVARIANT BREACH (preserve_all_exceptions)" in res.bypass_reason


def test_directional_relationship_semantics_conditional_on():
    """CONDITIONAL_ON: Retaining action while losing condition breaks relational condition."""
    ev_action = make_evidence("ev_act")
    ev_cond = make_evidence("ev_cnd", evidence_type=EvidenceType.CONDITION)
    rel = EvidenceRelationship(
        relationship_id="r_cnd",
        source_evidence_id="ev_act",
        target_evidence_id="ev_cnd",
        relationship_type=RelationshipType.CONDITIONAL_ON,
    )

    model = RiskModel()
    a_act = make_audit_result("p1", ["ev_act"], EvidencePreservationStatus.PRESERVED)
    a_cnd = make_audit_result("p2", ["ev_cnd"], EvidencePreservationStatus.MISSING, missing_evidence_ids=["ev_cnd"])

    res = model.assess(
        audit_results=[a_act, a_cnd],
        evidence_units=[ev_action, ev_cond],
        relationships=[rel],
    )
    assert len(res.relationship_audit_details) == 1
    assert res.relationship_audit_details[0].is_applicable is True
    assert res.relationship_audit_details[0].is_preserved is False
    assert res.relationship_audit_details[0].failure_category == "condition_missing_for_active_action"


# ============================================================================
# 8. Tolerance Invariant Overrides
# ============================================================================

def test_bypass_invariants_cannot_be_overridden_by_tolerances():
    """Generous non-mandatory tolerances or low composite scores NEVER override mandatory evidence safety."""
    ev_mand = make_evidence("ev_mand_fail", criticality=EvidenceCriticality.MANDATORY)
    ev_low = make_evidence("ev_low_1", criticality=EvidenceCriticality.LOW)

    a_mand = make_audit_result("p_mand", ["ev_mand_fail"], EvidencePreservationStatus.PARTIALLY_PRESERVED)
    a_low = make_audit_result("p_low", ["ev_low_1"], EvidencePreservationStatus.PRESERVED)

    # Policy allows 100% non-mandatory failure tolerance
    audit_policy = RABCOAuditPolicy(
        max_acceptable_non_mandatory_failure_rate=1.0,
    )

    model = RiskModel()
    assessment = model.assess(
        audit_results=[a_mand, a_low],
        evidence_units=[ev_mand, ev_low],
        audit_policy=audit_policy,
    )

    # Even though non-mandatory failure rate is 0.0, mandatory loss MUST force bypass
    assert assessment.non_mandatory_probe_failure_rate == 0.0
    assert assessment.mandatory_evidence_loss == 1
    assert assessment.bypass_recommended is True
    assert "STRICT SAFETY INVARIANT VIOLATION" in assessment.bypass_reason
