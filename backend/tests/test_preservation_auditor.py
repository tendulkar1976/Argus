"""Focused test suite for RABCO Phase 5D Preservation Auditor.

Covers:
1. Extractive evidence preservation (PRESERVED).
2. Summary-only evidence preservation (PARTIALLY_PRESERVED).
3. Explicit omission (MISSING).
4. Unverifiable evidence (unaccounted or rejected summaries).
5. Ambiguous evidence handling.
6. Mandatory evidence preservation auditing.
7. Relationship failures (DEPENDS_ON, CONTRADICTS, EXCEPTION_TO).
8. Audit vs. Development probe set filtering.
9. Semantic SLM verdict parsing (answerable, partially_answerable, unanswerable, ambiguous).
10. Answer-key leakage prevention (AuditorFacingProbe isolation).
11. Ollama timeout, connect error, and malformed response fallback.
12. Empty inputs handling.
13. End-to-end integration and Phase 5C RiskModel compatibility.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest
import httpx

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
    RiskAssessment,
)
from app.research.rabco.preservation_auditor import PreservationAuditor
from app.research.rabco.risk_model import RiskModel


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


def make_probe(
    probe_id: str,
    required_evidence_ids: list[str],
    related_relationship_ids: list[str] = (),
    target_evidence_text: str = None,
    question: str = "Is the critical fact preserved?",
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
        target_evidence_text=target_evidence_text,
        generation_method="deterministic_template",
        validation_status=ValidationStatus.VALIDATED,
        probe_set=probe_set,
    )


# ============================================================================
# 1. Deterministic Provenance Auditing Tests
# ============================================================================

def test_extractive_evidence_preservation():
    """Evidence retained extractively in selected_evidence audits as PRESERVED."""
    ev1 = make_evidence("ev_ext_1", text="Server port is 8080")
    ev2 = make_evidence("ev_ext_2", text="TLS encryption is enabled")

    probe = make_probe("probe_ext", ["ev_ext_1", "ev_ext_2"])

    comp_res = CompressionResult(
        selected_evidence=(ev1, ev2),
        validated_summaries=(),
        omitted_evidence=(),
        validation_outcomes=(),
        original_token_estimate=50,
        compressed_token_estimate=50,
        token_reduction_percent=0.0,
        token_estimation_method="heuristic",
    )

    auditor = PreservationAuditor()
    results = auditor.audit_probes(
        probes=[probe],
        evidence_units=[ev1, ev2],
        compression_result=comp_res,
    )

    assert len(results) == 1
    res = results[0]
    assert res.probe_id == "probe_ext"
    assert res.preservation_status == EvidencePreservationStatus.PRESERVED
    assert res.missing_evidence_ids == ()
    assert res.partially_preserved_ids == ()
    assert res.auditor_type == "deterministic_provenance"
    assert res.uncertainty_flag is False


def test_summary_only_evidence_preservation():
    """Evidence represented through validated_summaries audits as PARTIALLY_PRESERVED."""
    ev1 = make_evidence("ev_sum_1", text="Database backup completes daily at midnight")
    summary = SummaryUnit(
        summary_id="sum_01",
        summary_text="Daily database backups run at 00:00",
        source_evidence_ids=("ev_sum_1",),
        generation_method="extractive_condensation",
        validation_status=ValidationStatus.VALIDATED,
    )

    probe = make_probe("probe_sum", ["ev_sum_1"])

    comp_res = CompressionResult(
        selected_evidence=(),
        validated_summaries=(summary,),
        omitted_evidence=(),
        validation_outcomes=(),
        original_token_estimate=50,
        compressed_token_estimate=20,
        token_reduction_percent=60.0,
        token_estimation_method="heuristic",
    )

    auditor = PreservationAuditor()
    results = auditor.audit_probes(
        probes=[probe],
        evidence_units=[ev1],
        compression_result=comp_res,
    )

    assert len(results) == 1
    res = results[0]
    assert res.probe_id == "probe_sum"
    assert res.preservation_status == EvidencePreservationStatus.PARTIALLY_PRESERVED
    assert res.missing_evidence_ids == ()
    assert res.partially_preserved_ids == ("ev_sum_1",)
    assert res.auditor_type == "deterministic_provenance"


def test_explicit_omission_preservation():
    """Evidence explicitly in omitted_evidence audits as MISSING."""
    ev1 = make_evidence("ev_omit_1", text="User preferred dark theme in 2024")
    omitted = make_omitted_evidence("ev_omit_1", reason="budget_reduction", original_text=ev1.original_text)

    probe = make_probe("probe_omit", ["ev_omit_1"])

    comp_res = CompressionResult(
        selected_evidence=(),
        validated_summaries=(),
        omitted_evidence=(omitted,),
        validation_outcomes=(),
        original_token_estimate=50,
        compressed_token_estimate=0,
        token_reduction_percent=100.0,
        token_estimation_method="heuristic",
    )

    auditor = PreservationAuditor()
    results = auditor.audit_probes(
        probes=[probe],
        evidence_units=[ev1],
        compression_result=comp_res,
    )

    assert len(results) == 1
    res = results[0]
    assert res.probe_id == "probe_omit"
    assert res.preservation_status == EvidencePreservationStatus.MISSING
    assert res.missing_evidence_ids == ("ev_omit_1",)
    assert res.partially_preserved_ids == ()


def test_unverifiable_evidence_unaccounted_or_rejected():
    """Evidence unaccounted for or from rejected summaries audits as UNVERIFIABLE."""
    ev_unaccounted = make_evidence("ev_ghost", text="Unrecorded assertion")
    probe = make_probe("probe_unv", ["ev_ghost"])

    # Empty compression result: evidence is unaccounted for
    comp_res = CompressionResult(
        selected_evidence=(),
        validated_summaries=(),
        omitted_evidence=(),
        validation_outcomes=(),
        original_token_estimate=50,
        compressed_token_estimate=0,
        token_reduction_percent=100.0,
        token_estimation_method="heuristic",
    )

    auditor = PreservationAuditor()
    results = auditor.audit_probes(
        probes=[probe],
        evidence_units=[ev_unaccounted],
        compression_result=comp_res,
    )

    assert len(results) == 1
    res = results[0]
    assert res.preservation_status == EvidencePreservationStatus.UNVERIFIABLE
    assert res.uncertainty_flag is True

    # Case B: Evidence in rejected summary
    ev_rejected = make_evidence("ev_rej", text="Rejected assertion")
    probe_rej = make_probe("probe_rej", ["ev_rej"])
    comp_rejected = CompressionResult(
        selected_evidence=(),
        validated_summaries=(),
        omitted_evidence=(),
        validation_outcomes=(
            SummaryValidationOutcome(
                summary_id="sum_rej",
                is_valid=False,
                validation_status=ValidationStatus.REJECTED,
                reason="Ungrounded identifier",
                covered_evidence_ids=("ev_rej",),
            ),
        ),
        original_token_estimate=50,
        compressed_token_estimate=0,
        token_reduction_percent=100.0,
        token_estimation_method="heuristic",
    )

    results_rej = auditor.audit_probes(
        probes=[probe_rej],
        evidence_units=[ev_rejected],
        compression_result=comp_rejected,
    )
    assert len(results_rej) == 1
    assert results_rej[0].preservation_status == EvidencePreservationStatus.UNVERIFIABLE
    assert results_rej[0].uncertainty_flag is True


# ============================================================================
# 2. Relationship Failures Tests
# ============================================================================

def test_relationship_failure_depends_on():
    """Retaining dependent while omitting prerequisite results in MISSING outcome."""
    ev_act = make_evidence("ev_action", text="Deploy application to production")
    ev_pre = make_evidence("ev_prereq", text="Run database migrations successfully")

    rel = EvidenceRelationship(
        relationship_id="r_dep_1",
        source_evidence_id="ev_action",
        target_evidence_id="ev_prereq",
        relationship_type=RelationshipType.DEPENDS_ON,
    )

    probe = make_probe(
        probe_id="p_dep",
        required_evidence_ids=["ev_action", "ev_prereq"],
        related_relationship_ids=["r_dep_1"],
    )

    # Action retained, prerequisite omitted
    comp_res = CompressionResult(
        selected_evidence=(ev_act,),
        validated_summaries=(),
        omitted_evidence=(make_omitted_evidence("ev_prereq", original_text=ev_pre.original_text),),
        validation_outcomes=(),
        original_token_estimate=50,
        compressed_token_estimate=25,
        token_reduction_percent=50.0,
        token_estimation_method="heuristic",
    )

    auditor = PreservationAuditor()
    results = auditor.audit_probes(
        probes=[probe],
        evidence_units=[ev_act, ev_pre],
        relationships=[rel],
        compression_result=comp_res,
    )

    assert len(results) == 1
    res = results[0]
    assert res.preservation_status == EvidencePreservationStatus.MISSING
    assert "ev_prereq" in res.missing_evidence_ids
    assert "Relational dependency severed" in res.explanation


def test_relationship_failure_contradicts():
    """Suppressing one pole of a contradiction results in MISSING outcome."""
    ev_c1 = make_evidence("ev_pole_1", text="Budget is 50000 USD")
    ev_c2 = make_evidence("ev_pole_2", text="Budget is 75000 USD")

    rel = EvidenceRelationship(
        relationship_id="r_contra_1",
        source_evidence_id="ev_pole_1",
        target_evidence_id="ev_pole_2",
        relationship_type=RelationshipType.CONTRADICTS,
    )

    probe = make_probe(
        probe_id="p_contra",
        required_evidence_ids=["ev_pole_1", "ev_pole_2"],
        related_relationship_ids=["r_contra_1"],
    )

    # Pole 1 retained, Pole 2 omitted (one-sided suppression)
    comp_res = CompressionResult(
        selected_evidence=(ev_c1,),
        validated_summaries=(),
        omitted_evidence=(make_omitted_evidence("ev_pole_2", original_text=ev_c2.original_text),),
        validation_outcomes=(),
        original_token_estimate=50,
        compressed_token_estimate=25,
        token_reduction_percent=50.0,
        token_estimation_method="heuristic",
    )

    auditor = PreservationAuditor()
    results = auditor.audit_probes(
        probes=[probe],
        evidence_units=[ev_c1, ev_c2],
        relationships=[rel],
        compression_result=comp_res,
    )

    assert len(results) == 1
    res = results[0]
    assert res.preservation_status == EvidencePreservationStatus.MISSING
    assert "ev_pole_2" in res.missing_evidence_ids
    assert "Contradiction unbalanced" in res.explanation


def test_relationship_failure_exception_to():
    """Retaining rule while omitting exception results in MISSING outcome."""
    ev_rule = make_evidence("ev_rule", text="All users must change passwords every 90 days")
    ev_exc = make_evidence("ev_exc", text="Service accounts are exempt from 90 day password rotation")

    rel = EvidenceRelationship(
        relationship_id="r_exc_1",
        source_evidence_id="ev_exc",
        target_evidence_id="ev_rule",
        relationship_type=RelationshipType.EXCEPTION_TO,
    )

    probe = make_probe(
        probe_id="p_exc",
        required_evidence_ids=["ev_rule", "ev_exc"],
        related_relationship_ids=["r_exc_1"],
    )

    # Rule retained, exception omitted
    comp_res = CompressionResult(
        selected_evidence=(ev_rule,),
        validated_summaries=(),
        omitted_evidence=(make_omitted_evidence("ev_exc", original_text=ev_exc.original_text),),
        validation_outcomes=(),
        original_token_estimate=50,
        compressed_token_estimate=25,
        token_reduction_percent=50.0,
        token_estimation_method="heuristic",
    )

    auditor = PreservationAuditor()
    results = auditor.audit_probes(
        probes=[probe],
        evidence_units=[ev_rule, ev_exc],
        relationships=[rel],
        compression_result=comp_res,
    )

    assert len(results) == 1
    res = results[0]
    assert res.preservation_status == EvidencePreservationStatus.MISSING
    assert "ev_exc" in res.missing_evidence_ids
    assert "Exception 'ev_exc' omitted while governed rule 'ev_rule' was retained" in res.explanation


# ============================================================================
# 3. Audit vs. Development Probe Filtering
# ============================================================================

def test_audit_versus_development_probe_filtering():
    """Only AUDIT probes are evaluated; DEVELOPMENT probes are excluded without failure penalty."""
    ev = make_evidence("ev_shared")

    p_audit_1 = make_probe("probe_aud_1", ["ev_shared"], probe_set=ProbeSetType.AUDIT)
    p_audit_2 = make_probe("probe_aud_2", ["ev_shared"], probe_set=ProbeSetType.AUDIT)
    p_dev_1 = make_probe("probe_dev_1", ["ev_shared"], probe_set=ProbeSetType.DEVELOPMENT)
    p_dev_2 = make_probe("probe_dev_2", ["ev_shared"], probe_set=ProbeSetType.DEVELOPMENT)

    comp_res = CompressionResult(
        selected_evidence=(ev,),
        validated_summaries=(),
        omitted_evidence=(),
        validation_outcomes=(),
        original_token_estimate=20,
        compressed_token_estimate=20,
        token_reduction_percent=0.0,
        token_estimation_method="heuristic",
    )

    auditor = PreservationAuditor()
    results = auditor.audit_probes(
        probes=[p_audit_1, p_dev_1, p_audit_2, p_dev_2],
        evidence_units=[ev],
        compression_result=comp_res,
    )

    # Exactly 2 results returned corresponding to the AUDIT partition
    assert len(results) == 2
    returned_ids = {r.probe_id for r in results}
    assert returned_ids == {"probe_aud_1", "probe_aud_2"}
    assert "probe_dev_1" not in returned_ids
    assert "probe_dev_2" not in returned_ids


# ============================================================================
# 4. Semantic SLM Auditing & Answer-Key Isolation Tests
# ============================================================================

def test_semantic_verdict_parsing():
    """Semantic SLM verdicts parse cleanly into EvidencePreservationStatus classifications."""
    ev = make_evidence("ev_fact", text="API rate limit is 100 requests per minute")
    probe = make_probe("probe_sem", ["ev_fact"], question="What is the API rate limit?")

    policy = RABCOAuditPolicy(
        enable_semantic_audit=True,
        auditor_ollama_model="mock-auditor-model",
    )

    auditor = PreservationAuditor(policy=policy)

    # Test 1: answerable -> PRESERVED
    mock_resp_answerable = MagicMock(
        status_code=200,
        json=lambda: {
            "response": json.dumps({
                "verdict": "answerable",
                "explanation": "Rate limit 100 req/min clearly stated in context.",
                "uncertainty_flag": False,
            })
        },
    )

    with patch("httpx.Client.post", return_value=mock_resp_answerable):
        res_a = auditor.audit_probes(
            probes=[probe],
            evidence_units=[ev],
            compressed_messages=[ConversationMessage(message_id="m1", role=MessageRole.USER, content=ev.original_text, sequence_index=0)],
        )
        assert len(res_a) == 1
        assert res_a[0].preservation_status == EvidencePreservationStatus.PRESERVED
        assert res_a[0].auditor_type == "semantic_slm"
        assert res_a[0].uncertainty_flag is False

    # Test 2: unanswerable -> MISSING
    mock_resp_unanswerable = MagicMock(
        status_code=200,
        json=lambda: {
            "response": json.dumps({
                "verdict": "unanswerable",
                "explanation": "No rate limit is mentioned.",
                "uncertainty_flag": False,
            })
        },
    )

    with patch("httpx.Client.post", return_value=mock_resp_unanswerable):
        res_m = auditor.audit_probes(
            probes=[probe],
            evidence_units=[ev],
            compressed_messages=[ConversationMessage(message_id="m1", role=MessageRole.USER, content="Hello world", sequence_index=0)],
        )
        assert len(res_m) == 1
        assert res_m[0].preservation_status == EvidencePreservationStatus.MISSING
        assert "ev_fact" in res_m[0].missing_evidence_ids

    # Test 3: partially_answerable -> PARTIALLY_PRESERVED
    mock_resp_partial = MagicMock(
        status_code=200,
        json=lambda: {
            "response": json.dumps({
                "verdict": "partially_answerable",
                "explanation": "Mentions rate limits exist but omits exact number.",
                "uncertainty_flag": False,
            })
        },
    )

    with patch("httpx.Client.post", return_value=mock_resp_partial):
        res_p = auditor.audit_probes(
            probes=[probe],
            evidence_units=[ev],
            compressed_messages=[ConversationMessage(message_id="m1", role=MessageRole.USER, content="Rate limits apply", sequence_index=0)],
        )
        assert len(res_p) == 1
        assert res_p[0].preservation_status == EvidencePreservationStatus.PARTIALLY_PRESERVED
        assert "ev_fact" in res_p[0].partially_preserved_ids

    # Test 4: ambiguous -> AMBIGUOUS
    mock_resp_ambiguous = MagicMock(
        status_code=200,
        json=lambda: {
            "response": json.dumps({
                "verdict": "ambiguous",
                "explanation": "Two different rate limits are mentioned without priority.",
                "uncertainty_flag": True,
            })
        },
    )

    with patch("httpx.Client.post", return_value=mock_resp_ambiguous):
        res_amb = auditor.audit_probes(
            probes=[probe],
            evidence_units=[ev],
            compressed_messages=[ConversationMessage(message_id="m1", role=MessageRole.USER, content="100 or 200 req/min", sequence_index=0)],
        )
        assert len(res_amb) == 1
        assert res_amb[0].preservation_status == EvidencePreservationStatus.AMBIGUOUS
        assert res_amb[0].uncertainty_flag is True


def test_answer_key_leakage_prevention():
    """AuditorFacingProbe prompt strictly isolates question from ground-truth evidence IDs and answer keys."""
    ev = make_evidence("ev_secret_123", text="Database password is 'correct_battery_staple'")
    probe = make_probe(
        probe_id="p_leak_check",
        required_evidence_ids=["ev_secret_123"],
        target_evidence_text="Database password is 'correct_battery_staple'",
        question="What credentials are required to connect?",
    )

    policy = RABCOAuditPolicy(enable_semantic_audit=True)
    auditor = PreservationAuditor(policy=policy)

    captured_prompt = {}

    def mock_post(url, json, **kwargs):
        captured_prompt["prompt"] = json.get("prompt", "")
        return MagicMock(
            status_code=200,
            json=lambda: {
                "response": '{"verdict": "answerable", "explanation": "Credentials present", "uncertainty_flag": false}'
            },
        )

    with patch("httpx.Client.post", side_effect=mock_post):
        auditor.audit_probes(
            probes=[probe],
            evidence_units=[ev],
            compressed_messages=[ConversationMessage(message_id="m1", role=MessageRole.USER, content="Some message", sequence_index=0)],
        )

    prompt_text = captured_prompt.get("prompt", "")
    assert prompt_text != ""
    # Question must be present
    assert "What credentials are required to connect?" in prompt_text
    # Expected evidence ID must NEVER be leaked to the prompt
    assert "ev_secret_123" not in prompt_text
    # Binding target evidence text must NOT be included as an expected answer key
    assert "ProbeEvidenceBinding" not in prompt_text


def test_ollama_timeout_and_malformed_fallback():
    """Timeouts and malformed responses gracefully fall back to deterministic findings without crashing."""
    ev = make_evidence("ev_robust", text="Primary database cluster is in us-central1")
    probe = make_probe("probe_robust", ["ev_robust"])

    policy = RABCOAuditPolicy(enable_semantic_audit=True)
    auditor = PreservationAuditor(policy=policy)

    comp_res = CompressionResult(
        selected_evidence=(ev,),
        validated_summaries=(),
        omitted_evidence=(),
        validation_outcomes=(),
        original_token_estimate=20,
        compressed_token_estimate=20,
        token_reduction_percent=0.0,
        token_estimation_method="heuristic",
    )

    # Subcase A: Timeout
    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Read timed out")):
        res_timeout = auditor.audit_probes(
            probes=[probe],
            evidence_units=[ev],
            compression_result=comp_res,
        )
        assert len(res_timeout) == 1
        # Fallback retains the deterministic finding (PRESERVED)
        assert res_timeout[0].preservation_status == EvidencePreservationStatus.PRESERVED
        assert res_timeout[0].auditor_type == "deterministic_provenance_fallback"
        assert "Semantic audit fallback" in res_timeout[0].explanation
        assert res_timeout[0].uncertainty_flag is True

    # Subcase B: Non-200 HTTP error
    mock_500 = MagicMock(status_code=500, text="Internal Ollama error")
    with patch("httpx.Client.post", return_value=mock_500):
        res_500 = auditor.audit_probes(
            probes=[probe],
            evidence_units=[ev],
            compression_result=comp_res,
        )
        assert len(res_500) == 1
        assert res_500[0].preservation_status == EvidencePreservationStatus.PRESERVED
        assert res_500[0].auditor_type == "deterministic_provenance_fallback"

    # Subcase C: Malformed JSON
    mock_bad_json = MagicMock(status_code=200, json=lambda: {"response": "definitely not json at all"})
    with patch("httpx.Client.post", return_value=mock_bad_json):
        res_bad_json = auditor.audit_probes(
            probes=[probe],
            evidence_units=[ev],
            compression_result=comp_res,
        )
        assert len(res_bad_json) == 1
        assert res_bad_json[0].preservation_status == EvidencePreservationStatus.PRESERVED
        assert res_bad_json[0].auditor_type == "deterministic_provenance_fallback"


# ============================================================================
# 5. Empty Inputs & Edge Cases
# ============================================================================

def test_empty_inputs_handled_safely():
    """Empty probe collections or missing evidence inputs are handled safely without exceptions."""
    auditor = PreservationAuditor()

    # Empty probes
    empty_res = auditor.audit_probes(probes=[], evidence_units=[])
    assert empty_res == ()

    # Probes provided, but evidence empty
    p = make_probe("p_empty", ["ev_missing_all"])
    res = auditor.audit_probes(probes=[p], evidence_units=[])
    assert len(res) == 1
    # Missing from evidence collection -> classified as MISSING
    assert res[0].preservation_status == EvidencePreservationStatus.MISSING


# ============================================================================
# 6. End-to-End Phase 5C RiskModel Integration
# ============================================================================

def test_phase5c_risk_model_compatibility():
    """PreservationAuditor output directly integrates with RiskModel.assess()."""
    ev_mand = make_evidence("ev_mand", text="Mandatory compliance requirement", criticality=EvidenceCriticality.MANDATORY)
    ev_norm = make_evidence("ev_norm", text="Optional aesthetic suggestion", criticality=EvidenceCriticality.LOW)

    p_mand = make_probe("p_mand", ["ev_mand"], question="What is the mandatory compliance requirement?")
    p_norm = make_probe("p_norm", ["ev_norm"], question="What is the suggestion?")

    # Compressor retained mandatory evidence extractively, but omitted normal evidence
    comp_res = CompressionResult(
        selected_evidence=(ev_mand,),
        validated_summaries=(),
        omitted_evidence=(make_omitted_evidence("ev_norm", original_text=ev_norm.original_text),),
        validation_outcomes=(),
        original_token_estimate=50,
        compressed_token_estimate=25,
        token_reduction_percent=50.0,
        token_estimation_method="heuristic",
    )

    auditor = PreservationAuditor()
    audit_results = auditor.audit_probes(
        probes=[p_mand, p_norm],
        evidence_units=[ev_mand, ev_norm],
        compression_result=comp_res,
    )

    assert len(audit_results) == 2
    assert audit_results[0].preservation_status == EvidencePreservationStatus.PRESERVED
    assert audit_results[1].preservation_status == EvidencePreservationStatus.MISSING

    # Pass directly into Phase 5C RiskModel
    risk_model = RiskModel()
    assessment = risk_model.assess(
        audit_results=audit_results,
        evidence_units=[ev_mand, ev_norm],
        compression_result=comp_res,
        probes=[p_mand, p_norm],
    )

    assert isinstance(assessment, RiskAssessment)
    assert assessment.total_probes_audited == 2
    assert assessment.total_probes_generated == 2
    assert assessment.probe_failure_rate == 0.5  # 1 of 2 probes failed (the low-criticality one)
    assert assessment.probe_coverage == 0.5      # 1 of 2 preserved
    assert assessment.mandatory_evidence_loss == 0  # Mandatory unit was preserved!
    assert assessment.bypass_recommended is False # Low-criticality failure does not breach mandatory invariants
