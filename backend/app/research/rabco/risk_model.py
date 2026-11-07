"""RABCO Phase 5C: Explainable Risk Model and Preservation Assessment.

Project: Argus — Risk-Aware, Counterfactual-Bounded Context Optimization (RABCO).
Phase: 5C — Risk Model.

Purpose:
    Provide a deterministic, fully explainable risk assessment component that evaluates
    evidence preservation across compressed conversation context.

Architectural Guarantees:
1. Grounded Provenance & Independent Evaluation:
   Evidence preservation is derived from actual compressor outputs (CompressionResult)
   and verified counterfactual probe audit outcomes (ProbeAuditResult). Preservation
   is NEVER assumed simply because an evidence ID was referenced in a probe.
2. Strict Criticality Segregation & Mandatory Invariant:
   EvidenceCriticality.MANDATORY is distinctly separated from EvidenceCriticality.HIGH.
   Mandatory evidence must be fully preserved (PRESERVED) and independently verified.
   Any loss, partial preservation, ambiguity, or unverifiability of mandatory evidence
   (mandatory_evidence_loss > 0) is a strict safety invariant violation and unconditionally
   mandates original-context bypass (bypass_recommended = True).
3. Directional Relational Integrity:
   Preservation of relationships follows domain semantics and graph direction:
   - DEPENDS_ON: source (dependent) depends on target (prerequisite). If dependent is
     retained, prerequisite MUST be preserved.
   - CORRECTS: source (correction) corrects target (antecedent). Both must be preserved.
   - SUPERSEDES: source (current) supersedes target (previous). Both must be preserved.
   - CONTRADICTS: unresolved conflict between source and target. Both must be preserved.
   - EXCEPTION_TO: source (exception) modifies target (rule). Rule cannot be retained
     without its exception.
   - CONDITIONAL_ON: source (action) conditional on target (condition). Condition must
     not be discarded while retaining the action.
   When zero applicable dependencies or contradictions exist, indicators report None
   (unavailable) rather than fabricating perfect preservation or artificial penalties.
4. Exact Disjoint Counting & Probe Coverage:
   Probe failure calculations partition audited probes into mutually exclusive subsets:
   PRESERVED, MISSING, PARTIALLY_PRESERVED, AMBIGUOUS, UNVERIFIABLE.
   Each audited probe is counted exactly once. Probe coverage uses the total candidate
   probes as denominator when generated probes are provided, ensuring unaudited probes
   are never claimed as preserved.
5. Objective Recovery Accounting:
   recovery_effectiveness is None when recovery was not performed, avoiding false
   defaults. When performed, it reflects the true fraction of targeted missing evidence
   successfully restored.
6. Invariant Renormalization & Bypass Overrides:
   A low aggregate failure rate or composite risk score NEVER masks critical evidence loss.
   Composite risk scoring excludes unavailable dimensions and renormalizes remaining weights.
   If no dimensions are available, composite score is None.
7. Scientific Integrity & Disclaimers:
   Risk indicators represent observable sample ratios and counts, not calibrated probabilities.
8. Decoupled Architecture:
   Independent of network calls, external APIs, and LLM providers. Fully deterministic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

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
from .hybrid_compressor import (
    CompressionPolicy,
    CompressionResult,
    SummaryValidationOutcome,
)
from .schemas import (
    CounterfactualProbe,
    EvidencePreservationStatus,
    ProbeAuditResult,
    ProbeCategory,
    ProbeSetType,
    RABCOAuditPolicy,
    RABCORecoveryPolicy,
    RecoveryRecord,
    RecoveryResult,
    RelationshipAuditDetail,
    RiskAssessment,
)

log = logging.getLogger(__name__)


# ============================================================================
# Provisional Risk Scoring Weights Configuration
# ============================================================================

@dataclass(frozen=True)
class RiskScoringWeights:
    """Provisional heuristic weights for optional composite risk indicator calculation.

    Scientific Disclaimer:
    These weights are heuristic baseline research parameters and are NOT calibrated
    probabilities. The composite score provides a single summary indicator for experimental
    comparison only and must never override explicit invariant checks.
    """

    weight_probe_failure: float = field(
        default_factory=lambda: getattr(settings, "RABCO_RISK_WEIGHT_PROBE_FAILURE", 0.30)
    )
    weight_mandatory_loss: float = field(
        default_factory=lambda: getattr(settings, "RABCO_RISK_WEIGHT_MANDATORY_LOSS", 0.30)
    )
    weight_dependency_failure: float = field(
        default_factory=lambda: getattr(settings, "RABCO_RISK_WEIGHT_DEPENDENCY_FAILURE", 0.15)
    )
    weight_contradiction_failure: float = field(
        default_factory=lambda: getattr(settings, "RABCO_RISK_WEIGHT_CONTRADICTION_FAILURE", 0.15)
    )
    weight_summary_uncertainty: float = field(
        default_factory=lambda: getattr(settings, "RABCO_RISK_WEIGHT_SUMMARY_UNCERTAINTY", 0.10)
    )

    def __post_init__(self) -> None:
        """Validate individual weight ranges and aggregate sum constraint."""
        weights = (
            self.weight_probe_failure,
            self.weight_mandatory_loss,
            self.weight_dependency_failure,
            self.weight_contradiction_failure,
            self.weight_summary_uncertainty,
        )
        for w in weights:
            if not (0.0 <= w <= 1.0):
                raise ValueError(f"Risk scoring weight must be between 0.0 and 1.0, got {w}")
        total = sum(weights)
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Aggregate RiskScoringWeights must sum to 1.0 within 0.01 tolerance, got sum={total:.4f}"
            )


# ============================================================================
# Risk Model Implementation
# ============================================================================

class RiskModel:
    """Deterministic, explainable risk assessment engine for RABCO Phase 5C.

    Consumes:
    - Counterfactual probe audit results (ProbeAuditResult).
    - Original evidence units (EvidenceUnit) and relationship graph (EvidenceRelationship).
    - Compression outputs (CompressionResult) and policies (CompressionPolicy, RABCOAuditPolicy).
    - Bounded recovery records (RecoveryResult), when applicable.
    - Candidate generated probes (CounterfactualProbe) for coverage auditing.

    Produces:
    - RiskAssessment: immutable multi-indicator risk vector with complete calculation provenance,
      safety invariant checks, and fail-safe bypass recommendations.
    """

    def __init__(
        self,
        weights: Optional[RiskScoringWeights] = None,
        config=settings,
    ) -> None:
        """Initialize RiskModel with centralized configuration and heuristic weights.

        Args:
            weights: Optional RiskScoringWeights instance. Sourced from centralized settings by default.
            config: Application configuration instance.
        """
        self.weights = weights or RiskScoringWeights()
        self.config = config

    def assess(
        self,
        audit_results: Sequence[ProbeAuditResult],
        evidence_units: Sequence[EvidenceUnit],
        relationships: Sequence[EvidenceRelationship] = (),
        compression_result: Optional[CompressionResult] = None,
        audit_policy: Optional[RABCOAuditPolicy] = None,
        compression_policy: Optional[CompressionPolicy] = None,
        recovery_result: Optional[RecoveryResult] = None,
        probes: Sequence[CounterfactualProbe] = (),
    ) -> RiskAssessment:
        """Execute deterministic risk assessment across audited context.

        Parameters
        ----------
        audit_results:
            Preservation audit outcomes per probe.
        evidence_units:
            Full collection of extracted evidence units from original conversation.
        relationships:
            Relational links between evidence units (DEPENDS_ON, CORRECTS, etc.).
        compression_result:
            Optional compression output from HybridCompressor. If provided, establishes
            extractive vs summarized vs omitted status.
        audit_policy:
            Optional RABCOAuditPolicy defining non-mandatory failure tolerances.
        compression_policy:
            Optional CompressionPolicy defining core invariant preservation rules.
        recovery_result:
            Optional outcome from selective evidence recovery loop.
        probes:
            Optional collection of generated candidate probes for coverage auditing.

        Returns
        -------
        RiskAssessment
            Structured, explainable assessment vector.
        """
        eff_audit_policy = audit_policy or RABCOAuditPolicy()
        eff_comp_policy = compression_policy or CompressionPolicy()

        # Step 1: Index evidence units and segregate criticality tiers
        evidence_by_id: Dict[str, EvidenceUnit] = {u.evidence_id: u for u in evidence_units}
        mandatory_ids: Set[str] = {
            u.evidence_id for u in evidence_units if u.criticality == EvidenceCriticality.MANDATORY
        }
        high_criticality_ids: Set[str] = {
            u.evidence_id for u in evidence_units if u.criticality == EvidenceCriticality.HIGH
        }
        numerical_ids: Set[str] = {
            u.evidence_id for u in evidence_units if u.evidence_type == EvidenceType.NUMERICAL_VALUE
        }
        correction_ids: Set[str] = {
            u.evidence_id for u in evidence_units if u.evidence_type == EvidenceType.CORRECTION
        }

        # Step 2: Establish compressor survival status (Extractive vs Summarized vs Omitted)
        extractive_ids: Set[str] = set()
        summarized_ids: Set[str] = set()
        omitted_ids: Set[str] = set()
        invalid_summary_evidence_ids: Set[str] = set()

        if recovery_result is not None:
            extractive_ids = {u.evidence_id for u in recovery_result.final_selected_evidence}
            summarized_ids = {
                eid for s in recovery_result.final_validated_summaries for eid in s.source_evidence_ids
            }
        elif compression_result is not None:
            extractive_ids = {u.evidence_id for u in compression_result.selected_evidence}
            summarized_ids = {
                eid for s in compression_result.validated_summaries for eid in s.source_evidence_ids
            }
            omitted_ids = {o.evidence_id for o in compression_result.omitted_evidence}
            for outcome in compression_result.validation_outcomes:
                if not outcome.is_valid:
                    invalid_summary_evidence_ids.update(outcome.covered_evidence_ids)

        # Step 3: Classify probe audit outcomes into disjoint categories
        total_probes_audited = len(audit_results)
        total_probes_generated = len(probes)

        preserved_probes: List[ProbeAuditResult] = []
        missing_probes: List[ProbeAuditResult] = []
        partial_probes: List[ProbeAuditResult] = []
        ambiguous_probes: List[ProbeAuditResult] = []
        unverifiable_probes: List[ProbeAuditResult] = []

        # Track explicit evidence mentions in audit outcomes
        audit_reported_missing_evidence: Set[str] = set()
        audit_reported_partial_evidence: Set[str] = set()
        probes_by_required_evidence: Dict[str, List[ProbeAuditResult]] = {
            eid: [] for eid in evidence_by_id
        }

        for audit in audit_results:
            status = audit.preservation_status
            if status == EvidencePreservationStatus.PRESERVED:
                preserved_probes.append(audit)
            elif status == EvidencePreservationStatus.MISSING:
                missing_probes.append(audit)
            elif status == EvidencePreservationStatus.PARTIALLY_PRESERVED:
                partial_probes.append(audit)
            elif status == EvidencePreservationStatus.AMBIGUOUS:
                ambiguous_probes.append(audit)
            elif status == EvidencePreservationStatus.UNVERIFIABLE:
                unverifiable_probes.append(audit)
            else:
                unverifiable_probes.append(audit)

            # Collect explicitly flagged evidence IDs
            audit_reported_missing_evidence.update(audit.missing_evidence_ids)
            audit_reported_partial_evidence.update(audit.partially_preserved_ids)

            # Track which probes evaluated which evidence units
            for req_eid in audit.required_evidence_ids:
                if req_eid in probes_by_required_evidence:
                    probes_by_required_evidence[req_eid].append(audit)

        # Disjoint counts
        count_preserved = len(preserved_probes)
        count_missing = len(missing_probes)
        count_partial = len(partial_probes)
        count_ambiguous = len(ambiguous_probes)
        count_unverifiable = len(unverifiable_probes)
        total_failed_probes = count_missing + count_partial + count_ambiguous + count_unverifiable

        # Assert disjoint partitioning invariant
        assert count_preserved + total_failed_probes == total_probes_audited, (
            "Invariant violation: audited probe classification is not disjoint."
        )

        probe_failure_rate = (
            round(total_failed_probes / total_probes_audited, 4)
            if total_probes_audited > 0
            else 0.0
        )

        # Probe coverage:
        # If generated probes are provided, the denominator is max(total_probes_generated, total_probes_audited),
        # ensuring unaudited candidate probes are not falsely claimed as preserved.
        # If no generated probes are provided, denominator defaults to total_probes_audited.
        coverage_denominator = (
            max(total_probes_generated, total_probes_audited)
            if total_probes_generated > 0
            else total_probes_audited
        )
        probe_coverage = (
            round(count_preserved / coverage_denominator, 4)
            if coverage_denominator > 0
            else 0.0
        )

        audited_probe_ids = {a.probe_id for a in audit_results}
        unaudited_probes = (
            [p for p in probes if p.probe_id not in audited_probe_ids]
            if total_probes_generated > 0
            else []
        )

        # Step 4: Classify mandatory vs. non-mandatory probe failures
        mandatory_probes: List[ProbeAuditResult] = []
        non_mandatory_probes: List[ProbeAuditResult] = []

        for audit in audit_results:
            is_mand = any(req_id in mandatory_ids for req_id in audit.required_evidence_ids)
            if is_mand:
                mandatory_probes.append(audit)
            else:
                non_mandatory_probes.append(audit)

        failed_mandatory_probes = [
            p for p in mandatory_probes if p.preservation_status != EvidencePreservationStatus.PRESERVED
        ]
        failed_non_mandatory_probes = [
            p for p in non_mandatory_probes if p.preservation_status != EvidencePreservationStatus.PRESERVED
        ]

        non_mandatory_probe_failure_rate = (
            round(len(failed_non_mandatory_probes) / len(non_mandatory_probes), 4)
            if non_mandatory_probes
            else 0.0
        )

        # Step 5: Derive verified evidence preservation statuses
        # Synthesizes compressor selection and auditor verification without conflating them.
        verified_evidence_status: Dict[str, EvidencePreservationStatus] = {}
        for eid in evidence_by_id:
            # 1. Direct explicit auditor finding has highest precedence
            if eid in audit_reported_missing_evidence:
                verified_evidence_status[eid] = EvidencePreservationStatus.MISSING
                continue

            # 2. Explicit compressor omission
            if eid in omitted_ids and eid not in extractive_ids and eid not in summarized_ids:
                verified_evidence_status[eid] = EvidencePreservationStatus.MISSING
                continue

            # 3. Explicit auditor partial finding
            if eid in audit_reported_partial_evidence:
                verified_evidence_status[eid] = EvidencePreservationStatus.PARTIALLY_PRESERVED
                continue

            # 4. Probe-level outcomes for probes requiring this evidence
            probes_for_eid = probes_by_required_evidence.get(eid, [])
            if probes_for_eid:
                statuses = {p.preservation_status for p in probes_for_eid}
                if EvidencePreservationStatus.MISSING in statuses:
                    verified_evidence_status[eid] = EvidencePreservationStatus.MISSING
                elif EvidencePreservationStatus.AMBIGUOUS in statuses:
                    verified_evidence_status[eid] = EvidencePreservationStatus.AMBIGUOUS
                elif EvidencePreservationStatus.UNVERIFIABLE in statuses:
                    verified_evidence_status[eid] = EvidencePreservationStatus.UNVERIFIABLE
                elif EvidencePreservationStatus.PARTIALLY_PRESERVED in statuses:
                    verified_evidence_status[eid] = EvidencePreservationStatus.PARTIALLY_PRESERVED
                elif statuses == {EvidencePreservationStatus.PRESERVED}:
                    verified_evidence_status[eid] = EvidencePreservationStatus.PRESERVED
                else:
                    verified_evidence_status[eid] = EvidencePreservationStatus.AMBIGUOUS
                continue

            # 5. Untested evidence: inspect compressor output if available
            if compression_result is not None or recovery_result is not None:
                if eid in extractive_ids:
                    # Present extractively, but not independently audited by a probe
                    verified_evidence_status[eid] = EvidencePreservationStatus.PRESERVED
                elif eid in summarized_ids:
                    # Summarized, but not independently audited
                    verified_evidence_status[eid] = EvidencePreservationStatus.PARTIALLY_PRESERVED
                elif eid in invalid_summary_evidence_ids:
                    # Represented only in an unverified or rejected summary
                    verified_evidence_status[eid] = EvidencePreservationStatus.UNVERIFIABLE
                else:
                    verified_evidence_status[eid] = EvidencePreservationStatus.UNVERIFIABLE
            else:
                # Neither audited nor reported by compressor
                verified_evidence_status[eid] = EvidencePreservationStatus.UNVERIFIABLE

        # Step 6: Identify lost mandatory and high-criticality evidence
        lost_mandatory_ids: Set[str] = set()
        lost_high_criticality_ids: Set[str] = set()
        lost_numerical_ids: Set[str] = set()
        lost_correction_ids: Set[str] = set()

        for eid, st in verified_evidence_status.items():
            # Mandatory evidence safety invariant:
            # Mandatory evidence MUST be fully preserved (PRESERVED).
            # MISSING, PARTIALLY_PRESERVED, AMBIGUOUS, and UNVERIFIABLE are all safety failures!
            if eid in mandatory_ids:
                if st != EvidencePreservationStatus.PRESERVED:
                    lost_mandatory_ids.add(eid)

            # High-criticality, numerical, and correction tracking:
            is_lost_or_degraded = st in {
                EvidencePreservationStatus.MISSING,
                EvidencePreservationStatus.PARTIALLY_PRESERVED,
                EvidencePreservationStatus.AMBIGUOUS,
                EvidencePreservationStatus.UNVERIFIABLE,
            }
            if is_lost_or_degraded:
                if eid in high_criticality_ids:
                    lost_high_criticality_ids.add(eid)
                if eid in numerical_ids:
                    lost_numerical_ids.add(eid)
                if eid in correction_ids:
                    lost_correction_ids.add(eid)

        mandatory_evidence_loss = len(lost_mandatory_ids)

        # Step 7: Analyze directional relationship preservation
        rel_audit_details: List[RelationshipAuditDetail] = []
        applicable_dependencies = 0
        preserved_dependencies = 0
        applicable_contradictions = 0
        preserved_contradictions = 0
        broken_relationships: List[Dict[str, Any]] = []

        for rel in relationships:
            src_id = rel.source_evidence_id
            tgt_id = rel.target_evidence_id
            src_st = verified_evidence_status.get(src_id, EvidencePreservationStatus.UNVERIFIABLE)
            tgt_st = verified_evidence_status.get(tgt_id, EvidencePreservationStatus.UNVERIFIABLE)

            src_ok = src_st == EvidencePreservationStatus.PRESERVED
            tgt_ok = tgt_st == EvidencePreservationStatus.PRESERVED

            rel_type = rel.relationship_type
            is_app = False
            is_pres = False
            fail_cat: Optional[str] = None
            fail_expl: Optional[str] = None

            if rel_type == RelationshipType.DEPENDS_ON:
                # source depends on target (prerequisite).
                # Retaining source requires target to be preserved.
                is_app = src_ok
                if is_app:
                    applicable_dependencies += 1
                    if tgt_ok:
                        preserved_dependencies += 1
                        is_pres = True
                    else:
                        fail_cat = "prerequisite_missing"
                        fail_expl = (
                            f"Dependent evidence '{src_id}' is preserved, but prerequisite "
                            f"evidence '{tgt_id}' is not preserved (status={tgt_st.value})."
                        )
                else:
                    # Dependent was omitted; dependency is intact/non-applicable
                    is_pres = True

            elif rel_type == RelationshipType.CONTRADICTS:
                # Contradiction: both opposing poles must remain to avoid one-sided suppression.
                is_app = src_ok or tgt_ok
                if is_app:
                    applicable_contradictions += 1
                    if src_ok and tgt_ok:
                        preserved_contradictions += 1
                        is_pres = True
                    else:
                        fail_cat = "contradiction_pole_suppressed"
                        fail_expl = (
                            f"Contradiction between '{src_id}' (status={src_st.value}) and "
                            f"'{tgt_id}' (status={tgt_st.value}) was partially or fully suppressed."
                        )
                else:
                    is_pres = True

            elif rel_type in {RelationshipType.CORRECTS, RelationshipType.SUPERSEDES}:
                # Corrections & Supersessions: both antecedent and updated value must survive.
                is_app = src_ok or tgt_ok
                if is_app:
                    if src_ok and tgt_ok:
                        is_pres = True
                    else:
                        fail_cat = "correction_chain_broken"
                        fail_expl = (
                            f"Correction/supersession link between '{src_id}' (status={src_st.value}) "
                            f"and '{tgt_id}' (status={tgt_st.value}) is broken."
                        )
                else:
                    is_pres = True

            elif rel_type == RelationshipType.EXCEPTION_TO:
                # Exception: source is exception, target is rule.
                # If rule is preserved, exception MUST NOT be discarded.
                is_app = tgt_ok
                if is_app:
                    if src_ok:
                        is_pres = True
                    else:
                        fail_cat = "exception_missing_for_active_rule"
                        fail_expl = (
                            f"Rule '{tgt_id}' is preserved, but exception '{src_id}' "
                            f"was lost (status={src_st.value})."
                        )
                else:
                    is_pres = True

            elif rel_type == RelationshipType.CONDITIONAL_ON:
                # Condition: source is action, target is condition.
                # If action is preserved, condition MUST NOT be discarded.
                is_app = src_ok
                if is_app:
                    if tgt_ok:
                        is_pres = True
                    else:
                        fail_cat = "condition_missing_for_active_action"
                        fail_expl = (
                            f"Action '{src_id}' is preserved, but governing condition "
                            f"'{tgt_id}' was lost (status={tgt_st.value})."
                        )
                else:
                    is_pres = True

            else:
                # General association (applies_to, same_entity, etc.)
                is_app = src_ok and tgt_ok
                is_pres = is_app

            detail = RelationshipAuditDetail(
                relationship_id=rel.relationship_id,
                relationship_type=rel_type,
                source_evidence_id=src_id,
                target_evidence_id=tgt_id,
                source_preserved=src_ok,
                target_preserved=tgt_ok,
                is_applicable=is_app,
                is_preserved=is_pres,
                failure_category=fail_cat,
                failure_explanation=fail_expl,
            )
            rel_audit_details.append(detail)

            if fail_cat is not None:
                broken_relationships.append({
                    "relationship_id": rel.relationship_id,
                    "relationship_type": rel_type.value,
                    "source_evidence_id": src_id,
                    "target_evidence_id": tgt_id,
                    "failure_category": fail_cat,
                    "explanation": fail_expl,
                })

        # Represent zero applicable dependencies/contradictions as None (unavailable)
        # rather than reporting fabricated 1.0 (perfect) preservation.
        dependency_preservation: Optional[float] = (
            round(preserved_dependencies / applicable_dependencies, 4)
            if applicable_dependencies > 0
            else None
        )
        contradiction_preservation: Optional[float] = (
            round(preserved_contradictions / applicable_contradictions, 4)
            if applicable_contradictions > 0
            else None
        )

        # Step 8: Assess summary integrity from validation outcomes
        summary_pass_count = 0
        summary_uncertain_count = 0
        if compression_result is not None:
            for outcome in compression_result.validation_outcomes:
                if outcome.is_valid:
                    summary_pass_count += 1
                else:
                    summary_uncertain_count += 1

        # Step 9: Recovery effectiveness (None if recovery was not performed)
        recovery_effectiveness: Optional[float] = None
        if recovery_result is not None and recovery_result.total_recovery_attempts > 0:
            targeted_recovery_ids: Set[str] = set()
            for rec in recovery_result.recovery_attempts:
                targeted_recovery_ids.update(rec.missing_evidence_ids_targeted)

            if targeted_recovery_ids:
                final_restored_ids = {
                    u.evidence_id for u in recovery_result.final_selected_evidence
                } & targeted_recovery_ids
                recovery_effectiveness = round(
                    len(final_restored_ids) / len(targeted_recovery_ids), 4
                )
            else:
                recovery_effectiveness = 1.0

        # Step 10: Fail-Safe Bypass Evaluation & Invariant Overrides
        bypass_reasons: List[str] = []

        # 1. Mandatory Evidence Loss (Inviolable Safety Invariant)
        # Mandatory evidence must be fully preserved; partially preserved, ambiguous,
        # missing, and unverifiable mandatory units unconditionally mandate bypass.
        if mandatory_evidence_loss > 0:
            lost_mand_details = [
                f"{m_id} ({verified_evidence_status.get(m_id, EvidencePreservationStatus.UNVERIFIABLE).value})"
                for m_id in sorted(lost_mandatory_ids)
            ]
            bypass_reasons.append(
                f"STRICT SAFETY INVARIANT VIOLATION: {mandatory_evidence_loss} mandatory evidence "
                f"unit(s) not fully preserved (missing, partially preserved, ambiguous, or unverifiable): {lost_mand_details}"
            )

        # 2. Failed Mandatory Probes
        if failed_mandatory_probes:
            bypass_reasons.append(
                f"Mandatory probe audit failure: {len(failed_mandatory_probes)} mandatory probe(s) "
                f"failed preservation verification: {[p.probe_id for p in failed_mandatory_probes]}"
            )

        # 3. Core Relational Invariants (from CompressionPolicy)
        broken_deps = [r for r in broken_relationships if r["relationship_type"] == RelationshipType.DEPENDS_ON.value]
        if eff_comp_policy.preserve_all_dependencies and broken_deps:
            bypass_reasons.append(
                f"CORE INVARIANT BREACH (preserve_all_dependencies): {len(broken_deps)} prerequisite(s) "
                f"missing for retained dependent evidence: {[b['relationship_id'] for b in broken_deps]}"
            )

        broken_contras = [r for r in broken_relationships if r["relationship_type"] == RelationshipType.CONTRADICTS.value]
        if eff_comp_policy.preserve_all_contradictions and broken_contras:
            bypass_reasons.append(
                f"CORE INVARIANT BREACH (preserve_all_contradictions): {len(broken_contras)} contradiction "
                f"pair(s) partially or completely suppressed: {[b['relationship_id'] for b in broken_contras]}"
            )

        if eff_comp_policy.preserve_all_corrections:
            broken_corrs = [
                r for r in broken_relationships
                if r["relationship_type"] in {RelationshipType.CORRECTS.value, RelationshipType.SUPERSEDES.value}
            ]
            if broken_corrs:
                bypass_reasons.append(
                    f"CORE INVARIANT BREACH (preserve_all_corrections): {len(broken_corrs)} correction/supersession "
                    f"chain(s) severed: {[b['relationship_id'] for b in broken_corrs]}"
                )

        if eff_comp_policy.preserve_all_exceptions:
            broken_excs = [r for r in broken_relationships if r["relationship_type"] == RelationshipType.EXCEPTION_TO.value]
            if broken_excs:
                bypass_reasons.append(
                    f"CORE INVARIANT BREACH (preserve_all_exceptions): {len(broken_excs)} exception(s) "
                    f"discarded while governed rule was retained: {[b['relationship_id'] for b in broken_excs]}"
                )

        # 4. Compressor Safety State
        if compression_result is not None:
            if not compression_result.safety_invariants_enforced:
                bypass_reasons.append(
                    f"Compressor ran in unsafe experimental relaxation mode: {list(compression_result.safety_warnings)}"
                )
            if compression_result.budget_status == "budget_conflict":
                bypass_reasons.append(
                    f"Budget conflict in compressor: {compression_result.budget_conflict_details or 'mandatory context exceeds budget'}"
                )

        # 5. Recovery Exhaustion or Forced Bypass
        if recovery_result is not None and recovery_result.bypass_required:
            bypass_reasons.append(
                f"Selective recovery exhausted or failed to restore safety: {recovery_result.bypass_reason or 'recovery failed'}"
            )

        # 6. Non-Mandatory Probe Failure Tolerance
        max_tolerated_non_mandatory = eff_audit_policy.max_acceptable_non_mandatory_failure_rate
        if non_mandatory_probe_failure_rate > max_tolerated_non_mandatory:
            bypass_reasons.append(
                f"Non-mandatory probe failure rate ({non_mandatory_probe_failure_rate:.3f}) exceeds "
                f"policy tolerance threshold ({max_tolerated_non_mandatory:.3f})"
            )

        bypass_recommended = len(bypass_reasons) > 0
        bypass_reason = " | ".join(bypass_reasons) if bypass_reasons else None

        # Step 11: Compute Provisional Composite Score (for research benchmarking only)
        # Renormalize across available dimensions; if no dimensions are available, score is None.
        w_prob = self.weights.weight_probe_failure
        w_mand = self.weights.weight_mandatory_loss
        w_dep = self.weights.weight_dependency_failure
        w_contra = self.weights.weight_contradiction_failure
        w_sum = self.weights.weight_summary_uncertainty

        available_dims: Dict[str, Tuple[float, float]] = {}

        # Dimension 1: Probe failure (available if probes were audited)
        if total_probes_audited > 0:
            available_dims["probe_failure"] = (w_prob, probe_failure_rate)

        # Dimension 2: Mandatory loss (available if mandatory evidence units exist)
        if mandatory_ids:
            mand_loss_ratio = min(1.0, mandatory_evidence_loss / len(mandatory_ids))
            available_dims["mandatory_loss"] = (w_mand, mand_loss_ratio)

        # Dimension 3: Dependency failure (available if applicable dependencies exist)
        if dependency_preservation is not None:
            available_dims["dependency_failure"] = (w_dep, 1.0 - dependency_preservation)

        # Dimension 4: Contradiction failure (available if applicable contradictions exist)
        if contradiction_preservation is not None:
            available_dims["contradiction_failure"] = (w_contra, 1.0 - contradiction_preservation)

        # Dimension 5: Summary uncertainty (available if summaries exist)
        total_summaries = summary_pass_count + summary_uncertain_count
        if total_summaries > 0:
            sum_uncertainty_ratio = summary_uncertain_count / total_summaries
            available_dims["summary_uncertainty"] = (w_sum, sum_uncertainty_ratio)

        sum_available_weights = sum(w for w, _ in available_dims.values())
        if sum_available_weights > 0:
            raw_composite_score = round(
                sum(w * val for w, val in available_dims.values()) / sum_available_weights,
                4,
            )
            renormalized_weights = {
                dim: round(w / sum_available_weights, 4)
                for dim, (w, _) in available_dims.items()
            }
        else:
            raw_composite_score = None
            renormalized_weights = {}

        # Step 12: Assemble Indicator Calculation Methods and Diagnostic Telemetry
        methods_doc: Dict[str, Any] = {
            "probe_failure_rate": {
                "formula": "total_failed_probes / total_probes_audited",
                "total_probes_audited": total_probes_audited,
                "count_preserved": count_preserved,
                "count_missing": count_missing,
                "count_partial": count_partial,
                "count_ambiguous": count_ambiguous,
                "count_unverifiable": count_unverifiable,
                "failed_probe_ids": [p.probe_id for p in audit_results if p.preservation_status != EvidencePreservationStatus.PRESERVED],
            },
            "non_mandatory_probe_failure_rate": {
                "formula": "failed_non_mandatory_probes / total_non_mandatory_probes",
                "total_non_mandatory_probes": len(non_mandatory_probes),
                "failed_non_mandatory_count": len(failed_non_mandatory_probes),
                "tolerance_threshold": max_tolerated_non_mandatory,
                "tolerance_exceeded": non_mandatory_probe_failure_rate > max_tolerated_non_mandatory,
            },
            "probe_coverage": {
                "formula": (
                    "count_preserved / max(total_probes_generated, total_probes_audited)"
                    if total_probes_generated > 0
                    else "count_preserved / total_probes_audited"
                ),
                "total_probes_audited": total_probes_audited,
                "total_probes_generated": total_probes_generated,
                "count_preserved": count_preserved,
                "unaudited_probe_count": len(unaudited_probes),
                "unaudited_probe_ids": [p.probe_id for p in unaudited_probes],
                "coverage_value": probe_coverage,
            },
            "mandatory_evidence_loss": {
                "formula": "count(mandatory evidence where status in {MISSING, PARTIALLY_PRESERVED, AMBIGUOUS, UNVERIFIABLE})",
                "total_mandatory_units": len(mandatory_ids),
                "lost_count": mandatory_evidence_loss,
                "lost_mandatory_ids": sorted(lost_mandatory_ids),
                "mandatory_status_breakdown": {
                    eid: verified_evidence_status.get(eid, EvidencePreservationStatus.UNVERIFIABLE).value
                    for eid in mandatory_ids
                },
            },
            "dependency_preservation": {
                "formula": (
                    "preserved_applicable_dependencies / applicable_dependencies"
                    if applicable_dependencies > 0
                    else "unavailable (applicable_dependencies == 0)"
                ),
                "applicable_dependencies": applicable_dependencies,
                "preserved_dependencies": preserved_dependencies,
                "value": dependency_preservation,
            },
            "contradiction_preservation": {
                "formula": (
                    "preserved_contradiction_pairs / applicable_contradiction_pairs"
                    if applicable_contradictions > 0
                    else "unavailable (applicable_contradictions == 0)"
                ),
                "applicable_contradictions": applicable_contradictions,
                "preserved_contradictions": preserved_contradictions,
                "value": contradiction_preservation,
            },
            "summary_integrity": {
                "pass_count": summary_pass_count,
                "uncertain_count": summary_uncertain_count,
            },
            "recovery_effectiveness": {
                "formula": (
                    "restored_targeted_ids / targeted_missing_ids"
                    if recovery_effectiveness is not None
                    else "not_applicable (recovery was not performed)"
                ),
                "value": recovery_effectiveness,
            },
            "evidence_loss_breakdown": {
                "high_criticality_lost_ids": sorted(lost_high_criticality_ids),
                "numerical_lost_ids": sorted(lost_numerical_ids),
                "correction_lost_ids": sorted(lost_correction_ids),
            },
            "broken_relationships": broken_relationships,
            "bypass_evaluation": {
                "bypass_recommended": bypass_recommended,
                "critical_reasons": bypass_reasons,
            },
            "provisional_composite_risk_score": {
                "score": raw_composite_score,
                "available_dimensions": list(available_dims.keys()),
                "renormalized_weights": renormalized_weights,
                "configured_weights": {
                    "probe_failure": w_prob,
                    "mandatory_loss": w_mand,
                    "dependency_failure": w_dep,
                    "contradiction_failure": w_contra,
                    "summary_uncertainty": w_sum,
                },
                "disclaimer": (
                    "Composite risk score is a provisional research heuristic. "
                    "It is NOT a calibrated probability and does NOT override critical safety invariants."
                ),
            },
        }

        return RiskAssessment(
            total_probes_audited=total_probes_audited,
            total_probes_generated=total_probes_generated,
            probe_failure_rate=probe_failure_rate,
            non_mandatory_probe_failure_rate=non_mandatory_probe_failure_rate,
            probe_coverage=probe_coverage,
            mandatory_evidence_loss=mandatory_evidence_loss,
            dependency_preservation=dependency_preservation,
            contradiction_preservation=contradiction_preservation,
            summary_integrity_pass_count=summary_pass_count,
            summary_integrity_uncertain_count=summary_uncertain_count,
            recovery_effectiveness=recovery_effectiveness,
            bypass_recommended=bypass_recommended,
            bypass_reason=bypass_reason,
            composite_risk_score=raw_composite_score,
            relationship_audit_details=tuple(rel_audit_details),
            indicator_calculation_methods=methods_doc,
            probability_calibration_disclaimer=(
                "Indicators represent discrete observable counts and sample ratios. "
                "They are not calibrated statistical probabilities."
            ),
        )
