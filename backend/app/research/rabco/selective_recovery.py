"""RABCO Phase 5E: Selective Evidence Recovery.

Project: Argus — Risk-Aware, Counterfactual-Bounded Context Optimization (RABCO).
Phase: 5E — Selective Evidence Recovery.

Purpose:
    Provide a bounded, evidence-aware selective recovery engine that restores only evidence
    units identified as missing, partially preserved, or broken in critical relationships by
    Phase 5D preservation auditing, without unnecessarily restoring the entire original context.

Architectural Guarantees:
1. Evidence-Aware Selective Recovery:
   Targets only evidence units identified as missing, partially preserved, or unverified by
   Phase 5D audit findings. Never restores unneeded evidence already adequately preserved.
2. Bounded Iterations:
   Enforces a strict configurable maximum number of recovery attempts (settings.RABCO_MAX_RECOVERY_ATTEMPTS).
   Guarantees termination through convergence detection and loop prevention.
3. Authoritative Provenance:
   Restores exact original EvidenceUnits from original conversation history without fabrication,
   hallucination, or unsupported reconstruction. Original evidence IDs and character spans are preserved.
4. Relational Closure:
   Optionally pulls prerequisite dependencies (DEPENDS_ON), governing conditions (CONDITIONAL_ON),
   rules (EXCEPTION_TO), and opposing conflict poles (CONTRADICTS) into the restoration set
   to prevent severed relational chains.
5. Re-audit Verification:
   Integrates directly with Phase 5D PreservationAuditor. Every recovery iteration re-audits the
   recovered context against counterfactual probes. Recovery is reported as successful only when
   subsequent re-auditing verifies preservation.
6. Risk Model Safety Alignment:
   Synthesizes terminal recovery outcomes with Phase 5C RiskModel. Mandatory evidence loss is
   treated as an inviolable safety failure. If safety invariants cannot be restored within bounds,
   fail-safe bypass is recommended.
7. Structured Diagnostic Telemetry:
   Produces comprehensive telemetry: initially missing IDs, selected IDs, successfully recovered IDs,
   unresolved IDs, recovery records, remaining risk assessment, and re-audit results.
8. Centralized Configuration:
   All operational defaults are sourced from app.core.config.settings and RABCORecoveryPolicy
   with zero scattered hardcoding.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

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
    OmittedEvidence,
    SummaryValidationOutcome,
    estimate_tokens,
)
from .schemas import (
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
from .preservation_auditor import PreservationAuditor
from .risk_model import RiskModel, RiskScoringWeights

log = logging.getLogger(__name__)


# ============================================================================
# Relational Invariants and Prioritization Categories
# ============================================================================

_CRITICAL_RELATION_TYPES: FrozenSet[RelationshipType] = frozenset({
    RelationshipType.DEPENDS_ON,
    RelationshipType.CORRECTS,
    RelationshipType.SUPERSEDES,
    RelationshipType.CONTRADICTS,
    RelationshipType.CONDITIONAL_ON,
    RelationshipType.EXCEPTION_TO,
})

_HIGH_PRIORITY_EVIDENCE_TYPES: FrozenSet[EvidenceType] = frozenset({
    EvidenceType.CORRECTION,
    EvidenceType.NUMERICAL_VALUE,
    EvidenceType.CONSTRAINT,
    EvidenceType.CONDITION,
    EvidenceType.EXCEPTION,
})


# ============================================================================
# Diagnostic & Outcome Models
# ============================================================================

class RecoveryDiagnostics(BaseModel):
    """Detailed diagnostic telemetry documenting selective recovery execution."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    initially_missing_evidence_ids: Tuple[str, ...] = Field(
        ...,
        description="Evidence unit IDs identified as missing, partial, or degraded prior to recovery",
    )
    selected_evidence_ids: Tuple[str, ...] = Field(
        ...,
        description="Evidence unit IDs selected and restored from original source history",
    )
    successfully_recovered_ids: Tuple[str, ...] = Field(
        ...,
        description="Evidence unit IDs whose preservation was independently verified by post-recovery re-audit",
    )
    unresolved_evidence_ids: Tuple[str, ...] = Field(
        ...,
        description="Evidence unit IDs that could not be restored or failed post-recovery re-audit",
    )
    recovery_attempts: Tuple[RecoveryRecord, ...] = Field(
        ...,
        description="Chronological log of recovery iterations executed",
    )
    recovery_status: str = Field(
        ...,
        description="Terminal recovery outcome status (e.g. 'recovered_preserved', 'recovery_exhausted_bypassed', 'no_recovery_needed')",
    )
    remaining_risk: Optional[RiskAssessment] = Field(
        default=None,
        description="Post-recovery multi-indicator risk assessment from Phase 5C RiskModel",
    )
    re_audit_results: Tuple[ProbeAuditResult, ...] = Field(
        default_factory=tuple,
        description="Final re-audit outcomes across counterfactual probes",
    )


class SelectiveRecoveryOutcome(BaseModel):
    """Complete deliverable of selective evidence recovery containing result, updated compression, and diagnostics."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    recovery_result: RecoveryResult = Field(
        ...,
        description="Standard RABCO RecoveryResult contract compatible with RiskModel and Phase 5 pipelines",
    )
    updated_compression_result: CompressionResult = Field(
        ...,
        description="Updated CompressionResult reflecting verbatim restored evidence units and adjusted token footprints",
    )
    diagnostics: RecoveryDiagnostics = Field(
        ...,
        description="Structured recovery diagnostics and re-audit telemetry",
    )

    @property
    def final_selected_evidence(self) -> Tuple[EvidenceUnit, ...]:
        """Convenience property for final selected evidence units."""
        return self.recovery_result.final_selected_evidence

    @property
    def final_validated_summaries(self) -> Tuple[SummaryUnit, ...]:
        """Convenience property for final validated summaries."""
        return self.recovery_result.final_validated_summaries

    @property
    def bypass_required(self) -> bool:
        """Convenience property indicating whether safety invariants mandated original-context bypass."""
        return self.recovery_result.bypass_required

    @property
    def bypass_reason(self) -> Optional[str]:
        """Convenience property explaining bypass rationale if applicable."""
        return self.recovery_result.bypass_reason

    @property
    def total_recovery_attempts(self) -> int:
        """Convenience property for count of executed recovery attempts."""
        return self.recovery_result.total_recovery_attempts

    @property
    def re_audit_results(self) -> Tuple[ProbeAuditResult, ...]:
        """Convenience property for final probe re-audit results."""
        return self.diagnostics.re_audit_results

    @property
    def remaining_risk(self) -> Optional[RiskAssessment]:
        """Convenience property for post-recovery risk assessment."""
        return self.diagnostics.remaining_risk


# ============================================================================
# Recovery Helper Functions
# ============================================================================

def _identify_missing_evidence_ids(
    audit_results: Sequence[ProbeAuditResult],
    evidence_units: Sequence[EvidenceUnit],
    currently_selected_ids: Set[str],
) -> Tuple[str, ...]:
    """Identify unique evidence IDs requiring restoration based on audit findings and mandatory safety.

    Collects:
    1. Explicit missing_evidence_ids from failed probes.
    2. Explicit partially_preserved_ids from degraded probes.
    3. Required evidence IDs from unverifiable/ambiguous probes where specific IDs weren't resolved.
    4. Mandatory evidence units from evidence_units that are missing from currently_selected_ids.
    """
    candidates: Set[str] = set()

    for audit in audit_results:
        if audit.preservation_status != EvidencePreservationStatus.PRESERVED:
            # 1. Direct explicit findings
            if audit.missing_evidence_ids:
                candidates.update(audit.missing_evidence_ids)
            if audit.partially_preserved_ids:
                candidates.update(audit.partially_preserved_ids)

            # 2. If no explicit IDs flagged, target all required IDs not yet extractively preserved
            if not audit.missing_evidence_ids and not audit.partially_preserved_ids:
                for eid in audit.required_evidence_ids:
                    if eid not in currently_selected_ids:
                        candidates.add(eid)

    # 3. Inviolable safety check: mandatory evidence must never be omitted
    for u in evidence_units:
        if u.criticality == EvidenceCriticality.MANDATORY:
            if u.evidence_id not in currently_selected_ids:
                candidates.add(u.evidence_id)

    return tuple(sorted(candidates))


def _prioritize_evidence_units(
    candidate_ids: Sequence[str],
    evidence_by_id: Mapping[str, EvidenceUnit],
    relationships: Sequence[EvidenceRelationship],
    currently_selected_ids: Set[str],
) -> List[EvidenceUnit]:
    """Sort candidate evidence units by criticality, relational vulnerability, and semantic type.

    Deterministic Priority:
    1. Criticality: MANDATORY (0), HIGH (1), MEDIUM (2), LOW (3), UNKNOWN (4).
    2. Relational Vulnerability: Participating in critical relationships where an endpoint is missing (0 vs 1).
    3. Evidence Type: High-priority semantic types (corrections, numbers, constraints) (0 vs 1).
    4. Deterministic tie-breaker: evidence_id string comparison.
    """
    valid_candidates = [
        evidence_by_id[eid]
        for eid in candidate_ids
        if eid in evidence_by_id and eid not in currently_selected_ids
    ]

    # Map relationship endpoints for fast lookup
    rel_endpoints: Set[str] = set()
    for rel in relationships:
        if rel.relationship_type in _CRITICAL_RELATION_TYPES:
            rel_endpoints.add(rel.source_evidence_id)
            rel_endpoints.add(rel.target_evidence_id)

    def sort_key(u: EvidenceUnit) -> Tuple[int, int, int, str]:
        # 1. Criticality tier
        crit_order = {
            EvidenceCriticality.MANDATORY: 0,
            EvidenceCriticality.HIGH: 1,
            EvidenceCriticality.MEDIUM: 2,
            EvidenceCriticality.LOW: 3,
            EvidenceCriticality.UNKNOWN: 4,
        }
        c_rank = crit_order.get(u.criticality, 5)

        # 2. Relational participation
        rel_rank = 0 if u.evidence_id in rel_endpoints else 1

        # 3. Evidence type
        type_rank = 0 if u.evidence_type in _HIGH_PRIORITY_EVIDENCE_TYPES else 1

        # 4. Deterministic tie-breaker
        return (c_rank, rel_rank, type_rank, u.evidence_id)

    return sorted(valid_candidates, key=sort_key)


def _expand_relational_closure(
    seed_units: Sequence[EvidenceUnit],
    evidence_by_id: Mapping[str, EvidenceUnit],
    relationships: Sequence[EvidenceRelationship],
    currently_selected_ids: Set[str],
) -> List[EvidenceUnit]:
    """Expand restoration set by pulling prerequisite dependencies and governed conditions.

    Rules applied:
    - DEPENDS_ON: If dependent unit A is restored, prerequisite B must be restored.
    - CONDITIONAL_ON: If action A is restored, governing condition B must be restored.
    - EXCEPTION_TO: If governed rule B is retained, exception A must be restored.
    - CONTRADICTS / CORRECTS / SUPERSEDES: If one pole is restored, paired pole must be evaluated.
    """
    to_restore: Dict[str, EvidenceUnit] = {u.evidence_id: u for u in seed_units}
    active_ids = set(currently_selected_ids) | set(to_restore.keys())

    # Build relationship adjacency
    for rel in relationships:
        src = rel.source_evidence_id
        tgt = rel.target_evidence_id

        # 1. DEPENDS_ON: src depends on tgt. If src is active, tgt must be present.
        if rel.relationship_type == RelationshipType.DEPENDS_ON:
            if src in active_ids and tgt in evidence_by_id and tgt not in active_ids:
                to_restore[tgt] = evidence_by_id[tgt]
                active_ids.add(tgt)

        # 2. CONDITIONAL_ON: src is action, tgt is condition. If src is active, tgt must be present.
        elif rel.relationship_type == RelationshipType.CONDITIONAL_ON:
            if src in active_ids and tgt in evidence_by_id and tgt not in active_ids:
                to_restore[tgt] = evidence_by_id[tgt]
                active_ids.add(tgt)

        # 3. EXCEPTION_TO: src is exception, tgt is rule. If rule is active, exception must be present.
        elif rel.relationship_type == RelationshipType.EXCEPTION_TO:
            if tgt in active_ids and src in evidence_by_id and src not in active_ids:
                to_restore[src] = evidence_by_id[src]
                active_ids.add(src)

        # 4. CONTRADICTS: both opposing poles must be present to prevent one-sided bias.
        elif rel.relationship_type == RelationshipType.CONTRADICTS:
            if src in active_ids and tgt in evidence_by_id and tgt not in active_ids:
                to_restore[tgt] = evidence_by_id[tgt]
                active_ids.add(tgt)
            elif tgt in active_ids and src in evidence_by_id and src not in active_ids:
                to_restore[src] = evidence_by_id[src]
                active_ids.add(src)

        # 5. CORRECTS / SUPERSEDES: both antecedent and corrected value should survive.
        elif rel.relationship_type in {RelationshipType.CORRECTS, RelationshipType.SUPERSEDES}:
            if src in active_ids and tgt in evidence_by_id and tgt not in active_ids:
                to_restore[tgt] = evidence_by_id[tgt]
                active_ids.add(tgt)
            elif tgt in active_ids and src in evidence_by_id and src not in active_ids:
                to_restore[src] = evidence_by_id[src]
                active_ids.add(src)

    return list(to_restore.values())


def _calculate_restored_relationships(
    existing_retained_ids: Sequence[str],
    relationships: Sequence[EvidenceRelationship],
    preserved_ids: Set[str],
) -> Tuple[str, ...]:
    """Identify relationship IDs that are newly satisfied by restored evidence units."""
    existing_set = set(existing_retained_ids)
    restored: List[str] = []

    for rel in relationships:
        if rel.relationship_id not in existing_set:
            if rel.source_evidence_id in preserved_ids and rel.target_evidence_id in preserved_ids:
                restored.append(rel.relationship_id)

    return tuple(sorted(restored))


# ============================================================================
# Selective Evidence Recovery Engine
# ============================================================================

class SelectiveRecoveryEngine:
    """Bounded, evidence-aware selective recovery engine for RABCO Phase 5E.

    Coordinates targeted evidence restoration, provenance preservation, bounded
    retry loops, Phase 5D re-auditing, and Phase 5C risk assessment.
    """

    def __init__(
        self,
        recovery_policy: Optional[RABCORecoveryPolicy] = None,
        audit_policy: Optional[RABCOAuditPolicy] = None,
        auditor: Optional[PreservationAuditor] = None,
        risk_model: Optional[RiskModel] = None,
        config=settings,
    ) -> None:
        """Initialize SelectiveRecoveryEngine with centralized policies and components.

        Args:
            recovery_policy: Optional RABCORecoveryPolicy defining attempt limits and closure rules.
            audit_policy: Optional RABCOAuditPolicy defining re-audit failure tolerances.
            auditor: Optional PreservationAuditor instance for re-auditing.
            risk_model: Optional RiskModel instance for evaluating remaining post-recovery risk.
            config: Application configuration instance.
        """
        self.recovery_policy = recovery_policy or RABCORecoveryPolicy()
        self.audit_policy = audit_policy or RABCOAuditPolicy()
        self.auditor = auditor or PreservationAuditor(policy=self.audit_policy, config=config)
        self.risk_model = risk_model or RiskModel(config=config)
        self.config = config

    def recover(
        self,
        evidence_units: Sequence[EvidenceUnit],
        compression_result: Optional[CompressionResult] = None,
        audit_results: Optional[Sequence[ProbeAuditResult]] = None,
        probes: Sequence[CounterfactualProbe] = (),
        relationships: Sequence[EvidenceRelationship] = (),
        recovery_policy: Optional[RABCORecoveryPolicy] = None,
        audit_policy: Optional[RABCOAuditPolicy] = None,
        chars_per_token: Optional[float] = None,
    ) -> SelectiveRecoveryOutcome:
        """Execute bounded selective recovery across audited conversation context.

        Parameters
        ----------
        evidence_units:
            All extracted evidence units from original conversation (authoritative source).
        compression_result:
            Optional current compression output from HybridCompressor. If None, initialized empty.
        audit_results:
            Optional initial Phase 5D probe audit results. If None, computed from probes.
        probes:
            Counterfactual candidate probes used for auditing and re-auditing.
        relationships:
            Relational edges between evidence units (DEPENDS_ON, CORRECTS, etc.).
        recovery_policy:
            Optional override for recovery parameters.
        audit_policy:
            Optional override for audit parameters.
        chars_per_token:
            Optional heuristic character ratio for token estimation.

        Returns
        -------
        SelectiveRecoveryOutcome
            Complete outcome deliverable containing standard RecoveryResult, updated CompressionResult,
            and structured RecoveryDiagnostics.
        """
        eff_rec_policy = recovery_policy or self.recovery_policy
        eff_audit_policy = audit_policy or self.audit_policy
        eff_chars_per_token = (
            chars_per_token
            if chars_per_token is not None and chars_per_token > 0.0
            else eff_audit_policy.chars_per_token_heuristic
        )

        # Step 1: Index authoritative source evidence
        evidence_by_id: Dict[str, EvidenceUnit] = {u.evidence_id: u for u in evidence_units}

        # Step 2: Establish baseline compression result defensively
        if compression_result is None:
            total_orig_tokens = sum(
                estimate_tokens(u.original_text, eff_chars_per_token) for u in evidence_units
            )
            current_compression = CompressionResult(
                selected_evidence=(),
                validated_summaries=(),
                omitted_evidence=tuple(
                    OmittedEvidence(
                        evidence_id=u.evidence_id,
                        reason="initial_omission",
                        source_message_id=u.source_span.message_id,
                        criticality=u.criticality,
                        evidence_type=u.evidence_type,
                        original_text=u.original_text,
                    )
                    for u in evidence_units
                ),
                validation_outcomes=(),
                original_token_estimate=total_orig_tokens,
                compressed_token_estimate=0,
                token_reduction_percent=100.0,
                token_estimation_method="heuristic",
            )
        else:
            current_compression = compression_result

        # Step 3: Establish initial audit findings
        if audit_results is None:
            if probes:
                current_audits = self.auditor.audit_probes(
                    probes=probes,
                    evidence_units=evidence_units,
                    relationships=relationships,
                    compression_result=current_compression,
                    policy=eff_audit_policy,
                )
            else:
                current_audits = ()
        else:
            current_audits = tuple(audit_results)

        # Step 4: Identify initially missing evidence units
        extractive_ids: Set[str] = {u.evidence_id for u in current_compression.selected_evidence}
        initially_missing_ids: Tuple[str, ...] = _identify_missing_evidence_ids(
            audit_results=current_audits,
            evidence_units=evidence_units,
            currently_selected_ids=extractive_ids,
        )

        # Fast path: If no evidence is missing and all safety invariants hold, no recovery needed
        if not initially_missing_ids:
            # Verify remaining risk
            baseline_risk = self.risk_model.assess(
                audit_results=current_audits,
                evidence_units=evidence_units,
                relationships=relationships,
                compression_result=current_compression,
                audit_policy=eff_audit_policy,
                probes=probes,
            )
            if not baseline_risk.bypass_recommended:
                rec_result = RecoveryResult(
                    final_selected_evidence=current_compression.selected_evidence,
                    final_validated_summaries=current_compression.validated_summaries,
                    recovery_attempts=(),
                    total_recovery_attempts=0,
                    recovery_limit_reached=False,
                    final_preservation_status="no_recovery_needed",
                    bypass_required=False,
                    bypass_reason=None,
                )
                diagnostics = RecoveryDiagnostics(
                    initially_missing_evidence_ids=(),
                    selected_evidence_ids=(),
                    successfully_recovered_ids=(),
                    unresolved_evidence_ids=(),
                    recovery_attempts=(),
                    recovery_status="no_recovery_needed",
                    remaining_risk=baseline_risk,
                    re_audit_results=current_audits,
                )
                return SelectiveRecoveryOutcome(
                    recovery_result=rec_result,
                    updated_compression_result=current_compression,
                    diagnostics=diagnostics,
                )

        # Step 5: Bounded Recovery Loop
        max_attempts = eff_rec_policy.max_recovery_attempts
        recovery_records: List[RecoveryRecord] = []
        all_restored_ids: Set[str] = set()
        attempt = 0

        while attempt < max_attempts:
            attempt += 1

            # Currently preserved IDs
            currently_selected = {u.evidence_id for u in current_compression.selected_evidence}

            # Find candidates requiring recovery from current audit results
            target_ids = _identify_missing_evidence_ids(
                audit_results=current_audits,
                evidence_units=evidence_units,
                currently_selected_ids=currently_selected,
            )

            # Filter candidates that are already selected or not in authoritative evidence pool
            new_candidate_ids = [
                eid for eid in target_ids
                if eid in evidence_by_id and eid not in currently_selected
            ]

            if not new_candidate_ids:
                # No further actionable recovery can be made from authoritative pool
                log.info(
                    "SelectiveRecovery: No new recoverable evidence identified at attempt %d; breaking loop.",
                    attempt,
                )
                break

            # Prioritize candidates deterministically
            prioritized_units = _prioritize_evidence_units(
                candidate_ids=new_candidate_ids,
                evidence_by_id=evidence_by_id,
                relationships=relationships,
                currently_selected_ids=currently_selected,
            )

            # Apply relational closure if enabled
            if eff_rec_policy.enable_relational_closure:
                units_to_restore = _expand_relational_closure(
                    seed_units=prioritized_units,
                    evidence_by_id=evidence_by_id,
                    relationships=relationships,
                    currently_selected_ids=currently_selected,
                )
            else:
                units_to_restore = prioritized_units

            # Re-filter to ensure strict deduplication and exclude already selected
            deduped_units_to_restore: List[EvidenceUnit] = []
            seen_in_attempt: Set[str] = set()
            for u in units_to_restore:
                if u.evidence_id not in currently_selected and u.evidence_id not in seen_in_attempt:
                    deduped_units_to_restore.append(u)
                    seen_in_attempt.add(u.evidence_id)

            if not deduped_units_to_restore:
                log.info(
                    "SelectiveRecovery: All candidates already selected at attempt %d; breaking loop.",
                    attempt,
                )
                break

            # Compute footprints and restored relationships
            token_before = current_compression.compressed_token_estimate
            new_tokens = sum(
                estimate_tokens(u.original_text, eff_chars_per_token)
                for u in deduped_units_to_restore
            )
            token_after = token_before + new_tokens

            restored_eids = tuple(u.evidence_id for u in deduped_units_to_restore)
            all_preserved_ids = currently_selected | set(restored_eids)
            for s in current_compression.validated_summaries:
                all_preserved_ids.update(s.source_evidence_ids)

            restored_rel_ids = _calculate_restored_relationships(
                existing_retained_ids=current_compression.retained_relationship_ids,
                relationships=relationships,
                preserved_ids=all_preserved_ids,
            )

            # Build RecoveryRecord for this iteration
            record = RecoveryRecord(
                attempt_number=attempt,
                restored_evidence_ids=restored_eids,
                restored_relationship_ids=restored_rel_ids,
                missing_evidence_ids_targeted=tuple(target_ids),
                token_estimate_before=token_before,
                token_estimate_after=token_after,
                restoration_reason=(
                    f"Selective restoration cycle {attempt}: restored {len(deduped_units_to_restore)} "
                    f"authoritative units ({list(restored_eids)}) to resolve failed probes."
                ),
                timestamp=datetime.now(timezone.utc),
            )
            recovery_records.append(record)
            all_restored_ids.update(restored_eids)

            # Update CompressionResult state
            updated_selected = list(current_compression.selected_evidence) + deduped_units_to_restore
            updated_omitted = [
                o for o in current_compression.omitted_evidence
                if o.evidence_id not in restored_eids
            ]
            all_retained_rels = tuple(
                sorted(set(current_compression.retained_relationship_ids) | set(restored_rel_ids))
            )

            orig_tokens = current_compression.original_token_estimate
            token_reduction = (
                round(100.0 * (1.0 - (token_after / orig_tokens)), 2)
                if orig_tokens > 0
                else 0.0
            )

            # Evaluate budget status
            budget_status = current_compression.budget_status
            budget_details = current_compression.budget_conflict_details
            if orig_tokens > 0 and token_after > orig_tokens:
                budget_status = "budget_conflict"
                budget_details = (
                    f"Restored context token estimate ({token_after}) exceeds original token footprint ({orig_tokens})."
                )

            current_compression = CompressionResult(
                selected_evidence=tuple(updated_selected),
                validated_summaries=current_compression.validated_summaries,
                omitted_evidence=tuple(updated_omitted),
                validation_outcomes=current_compression.validation_outcomes,
                original_token_estimate=orig_tokens,
                compressed_token_estimate=token_after,
                token_reduction_percent=token_reduction,
                token_estimation_method=current_compression.token_estimation_method,
                retained_relationship_ids=all_retained_rels,
                budget_status=budget_status,
                budget_conflict_details=budget_details,
                safety_invariants_enforced=True,
                safety_warnings=current_compression.safety_warnings,
                processing_time_ms=current_compression.processing_time_ms,
            )

            # Step 5D Re-Auditing
            if probes:
                current_audits = self.auditor.audit_probes(
                    probes=probes,
                    evidence_units=evidence_units,
                    relationships=relationships,
                    compression_result=current_compression,
                    policy=eff_audit_policy,
                )

            # Convergence Check: assess current state against safety invariants
            interim_risk = self.risk_model.assess(
                audit_results=current_audits,
                evidence_units=evidence_units,
                relationships=relationships,
                compression_result=current_compression,
                audit_policy=eff_audit_policy,
                probes=probes,
            )

            if not interim_risk.bypass_recommended:
                log.info(
                    "SelectiveRecovery: Preservation successfully verified at attempt %d; exiting loop.",
                    attempt,
                )
                break

        # Step 6: Terminal Risk Evaluation and Deliverables Assembly
        terminal_risk = self.risk_model.assess(
            audit_results=current_audits,
            evidence_units=evidence_units,
            relationships=relationships,
            compression_result=current_compression,
            audit_policy=eff_audit_policy,
            probes=probes,
        )

        # Determine terminal status and bypass requirements
        total_attempts = len(recovery_records)
        limit_reached = total_attempts >= max_attempts and terminal_risk.bypass_recommended

        if not terminal_risk.bypass_recommended:
            bypass_required = False
            bypass_reason = None
            final_status = "recovered_preserved" if total_attempts > 0 else "audit_passed"
        else:
            if eff_rec_policy.bypass_on_failure:
                bypass_required = True
                bypass_reason = (
                    f"Selective recovery exhausted ({total_attempts} attempts) without satisfying "
                    f"safety invariants: {terminal_risk.bypass_reason or 'unresolved preservation risk'}"
                )
                final_status = "recovery_exhausted_bypassed"
            else:
                bypass_required = False
                bypass_reason = terminal_risk.bypass_reason
                final_status = "recovery_exhausted"

        # Determine successfully recovered vs unresolved evidence IDs
        final_extractive = {u.evidence_id for u in current_compression.selected_evidence}

        # An evidence unit is successfully recovered if it was restored and is not flagged missing/partial
        failed_audit_eids: Set[str] = set()
        for a in current_audits:
            if a.preservation_status != EvidencePreservationStatus.PRESERVED:
                failed_audit_eids.update(a.missing_evidence_ids)
                failed_audit_eids.update(a.partially_preserved_ids)
                if not a.missing_evidence_ids and not a.partially_preserved_ids:
                    failed_audit_eids.update(a.required_evidence_ids)

        successfully_recovered_ids = tuple(sorted(
            eid for eid in all_restored_ids
            if eid in final_extractive and eid not in failed_audit_eids
        ))

        unresolved_ids = tuple(sorted(
            eid for eid in initially_missing_ids
            if eid not in successfully_recovered_ids
        ))

        # Build final standard RecoveryResult
        final_recovery_result = RecoveryResult(
            final_selected_evidence=current_compression.selected_evidence,
            final_validated_summaries=current_compression.validated_summaries,
            recovery_attempts=tuple(recovery_records),
            total_recovery_attempts=total_attempts,
            recovery_limit_reached=limit_reached,
            final_preservation_status=final_status,
            bypass_required=bypass_required,
            bypass_reason=bypass_reason,
        )

        # Build detailed diagnostics
        diagnostics = RecoveryDiagnostics(
            initially_missing_evidence_ids=initially_missing_ids,
            selected_evidence_ids=tuple(sorted(all_restored_ids)),
            successfully_recovered_ids=successfully_recovered_ids,
            unresolved_evidence_ids=unresolved_ids,
            recovery_attempts=tuple(recovery_records),
            recovery_status=final_status,
            remaining_risk=terminal_risk,
            re_audit_results=current_audits,
        )

        return SelectiveRecoveryOutcome(
            recovery_result=final_recovery_result,
            updated_compression_result=current_compression,
            diagnostics=diagnostics,
        )
