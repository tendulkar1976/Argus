"""RABCO Phase 5F: End-to-End Orchestration Layer.

Project: Argus — Risk-Aware, Counterfactual-Bounded Context Optimization (RABCO).
Phase: 5F — End-to-End Orchestration.

Purpose:
    Provide a unified, production-quality orchestration pipeline connecting all RABCO
    phases (Phase 2 extraction, Phase 3 relationships, Phase 4 compression, Phase 5B probe
    generation, Phase 5C risk assessment, Phase 5D preservation auditing, and Phase 5E
    selective recovery) into an integrated, deterministic, fail-safe context optimization pipeline.

Architectural Guarantees:
1. End-to-End Coordination:
   Coordinates Phase 2 through Phase 5E sequentially without duplicating component logic:
   messages -> evidence -> relationships -> compression -> probes -> audit -> risk -> recovery -> decision.
2. Distinct Decision Taxonomy:
   Emits clearly distinguishable terminal decisions:
   - SAFE_TO_CONTINUE: Initial compression verified as safe and preserved.
   - RECOVERY_COMPLETED: Initial context required restoration; recovery successfully restored safety.
   - RECOVERY_UNSUCCESSFUL: Recovery exhausted attempts without satisfying safety requirements.
   - BYPASS_REQUIRED: Core safety invariants violated; reverting to original context.
   - EXECUTION_FAILED: Stage-level runtime exception encountered; fail-safe bypass engaged.
3. Inviolable Mandatory Safety:
   Mandatory evidence must never be silently discarded. If any mandatory evidence is lost,
   partially preserved, ambiguous, or unverifiable, the pipeline unconditionally mandates bypass.
4. Fail-Safe Bypass Guarantee:
   Whenever bypass is triggered, the final delivered context is guaranteed to be the immutable
   verbatim original conversation messages (original_messages_ref).
5. Granular Stage Isolation:
   Tracks execution milestones through PipelineStage and logs stage-level diagnostics,
   pinpointing the exact stage if a failure or exception occurs.
6. Strict Contract Compatibility:
   Produces both a specialized RABCOOrchestrationResult and standard schemas.RABCOPhase5Result.
7. Dependency Injection & Testability:
   Accepts optional pre-configured component instances and policy overrides for clean isolation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.schemas.rabco import (
    ConversationMessage,
    EvidenceCriticality,
    EvidenceRelationship,
    EvidenceType,
    EvidenceUnit,
    MessageRole,
    RelationshipType,
    SummaryUnit,
    ValidationStatus,
)
from .evidence_extractor import EvidenceExtractor
from .relationship_mapper import RelationshipMapper
from .hybrid_compressor import (
    CompressionPolicy,
    CompressionResult,
    HybridCompressor,
    OmittedEvidence,
    estimate_tokens,
)
from .counterfactual_generator import CounterfactualProbeGenerator
from .preservation_auditor import PreservationAuditor
from .risk_model import RiskModel, RiskScoringWeights
from .selective_recovery import (
    RecoveryDiagnostics,
    SelectiveRecoveryEngine,
    SelectiveRecoveryOutcome,
)
from .schemas import (
    CounterfactualProbe,
    EvidencePreservationStatus,
    ProbeAuditResult,
    ProbeCategory,
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

log = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class OrchestrationDecision(str, Enum):
    """Final high-level safety decision produced by the RABCO orchestrator."""

    SAFE_TO_CONTINUE = "safe_to_continue"
    RECOVERY_COMPLETED = "recovery_completed"
    RECOVERY_UNSUCCESSFUL = "recovery_unsuccessful"
    BYPASS_REQUIRED = "bypass_required"
    EXECUTION_FAILED = "execution_failed"


class PipelineStage(str, Enum):
    """Pipeline stage taxonomy for execution tracking and failure isolation."""

    INITIALIZATION = "initialization"
    EXTRACTION = "extraction"
    RELATIONSHIP_MAPPING = "relationship_mapping"
    COMPRESSION = "compression"
    PROBE_GENERATION = "probe_generation"
    PRESERVATION_AUDITING = "preservation_auditing"
    INITIAL_RISK_ASSESSMENT = "initial_risk_assessment"
    SELECTIVE_RECOVERY = "selective_recovery"
    TERMINAL_RISK_ASSESSMENT = "terminal_risk_assessment"
    DECISION_AND_ASSEMBLY = "decision_and_assembly"
    COMPLETED = "completed"


# ============================================================================
# Exceptions
# ============================================================================

class RABCOPipelineError(Exception):
    """Raised when an unrecoverable error occurs in a specific RABCO pipeline stage."""

    def __init__(self, stage: PipelineStage, message: str, cause: Optional[Exception] = None) -> None:
        super().__init__(f"[{stage.value}] {message}")
        self.stage = stage
        self.message = message
        self.cause = cause


# ============================================================================
# Context Rendering Helper
# ============================================================================

def _render_optimized_context_messages(
    compression_result: CompressionResult,
    original_messages: Sequence[ConversationMessage],
) -> Tuple[ConversationMessage, ...]:
    """Render optimized context into structured ConversationMessage turns.

    Produces:
    - A system message summarizing verified context (summaries and verbatim evidence).
    - Preserves trailing user/assistant turns if relevant to active query grounding.
    """
    sections: List[str] = []

    if compression_result.validated_summaries:
        summary_lines = [f"- {s.summary_text}" for s in compression_result.validated_summaries]
        sections.append("Context Summaries:\n" + "\n".join(summary_lines))

    if compression_result.selected_evidence:
        extractive_lines = [f"- {u.original_text}" for u in compression_result.selected_evidence]
        sections.append("Verified Retained Facts:\n" + "\n".join(extractive_lines))

    rendered_content = "\n\n".join(sections) if sections else "No retained context."

    optimized_system_msg = ConversationMessage(
        message_id="msg_optimized_context",
        role=MessageRole.SYSTEM,
        content=rendered_content,
        sequence_index=0,
        timestamp=datetime.now(timezone.utc),
    )

    # If the original conversation ended with an active user query, append it
    if original_messages and original_messages[-1].role == MessageRole.USER:
        last_user_msg = original_messages[-1]
        active_query_msg = ConversationMessage(
            message_id=f"msg_active_{last_user_msg.message_id}",
            role=MessageRole.USER,
            content=last_user_msg.content,
            sequence_index=1,
            timestamp=datetime.now(timezone.utc),
        )
        return (optimized_system_msg, active_query_msg)

    return (optimized_system_msg,)


# ============================================================================
# Deliverable Model
# ============================================================================

class RABCOOrchestrationResult(BaseModel):
    """Complete deliverable of Phase 5F end-to-end orchestration."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    decision: OrchestrationDecision = Field(
        ...,
        description="High-level safety decision: safe_to_continue, recovery_completed, recovery_unsuccessful, bypass_required, execution_failed",
    )
    phase5_result: RABCOPhase5Result = Field(
        ...,
        description="Authoritative Phase 5 contract deliverable conforming to schemas.RABCOPhase5Result",
    )
    failed_stage: Optional[PipelineStage] = Field(
        default=None,
        description="Pipeline stage where a failure occurred, if any",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Detailed diagnostic error message if execution failed",
    )
    execution_trace: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Chronological log of executed pipeline stages and milestones",
    )
    evidence_units: Tuple[EvidenceUnit, ...] = Field(
        default_factory=tuple,
        description="All extracted or provided evidence units",
    )
    relationships: Tuple[EvidenceRelationship, ...] = Field(
        default_factory=tuple,
        description="All mapped or provided evidence relationships",
    )
    compression_result: Optional[CompressionResult] = Field(
        default=None,
        description="Compression outcome before recovery",
    )
    recovery_outcome: Optional[SelectiveRecoveryOutcome] = Field(
        default=None,
        description="Selective recovery outcome and diagnostics, if recovery was invoked",
    )

    @property
    def final_context_messages(self) -> Tuple[ConversationMessage, ...]:
        """Convenience property for downstream context messages."""
        return self.phase5_result.final_context_messages

    @property
    def bypassed(self) -> bool:
        """Convenience property indicating whether original-context bypass was engaged."""
        return self.phase5_result.bypassed

    @property
    def bypass_reason(self) -> Optional[str]:
        """Convenience property for bypass rationale."""
        return self.phase5_result.bypass_reason

    @property
    def risk_assessment(self) -> RiskAssessment:
        """Convenience property for terminal risk assessment."""
        return self.phase5_result.risk_assessment

    @property
    def probes(self) -> Tuple[CounterfactualProbe, ...]:
        """Convenience property for generated audit probes."""
        return self.phase5_result.probes

    @property
    def audit_results(self) -> Tuple[ProbeAuditResult, ...]:
        """Convenience property for final probe audit results."""
        return self.phase5_result.audit_results

    def to_phase5_result(self) -> RABCOPhase5Result:
        """Extract authoritative standard Phase 5 result contract."""
        return self.phase5_result


# ============================================================================
# Orchestrator Core Implementation
# ============================================================================

class RABCOOrchestrator:
    """Central orchestration layer for the RABCO research engine (Phase 5F).

    Coordinates:
    - Phase 2: EvidenceExtractor
    - Phase 3: RelationshipMapper
    - Phase 4: HybridCompressor
    - Phase 5B: CounterfactualProbeGenerator
    - Phase 5C: RiskModel
    - Phase 5D: PreservationAuditor
    - Phase 5E: SelectiveRecoveryEngine
    - Phase 5F: Safety Decision, Invariant Verification, and Result Assembly
    """

    def __init__(
        self,
        extractor: Optional[EvidenceExtractor] = None,
        mapper: Optional[RelationshipMapper] = None,
        compressor: Optional[HybridCompressor] = None,
        probe_generator: Optional[CounterfactualProbeGenerator] = None,
        auditor: Optional[PreservationAuditor] = None,
        risk_model: Optional[RiskModel] = None,
        recovery_engine: Optional[SelectiveRecoveryEngine] = None,
        compression_policy: Optional[CompressionPolicy] = None,
        audit_policy: Optional[RABCOAuditPolicy] = None,
        recovery_policy: Optional[RABCORecoveryPolicy] = None,
        config=settings,
    ) -> None:
        """Initialize orchestrator with components, policies, and centralized configuration."""
        self.compression_policy = compression_policy or CompressionPolicy()
        self.audit_policy = audit_policy or RABCOAuditPolicy()
        self.recovery_policy = recovery_policy or RABCORecoveryPolicy()
        self.config = config

        # Component dependency injection with centralized fallbacks
        self.extractor = extractor or EvidenceExtractor()
        self.mapper = mapper or RelationshipMapper()
        self.compressor = compressor or HybridCompressor(policy=self.compression_policy)
        self.probe_generator = probe_generator or CounterfactualProbeGenerator(config=config)
        self.auditor = auditor or PreservationAuditor(policy=self.audit_policy, config=config)
        self.risk_model = risk_model or RiskModel(config=config)
        self.recovery_engine = recovery_engine or SelectiveRecoveryEngine(
            recovery_policy=self.recovery_policy,
            audit_policy=self.audit_policy,
            auditor=self.auditor,
            risk_model=self.risk_model,
            config=config,
        )

    def orchestrate(
        self,
        messages: Sequence[ConversationMessage],
        user_query: Optional[str] = None,
        evidence_units: Optional[Sequence[EvidenceUnit]] = None,
        relationships: Optional[Sequence[EvidenceRelationship]] = None,
        compression_result: Optional[CompressionResult] = None,
        probes: Optional[Sequence[CounterfactualProbe]] = None,
        compression_policy: Optional[CompressionPolicy] = None,
        audit_policy: Optional[RABCOAuditPolicy] = None,
        recovery_policy: Optional[RABCORecoveryPolicy] = None,
        raise_on_failure: bool = False,
    ) -> RABCOOrchestrationResult:
        """Execute complete end-to-end RABCO context optimization pipeline.

        Parameters
        ----------
        messages:
            Original immutable conversation history turns.
        user_query:
            Active user query prompting optimization. Defaults to last message content or fallback.
        evidence_units:
            Optional pre-extracted evidence units. If None, extracted using Phase 2 EvidenceExtractor.
        relationships:
            Optional pre-mapped relationships. If None, mapped using Phase 3 RelationshipMapper.
        compression_result:
            Optional pre-computed compression result. If None, computed using Phase 4 HybridCompressor.
        probes:
            Optional pre-generated probes. If None, generated using Phase 5B CounterfactualProbeGenerator.
        compression_policy:
            Optional override for compression parameters.
        audit_policy:
            Optional override for audit parameters.
        recovery_policy:
            Optional override for recovery parameters.
        raise_on_failure:
            Whether to re-raise stage exceptions or return a fail-safe EXECUTION_FAILED result.

        Returns
        -------
        RABCOOrchestrationResult
            Complete, typed orchestration deliverable with decision, Phase 5 contract, and diagnostics.
        """
        start_time = time.perf_counter()
        trace: List[str] = []
        current_stage = PipelineStage.INITIALIZATION
        trace.append(f"Stage: {current_stage.value}")

        eff_comp_policy = compression_policy or self.compression_policy
        eff_audit_policy = audit_policy or self.audit_policy
        eff_rec_policy = recovery_policy or self.recovery_policy

        # Resolve query
        effective_query = user_query
        if not effective_query:
            for m in reversed(messages):
                if m.role == MessageRole.USER:
                    effective_query = m.content
                    break
            if not effective_query:
                effective_query = "Preserve critical conversation facts and constraints."

        # Defensive fallback for empty conversation
        orig_messages_tuple = tuple(messages)
        if not orig_messages_tuple:
            trace.append("Empty conversation input; exiting safely.")
            total_time_ms = round((time.perf_counter() - start_time) * 1000)
            empty_risk = self.risk_model.assess(
                audit_results=(),
                evidence_units=(),
                relationships=(),
                compression_result=None,
                audit_policy=eff_audit_policy,
            )
            p5_res = RABCOPhase5Result(
                final_context_messages=(),
                original_messages_ref=(),
                probes=(),
                audit_results=(),
                missing_evidence_ids=(),
                restored_evidence_ids=(),
                recovery_attempt_count=0,
                recovery_records=(),
                final_preservation_status="empty_context_safe",
                budget_status="feasible",
                budget_conflict_details=None,
                risk_assessment=empty_risk,
                original_token_estimate=0,
                optimized_token_estimate=0,
                token_reduction_percent=0.0,
                token_estimation_method="heuristic",
                bypassed=False,
                bypass_reason=None,
                auditor_configuration=self._build_auditor_config(eff_audit_policy, eff_rec_policy),
                processing_time_ms=total_time_ms,
            )
            return RABCOOrchestrationResult(
                decision=OrchestrationDecision.SAFE_TO_CONTINUE,
                phase5_result=p5_res,
                failed_stage=None,
                error_message=None,
                execution_trace=tuple(trace),
                evidence_units=(),
                relationships=(),
                compression_result=None,
                recovery_outcome=None,
            )

        heuristic = eff_comp_policy.chars_per_token_heuristic
        original_tokens = sum(estimate_tokens(m.content, heuristic) for m in orig_messages_tuple)

        # --------------------------------------------------------------------
        # Stage-by-Stage Pipeline Execution with Error Isolation
        # --------------------------------------------------------------------
        try:
            # Stage 1: Evidence Extraction
            current_stage = PipelineStage.EXTRACTION
            trace.append(f"Stage: {current_stage.value}")
            if evidence_units is not None:
                all_evidence = tuple(evidence_units)
                trace.append(f"Extraction bypassed: using {len(all_evidence)} pre-supplied units.")
            else:
                all_evidence = tuple(self.extractor.extract(orig_messages_tuple))
                trace.append(f"Extracted {len(all_evidence)} evidence units from messages.")

            # Stage 2: Relationship Mapping
            current_stage = PipelineStage.RELATIONSHIP_MAPPING
            trace.append(f"Stage: {current_stage.value}")
            if relationships is not None:
                all_relationships = tuple(relationships)
                trace.append(f"Relationship mapping bypassed: using {len(all_relationships)} pre-supplied edges.")
            else:
                all_relationships = tuple(
                    self.mapper.map_relationships(all_evidence, orig_messages_tuple)
                )
                trace.append(f"Mapped {len(all_relationships)} relationship edges.")

            # Stage 3: Context Compression
            current_stage = PipelineStage.COMPRESSION
            trace.append(f"Stage: {current_stage.value}")
            if compression_result is not None:
                comp_res = compression_result
                trace.append("Compression bypassed: using pre-supplied CompressionResult.")
            else:
                comp_res = self.compressor.compress(
                    evidence_units=all_evidence,
                    relationships=all_relationships,
                    messages=orig_messages_tuple,
                    current_query=effective_query,
                    policy=eff_comp_policy,
                )
                trace.append(
                    f"Compressed context: {len(comp_res.selected_evidence)} extractive, "
                    f"{len(comp_res.validated_summaries)} summaries, {len(comp_res.omitted_evidence)} omitted."
                )

            # Stage 4: Probe Generation
            current_stage = PipelineStage.PROBE_GENERATION
            trace.append(f"Stage: {current_stage.value}")
            if probes is not None:
                audit_probes = tuple(probes)
                trace.append(f"Probe generation bypassed: using {len(audit_probes)} pre-supplied probes.")
            else:
                gen_result = self.probe_generator.generate_probes(
                    original_messages=orig_messages_tuple,
                    evidence_units=all_evidence,
                    relationships=all_relationships,
                    user_query=effective_query,
                    probe_set=ProbeSetType.AUDIT,
                )
                audit_probes = gen_result.probes
                trace.append(f"Generated {len(audit_probes)} AUDIT probes.")

            # Stage 5: Preservation Auditing (Phase 5D)
            current_stage = PipelineStage.PRESERVATION_AUDITING
            trace.append(f"Stage: {current_stage.value}")
            initial_audits = self.auditor.audit_probes(
                probes=audit_probes,
                evidence_units=all_evidence,
                relationships=all_relationships,
                compression_result=comp_res,
                policy=eff_audit_policy,
            )
            trace.append(f"Preservation audit complete across {len(initial_audits)} probes.")

            # Stage 6: Initial Risk Assessment (Phase 5C)
            current_stage = PipelineStage.INITIAL_RISK_ASSESSMENT
            trace.append(f"Stage: {current_stage.value}")
            initial_risk = self.risk_model.assess(
                audit_results=initial_audits,
                evidence_units=all_evidence,
                relationships=all_relationships,
                compression_result=comp_res,
                audit_policy=eff_audit_policy,
                probes=audit_probes,
            )
            trace.append(
                f"Initial risk assessment: bypass_recommended={initial_risk.bypass_recommended}, "
                f"mandatory_loss={initial_risk.mandatory_evidence_loss}, "
                f"probe_failure_rate={initial_risk.probe_failure_rate:.3f}."
            )

            # Stage 7: Selective Recovery (Phase 5E), if required
            recovery_outcome: Optional[SelectiveRecoveryOutcome] = None
            active_compression = comp_res
            active_audits = initial_audits
            active_risk = initial_risk

            needs_recovery = (
                initial_risk.bypass_recommended
                or initial_risk.mandatory_evidence_loss > 0
                or initial_risk.probe_failure_rate > 0.0
                or any(a.preservation_status != EvidencePreservationStatus.PRESERVED for a in initial_audits)
            )

            if needs_recovery:
                current_stage = PipelineStage.SELECTIVE_RECOVERY
                trace.append(f"Stage: {current_stage.value}")
                trace.append("Initial context degraded or unsafe; invoking selective recovery.")

                recovery_outcome = self.recovery_engine.recover(
                    evidence_units=all_evidence,
                    compression_result=comp_res,
                    audit_results=initial_audits,
                    probes=audit_probes,
                    relationships=all_relationships,
                    recovery_policy=eff_rec_policy,
                    audit_policy=eff_audit_policy,
                    chars_per_token=heuristic,
                )

                active_compression = recovery_outcome.updated_compression_result
                active_audits = recovery_outcome.re_audit_results
                trace.append(
                    f"Selective recovery completed with status '{recovery_outcome.recovery_result.final_preservation_status}' "
                    f"after {recovery_outcome.total_recovery_attempts} attempt(s)."
                )

                # Stage 8: Terminal Risk Assessment (Phase 5C)
                current_stage = PipelineStage.TERMINAL_RISK_ASSESSMENT
                trace.append(f"Stage: {current_stage.value}")
                active_risk = self.risk_model.assess(
                    audit_results=active_audits,
                    evidence_units=all_evidence,
                    relationships=all_relationships,
                    compression_result=active_compression,
                    audit_policy=eff_audit_policy,
                    recovery_result=recovery_outcome.recovery_result,
                    probes=audit_probes,
                )
                trace.append(
                    f"Terminal risk assessment: bypass_recommended={active_risk.bypass_recommended}, "
                    f"mandatory_loss={active_risk.mandatory_evidence_loss}, "
                    f"probe_failure_rate={active_risk.probe_failure_rate:.3f}."
                )

            # Stage 9: Final Decision and Contract Assembly
            current_stage = PipelineStage.DECISION_AND_ASSEMBLY
            trace.append(f"Stage: {current_stage.value}")

            decision, bypassed, bypass_reason, final_status = self._evaluate_final_safety_decision(
                initial_risk=initial_risk,
                active_risk=active_risk,
                recovery_outcome=recovery_outcome,
                all_evidence=all_evidence,
                active_compression=active_compression,
            )
            trace.append(f"Final safety decision: {decision.value} (bypassed={bypassed}).")

            # Final context messages: revert to original immutable history on bypass
            if bypassed:
                final_messages = orig_messages_tuple
                optimized_tokens = original_tokens
                token_reduction = 0.0
            else:
                final_messages = _render_optimized_context_messages(active_compression, orig_messages_tuple)
                optimized_tokens = sum(estimate_tokens(m.content, heuristic) for m in final_messages)
                token_reduction = (
                    round(((original_tokens - optimized_tokens) / original_tokens) * 100.0, 2)
                    if original_tokens > 0
                    else 0.0
                )

            # Metadata and recovery attributes
            missing_eids = (
                recovery_outcome.diagnostics.unresolved_evidence_ids
                if recovery_outcome is not None
                else tuple(sorted({eid for a in active_audits for eid in a.missing_evidence_ids}))
            )
            restored_eids = (
                recovery_outcome.diagnostics.selected_evidence_ids
                if recovery_outcome is not None
                else ()
            )
            recovery_records = (
                recovery_outcome.recovery_result.recovery_attempts
                if recovery_outcome is not None
                else ()
            )
            recovery_attempts_count = len(recovery_records)

            total_duration_ms = round((time.perf_counter() - start_time) * 1000)
            auditor_cfg = self._build_auditor_config(eff_audit_policy, eff_rec_policy)

            phase5_result = RABCOPhase5Result(
                final_context_messages=final_messages,
                original_messages_ref=orig_messages_tuple,
                probes=audit_probes,
                audit_results=active_audits,
                missing_evidence_ids=missing_eids,
                restored_evidence_ids=restored_eids,
                recovery_attempt_count=recovery_attempts_count,
                recovery_records=recovery_records,
                final_preservation_status=final_status,
                budget_status=active_compression.budget_status,
                budget_conflict_details=active_compression.budget_conflict_details,
                risk_assessment=active_risk,
                original_token_estimate=original_tokens,
                optimized_token_estimate=optimized_tokens,
                token_reduction_percent=token_reduction,
                token_estimation_method="heuristic",
                bypassed=bypassed,
                bypass_reason=bypass_reason,
                auditor_configuration=auditor_cfg,
                processing_time_ms=total_duration_ms,
            )

            current_stage = PipelineStage.COMPLETED
            trace.append(f"Stage: {current_stage.value}")

            return RABCOOrchestrationResult(
                decision=decision,
                phase5_result=phase5_result,
                failed_stage=None,
                error_message=None,
                execution_trace=tuple(trace),
                evidence_units=all_evidence,
                relationships=all_relationships,
                compression_result=active_compression,
                recovery_outcome=recovery_outcome,
            )

        except Exception as exc:
            log.exception("RABCO pipeline failure at stage '%s': %s", current_stage.value, exc)
            trace.append(f"ERROR at stage {current_stage.value}: {exc}")

            if raise_on_failure:
                raise RABCOPipelineError(stage=current_stage, message=str(exc), cause=exc) from exc

            # Fail-safe failover: engagement of fail-safe original context bypass
            total_duration_ms = round((time.perf_counter() - start_time) * 1000)
            bypass_msg = f"Pipeline execution failed at stage '{current_stage.value}': {exc}"

            fallback_risk = self.risk_model.assess(
                audit_results=(),
                evidence_units=(),
                relationships=(),
                compression_result=None,
                audit_policy=eff_audit_policy,
            )

            fallback_p5_result = RABCOPhase5Result(
                final_context_messages=orig_messages_tuple,
                original_messages_ref=orig_messages_tuple,
                probes=(),
                audit_results=(),
                missing_evidence_ids=(),
                restored_evidence_ids=(),
                recovery_attempt_count=0,
                recovery_records=(),
                final_preservation_status="execution_failed_bypassed",
                budget_status="budget_conflict",
                budget_conflict_details=bypass_msg,
                risk_assessment=fallback_risk,
                original_token_estimate=original_tokens,
                optimized_token_estimate=original_tokens,
                token_reduction_percent=0.0,
                token_estimation_method="heuristic",
                bypassed=True,
                bypass_reason=bypass_msg,
                auditor_configuration=self._build_auditor_config(eff_audit_policy, eff_rec_policy),
                processing_time_ms=total_duration_ms,
            )

            return RABCOOrchestrationResult(
                decision=OrchestrationDecision.EXECUTION_FAILED,
                phase5_result=fallback_p5_result,
                failed_stage=current_stage,
                error_message=str(exc),
                execution_trace=tuple(trace),
                evidence_units=(),
                relationships=(),
                compression_result=None,
                recovery_outcome=None,
            )

    run = orchestrate

    # ------------------------------------------------------------------------
    # Safety Decision Logic
    # ------------------------------------------------------------------------

    def _evaluate_final_safety_decision(
        self,
        initial_risk: RiskAssessment,
        active_risk: RiskAssessment,
        recovery_outcome: Optional[SelectiveRecoveryOutcome],
        all_evidence: Sequence[EvidenceUnit],
        active_compression: CompressionResult,
    ) -> Tuple[OrchestrationDecision, bool, Optional[str], str]:
        """Evaluate terminal pipeline safety and determine whether bypass is required.

        Invariants:
        1. Mandatory Evidence Integrity:
           Mandatory evidence must never be silently discarded. If mandatory_evidence_loss > 0,
           the pipeline MUST mandate bypass and reject safe continuation.
        2. Recovery Exhaustion:
           If recovery was performed but failed to restore safety invariants, bypass is required.
        3. Fail-Safe Bypass:
           When bypass is engaged, bypassed is True, and the final context reverts to original messages.
        """
        # 1. Inviolable Safety Check: Mandatory evidence loss
        if active_risk.mandatory_evidence_loss > 0:
            lost_mand_count = active_risk.mandatory_evidence_loss
            reason = (
                f"STRICT SAFETY INVARIANT BREACH: {lost_mand_count} mandatory evidence unit(s) "
                f"not fully preserved. Reverting to original immutable conversation history."
            )
            decision = (
                OrchestrationDecision.RECOVERY_UNSUCCESSFUL
                if recovery_outcome is not None and recovery_outcome.total_recovery_attempts > 0
                else OrchestrationDecision.BYPASS_REQUIRED
            )
            return decision, True, reason, "mandatory_loss_bypassed"

        # 2. Check if recovery was executed
        if recovery_outcome is not None:
            if recovery_outcome.bypass_required or active_risk.bypass_recommended:
                reason = (
                    recovery_outcome.bypass_reason
                    or active_risk.bypass_reason
                    or "Selective recovery exhausted without satisfying preservation invariants."
                )
                status = recovery_outcome.recovery_result.final_preservation_status
                return OrchestrationDecision.RECOVERY_UNSUCCESSFUL, True, reason, status

            # Recovery succeeded and restored all safety invariants
            status = "recovered_preserved"
            return OrchestrationDecision.RECOVERY_COMPLETED, False, None, status

        # 3. No recovery was needed: check initial assessment
        if initial_risk.bypass_recommended:
            reason = initial_risk.bypass_reason or "Context compression failed safety invariants."
            return OrchestrationDecision.BYPASS_REQUIRED, True, reason, "initial_assessment_bypassed"

        # Initial compression passed all safety checks cleanly
        return OrchestrationDecision.SAFE_TO_CONTINUE, False, None, "verified_preserved"

    # ------------------------------------------------------------------------
    # Telemetry Helper
    # ------------------------------------------------------------------------

    @staticmethod
    def _build_auditor_config(
        audit_policy: RABCOAuditPolicy,
        recovery_policy: RABCORecoveryPolicy,
    ) -> Dict[str, Any]:
        """Construct immutable telemetry dictionary documenting active policy parameters."""
        return {
            "enable_semantic_audit": audit_policy.enable_semantic_audit,
            "auditor_model": audit_policy.auditor_ollama_model,
            "auditor_base_url": audit_policy.auditor_ollama_base_url,
            "auditor_timeout_seconds": audit_policy.auditor_timeout_seconds,
            "auditor_temperature": audit_policy.auditor_temperature,
            "max_acceptable_non_mandatory_failure_rate": audit_policy.max_acceptable_non_mandatory_failure_rate,
            "max_recovery_attempts": recovery_policy.max_recovery_attempts,
            "enable_relational_closure": recovery_policy.enable_relational_closure,
            "bypass_on_failure": recovery_policy.bypass_on_failure,
        }
