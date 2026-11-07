"""RABCO Phase 5D: Preservation Auditor.

Project: Argus — Risk-Aware, Counterfactual-Bounded Context Optimization (RABCO).
Phase: 5D — Preservation Auditor.

Purpose:
    Provide an independent evaluation engine that determines whether counterfactual follow-up
    questions (probes) remain answerable after context compression.

Architectural Guarantees:
1. Deterministic Primary Evaluation:
   Deterministic provenance auditing maps each probe's required evidence against the actual
   compression outputs (extractive evidence, validated summaries, and omitted evidence).
   - Verbatim extractive retention: PRESERVED.
   - Summary-only representation: PARTIALLY_PRESERVED.
   - Explicit omission: MISSING.
   - Broken relational prerequisites/contradictions: MISSING or PARTIALLY_PRESERVED.
   - Unverified summaries or unaccounted evidence: UNVERIFIABLE.
2. Decoupled Semantic SLM Auditing:
   When enabled via policy (enable_semantic_audit=True), queries the independent local SLM
   (Ollama) using ONLY the AuditorFacingProbe (question text) and the compressed context.
   NEVER exposes expected evidence IDs, ProbeEvidenceBindings, ground-truth evidence text,
   or expected answers to the semantic auditor.
3. Strict Grounding Re-association:
   Uses ProbeEvidenceBinding post-hoc to attribute missing or partially preserved evidence IDs.
   Evidence attribution is never invented or assumed from probe presence.
4. Resilient Fallback:
   Network errors, model timeouts, or malformed JSON responses from Ollama never crash the pipeline;
   the auditor gracefully falls back to deterministic provenance results with uncertainty tracking.
5. Partition Isolation:
   Only probes belonging to the AUDIT set are evaluated. DEVELOPMENT probes are excluded and
   never counted as preservation failures.
6. Seamless Phase 5C Compatibility:
   The primary output is Tuple[ProbeAuditResult, ...], directly consumable by RiskModel.assess().
7. Zero-Hardcoding:
   Model names, endpoints, timeouts, and temperatures are centrally sourced from
   app.core.config.settings and RABCOAuditPolicy.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import httpx

from app.core.config import settings
from app.schemas.rabco import (
    ConversationMessage,
    EvidenceRelationship,
    EvidenceType,
    EvidenceUnit,
    RelationshipType,
)
from .hybrid_compressor import (
    CompressionResult,
    SummaryValidationOutcome,
)
from .schemas import (
    AuditorFacingProbe,
    CounterfactualProbe,
    EvidencePreservationStatus,
    ProbeAuditResult,
    ProbeCategory,
    ProbeEvidenceBinding,
    ProbeSetType,
    RABCOAuditPolicy,
)

log = logging.getLogger(__name__)


# ============================================================================
# Verdict Mapping Constants
# ============================================================================

_SEMANTIC_VERDICT_TO_STATUS: Mapping[str, EvidencePreservationStatus] = {
    "answerable": EvidencePreservationStatus.PRESERVED,
    "partially_answerable": EvidencePreservationStatus.PARTIALLY_PRESERVED,
    "unanswerable": EvidencePreservationStatus.MISSING,
    "ambiguous": EvidencePreservationStatus.AMBIGUOUS,
}


# ============================================================================
# Context Formatting Helper
# ============================================================================

def _format_compressed_context(
    compression_result: Optional[CompressionResult] = None,
    compressed_messages: Optional[Sequence[ConversationMessage]] = None,
) -> str:
    """Format compressed context into structured text for semantic evaluation."""
    if compressed_messages:
        lines = []
        for msg in compressed_messages:
            lines.append(f"[{msg.role.value.upper()}]: {msg.content}")
        return "\n".join(lines)

    if compression_result is not None:
        sections: List[str] = []
        if compression_result.selected_evidence:
            extractive_lines = [
                f"- {u.original_text}"
                for u in compression_result.selected_evidence
            ]
            sections.append("Extractive Evidence:\n" + "\n".join(extractive_lines))

        if compression_result.validated_summaries:
            summary_lines = [
                f"- {s.summary_text}"
                for s in compression_result.validated_summaries
            ]
            sections.append("Context Summaries:\n" + "\n".join(summary_lines))

        return "\n\n".join(sections) if sections else "No retained context."

    return "No context provided."


# ============================================================================
# Semantic Auditor Prompt Construction (Decoupled from Answer Keys)
# ============================================================================

def _build_semantic_audit_prompt(
    probe: AuditorFacingProbe,
    compressed_text: str,
) -> str:
    """Synthesize independent auditor prompt strictly without ground-truth answer keys.

    Architectural Principle:
    The independent auditor evaluates whether a question is answerable based solely
    on the provided compressed context. Ground-truth evidence IDs, bindings, and
    expected answers are never included in the prompt.
    """
    return (
        "You are an independent conversation preservation auditor.\n"
        "Your task is to evaluate whether a specific question can be answered based strictly "
        "and solely on the provided conversation context.\n\n"
        "STRICT EVALUATION RULES:\n"
        "1. Base your judgment ONLY on the facts, constraints, and statements directly present in the context below.\n"
        "2. Do NOT assume, extrapolate, or use outside knowledge.\n"
        "3. If the context contains sufficient information to completely and unambiguously answer the question, verdict is 'answerable'.\n"
        "4. If the context contains some relevant points or hints but lacks critical specifics needed for a complete answer, verdict is 'partially_answerable'.\n"
        "5. If the context does not contain the information needed to answer the question, verdict is 'unanswerable'.\n"
        "6. If the context contains contradictory, conflicting, or genuinely ambiguous information regarding the question, verdict is 'ambiguous'.\n\n"
        "You must respond with a JSON object in this exact format:\n"
        "{\n"
        '  "verdict": "answerable" | "partially_answerable" | "unanswerable" | "ambiguous",\n'
        '  "explanation": "Concise factual reason for the verdict",\n'
        '  "uncertainty_flag": true | false\n'
        "}\n\n"
        f"--- CONTEXT ---\n{compressed_text}\n\n"
        f"--- QUESTION ---\n{probe.question}\n\n"
        "JSON Verdict:"
    )


# ============================================================================
# Ollama Client Query Wrapper
# ============================================================================

def _query_ollama_auditor(
    prompt: str,
    model: str,
    base_url: str,
    timeout: float,
    temperature: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Query local Ollama instance with structured JSON prompt and defensive error handling.

    Returns:
        (parsed_dict, error_message)
        error_message is None on successful JSON parse.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
        },
    }
    url = f"{base_url.rstrip('/')}/api/generate"

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
            if resp.status_code != 200:
                return None, f"Ollama HTTP {resp.status_code}: {resp.text[:120]}"

            data = resp.json()
            raw_response = data.get("response", "")
            if not isinstance(raw_response, str):
                return None, "Ollama response.response is not a string"

            # Strip markdown code fences if present
            stripped = raw_response.strip()
            for fence in ("```json", "```"):
                if stripped.startswith(fence):
                    stripped = stripped[len(fence):]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
            stripped = stripped.strip()

            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                return None, f"Ollama response root must be JSON object, got {type(parsed).__name__}"

            return parsed, None

    except httpx.TimeoutException:
        return None, f"Ollama request timed out after {timeout} seconds"
    except httpx.ConnectError:
        return None, f"Unable to connect to Ollama at {base_url}"
    except json.JSONDecodeError as exc:
        return None, f"Malformed JSON from Ollama: {exc}"
    except Exception as exc:
        return None, f"Unexpected error querying Ollama: {exc}"


# ============================================================================
# Preservation Auditor Implementation
# ============================================================================

class PreservationAuditor:
    """Independent preservation auditor for RABCO Phase 5D.

    Evaluates counterfactual follow-up probes against compressed context using a
    deterministic-first layered approach, with optional independent SLM verification.
    """

    def __init__(
        self,
        policy: Optional[RABCOAuditPolicy] = None,
        config=settings,
    ) -> None:
        """Initialize PreservationAuditor with centralized policy and configuration.

        Args:
            policy: Optional RABCOAuditPolicy instance. Defaults to policy from centralized settings.
            config: Application configuration instance.
        """
        self.policy = policy or RABCOAuditPolicy()
        self.config = config

    def audit_probes(
        self,
        probes: Sequence[CounterfactualProbe],
        evidence_units: Sequence[EvidenceUnit],
        relationships: Sequence[EvidenceRelationship] = (),
        compression_result: Optional[CompressionResult] = None,
        compressed_messages: Optional[Sequence[ConversationMessage]] = None,
        policy: Optional[RABCOAuditPolicy] = None,
    ) -> Tuple[ProbeAuditResult, ...]:
        """Execute preservation audit across eligible counterfactual probes.

        Parameters
        ----------
        probes:
            Collection of counterfactual candidate probes from Phase 5B.
        evidence_units:
            All extracted evidence units from original conversation.
        relationships:
            Relational edges between evidence units (DEPENDS_ON, CORRECTS, etc.).
        compression_result:
            Optional structured compression output from HybridCompressor.
        compressed_messages:
            Optional final optimized context messages.
        policy:
            Optional audit policy override.

        Returns
        -------
        Tuple[ProbeAuditResult, ...]
            Immutable sequence of independent audit results, directly consumable by RiskModel.assess().
        """
        eff_policy = policy or self.policy

        # Step 1: Filter probes to AUDIT partition only.
        # DEVELOPMENT probes must not be evaluated here or counted as preservation failures.
        eligible_probes = [p for p in probes if p.probe_set == ProbeSetType.AUDIT]
        if not eligible_probes:
            log.info("PreservationAuditor: No AUDIT probes provided; returning empty results.")
            return ()

        # Step 2: Index evidence units and relationship graph
        evidence_by_id: Dict[str, EvidenceUnit] = {u.evidence_id: u for u in evidence_units}
        rel_by_id: Dict[str, EvidenceRelationship] = {r.relationship_id: r for r in relationships}

        # Step 3: Establish compressor survival status from CompressionResult
        extractive_ids: Set[str] = set()
        summarized_ids: Set[str] = set()
        omitted_ids: Set[str] = set()
        invalid_summary_evidence_ids: Set[str] = set()

        if compression_result is not None:
            extractive_ids = {u.evidence_id for u in compression_result.selected_evidence}
            summarized_ids = {
                eid for s in compression_result.validated_summaries for eid in s.source_evidence_ids
            }
            omitted_ids = {o.evidence_id for o in compression_result.omitted_evidence}
            for outcome in compression_result.validation_outcomes:
                if not outcome.is_valid:
                    invalid_summary_evidence_ids.update(outcome.covered_evidence_ids)

        elif compressed_messages is not None:
            # Fallback when only messages are passed: evaluate verbatim substring presence
            combined_msg_text = " ".join(m.content for m in compressed_messages)
            for eid, u in evidence_by_id.items():
                if u.original_text in combined_msg_text:
                    extractive_ids.add(eid)
                else:
                    omitted_ids.add(eid)

        # Step 4: Format compressed context text for semantic auditing
        compressed_text = _format_compressed_context(compression_result, compressed_messages)

        # Step 5: Audit each eligible probe
        audit_results: List[ProbeAuditResult] = []

        for probe in eligible_probes:
            # Evaluate deterministic baseline
            (
                det_status,
                det_missing_ids,
                det_partial_ids,
                det_explanation,
                det_uncertainty,
            ) = self._audit_probe_deterministic(
                probe=probe,
                evidence_by_id=evidence_by_id,
                rel_by_id=rel_by_id,
                extractive_ids=extractive_ids,
                summarized_ids=summarized_ids,
                omitted_ids=omitted_ids,
                invalid_summary_evidence_ids=invalid_summary_evidence_ids,
            )

            # Determine whether semantic evaluation should be performed
            if eff_policy.enable_semantic_audit:
                sem_result = self._audit_probe_semantic(
                    probe=probe,
                    compressed_text=compressed_text,
                    policy=eff_policy,
                    det_status=det_status,
                    det_missing_ids=det_missing_ids,
                    det_partial_ids=det_partial_ids,
                    det_explanation=det_explanation,
                    det_uncertainty=det_uncertainty,
                    extractive_ids=extractive_ids,
                    omitted_ids=omitted_ids,
                )
                audit_results.append(sem_result)
            else:
                # Deterministic-only audit
                audit_results.append(
                    ProbeAuditResult(
                        probe_id=probe.probe_id,
                        question=probe.question,
                        preservation_status=det_status,
                        required_evidence_ids=probe.required_evidence_ids,
                        missing_evidence_ids=det_missing_ids,
                        partially_preserved_ids=det_partial_ids,
                        auditor_type="deterministic_provenance",
                        explanation=det_explanation,
                        uncertainty_flag=det_uncertainty,
                    )
                )

        return tuple(audit_results)

    # ------------------------------------------------------------------------
    # Deterministic Provenance Auditing Layer
    # ------------------------------------------------------------------------

    def _audit_probe_deterministic(
        self,
        probe: CounterfactualProbe,
        evidence_by_id: Mapping[str, EvidenceUnit],
        rel_by_id: Mapping[str, EvidenceRelationship],
        extractive_ids: Set[str],
        summarized_ids: Set[str],
        omitted_ids: Set[str],
        invalid_summary_evidence_ids: Set[str],
    ) -> Tuple[EvidencePreservationStatus, Tuple[str, ...], Tuple[str, ...], str, bool]:
        """Perform structural and provenance audit for an individual probe.

        Returns:
            (status, missing_evidence_ids, partially_preserved_ids, explanation, uncertainty_flag)
        """
        binding = probe.to_evidence_binding()
        req_ids = binding.required_evidence_ids

        # 1. Evaluate relational edge integrity if this probe is relationship-grounded
        for rel_id in binding.related_relationship_ids:
            if rel_id in rel_by_id:
                rel = rel_by_id[rel_id]
                src_id = rel.source_evidence_id
                tgt_id = rel.target_evidence_id

                src_ext = src_id in extractive_ids
                tgt_ext = tgt_id in extractive_ids
                src_omit = src_id in omitted_ids
                tgt_omit = tgt_id in omitted_ids

                # DEPENDS_ON: Dependent retained, prerequisite omitted
                if rel.relationship_type == RelationshipType.DEPENDS_ON:
                    if src_ext and tgt_omit:
                        missing = (tgt_id,) if tgt_id in req_ids else req_ids
                        return (
                            EvidencePreservationStatus.MISSING,
                            missing,
                            (),
                            f"Relational dependency severed: prerequisite evidence '{tgt_id}' was "
                            f"omitted while dependent '{src_id}' was retained.",
                            False,
                        )

                # CONTRADICTS: One pole retained, opposing pole omitted (suppression)
                elif rel.relationship_type == RelationshipType.CONTRADICTS:
                    if (src_ext and tgt_omit) or (tgt_ext and src_omit):
                        omitted_pole = tgt_id if tgt_omit else src_id
                        missing = (omitted_pole,) if omitted_pole in req_ids else req_ids
                        return (
                            EvidencePreservationStatus.MISSING,
                            missing,
                            (),
                            f"Contradiction unbalanced: opposing evidence pole '{omitted_pole}' was "
                            f"omitted, suppressing conflict representation.",
                            False,
                        )

                # CORRECTS / SUPERSEDES: Severed correction chain
                elif rel.relationship_type in {RelationshipType.CORRECTS, RelationshipType.SUPERSEDES}:
                    if (src_ext and tgt_omit) or (tgt_ext and src_omit):
                        broken_id = tgt_id if tgt_omit else src_id
                        missing = (broken_id,) if broken_id in req_ids else req_ids
                        return (
                            EvidencePreservationStatus.MISSING,
                            missing,
                            (),
                            f"Correction chain broken: link between '{src_id}' and '{tgt_id}' severed.",
                            False,
                        )

                # EXCEPTION_TO: Rule retained while exception is omitted
                elif rel.relationship_type == RelationshipType.EXCEPTION_TO:
                    if tgt_ext and src_omit:
                        missing = (src_id,) if src_id in req_ids else req_ids
                        return (
                            EvidencePreservationStatus.MISSING,
                            missing,
                            (),
                            f"Exception '{src_id}' omitted while governed rule '{tgt_id}' was retained.",
                            False,
                        )

                # CONDITIONAL_ON: Action retained while governing condition is omitted
                elif rel.relationship_type == RelationshipType.CONDITIONAL_ON:
                    if src_ext and tgt_omit:
                        missing = (tgt_id,) if tgt_id in req_ids else req_ids
                        return (
                            EvidencePreservationStatus.MISSING,
                            missing,
                            (),
                            f"Condition '{tgt_id}' omitted while governed action '{src_id}' was retained.",
                            False,
                        )

        # 2. Evaluate required evidence units status
        missing_ids_list: List[str] = []
        partial_ids_list: List[str] = []
        extractive_count = 0
        unverifiable_count = 0

        for eid in req_ids:
            if eid in extractive_ids:
                extractive_count += 1
            elif eid in omitted_ids:
                missing_ids_list.append(eid)
            elif eid in summarized_ids:
                partial_ids_list.append(eid)
            elif eid in invalid_summary_evidence_ids:
                unverifiable_count += 1
            elif eid not in evidence_by_id:
                missing_ids_list.append(eid)
            else:
                # Evidence exists in original pool but is neither extractive, nor summarized, nor omitted
                unverifiable_count += 1

        total_req = len(req_ids)

        # Decision tree
        if extractive_count == total_req:
            return (
                EvidencePreservationStatus.PRESERVED,
                (),
                (),
                f"All required evidence units ({list(req_ids)}) verified as preserved extractively.",
                False,
            )

        if len(missing_ids_list) == total_req:
            return (
                EvidencePreservationStatus.MISSING,
                tuple(missing_ids_list),
                (),
                f"All required evidence units ({missing_ids_list}) explicitly omitted from context.",
                False,
            )

        if missing_ids_list:
            # Some units missing, others retained or summarized
            return (
                EvidencePreservationStatus.PARTIALLY_PRESERVED,
                tuple(missing_ids_list),
                tuple(partial_ids_list),
                f"Evidence partially preserved: missing={missing_ids_list}, partial={partial_ids_list}.",
                False,
            )

        if partial_ids_list and (extractive_count + len(partial_ids_list) == total_req):
            return (
                EvidencePreservationStatus.PARTIALLY_PRESERVED,
                (),
                tuple(partial_ids_list),
                f"Evidence units ({partial_ids_list}) represented through summary narrative without verbatim text.",
                False,
            )

        if unverifiable_count > 0:
            return (
                EvidencePreservationStatus.UNVERIFIABLE,
                (),
                (),
                f"Insufficient provenance data to verify survival of required evidence units ({list(req_ids)}).",
                True,
            )

        # Default fallback
        return (
            EvidencePreservationStatus.AMBIGUOUS,
            (),
            (),
            f"Ambiguous preservation status for evidence units ({list(req_ids)}).",
            True,
        )

    # ------------------------------------------------------------------------
    # Semantic SLM Auditing Layer
    # ------------------------------------------------------------------------

    def _audit_probe_semantic(
        self,
        probe: CounterfactualProbe,
        compressed_text: str,
        policy: RABCOAuditPolicy,
        det_status: EvidencePreservationStatus,
        det_missing_ids: Tuple[str, ...],
        det_partial_ids: Tuple[str, ...],
        det_explanation: str,
        det_uncertainty: bool,
        extractive_ids: Set[str],
        omitted_ids: Set[str],
    ) -> ProbeAuditResult:
        """Perform decoupled semantic question answering audit via local Ollama SLM."""
        # Derive strictly decoupled auditor-facing view (stripped of answer keys)
        auditor_probe = probe.to_auditor_view()
        prompt = _build_semantic_audit_prompt(auditor_probe, compressed_text)

        model = policy.auditor_ollama_model
        base_url = policy.auditor_ollama_base_url
        timeout = policy.auditor_timeout_seconds
        temperature = policy.auditor_temperature

        parsed_dict, err = _query_ollama_auditor(
            prompt=prompt,
            model=model,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
        )

        if err is not None or parsed_dict is None:
            # Fallback gracefully to deterministic finding
            log.warning(
                "PreservationAuditor: Semantic audit failed for probe '%s' (%s). "
                "Falling back to deterministic result.",
                probe.probe_id,
                err,
            )
            return ProbeAuditResult(
                probe_id=probe.probe_id,
                question=probe.question,
                preservation_status=det_status,
                required_evidence_ids=probe.required_evidence_ids,
                missing_evidence_ids=det_missing_ids,
                partially_preserved_ids=det_partial_ids,
                auditor_type="deterministic_provenance_fallback",
                explanation=f"{det_explanation} [Semantic audit fallback: {err}]",
                uncertainty_flag=True,
            )

        verdict_str = str(parsed_dict.get("verdict", "")).strip().lower()
        explanation = str(parsed_dict.get("explanation", "")).strip() or "Semantic evaluation completed."
        uncertainty = bool(parsed_dict.get("uncertainty_flag", False))

        mapped_status = _SEMANTIC_VERDICT_TO_STATUS.get(verdict_str)
        if mapped_status is None:
            log.warning(
                "PreservationAuditor: Unrecognized semantic verdict '%s' for probe '%s'. "
                "Falling back to deterministic result.",
                verdict_str,
                probe.probe_id,
            )
            return ProbeAuditResult(
                probe_id=probe.probe_id,
                question=probe.question,
                preservation_status=det_status,
                required_evidence_ids=probe.required_evidence_ids,
                missing_evidence_ids=det_missing_ids,
                partially_preserved_ids=det_partial_ids,
                auditor_type="deterministic_provenance_fallback",
                explanation=f"{det_explanation} [Unrecognized verdict: '{verdict_str}']",
                uncertainty_flag=True,
            )

        # Post-hoc evidence re-association via ground-truth ProbeEvidenceBinding
        binding = probe.to_evidence_binding()
        missing_ids: Tuple[str, ...] = ()
        partial_ids: Tuple[str, ...] = ()

        if mapped_status == EvidencePreservationStatus.MISSING:
            # Re-associate unanswerable verdict with required evidence that was omitted or non-extractive
            non_extractive = [eid for eid in binding.required_evidence_ids if eid not in extractive_ids]
            missing_ids = tuple(non_extractive) if non_extractive else binding.required_evidence_ids

        elif mapped_status == EvidencePreservationStatus.PARTIALLY_PRESERVED:
            # Re-associate partially answerable verdict
            omitted = [eid for eid in binding.required_evidence_ids if eid in omitted_ids]
            partially_represented = [eid for eid in binding.required_evidence_ids if eid not in omitted]
            missing_ids = tuple(omitted)
            partial_ids = tuple(partially_represented)

        elif mapped_status in {EvidencePreservationStatus.AMBIGUOUS, EvidencePreservationStatus.UNVERIFIABLE}:
            uncertainty = True

        return ProbeAuditResult(
            probe_id=probe.probe_id,
            question=probe.question,
            preservation_status=mapped_status,
            required_evidence_ids=probe.required_evidence_ids,
            missing_evidence_ids=missing_ids,
            partially_preserved_ids=partial_ids,
            auditor_type="semantic_slm",
            explanation=f"Semantic SLM audit ({model}): {explanation}",
            uncertainty_flag=uncertainty,
        )
