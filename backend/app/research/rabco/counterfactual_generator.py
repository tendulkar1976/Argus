"""RABCO Phase 5B: Hybrid Counterfactual Probe Generator.

Project: Argus — Risk-Aware, Counterfactual-Bounded Context Optimization (RABCO).
Phase: 5B — Counterfactual Probe Generator.

Purpose:
    Generate a finite, controlled set of plausible follow-up questions (probes) that
    assess whether a compressed conversation context retains sufficient evidence to answer
    questions derived from the original conversation history.

Architectural Guarantees:
1. No Hardcoding: All operational constants (model, URL, timeout, limits) are sourced
   from app.core.config.settings with explicit provisional-default documentation.
2. Deterministic First: Template-based probes are always produced and always succeed.
   LLM-assisted generation is strictly optional and additive; its failure cannot
   degrade deterministic output.
3. Relationship-Aware: Deterministic probes are generated from both evidence type and
   the actual EvidenceRelationship graph. Each relationship type maps to a specific
   probe template that reflects the relational semantics.
4. Grounded Evidence: Every probe must reference at least one valid, existing evidence ID.
   LLM-supplied evidence references are never trusted without structural verification
   against the actual evidence collection. The LLM cannot create or modify ground truth.
5. Probe-Set Isolation: DEVELOPMENT, AUDIT, and HELD_OUT are distinct. HELD_OUT is never
   generated automatically. Cross-set mixing is explicitly rejected at construction time.
   The optimizer must never receive held-out probes.
6. Provenance: Source message IDs are derived deterministically from the referenced
   EvidenceUnit.source_span.message_id — never from LLM output.
7. Research Integrity: Probes represent a finite sample. A passing preservation score
   indicates preservation across the generated probe set only, not across all possible
   future questions from the original conversation.
8. Separation: This module does not implement auditing, risk calculation, or recovery.
   Those are Phase 5C–5F responsibilities.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

from pydantic import ValidationError

from app.core.config import settings
from app.schemas.rabco import (
    ConversationMessage,
    EvidenceRelationship,
    EvidenceType,
    EvidenceUnit,
    RelationshipType,
    ValidationStatus,
)
from .schemas import (
    CounterfactualProbe,
    ProbeCategory,
    ProbeGenerationResult,
    ProbeSetType,
)

log = logging.getLogger(__name__)


# ============================================================================
# Constants — relationship type to probe category mapping
# ============================================================================

# Maps each RelationshipType that warrants a dedicated probe template to its
# ProbeCategory. Categories without a relationship-specific mapping fall back to
# the evidence-type template engine.
_RELATIONSHIP_TO_PROBE_CATEGORY: Mapping[RelationshipType, ProbeCategory] = {
    RelationshipType.CORRECTS: ProbeCategory.CORRECTION,
    RelationshipType.SUPERSEDES: ProbeCategory.HISTORICAL_STATE,
    RelationshipType.CONTRADICTS: ProbeCategory.CONTRADICTION,
    RelationshipType.DEPENDS_ON: ProbeCategory.DEPENDENCY,
    RelationshipType.CONDITIONAL_ON: ProbeCategory.CONSTRAINT,
    RelationshipType.EXCEPTION_TO: ProbeCategory.EXCEPTION,
    RelationshipType.APPLIES_TO: ProbeCategory.GENERAL,
    RelationshipType.SAME_ENTITY: ProbeCategory.GENERAL,
    # REFERS_TO and RELATED_TO are excluded from dedicated relationship probes;
    # they are weak associations that do not warrant targeted preservation probes.
}

# Maps EvidenceType to ProbeCategory for the type-based deterministic engine.
_EVIDENCE_TYPE_TO_PROBE_CATEGORY: Mapping[EvidenceType, ProbeCategory] = {
    EvidenceType.NUMERICAL_VALUE: ProbeCategory.NUMERICAL_VALUE,
    EvidenceType.CORRECTION: ProbeCategory.CORRECTION,
    EvidenceType.CONSTRAINT: ProbeCategory.CONSTRAINT,
    EvidenceType.CONDITION: ProbeCategory.CONSTRAINT,
    EvidenceType.EXCEPTION: ProbeCategory.EXCEPTION,
    EvidenceType.HISTORICAL_STATE: ProbeCategory.HISTORICAL_STATE,
    EvidenceType.PREFERENCE: ProbeCategory.PREFERENCE_UPDATE,
    EvidenceType.FACT: ProbeCategory.GENERAL,
    EvidenceType.INSTRUCTION: ProbeCategory.GENERAL,
    EvidenceType.GOAL: ProbeCategory.GENERAL,
    EvidenceType.DECISION: ProbeCategory.GENERAL,
    EvidenceType.INTERMEDIATE_RESULT: ProbeCategory.GENERAL,
    EvidenceType.OTHER: ProbeCategory.GENERAL,
}


# ============================================================================
# Probe-set isolation guard
# ============================================================================

# Probe sets that may be automatically generated during normal RABCO operation.
_GENERATABLE_PROBE_SETS: FrozenSet[ProbeSetType] = frozenset({
    ProbeSetType.DEVELOPMENT,
    ProbeSetType.AUDIT,
})


def _assert_generatable_probe_set(probe_set: ProbeSetType) -> None:
    """Raise ValueError if the requested probe set may not be generated automatically.

    HELD_OUT probes are reserved exclusively for final evaluation and must never be
    produced by the automatic generator. This prevents accidental optimizer exposure
    to held-out evaluation data.
    """
    if probe_set not in _GENERATABLE_PROBE_SETS:
        raise ValueError(
            f"Probe set '{probe_set.value}' cannot be generated automatically. "
            f"HELD_OUT is reserved exclusively for final evaluation and must never "
            f"be produced by the CounterfactualProbeGenerator. "
            f"Generatable sets: {[s.value for s in _GENERATABLE_PROBE_SETS]}."
        )


# ============================================================================
# Deduplication
# ============================================================================

def _normalize_question(question: str) -> str:
    """Return a canonical key for question deduplication.

    Strips leading/trailing whitespace, lowercases, and removes all
    non-alphanumeric characters. This catches minor wording variants of
    the same question without requiring semantic similarity computation.
    """
    return re.sub(r"\W+", "", question.strip().lower())


# ============================================================================
# Evidence grounding validation
# ============================================================================

def _derive_source_message_ids(
    evidence_ids: Tuple[str, ...],
    evidence_by_id: Mapping[str, EvidenceUnit],
) -> Tuple[str, ...]:
    """Derive source message IDs from evidence provenance — never from LLM output.

    For each evidence ID, extracts source_span.message_id from the actual
    EvidenceUnit. Any missing IDs are skipped (evidence validation must have
    already confirmed existence before calling this function).
    """
    seen: Set[str] = set()
    ordered: List[str] = []
    for ev_id in evidence_ids:
        ev = evidence_by_id.get(ev_id)
        if ev is None:
            continue
        msg_id = ev.source_span.message_id
        if msg_id not in seen:
            seen.add(msg_id)
            ordered.append(msg_id)
    return tuple(ordered)


def _validate_evidence_references(
    required_evidence_ids: Tuple[str, ...],
    evidence_by_id: Mapping[str, EvidenceUnit],
    target_evidence_text: Optional[str],
) -> Tuple[bool, str]:
    """Validate that all referenced evidence IDs exist and optional anchor text is genuine.

    Returns:
        (valid: bool, rejection_reason: str)
        rejection_reason is empty when valid is True.

    Rules:
    - Every ID in required_evidence_ids must exist in evidence_by_id.
    - If target_evidence_text is provided, it must appear verbatim (substring)
      in the original_text of at least one of the referenced evidence units.
      This prevents LLM-fabricated anchor text.
    - Blank IDs are rejected.
    """
    for ev_id in required_evidence_ids:
        if not ev_id.strip():
            return False, f"Blank evidence ID in required_evidence_ids"
        if ev_id not in evidence_by_id:
            return False, f"Evidence ID '{ev_id}' does not exist in provided evidence collection"

    if target_evidence_text is not None:
        anchor_found = any(
            target_evidence_text in evidence_by_id[ev_id].original_text
            for ev_id in required_evidence_ids
            if ev_id in evidence_by_id
        )
        if not anchor_found:
            return False, (
                f"target_evidence_text does not appear verbatim in any of the "
                f"referenced evidence units: {required_evidence_ids}"
            )

    return True, ""


# ============================================================================
# Unique probe ID generation
# ============================================================================

def _make_probe_id(prefix: str, existing_ids: Set[str]) -> str:
    """Generate a unique probe ID with the given prefix, retrying on collision."""
    for _ in range(16):
        candidate = f"{prefix}_{uuid.uuid4().hex}"
        if candidate not in existing_ids:
            existing_ids.add(candidate)
            return candidate
    # Statistically impossible collision after 16 UUIDs; raise for safety.
    raise RuntimeError("Could not generate a unique probe ID after 16 attempts")


# ============================================================================
# Deterministic template engine — evidence-type probes
# ============================================================================

# Template strings per ProbeCategory. The {snippet} placeholder receives the
# first 160 characters of the evidence's original_text (stripped of newlines).
_TYPE_TEMPLATES: Mapping[ProbeCategory, str] = {
    ProbeCategory.NUMERICAL_VALUE: (
        "What is the numeric value stated in: '{snippet}'?"
    ),
    ProbeCategory.CORRECTION: (
        "What correction was made in: '{snippet}', and what was the original value?"
    ),
    ProbeCategory.CONSTRAINT: (
        "What constraint or condition is imposed by: '{snippet}'?"
    ),
    ProbeCategory.EXCEPTION: (
        "What exception is described in: '{snippet}', and to which rule does it apply?"
    ),
    ProbeCategory.DEPENDENCY: (
        "What prerequisite dependency is expressed in: '{snippet}'?"
    ),
    ProbeCategory.CONTRADICTION: (
        "What conflicting assertion is present in: '{snippet}'?"
    ),
    ProbeCategory.HISTORICAL_STATE: (
        "What historical state or previous value is described in: '{snippet}'?"
    ),
    ProbeCategory.PREFERENCE_UPDATE: (
        "What preference update is reflected in: '{snippet}'?"
    ),
    ProbeCategory.GENERAL: (
        "What information is conveyed in: '{snippet}'?"
    ),
}


def _build_type_based_probes(
    evidence_units: Sequence[EvidenceUnit],
    probe_set: ProbeSetType,
    seen_normalized: Set[str],
    seen_ids: Set[str],
) -> List[CounterfactualProbe]:
    """Generate one probe per evidence unit based on its EvidenceType.

    Each probe uses a category-specific template question derived from the first
    160 characters of original_text. Only evidence with a mapped ProbeCategory
    receives a probe. The question is deduplicated against seen_normalized.

    Args:
        evidence_units: Sequence of evidence units to generate probes from.
        probe_set: Target partition (DEVELOPMENT or AUDIT).
        seen_normalized: Mutable set of normalized question strings for deduplication.
        seen_ids: Mutable set of probe IDs already allocated.

    Returns:
        List of accepted CounterfactualProbe instances.
    """
    probes: List[CounterfactualProbe] = []

    for ev in evidence_units:
        cat = _EVIDENCE_TYPE_TO_PROBE_CATEGORY.get(ev.evidence_type)
        if cat is None:
            # Evidence type explicitly has no mapped category; skip.
            continue

        tmpl = _TYPE_TEMPLATES.get(cat)
        if tmpl is None:
            continue

        snippet = ev.original_text[:160].replace("\n", " ").strip()
        question = tmpl.format(snippet=snippet)

        norm = _normalize_question(question)
        if norm in seen_normalized:
            continue
        seen_normalized.add(norm)

        source_msg_ids = (ev.source_span.message_id,)

        try:
            probe = CounterfactualProbe(
                probe_id=_make_probe_id("det_type", seen_ids),
                question=question,
                probe_category=cat,
                required_evidence_ids=(ev.evidence_id,),
                related_relationship_ids=(),
                generation_method="deterministic_type_template",
                source_message_ids=source_msg_ids,
                validation_status=ValidationStatus.VALIDATED,
                probe_set=probe_set,
                target_evidence_text=ev.original_text,
                metadata={
                    "template_source": "type_based",
                    "evidence_type": ev.evidence_type.value,
                    "probe_category": cat.value,
                },
            )
            probes.append(probe)
        except (ValidationError, ValueError) as exc:
            log.warning(
                "Type-based probe construction failed for evidence_id=%s: %s",
                ev.evidence_id,
                exc,
            )

    return probes


# ============================================================================
# Deterministic template engine — relationship-aware probes
# ============================================================================

# Relationship-aware probe templates.
# Each template uses named placeholders:
#   {src_snippet}: first 120 chars of source evidence original_text
#   {tgt_snippet}: first 120 chars of target evidence original_text
_RELATIONSHIP_TEMPLATES: Mapping[RelationshipType, str] = {
    RelationshipType.CORRECTS: (
        "The statement '{src_snippet}' was described as correcting '{tgt_snippet}'. "
        "What was the original value and what is the corrected value?"
    ),
    RelationshipType.SUPERSEDES: (
        "The statement '{src_snippet}' supersedes '{tgt_snippet}'. "
        "What is the current state and what was the previous state?"
    ),
    RelationshipType.CONTRADICTS: (
        "'{src_snippet}' and '{tgt_snippet}' are described as contradicting each other. "
        "What unresolved conflict exists between these two statements?"
    ),
    RelationshipType.DEPENDS_ON: (
        "'{src_snippet}' depends on '{tgt_snippet}'. "
        "What prerequisite relationship must be preserved?"
    ),
    RelationshipType.CONDITIONAL_ON: (
        "'{src_snippet}' is conditional on '{tgt_snippet}'. "
        "Under what condition does this statement apply?"
    ),
    RelationshipType.EXCEPTION_TO: (
        "'{src_snippet}' is an exception to '{tgt_snippet}'. "
        "What rule or general principle does this statement create an exception to?"
    ),
    RelationshipType.APPLIES_TO: (
        "'{src_snippet}' applies to '{tgt_snippet}'. "
        "What is the scope or domain to which this statement applies?"
    ),
    RelationshipType.SAME_ENTITY: (
        "'{src_snippet}' and '{tgt_snippet}' refer to the same entity. "
        "What entity do these two statements both reference?"
    ),
}


def _build_relationship_probes(
    relationships: Sequence[EvidenceRelationship],
    evidence_by_id: Mapping[str, EvidenceUnit],
    probe_set: ProbeSetType,
    seen_normalized: Set[str],
    seen_ids: Set[str],
) -> List[CounterfactualProbe]:
    """Generate probes from the EvidenceRelationship graph.

    For each relationship whose type has a registered template, produces a probe
    that explicitly captures the relational semantics (e.g., old vs. corrected value,
    current vs. previous state, conflicting assertions).

    Validity requirements:
    - Both source_evidence_id and target_evidence_id must exist in evidence_by_id.
    - The relationship type must have a registered template in _RELATIONSHIP_TEMPLATES.
    - The generated question must not be a duplicate of an already-seen normalized form.

    Args:
        relationships: All EvidenceRelationship edges from the Phase 3 mapper.
        evidence_by_id: Index of EvidenceUnit by evidence_id.
        probe_set: Target partition (DEVELOPMENT or AUDIT).
        seen_normalized: Mutable deduplication set.
        seen_ids: Mutable probe ID allocation set.

    Returns:
        List of accepted CounterfactualProbe instances.
    """
    probes: List[CounterfactualProbe] = []

    for rel in relationships:
        # Validate both referenced evidence IDs exist
        src_ev = evidence_by_id.get(rel.source_evidence_id)
        tgt_ev = evidence_by_id.get(rel.target_evidence_id)

        if src_ev is None:
            log.debug(
                "Relationship %s references unknown source_evidence_id=%s; skipping.",
                rel.relationship_id,
                rel.source_evidence_id,
            )
            continue

        if tgt_ev is None:
            log.debug(
                "Relationship %s references unknown target_evidence_id=%s; skipping.",
                rel.relationship_id,
                rel.target_evidence_id,
            )
            continue

        tmpl = _RELATIONSHIP_TEMPLATES.get(rel.relationship_type)
        if tmpl is None:
            # Relationship type has no dedicated probe template (e.g. REFERS_TO, RELATED_TO).
            continue

        cat = _RELATIONSHIP_TO_PROBE_CATEGORY.get(rel.relationship_type, ProbeCategory.GENERAL)
        src_snippet = src_ev.original_text[:120].replace("\n", " ").strip()
        tgt_snippet = tgt_ev.original_text[:120].replace("\n", " ").strip()
        question = tmpl.format(src_snippet=src_snippet, tgt_snippet=tgt_snippet)

        norm = _normalize_question(question)
        if norm in seen_normalized:
            continue
        seen_normalized.add(norm)

        # Provenance is derived deterministically from the evidence units.
        source_message_ids = _derive_source_message_ids(
            (rel.source_evidence_id, rel.target_evidence_id),
            evidence_by_id,
        )

        try:
            probe = CounterfactualProbe(
                probe_id=_make_probe_id("det_rel", seen_ids),
                question=question,
                probe_category=cat,
                required_evidence_ids=(rel.source_evidence_id, rel.target_evidence_id),
                related_relationship_ids=(rel.relationship_id,),
                generation_method="deterministic_relationship_template",
                source_message_ids=source_message_ids,
                validation_status=ValidationStatus.VALIDATED,
                probe_set=probe_set,
                target_evidence_text=None,  # Relationship probes span two units; no single anchor.
                metadata={
                    "template_source": "relationship_based",
                    "relationship_type": rel.relationship_type.value,
                    "relationship_id": rel.relationship_id,
                    "probe_category": cat.value,
                },
            )
            probes.append(probe)
        except (ValidationError, ValueError) as exc:
            log.warning(
                "Relationship probe construction failed for relationship_id=%s: %s",
                rel.relationship_id,
                exc,
            )

    return probes


# ============================================================================
# LLM-assisted probe generation (optional, additive)
# ============================================================================

def _build_ollama_prompt(
    evidence_units: Sequence[EvidenceUnit],
    max_candidates: int,
) -> str:
    """Build the Ollama prompt requesting LLM-generated probe candidates.

    The prompt explicitly instructs the LLM to use only the provided evidence IDs
    and to not invent IDs, entities, or facts not present in the snippets.
    The expected response is a JSON array of objects.
    """
    snippets = "\n".join(
        f"- [{ev.evidence_id}] {ev.original_text[:200].replace(chr(10), ' ')}"
        for ev in evidence_units
    )
    return (
        f"Given the following evidence snippets, propose up to {max_candidates} concise "
        f"follow-up questions that could be answered using ONLY the evidence listed below.\n\n"
        f"Rules:\n"
        f"- Only reference evidence IDs that appear in the list below.\n"
        f"- Do not invent facts, entities, or values not present in the snippets.\n"
        f"- Each question must be clearly answerable from the referenced evidence.\n\n"
        f"Return a JSON array only. Each element must be an object with exactly these keys:\n"
        f"  question (string): The follow-up question.\n"
        f"  required_evidence_ids (array of strings): IDs from the list below that answer it.\n"
        f"  target_evidence_text (string or null): An exact verbatim substring from one of "
        f"the referenced snippets, or null if no short anchor applies.\n\n"
        f"Evidence:\n{snippets}\n"
    )


def _call_ollama_for_probes(prompt: str) -> Tuple[List[Dict[str, Any]], str]:
    """Call the Ollama API for probe candidate generation.

    Returns:
        (candidates: list of raw dicts, error_detail: str)
        candidates is empty and error_detail is non-empty on any failure.

    Failure modes handled:
    - Network/connection failure (requests.RequestException)
    - Non-200 HTTP status
    - Response body is not valid JSON
    - JSON root is not a list
    - Any unexpected exception

    In all failure cases the calling code must proceed with deterministic probes only.
    LLM failure must never corrupt deterministic output.
    """
    # Lazy import to avoid hard dependency when LLM generation is disabled.
    try:
        import requests
    except ImportError:
        return [], "requests library not available; LLM probe generation disabled"

    model = settings.RABCO_PROBE_GENERATION_MODEL or settings.OLLAMA_MODEL
    base_url = settings.RABCO_PROBE_GENERATION_BASE_URL or settings.OLLAMA_BASE_URL
    timeout = settings.RABCO_PROBE_GENERATION_TIMEOUT_SECONDS
    url = f"{base_url}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            # Use the same temperature as the auditor for deterministic-leaning output.
            "temperature": settings.RABCO_AUDITOR_TEMPERATURE,
        },
    }

    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
    except Exception as exc:
        return [], f"Ollama request failed: {exc}"

    try:
        data = response.json()
    except ValueError as exc:
        return [], f"Ollama response body is not valid JSON: {exc}"

    raw_text = data.get("response", "")
    if not isinstance(raw_text, str):
        return [], "Ollama response.response is not a string"

    # Strip Markdown code fences that the model may add around the JSON output.
    stripped = raw_text.strip()
    for fence in ("```json", "```"):
        if stripped.startswith(fence):
            stripped = stripped[len(fence):]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    stripped = stripped.strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return [], f"LLM response is not valid JSON after stripping: {exc}"

    if not isinstance(parsed, list):
        return [], f"LLM response root must be a JSON array, got {type(parsed).__name__}"

    return parsed, ""


def _process_llm_candidates(
    raw_candidates: List[Dict[str, Any]],
    evidence_by_id: Mapping[str, EvidenceUnit],
    probe_set: ProbeSetType,
    seen_normalized: Set[str],
    seen_ids: Set[str],
) -> Tuple[List[CounterfactualProbe], int, int, int]:
    """Validate and convert raw LLM candidate dicts to CounterfactualProbe instances.

    Validation steps (per candidate):
    1. Confirm `question` is a non-empty string.
    2. Confirm `required_evidence_ids` is a non-empty list of strings.
    3. Validate every referenced evidence ID exists in evidence_by_id.
    4. If `target_evidence_text` is present, verify it appears verbatim in
       at least one referenced evidence unit's original_text.
    5. Derive source_message_ids from the actual evidence (never from LLM output).
    6. Reject duplicates via normalized question deduplication.

    Returns:
        (accepted_probes, duplicate_rejected, invalid_evidence_rejected)
    """
    accepted: List[CounterfactualProbe] = []
    duplicate_rejected = 0
    invalid_evidence_rejected = 0

    for idx, cand in enumerate(raw_candidates):
        if not isinstance(cand, dict):
            log.debug("LLM candidate %d is not a dict; rejected.", idx)
            invalid_evidence_rejected += 1
            continue

        # 1. Validate question
        question = cand.get("question", "")
        if not isinstance(question, str) or not question.strip():
            log.debug("LLM candidate %d has missing or blank question; rejected.", idx)
            invalid_evidence_rejected += 1
            continue

        # 2. Validate required_evidence_ids
        raw_ids = cand.get("required_evidence_ids", [])
        if not isinstance(raw_ids, list) or not raw_ids:
            log.debug(
                "LLM candidate %d has missing or empty required_evidence_ids; rejected.", idx
            )
            invalid_evidence_rejected += 1
            continue

        required_ids = tuple(str(eid) for eid in raw_ids)

        # 3+4. Structural evidence grounding (anchor text if present)
        target_text: Optional[str] = cand.get("target_evidence_text")
        if target_text is not None and not isinstance(target_text, str):
            target_text = None  # Discard non-string anchor text rather than reject entirely

        valid, rejection_reason = _validate_evidence_references(
            required_ids, evidence_by_id, target_text
        )
        if not valid:
            log.debug(
                "LLM candidate %d failed evidence grounding: %s; rejected.", idx, rejection_reason
            )
            invalid_evidence_rejected += 1
            continue

        # 5. Derive provenance from actual evidence (never from LLM output)
        source_message_ids = _derive_source_message_ids(required_ids, evidence_by_id)

        # 6. Deduplication (conditional based on configuration)
        norm = _normalize_question(question)
        if settings.RABCO_PROBE_DEDUPLICATION_ENABLED:
            if norm in seen_normalized:
                log.debug("LLM candidate %d is a duplicate question; rejected.", idx)
                duplicate_rejected += 1
                continue
        # When deduplication is disabled, duplicates are accepted.
        # Add to seen set only if deduplication is enabled to avoid affecting later checks when disabled.
        if settings.RABCO_PROBE_DEDUPLICATION_ENABLED:
            seen_normalized.add(norm)

        # Construct probe
        try:
            probe = CounterfactualProbe(
                probe_id=_make_probe_id("llm", seen_ids),
                question=question.strip(),
                probe_category=ProbeCategory.GENERAL,
                required_evidence_ids=required_ids,
                related_relationship_ids=(),
                generation_method="selective_llm_ollama",
                source_message_ids=source_message_ids,
                validation_status=ValidationStatus.VALIDATED,
                probe_set=probe_set,
                target_evidence_text=target_text,
                metadata={
                    "template_source": "llm_ollama",
                    "model": settings.RABCO_PROBE_GENERATION_MODEL or settings.OLLAMA_MODEL,
                },
            )
            accepted.append(probe)
        except (ValidationError, ValueError) as exc:
            log.warning("LLM probe construction failed for candidate %d: %s", idx, exc)
            invalid_evidence_rejected += 1

    return accepted, duplicate_rejected, invalid_evidence_rejected


# ============================================================================
# Main generator class
# ============================================================================

class CounterfactualProbeGenerator:
    """Hybrid counterfactual probe generator for RABCO Phase 5B.

    Combines:
    1. Deterministic evidence-type probes: one probe per EvidenceUnit, using
       category-specific question templates derived from EvidenceType.
    2. Deterministic relationship-aware probes: one probe per EvidenceRelationship
       with a registered template, capturing relational semantics (corrections,
       supersessions, contradictions, dependencies, conditions, exceptions, scope,
       entity co-reference).
    3. Optional LLM-assisted probes (Ollama): requested only when
       settings.RABCO_PROBE_GENERATION_ENABLED is True. Candidates are strictly
       validated against the evidence collection before acceptance.

    Probe-set isolation:
    - DEVELOPMENT and AUDIT probes may be generated.
    - HELD_OUT is explicitly rejected and raises ValueError.
    - Probes are never silently converted between sets.

    Research integrity:
    - Generated probes represent a finite sample of possible follow-up questions.
    - A passing preservation score indicates preservation only across this probe set.
    - No auditing, risk calculation, or recovery is performed here.
    """

    def __init__(self, config=settings) -> None:
        """Initialize the generator with centralized configuration.

        Args:
            config: Settings instance. Defaults to the application-wide settings
                    singleton. Accepts a substitution for testing purposes.
        """
        self._config = config

    def generate_probes(
        self,
        original_messages: Tuple[ConversationMessage, ...],
        evidence_units: Tuple[EvidenceUnit, ...],
        relationships: Tuple[EvidenceRelationship, ...],
        user_query: str,
        probe_set: ProbeSetType,
    ) -> ProbeGenerationResult:
        """Generate a set of counterfactual probes for preservation auditing.

        Parameters
        ----------
        original_messages:
            All conversation messages. Retained for provenance context;
            message IDs are cross-referenced against evidence spans.
        evidence_units:
            Extracted evidence units from Phase 2. Used as the grounding
            corpus for all probe construction.
        relationships:
            Evidence relationship edges from Phase 3. Used to generate
            relationship-aware probes.
        user_query:
            The active user query that prompted probe generation.
            Retained in result metadata for traceability.
        probe_set:
            Target partition for all generated probes.
            Must be DEVELOPMENT or AUDIT. HELD_OUT raises ValueError.

        Returns
        -------
        ProbeGenerationResult
            Immutable result containing all validated probes and generation
            statistics. Never raises due to LLM failure; falls back to
            deterministic probes only.

        Raises
        ------
        ValueError
            If probe_set is HELD_OUT.
        """
        # Guard: reject HELD_OUT before any generation
        _assert_generatable_probe_set(probe_set)

        # Early return for empty evidence
        if not evidence_units:
            log.info(
                "CounterfactualProbeGenerator: no evidence units provided; "
                "returning empty ProbeGenerationResult."
            )
            return ProbeGenerationResult(
                probes=(),
                deterministic_count=0,
                llm_candidate_count=0,
                llm_accepted_count=0,
                rejected_candidate_count=0,
                duplicate_rejected_count=0,
                invalid_evidence_rejected_count=0,
                total_accepted_count=0,
                generation_method_summary="empty_input",
            )

        # Build evidence index for O(1) lookups
        evidence_by_id: Dict[str, EvidenceUnit] = {
            ev.evidence_id: ev for ev in evidence_units
        }

        # Shared mutable state for deduplication and ID allocation
        # across both deterministic engines and LLM candidates.
        seen_normalized: Set[str] = set()
        seen_ids: Set[str] = set()

        # ------------------------------------------------------------------
        # Phase 1: Evidence-type deterministic probes
        # ------------------------------------------------------------------
        type_probes = _build_type_based_probes(
            evidence_units=evidence_units,
            probe_set=probe_set,
            seen_normalized=seen_normalized,
            seen_ids=seen_ids,
        )

        # ------------------------------------------------------------------
        # Phase 2: Relationship-aware deterministic probes
        # ------------------------------------------------------------------
        rel_probes = _build_relationship_probes(
            relationships=relationships,
            evidence_by_id=evidence_by_id,
            probe_set=probe_set,
            seen_normalized=seen_normalized,
            seen_ids=seen_ids,
        )

        deterministic_all = type_probes + rel_probes
        deterministic_count = len(deterministic_all)

        # ------------------------------------------------------------------
        # Phase 3: Optional LLM-assisted probe generation
        # ------------------------------------------------------------------
        llm_candidate_count = 0
        llm_accepted_count = 0
        duplicate_rejected_count = 0
        invalid_evidence_rejected_count = 0
        llm_probes: List[CounterfactualProbe] = []
        generation_method = "deterministic_only"

        if self._config.RABCO_PROBE_GENERATION_ENABLED:
            max_cands = self._config.RABCO_PROBE_GENERATION_MAX_CANDIDATES
            prompt = _build_ollama_prompt(evidence_units, max_cands)
            raw_candidates, error_detail = _call_ollama_for_probes(prompt)

            if error_detail:
                log.warning(
                    "LLM probe generation unavailable or failed; "
                    "proceeding with deterministic probes only. Reason: %s",
                    error_detail,
                )
            else:
                llm_candidate_count = len(raw_candidates)
                llm_probes, dup_rej, inv_rej = _process_llm_candidates(
                    raw_candidates=raw_candidates,
                    evidence_by_id=evidence_by_id,
                    probe_set=probe_set,
                    seen_normalized=seen_normalized,
                    seen_ids=seen_ids,
                )
                llm_accepted_count = len(llm_probes)
                duplicate_rejected_count = dup_rej
                invalid_evidence_rejected_count = inv_rej

                if llm_probes:
                    generation_method = "hybrid_deterministic_plus_ollama"

        total_rejected = (
            (llm_candidate_count - llm_accepted_count)
            if llm_candidate_count > 0
            else 0
        )

        all_probes = tuple(deterministic_all + llm_probes)
        total_accepted = len(all_probes)

        log.info(
            "CounterfactualProbeGenerator: probe_set=%s, deterministic=%d, "
            "llm_candidates=%d, llm_accepted=%d, llm_rejected=%d "
            "(duplicate=%d, invalid_evidence=%d), total=%d",
            probe_set.value,
            deterministic_count,
            llm_candidate_count,
            llm_accepted_count,
            total_rejected,
            duplicate_rejected_count,
            invalid_evidence_rejected_count,
            total_accepted,
        )

        return ProbeGenerationResult(
            probes=all_probes,
            deterministic_count=deterministic_count,
            llm_candidate_count=llm_candidate_count,
            llm_accepted_count=llm_accepted_count,
            rejected_candidate_count=total_rejected,
            duplicate_rejected_count=duplicate_rejected_count,
            invalid_evidence_rejected_count=invalid_evidence_rejected_count,
            total_accepted_count=total_accepted,
            generation_method_summary=generation_method,
        )
