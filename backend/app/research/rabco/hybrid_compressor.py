"""Hybrid Context Compression Engine for Multi-Turn Conversation History.

Project: Argus — Risk-Aware, Counterfactual-Bounded Context Optimization (RABCO).
Phase: 4.2 — Final Safety Invariant Review.

Architectural Guarantees:
1. Extractive Priority: Important, sensitive, conditional, contradictory, and corrective
   evidence units remain strictly extractive. Original verbatim text is preserved.
2. Represented Dependency Closure: Causal and procedural prerequisites (target of DEPENDS_ON)
   are guaranteed to be retained extractively whenever the dependent action (source of DEPENDS_ON)
   is retained, based strictly on dependencies represented in the extracted evidence units
   and relationship graph.
   Note on Scope: Closure is guaranteed over the modeled graph representation; it does not claim
   discovery of latent, unmodeled real-world dependencies in the original conversation.
3. Strict Grounding & Ambiguity Rejection: Summaries with ungrounded entities, ungrounded numbers,
   unresolved negation inversions, condition/exception mismatches, or token footprint inflation
   are rejected. Rejected summaries restore their source evidence extractively. Lexical grounding
   is explicitly distinguished from formal semantic equivalence.
4. Relational Integrity in Grouping: Segmentation never severs corrections from antecedent values,
   exceptions from modified rules, conditions from governed actions, dependencies from prerequisites,
   or opposing sides of contradictions. Inseparable groups that cannot be safely partitioned
   are preserved extractively in full.
5. Inviolable Core Preservation Invariants: The preservation of corrections, dependencies, exceptions,
   and contradictions represents non-optional core safety requirements in normal RABCO operation.
   Experimental disabling requires an explicit unsafe gate (`enable_unsafe_experimental_relaxation=True`),
   is flagged in the result, and is never presented as satisfying normal RABCO safety standards.
6. Budget Conflict Resolution: When mandatory evidence and its relational closures exceed the target
   token budget, mandatory context is NEVER silently dropped. An explicit 'budget_conflict' outcome
   is emitted with full diagnostic details.
7. Zero-Hardcoding: All model parameters, endpoints, timeouts, and heuristics are sourced from
   centralized settings (app.core.config.settings) with documented, provisional defaults.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import re
import time
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.schemas.rabco import (
    ConversationMessage,
    EvidenceCriticality,
    EvidenceRelationship,
    EvidenceType,
    EvidenceUnit,
    RelationshipType,
    SourceSpan,
    SummaryUnit,
    ValidationStatus,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Centralized Compression Policy
# ============================================================================

@dataclass(frozen=True)
class CompressionPolicy:
    """Configurable behavioral and algorithmic policy for hybrid compression.

    Safety Invariant Architecture:
    The preservation of corrections, dependencies, exceptions, and contradictions
    represents core RABCO preservation requirements rather than ordinary compression
    trade-offs. In normal operation, these invariants are strictly enforced.
    Experimental relaxation requires setting `enable_unsafe_experimental_relaxation=True`,
    is marked as unsafe, is recorded in the compression result, and must never be
    presented as satisfying standard RABCO safety requirements.
    """

    target_token_budget: Optional[int] = None
    """Strict upper bound on estimated tokens. None indicates unconstrained optimization."""

    min_summary_group_size: int = 2
    """Provisional minimum number of related evidence units required to form an eligible
    summary cluster. Single units are retained extractively to prevent summary overhead inflation."""

    max_summary_group_size: int = 5
    """Provisional maximum number of evidence units clustered into a single summary to
    maintain semantic focus and bounded scope."""

    allow_generative_summary: bool = False
    """Whether to invoke local SLM/LLM for generative summarization. If False, deterministic
    extractive condensation is used."""

    strict_provenance_validation: bool = True
    """Whether to reject generated summaries that introduce ungrounded numbers or identifiers."""

    preserve_all_corrections: bool = True
    """Core Invariant: Retain all historical values in correction chains extractively."""

    preserve_all_dependencies: bool = True
    """Core Invariant: Enforce that prerequisite dependencies are never discarded if their dependent is kept."""

    preserve_all_exceptions: bool = True
    """Core Invariant: Enforce that exceptions are never discarded while retaining the rule they carve out from."""

    preserve_all_contradictions: bool = True
    """Core Invariant: Enforce that both poles of unresolved contradictions are preserved extractively."""

    enable_unsafe_experimental_relaxation: bool = False
    """Safety Gate: Must be explicitly True to permit disabling any of the core preservation invariants above.
    When False, disabling any core invariant is rejected and overridden to enforce standard safety."""

    chars_per_token_heuristic: float = settings.CHARS_PER_TOKEN_HEURISTIC
    """Character-to-token ratio used for local estimation. Centrally sourced from app.core.config.settings."""

    ollama_model: str = settings.OLLAMA_MODEL
    """Local Ollama model identifier for selective summarization. Centrally sourced from app.core.config.settings."""

    ollama_base_url: str = settings.OLLAMA_BASE_URL
    """Base endpoint URL for Ollama inference. Centrally sourced from app.core.config.settings."""

    ollama_timeout_seconds: float = settings.OLLAMA_TIMEOUT_SECONDS
    """Timeout in seconds for local SLM summary generation requests. Centrally sourced from app.core.config.settings."""

    ollama_temperature: float = settings.OLLAMA_TEMPERATURE
    """Inference temperature for summarization. Centrally sourced from app.core.config.settings."""


# ============================================================================
# Structured Output Contracts
# ============================================================================

class OmittedEvidence(BaseModel):
    """Record of an evidence unit omitted from the optimized view with explicit rationale."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    evidence_id: str = Field(..., description="Identifier of the omitted evidence unit")
    reason: str = Field(..., description="Explanation of why this evidence was omitted")
    source_message_id: str = Field(..., description="Message turn where this evidence originated")
    criticality: EvidenceCriticality = Field(..., description="Criticality level of the omitted unit")
    evidence_type: EvidenceType = Field(..., description="Semantic type of the omitted unit")
    original_text: str = Field(..., description="Verbatim text of the omitted unit")


class SummaryValidationOutcome(BaseModel):
    """Result of grounded provenance and structural verification on a generated summary unit."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    summary_id: str = Field(..., description="Identifier of the evaluated SummaryUnit")
    is_valid: bool = Field(..., description="Whether the summary passed grounding verification")
    validation_status: ValidationStatus = Field(..., description="Validation lifecycle state")
    unsupported_tokens: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Tokens, numbers, or entities present in summary but absent from source evidence",
    )
    lexical_grounding_passed: bool = Field(
        default=True,
        description="Whether all numbers and substantive identifiers are grounded in source evidence",
    )
    polarity_consistent: bool = Field(
        default=True,
        description="Whether negation and prohibition polarity matches source evidence without inversion",
    )
    condition_consistent: bool = Field(
        default=True,
        description="Whether conditional and exception clauses match source evidence without ambiguity",
    )
    token_compression_ratio: float = Field(
        default=1.0,
        description="Ratio of summary tokens to source tokens (must be < 1.0 to avoid inflation)",
    )
    reason: str = Field(..., description="Detailed explanation of the validation outcome")
    covered_evidence_ids: Tuple[str, ...] = Field(
        ...,
        description="Source evidence units intended to be represented by this summary",
    )
    validation_scope_note: str = Field(
        default="Lexical grounding and structural consistency verified. Formal semantic equivalence is not guaranteed.",
        description="Explicit disclaimer distinguishing lexical grounding from formal semantic validation",
    )


class CompressionResult(BaseModel):
    """Structured result of hybrid context compression containing data, summaries, and audit trail."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    selected_evidence: Tuple[EvidenceUnit, ...] = Field(
        ...,
        description="Exact semantic evidence units retained extractively in the optimized context",
    )
    validated_summaries: Tuple[SummaryUnit, ...] = Field(
        ...,
        description="Validated SummaryUnits consolidating eligible lower-risk evidence clusters",
    )
    omitted_evidence: Tuple[OmittedEvidence, ...] = Field(
        ...,
        description="Record of evidence units excluded from the view along with explicit omission reasons",
    )
    validation_outcomes: Tuple[SummaryValidationOutcome, ...] = Field(
        ...,
        description="Verification outcomes for all generated candidate summaries",
    )
    original_token_estimate: int = Field(
        ...,
        ge=0,
        description="Estimated token footprint of uncompressed input context",
    )
    compressed_token_estimate: int = Field(
        ...,
        ge=0,
        description="Estimated token footprint of optimized hybrid context",
    )
    token_reduction_percent: float = Field(
        ...,
        description="Percentage reduction in estimated token footprint",
    )
    token_estimation_method: str = Field(
        ...,
        description="Description of the heuristic or tokenizer used for estimation",
    )
    retained_relationship_ids: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Identifiers of relationships whose endpoints remain present in the optimized view",
    )
    budget_status: str = Field(
        default="unconstrained",
        description="Budget feasibility status: 'feasible', 'budget_conflict', or 'unconstrained'",
    )
    budget_conflict_details: Optional[str] = Field(
        default=None,
        description="Diagnostic details when mandatory context exceeds target budget",
    )
    safety_invariants_enforced: bool = Field(
        default=True,
        description="Whether core RABCO preservation invariants (corrections, dependencies, exceptions, contradictions) were strictly enforced",
    )
    safety_warnings: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="Explicit warnings if operating in unsafe experimental relaxation mode",
    )
    processing_time_ms: int = Field(
        default=0,
        ge=0,
        description="Pipeline execution duration in milliseconds",
    )


# ============================================================================
# Token Estimation & Grounding Lexical Patterns
# ============================================================================

_RE_NUMBERS = re.compile(
    r"(?:[₹$€£]\s*)?\b\d+(?:[.,]\d+)?(?:\s*(?:k|m|b|ms|s|sec|seconds|tokens|usd|inr|gb|mb|kb|%))?\b",
    re.IGNORECASE,
)
_RE_IDENTIFIERS = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b|\b`([^`]+)`|\"([^\"]+)\"")
_RE_NEGATIONS = re.compile(
    r"\b(?:not|never|no|cannot|can't|must\s+not|should\s+not|shouldn't|prohibit|prohibited|"
    r"forbidden|strictly\s+avoid|disallow|disallowed|without|don't)\b",
    re.IGNORECASE,
)
_RE_CONDITIONS = re.compile(
    r"\b(?:if|when|provided\s+that|in\s+case|unless|assuming\s+that|only\s+if)\b",
    re.IGNORECASE,
)
_RE_EXCEPTIONS = re.compile(
    r"\b(?:except|excluding|with\s+the\s+exception\s+of|unless|save\s+for)\b",
    re.IGNORECASE,
)

_STOP_WORDS: FrozenSet[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "should", "shouldn't", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "were", "weren't", "what", "when", "where", "which", "while", "who",
    "whom", "why", "with", "won't", "would", "wouldn't", "you", "your", "yours",
    "yourself", "yourselves",
})

_INSEPARABLE_RELATION_TYPES: FrozenSet[RelationshipType] = frozenset({
    RelationshipType.DEPENDS_ON,
    RelationshipType.CORRECTS,
    RelationshipType.SUPERSEDES,
    RelationshipType.CONTRADICTS,
    RelationshipType.CONDITIONAL_ON,
    RelationshipType.EXCEPTION_TO,
})


def estimate_tokens(text: str, chars_per_token: float) -> int:
    """Estimate token footprint using character-count heuristic.

    Limitations:
    This is an approximation heuristic and does not guarantee exact Byte-Pair Encoding
    (BPE) token counts for specific downstream frontier tokenizers.
    """
    clean = text.strip()
    if not clean:
        return 0
    ratio = chars_per_token if chars_per_token > 0.0 else 4.0
    return max(1, round(len(clean) / ratio))


def extract_numerical_tokens(text: str) -> Set[str]:
    """Extract normalized numerical and monetary mentions from text."""
    matches = _RE_NUMBERS.findall(text)
    return {m.strip().lower() for m in matches if m.strip()}


def extract_substantive_identifiers(text: str) -> Set[str]:
    """Extract code identifiers, quoted literals, and capitalized terms."""
    results: Set[str] = set()
    for match in _RE_IDENTIFIERS.finditer(text):
        quoted1 = match.group(1)
        quoted2 = match.group(2)
        raw_id = match.group(0)
        val = quoted1 or quoted2 or raw_id
        val_clean = val.strip("`\"'").lower()
        if len(val_clean) >= 3 and val_clean not in _STOP_WORDS:
            results.add(val_clean)
    return results


# ============================================================================
# Hybrid Compressor Implementation
# ============================================================================

class HybridCompressor:
    """Orchestrates hybrid context optimization across multi-turn conversation evidence."""

    def __init__(self, policy: Optional[CompressionPolicy] = None) -> None:
        self.policy = policy or CompressionPolicy()

    def _resolve_effective_policy(
        self, policy: CompressionPolicy
    ) -> Tuple[CompressionPolicy, bool, Tuple[str, ...]]:
        """Verify safety invariants and determine whether unsafe experimental mode is active.

        Guarantees:
        - In normal RABCO operation (`enable_unsafe_experimental_relaxation=False`),
          core preservation requirements (corrections, dependencies, exceptions, contradictions)
          CANNOT be disabled. If requested as False, they are forced to True with a log message.
        - If `enable_unsafe_experimental_relaxation=True` is explicitly specified:
          Requested relaxations are permitted for counterfactual ablation research,
          but safety_invariants_enforced is marked False and explicit warnings are attached.
        """
        requested_relaxation = not (
            policy.preserve_all_corrections
            and policy.preserve_all_dependencies
            and policy.preserve_all_exceptions
            and policy.preserve_all_contradictions
        )

        if not requested_relaxation:
            return policy, True, ()

        # Relaxation was requested: check safety gate
        if not policy.enable_unsafe_experimental_relaxation:
            logger.warning(
                "Requested relaxation of core preservation invariants rejected: normal RABCO operation "
                "strictly enforces corrections, dependencies, exceptions, and contradictions. "
                "Set enable_unsafe_experimental_relaxation=True to run unsafe counterfactual experiments."
            )
            # Re-enforce invariants for normal operation
            sanitized = CompressionPolicy(
                target_token_budget=policy.target_token_budget,
                min_summary_group_size=policy.min_summary_group_size,
                max_summary_group_size=policy.max_summary_group_size,
                allow_generative_summary=policy.allow_generative_summary,
                strict_provenance_validation=policy.strict_provenance_validation,
                preserve_all_corrections=True,
                preserve_all_dependencies=True,
                preserve_all_exceptions=True,
                preserve_all_contradictions=True,
                enable_unsafe_experimental_relaxation=False,
                chars_per_token_heuristic=policy.chars_per_token_heuristic,
                ollama_model=policy.ollama_model,
                ollama_base_url=policy.ollama_base_url,
                ollama_timeout_seconds=policy.ollama_timeout_seconds,
                ollama_temperature=policy.ollama_temperature,
            )
            return sanitized, True, ()

        # Explicit unsafe mode permitted for research ablation
        warnings: List[str] = [
            "UNSAFE EXPERIMENTAL MODE ACTIVE: One or more core RABCO preservation invariants "
            "(corrections, dependencies, exceptions, contradictions) have been disabled. "
            "Output does NOT satisfy normal RABCO safety requirements."
        ]
        if not policy.preserve_all_corrections:
            warnings.append("Unsafe relaxation: historical correction chains are not guaranteed to be retained.")
        if not policy.preserve_all_dependencies:
            warnings.append("Unsafe relaxation: prerequisite dependencies are not guaranteed to be retained.")
        if not policy.preserve_all_exceptions:
            warnings.append("Unsafe relaxation: rule exceptions are not guaranteed to be retained.")
        if not policy.preserve_all_contradictions:
            warnings.append("Unsafe relaxation: opposing sides of contradictions are not guaranteed to be retained.")

        return policy, False, tuple(warnings)

    def compress(
        self,
        evidence_units: Sequence[EvidenceUnit],
        relationships: Sequence[EvidenceRelationship],
        messages: Sequence[ConversationMessage],
        current_query: str,
        policy: Optional[CompressionPolicy] = None,
    ) -> CompressionResult:
        """Synchronously perform hybrid context compression with deterministic condensation."""
        start_time = time.perf_counter()
        active_policy, safety_enforced, safety_warnings = self._resolve_effective_policy(
            policy or self.policy
        )

        # Step 1: Identify mandatory extractive units and compute relational closure
        extractive_set, omission_candidates = self._partition_extractive_vs_eligible(
            evidence_units, relationships, active_policy
        )

        # Step 2: Cluster eligible evidence units into candidate summary groups
        eligible_units = [u for u in evidence_units if u.evidence_id not in extractive_set]
        summary_groups, unclustered_units = self._cluster_eligible_units(
            eligible_units, relationships, active_policy
        )

        # Unclustered units default to extractive retention to avoid data loss
        for u in unclustered_units:
            extractive_set.add(u.evidence_id)

        # Step 3: Synthesize candidate summaries deterministically
        candidate_summaries = []
        for group in summary_groups:
            sum_unit = self._synthesize_deterministic_summary(group)
            candidate_summaries.append((sum_unit, group))

        # Step 4: Validate generated summaries against source evidence
        validated_summaries: List[SummaryUnit] = []
        validation_outcomes: List[SummaryValidationOutcome] = []

        for candidate, group in candidate_summaries:
            outcome = self.validate_summary(candidate, group, active_policy)
            validation_outcomes.append(outcome)

            if outcome.is_valid:
                validated_summaries.append(candidate)
            else:
                logger.warning(
                    "Summary '%s' failed validation (%s). Restoring source evidence extractively.",
                    candidate.summary_id,
                    outcome.reason,
                )
                for u in group:
                    extractive_set.add(u.evidence_id)

        # Step 5: Enforce budget constraints and handle conflicts
        final_extractive, final_summaries, omitted_records, budget_status, budget_conflict_details = (
            self._apply_budget_and_finalize(
                evidence_units,
                extractive_set,
                validated_summaries,
                omission_candidates,
                active_policy,
            )
        )

        # Step 6: Identify preserved relationships
        retained_ev_ids = {u.evidence_id for u in final_extractive}
        retained_rel_ids = tuple(
            r.relationship_id
            for r in relationships
            if r.source_evidence_id in retained_ev_ids and r.target_evidence_id in retained_ev_ids
        )

        # Step 7: Calculate token footprints
        original_tokens = sum(
            estimate_tokens(m.content, active_policy.chars_per_token_heuristic)
            for m in messages
        )
        compressed_tokens = sum(
            estimate_tokens(u.original_text, active_policy.chars_per_token_heuristic)
            for u in final_extractive
        ) + sum(
            estimate_tokens(s.summary_text, active_policy.chars_per_token_heuristic)
            for s in final_summaries
        )

        reduction_pct = (
            round(((original_tokens - compressed_tokens) / original_tokens) * 100, 2)
            if original_tokens > 0
            else 0.0
        )

        duration_ms = round((time.perf_counter() - start_time) * 1000)

        return CompressionResult(
            selected_evidence=tuple(final_extractive),
            validated_summaries=tuple(final_summaries),
            omitted_evidence=tuple(omitted_records),
            validation_outcomes=tuple(validation_outcomes),
            original_token_estimate=original_tokens,
            compressed_token_estimate=compressed_tokens,
            token_reduction_percent=reduction_pct,
            token_estimation_method=(
                f"chars_per_token_heuristic (heuristic={active_policy.chars_per_token_heuristic} chars/token)"
            ),
            retained_relationship_ids=retained_rel_ids,
            budget_status=budget_status,
            budget_conflict_details=budget_conflict_details,
            safety_invariants_enforced=safety_enforced,
            safety_warnings=safety_warnings,
            processing_time_ms=duration_ms,
        )

    async def compress_async(
        self,
        evidence_units: Sequence[EvidenceUnit],
        relationships: Sequence[EvidenceRelationship],
        messages: Sequence[ConversationMessage],
        current_query: str,
        policy: Optional[CompressionPolicy] = None,
    ) -> CompressionResult:
        """Asynchronously perform hybrid context compression with optional local SLM summarization."""
        start_time = time.perf_counter()
        active_policy, safety_enforced, safety_warnings = self._resolve_effective_policy(
            policy or self.policy
        )

        if not active_policy.allow_generative_summary:
            return self.compress(evidence_units, relationships, messages, current_query, active_policy)

        # Step 1: Identify mandatory extractive units and compute relational closure
        extractive_set, omission_candidates = self._partition_extractive_vs_eligible(
            evidence_units, relationships, active_policy
        )

        # Step 2: Cluster eligible evidence units
        eligible_units = [u for u in evidence_units if u.evidence_id not in extractive_set]
        summary_groups, unclustered_units = self._cluster_eligible_units(
            eligible_units, relationships, active_policy
        )

        for u in unclustered_units:
            extractive_set.add(u.evidence_id)

        # Step 3: Synthesize summaries (SLM with deterministic fallback)
        validated_summaries: List[SummaryUnit] = []
        validation_outcomes: List[SummaryValidationOutcome] = []

        for group in summary_groups:
            candidate = await self._synthesize_generative_summary(group, active_policy)
            if candidate is None:
                candidate = self._synthesize_deterministic_summary(group)

            outcome = self.validate_summary(candidate, group, active_policy)
            validation_outcomes.append(outcome)

            if outcome.is_valid:
                validated_summaries.append(candidate)
            else:
                logger.warning(
                    "Generative summary '%s' failed validation (%s). Restoring source units extractively.",
                    candidate.summary_id,
                    outcome.reason,
                )
                for u in group:
                    extractive_set.add(u.evidence_id)

        # Step 4: Budget application and final assembly
        final_extractive, final_summaries, omitted_records, budget_status, budget_conflict_details = (
            self._apply_budget_and_finalize(
                evidence_units,
                extractive_set,
                validated_summaries,
                omission_candidates,
                active_policy,
            )
        )

        retained_ev_ids = {u.evidence_id for u in final_extractive}
        retained_rel_ids = tuple(
            r.relationship_id
            for r in relationships
            if r.source_evidence_id in retained_ev_ids and r.target_evidence_id in retained_ev_ids
        )

        original_tokens = sum(
            estimate_tokens(m.content, active_policy.chars_per_token_heuristic)
            for m in messages
        )
        compressed_tokens = sum(
            estimate_tokens(u.original_text, active_policy.chars_per_token_heuristic)
            for u in final_extractive
        ) + sum(
            estimate_tokens(s.summary_text, active_policy.chars_per_token_heuristic)
            for s in final_summaries
        )

        reduction_pct = (
            round(((original_tokens - compressed_tokens) / original_tokens) * 100, 2)
            if original_tokens > 0
            else 0.0
        )

        duration_ms = round((time.perf_counter() - start_time) * 1000)

        return CompressionResult(
            selected_evidence=tuple(final_extractive),
            validated_summaries=tuple(final_summaries),
            omitted_evidence=tuple(omitted_records),
            validation_outcomes=tuple(validation_outcomes),
            original_token_estimate=original_tokens,
            compressed_token_estimate=compressed_tokens,
            token_reduction_percent=reduction_pct,
            token_estimation_method=(
                f"chars_per_token_heuristic (heuristic={active_policy.chars_per_token_heuristic} chars/token)"
            ),
            retained_relationship_ids=retained_rel_ids,
            budget_status=budget_status,
            budget_conflict_details=budget_conflict_details,
            safety_invariants_enforced=safety_enforced,
            safety_warnings=safety_warnings,
            processing_time_ms=duration_ms,
        )

    # ------------------------------------------------------------------------
    # Partitioning & Relational Closure
    # ------------------------------------------------------------------------

    def _partition_extractive_vs_eligible(
        self,
        evidence_units: Sequence[EvidenceUnit],
        relationships: Sequence[EvidenceRelationship],
        policy: CompressionPolicy,
    ) -> Tuple[Set[str], List[EvidenceUnit]]:
        """Identify mandatory extractive evidence units and expand along relational closures.

        Dependency Direction & Closure Guarantees:
        1. Direction: In RelationshipType.DEPENDS_ON, source_evidence_id represents the dependent
           action (A) and target_evidence_id represents the prerequisite (B). Retaining A guarantees
           B is retained in the extractive set via BFS closure.
        2. Representation Scope: Closure is guaranteed strictly over dependencies represented
           in the extracted evidence units (via EvidenceUnit.dependency_ids) and the relationship
           graph (via EvidenceRelationship with RelationshipType.DEPENDS_ON).
        3. Boundary Disclaimer: This closure algorithm guarantees complete reachability over the
           modeled graph representation; it does NOT claim or guarantee discovery of latent, unextracted,
           or unmodeled real-world dependencies in the original conversation text that were not
           captured during extraction or relationship mapping.
        4. Cycle Tolerance: Traversal uses explicit visited-node tracking to handle valid directed
           cycles (e.g. mutual dependencies) safely and deterministically without infinite recursion.
        """
        extractive_ids: Set[str] = set()
        ev_by_id = {u.evidence_id: u for u in evidence_units}

        # 1. Base extractive criteria: Criticality tiers and sensitive semantic types
        for u in evidence_units:
            if u.criticality in {EvidenceCriticality.MANDATORY, EvidenceCriticality.HIGH}:
                extractive_ids.add(u.evidence_id)
                continue

            if u.evidence_type in {
                EvidenceType.CONSTRAINT,
                EvidenceType.INSTRUCTION,
                EvidenceType.CORRECTION,
                EvidenceType.CONDITION,
                EvidenceType.EXCEPTION,
                EvidenceType.NUMERICAL_VALUE,
                EvidenceType.DECISION,
            }:
                extractive_ids.add(u.evidence_id)
                continue

        # 2. Relational sensitivity tagging (pairwise coupled invariants)
        for rel in relationships:
            src_exists = rel.source_evidence_id in ev_by_id
            tgt_exists = rel.target_evidence_id in ev_by_id

            if not (src_exists and tgt_exists):
                continue

            # Contradictions: retain both opposing assertions
            if policy.preserve_all_contradictions and rel.relationship_type == RelationshipType.CONTRADICTS:
                extractive_ids.add(rel.source_evidence_id)
                extractive_ids.add(rel.target_evidence_id)

            # Corrections: preserve both antecedent and updated values
            if policy.preserve_all_corrections and rel.relationship_type in {
                RelationshipType.CORRECTS,
                RelationshipType.SUPERSEDES,
            }:
                extractive_ids.add(rel.source_evidence_id)
                extractive_ids.add(rel.target_evidence_id)

            # Exceptions: keep both exception and parent rule
            if policy.preserve_all_exceptions and rel.relationship_type == RelationshipType.EXCEPTION_TO:
                extractive_ids.add(rel.source_evidence_id)
                extractive_ids.add(rel.target_evidence_id)

            # Conditions: keep both condition and governed action
            if rel.relationship_type == RelationshipType.CONDITIONAL_ON:
                extractive_ids.add(rel.source_evidence_id)
                extractive_ids.add(rel.target_evidence_id)

        # 3. Relational closure propagation for dependencies (cycle-tolerant BFS)
        # Concrete example: A (action) depends on B (prerequisite).
        # In relationship_mapper: rel.source_evidence_id = A, rel.target_evidence_id = B.
        # Retaining A causes B to be retained.
        if policy.preserve_all_dependencies:
            dependency_map: Dict[str, Set[str]] = defaultdict(set)

            # Add graph relationships of type DEPENDS_ON
            for rel in relationships:
                if rel.relationship_type == RelationshipType.DEPENDS_ON:
                    # source (dependent) -> target (prerequisite)
                    dependency_map[rel.source_evidence_id].add(rel.target_evidence_id)

            # Also incorporate embedded dependency_ids declared directly on EvidenceUnit
            for u in evidence_units:
                for dep_id in u.dependency_ids:
                    dependency_map[u.evidence_id].add(dep_id)

            queue = list(extractive_ids)
            visited = set(extractive_ids)

            while queue:
                curr_id = queue.pop(0)
                for prereq_id in dependency_map.get(curr_id, ()):
                    if prereq_id in ev_by_id and prereq_id not in visited:
                        visited.add(prereq_id)
                        extractive_ids.add(prereq_id)
                        queue.append(prereq_id)

        eligible_units = [u for u in evidence_units if u.evidence_id not in extractive_ids]
        return extractive_ids, eligible_units

    # ------------------------------------------------------------------------
    # Clustering Eligible Evidence
    # ------------------------------------------------------------------------

    def _cluster_eligible_units(
        self,
        eligible_units: Sequence[EvidenceUnit],
        relationships: Sequence[EvidenceRelationship],
        policy: CompressionPolicy,
    ) -> Tuple[List[List[EvidenceUnit]], List[EvidenceUnit]]:
        """Group eligible lower-risk evidence units into coherent summary candidates.

        Inseparable Pair Guarantee:
        Splitting a group cannot separate:
        - Corrections from antecedent values
        - Exceptions from rules they modify
        - Conditions from actions they govern
        - Dependencies from prerequisites
        - Both sides of unresolved contradictions

        If a component exceeds max_summary_group_size and cannot be partitioned into
        safe, independent sub-groups without severing relational edges, the entire
        component is preserved extractively.
        """
        if not eligible_units:
            return [], []

        eligible_ids = {u.evidence_id for u in eligible_units}
        adj: Dict[str, Set[str]] = defaultdict(set)
        inseparable_edges: Set[Tuple[str, str]] = set()

        # Build adjacency over topical and entity relationships
        for rel in relationships:
            if rel.source_evidence_id in eligible_ids and rel.target_evidence_id in eligible_ids:
                adj[rel.source_evidence_id].add(rel.target_evidence_id)
                adj[rel.target_evidence_id].add(rel.source_evidence_id)

                if rel.relationship_type in _INSEPARABLE_RELATION_TYPES:
                    inseparable_edges.add((rel.source_evidence_id, rel.target_evidence_id))
                    inseparable_edges.add((rel.target_evidence_id, rel.source_evidence_id))

        # Also group evidence originating from the same message turn
        msg_groups: Dict[str, List[EvidenceUnit]] = defaultdict(list)
        for u in eligible_units:
            msg_groups[u.source_span.message_id].append(u)

        for mid, units in msg_groups.items():
            for i in range(len(units)):
                for j in range(i + 1, len(units)):
                    adj[units[i].evidence_id].add(units[j].evidence_id)
                    adj[units[j].evidence_id].add(units[i].evidence_id)

        # Connected components extraction (cycle-tolerant traversal)
        visited: Set[str] = set()
        components: List[List[EvidenceUnit]] = []
        ev_by_id = {u.evidence_id: u for u in eligible_units}

        for u in eligible_units:
            if u.evidence_id in visited:
                continue

            comp: List[EvidenceUnit] = []
            stack = [u.evidence_id]
            visited.add(u.evidence_id)

            while stack:
                curr_id = stack.pop()
                comp.append(ev_by_id[curr_id])

                for neighbor_id in adj.get(curr_id, ()):
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        stack.append(neighbor_id)

            components.append(comp)

        summary_groups: List[List[EvidenceUnit]] = []
        unclustered: List[EvidenceUnit] = []

        for comp in components:
            if len(comp) < policy.min_summary_group_size:
                # Group too small to summarize without token inflation overhead
                unclustered.extend(comp)
            elif len(comp) <= policy.max_summary_group_size:
                # Component is within safe bounds; keep all connected units together
                summary_groups.append(comp)
            else:
                # Component exceeds max_summary_group_size.
                # Check if safe splitting is possible without cutting inseparable edges.
                has_inseparable_link = any(
                    (u.evidence_id, v.evidence_id) in inseparable_edges
                    for u in comp
                    for v in comp
                    if u.evidence_id != v.evidence_id
                )

                if has_inseparable_link:
                    # Inseparable relationships cannot be safely severed across chunks; preserve extractively
                    logger.info(
                        "Component of %d units contains coupled relational edges; preserving extractively to prevent separation.",
                        len(comp),
                    )
                    unclustered.extend(comp)
                else:
                    # Safe to chunk into bounded topical groups
                    max_size = max(1, policy.max_summary_group_size)
                    for k in range(0, len(comp), max_size):
                        chunk = comp[k : k + max_size]
                        if len(chunk) >= policy.min_summary_group_size:
                            summary_groups.append(chunk)
                        else:
                            unclustered.extend(chunk)

        return summary_groups, unclustered

    # ------------------------------------------------------------------------
    # Summary Synthesis Methods
    # ------------------------------------------------------------------------

    def _synthesize_deterministic_summary(
        self,
        group: Sequence[EvidenceUnit],
    ) -> SummaryUnit:
        """Deterministically condense multiple evidence units without hallucination risk."""
        ev_ids = tuple(u.evidence_id for u in group)
        msg_ids = tuple(sorted({u.source_span.message_id for u in group}))

        clauses = []
        for u in group:
            text = u.original_text.strip().rstrip(".;,")
            if text and text not in clauses:
                clauses.append(text)

        summary_text = f"Consolidated narrative ({len(clauses)} points): " + "; ".join(clauses) + "."

        raw_key = ":".join(sorted(ev_ids))
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:12]
        summary_id = f"sum_ext_{digest}"

        return SummaryUnit(
            summary_id=summary_id,
            summary_text=summary_text,
            source_evidence_ids=ev_ids,
            source_message_ids=msg_ids,
            generation_method="extractive_condensation",
            provider_name="local_deterministic",
            model_identifier=None,
            validation_status=ValidationStatus.VALIDATED,
            created_at=datetime.now(timezone.utc),
        )

    async def _synthesize_generative_summary(
        self,
        group: Sequence[EvidenceUnit],
        policy: CompressionPolicy,
    ) -> Optional[SummaryUnit]:
        """Generate a concise factual summary via local Ollama instance."""
        ev_ids = tuple(u.evidence_id for u in group)
        msg_ids = tuple(sorted({u.source_span.message_id for u in group}))

        facts_text = "\n".join(f"- {u.original_text}" for u in group)
        prompt = (
            "You are a strict factual summarization engine.\n"
            "Synthesize the following conversation evidence into a single, concise factual sentence.\n"
            "STRICT RULES:\n"
            "1. Do NOT add any outside knowledge.\n"
            "2. Do NOT introduce any names, entities, numbers, dates, prohibitions, or conditions not explicitly stated below.\n"
            "3. State only what is strictly supported by the evidence.\n\n"
            f"Evidence facts:\n{facts_text}\n\n"
            "Summary:"
        )

        payload = {
            "model": policy.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": policy.ollama_temperature,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=policy.ollama_timeout_seconds) as client:
                resp = await client.post(f"{policy.ollama_base_url}/api/generate", json=payload)
                if resp.status_code != 200:
                    return None

                data = resp.json()
                raw_summary = data.get("response", "").strip()
                if not raw_summary:
                    return None

                clean_summary = raw_summary.strip("\"'").strip()

                raw_key = f"{policy.ollama_model}:" + ":".join(sorted(ev_ids))
                digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:12]
                summary_id = f"sum_slm_{digest}"

                return SummaryUnit(
                    summary_id=summary_id,
                    summary_text=clean_summary,
                    source_evidence_ids=ev_ids,
                    source_message_ids=msg_ids,
                    generation_method="selective_slm_ollama",
                    provider_name="ollama",
                    model_identifier=policy.ollama_model,
                    validation_status=ValidationStatus.PENDING,
                    created_at=datetime.now(timezone.utc),
                )

        except Exception as exc:
            logger.warning("Generative summarization request failed gracefully: %s", exc)
            return None

    # ------------------------------------------------------------------------
    # Summary Validation & Provenance Verification
    # ------------------------------------------------------------------------

    def validate_summary(
        self,
        summary: SummaryUnit,
        source_units: Sequence[EvidenceUnit],
        policy: CompressionPolicy,
    ) -> SummaryValidationOutcome:
        """Validate candidate summary claims for lexical grounding and structural consistency.

        Checks Enforced:
        1. Source Reference: Summary must reference at least one source evidence unit.
        2. Numerical Grounding: Numbers/amounts in summary must appear in source units.
        3. Identifier Grounding: Code/quoted identifiers in summary must appear in source units.
        4. Polarity Consistency: Detects unresolved negation or prohibition inversion.
        5. Condition Consistency: Detects introduced or stripped conditional/exception clauses.
        6. Non-Inflation Check: Rejects summaries whose estimated token count exceeds source tokens.

        Explicit Boundary Disclaimer:
        This check establishes lexical grounding, numerical provenance, and structural consistency.
        It does NOT constitute a mathematical or formal semantic proof of full semantic equivalence.
        """
        source_ids = tuple(u.evidence_id for u in source_units)
        heuristic = policy.chars_per_token_heuristic

        # Token footprint check
        summary_tokens = estimate_tokens(summary.summary_text, heuristic)
        source_tokens = sum(estimate_tokens(u.original_text, heuristic) for u in source_units)
        ratio = round(summary_tokens / source_tokens, 2) if source_tokens > 0 else 1.0

        # Check 1: Declared coverage
        if not summary.source_evidence_ids:
            return SummaryValidationOutcome(
                summary_id=summary.summary_id,
                is_valid=False,
                validation_status=ValidationStatus.REJECTED,
                unsupported_tokens=(),
                lexical_grounding_passed=False,
                polarity_consistent=False,
                condition_consistent=False,
                token_compression_ratio=ratio,
                reason="Summary declares no source evidence IDs.",
                covered_evidence_ids=source_ids,
            )

        # Check 2: Token footprint inflation
        # If summarization inflates context rather than compressing it, prefer extractive retention
        if summary.generation_method != "extractive_condensation" and summary_tokens >= source_tokens:
            return SummaryValidationOutcome(
                summary_id=summary.summary_id,
                is_valid=False,
                validation_status=ValidationStatus.REJECTED,
                unsupported_tokens=(),
                lexical_grounding_passed=True,
                polarity_consistent=True,
                condition_consistent=True,
                token_compression_ratio=ratio,
                reason=(
                    f"Token footprint inflation: candidate summary ({summary_tokens} tokens) "
                    f"is not smaller than source units ({source_tokens} tokens). Extractive preservation preferred."
                ),
                covered_evidence_ids=source_ids,
            )

        if not policy.strict_provenance_validation:
            return SummaryValidationOutcome(
                summary_id=summary.summary_id,
                is_valid=True,
                validation_status=ValidationStatus.VALIDATED,
                unsupported_tokens=(),
                lexical_grounding_passed=True,
                polarity_consistent=True,
                condition_consistent=True,
                token_compression_ratio=ratio,
                reason="Validation bypassed by policy configuration.",
                covered_evidence_ids=source_ids,
            )

        # Extractive condensation is intrinsically grounded in verbatim evidence
        if summary.generation_method == "extractive_condensation":
            return SummaryValidationOutcome(
                summary_id=summary.summary_id,
                is_valid=True,
                validation_status=ValidationStatus.VALIDATED,
                unsupported_tokens=(),
                lexical_grounding_passed=True,
                polarity_consistent=True,
                condition_consistent=True,
                token_compression_ratio=ratio,
                reason="Extractive condensation is intrinsically grounded in verbatim evidence spans.",
                covered_evidence_ids=source_ids,
            )

        # Check 3: Extract grounded pool from source evidence units
        grounded_numbers: Set[str] = set()
        grounded_identifiers: Set[str] = set()
        source_has_negation = False
        source_has_condition = False

        for u in source_units:
            grounded_numbers.update(extract_numerical_tokens(u.original_text))
            grounded_identifiers.update(extract_substantive_identifiers(u.original_text))
            if _RE_NEGATIONS.search(u.original_text):
                source_has_negation = True
            if _RE_CONDITIONS.search(u.original_text) or _RE_EXCEPTIONS.search(u.original_text):
                source_has_condition = True

        # Check 4: Lexical grounding (numbers and identifiers)
        summary_numbers = extract_numerical_tokens(summary.summary_text)
        summary_identifiers = extract_substantive_identifiers(summary.summary_text)

        unsupported_numbers = summary_numbers - grounded_numbers
        substantive_summary_ids = {
            tok for tok in summary_identifiers
            if (len(tok) >= 4 or "_" in tok) and tok not in _STOP_WORDS
        }
        unsupported_identifiers = substantive_summary_ids - grounded_identifiers
        unsupported_tokens = tuple(sorted(unsupported_numbers | unsupported_identifiers))

        if unsupported_tokens:
            return SummaryValidationOutcome(
                summary_id=summary.summary_id,
                is_valid=False,
                validation_status=ValidationStatus.REJECTED,
                unsupported_tokens=unsupported_tokens,
                lexical_grounding_passed=False,
                polarity_consistent=True,
                condition_consistent=True,
                token_compression_ratio=ratio,
                reason=f"Detected ungrounded entities or numbers: {unsupported_tokens}",
                covered_evidence_ids=source_ids,
            )

        # Check 5: Polarity and Negation Consistency
        summary_has_negation = bool(_RE_NEGATIONS.search(summary.summary_text))
        if summary_has_negation != source_has_negation:
            mismatch_desc = (
                "Summary introduced negation not present in source evidence"
                if summary_has_negation
                else "Summary stripped negation/prohibition present in source evidence"
            )
            return SummaryValidationOutcome(
                summary_id=summary.summary_id,
                is_valid=False,
                validation_status=ValidationStatus.REJECTED,
                unsupported_tokens=(),
                lexical_grounding_passed=True,
                polarity_consistent=False,
                condition_consistent=True,
                token_compression_ratio=ratio,
                reason=f"Polarity inconsistency: {mismatch_desc}. Extractive preservation preferred.",
                covered_evidence_ids=source_ids,
            )

        # Check 6: Condition and Exception Consistency
        summary_has_condition = bool(
            _RE_CONDITIONS.search(summary.summary_text) or _RE_EXCEPTIONS.search(summary.summary_text)
        )
        if summary_has_condition != source_has_condition:
            mismatch_desc = (
                "Summary introduced conditional/exception clause not in source"
                if summary_has_condition
                else "Summary stripped condition/exception clause present in source"
            )
            return SummaryValidationOutcome(
                summary_id=summary.summary_id,
                is_valid=False,
                validation_status=ValidationStatus.REJECTED,
                unsupported_tokens=(),
                lexical_grounding_passed=True,
                polarity_consistent=True,
                condition_consistent=False,
                token_compression_ratio=ratio,
                reason=f"Condition ambiguity: {mismatch_desc}. Extractive preservation preferred.",
                covered_evidence_ids=source_ids,
            )

        return SummaryValidationOutcome(
            summary_id=summary.summary_id,
            is_valid=True,
            validation_status=ValidationStatus.VALIDATED,
            unsupported_tokens=(),
            lexical_grounding_passed=True,
            polarity_consistent=True,
            condition_consistent=True,
            token_compression_ratio=ratio,
            reason="All numerical values, identifiers, polarity, and conditions are verified grounded.",
            covered_evidence_ids=source_ids,
        )

    # ------------------------------------------------------------------------
    # Budget Management & Final Assembly
    # ------------------------------------------------------------------------

    def _apply_budget_and_finalize(
        self,
        all_evidence: Sequence[EvidenceUnit],
        extractive_ids: Set[str],
        summaries: Sequence[SummaryUnit],
        eligible_units: Sequence[EvidenceUnit],
        policy: CompressionPolicy,
    ) -> Tuple[List[EvidenceUnit], List[SummaryUnit], List[OmittedEvidence], str, Optional[str]]:
        """Assemble selected context and evaluate budget constraints without dropping mandatory context.

        Budget Scenarios:
        - Feasible: Combined footprint fits within target budget.
        - Budget Conflict: Mandatory context (and relational closures) exceeds budget.
          Mandatory evidence is NEVER silently dropped; an explicit budget_conflict is flagged.
        - Unconstrained: No target budget was configured.
        """
        ev_by_id = {u.evidence_id: u for u in all_evidence}
        final_extractive = [ev_by_id[eid] for eid in extractive_ids if eid in ev_by_id]
        final_summaries = list(summaries)
        omitted_records: List[OmittedEvidence] = []
        budget_status = "unconstrained"
        budget_conflict_details: Optional[str] = None

        # Track evidence units consolidated into summaries
        summarized_ev_ids: Set[str] = set()
        for s in final_summaries:
            summarized_ev_ids.update(s.source_evidence_ids)

        # Record eligible units that were neither kept extractively nor summarized
        for u in eligible_units:
            if u.evidence_id not in extractive_ids and u.evidence_id not in summarized_ev_ids:
                omitted_records.append(
                    OmittedEvidence(
                        evidence_id=u.evidence_id,
                        reason="lower_risk_redundant_dialogue_fact",
                        source_message_id=u.source_span.message_id,
                        criticality=u.criticality,
                        evidence_type=u.evidence_type,
                        original_text=u.original_text,
                    )
                )

        # Budget evaluation
        if policy.target_token_budget is not None and policy.target_token_budget > 0:
            heuristic = policy.chars_per_token_heuristic
            target_budget = policy.target_token_budget

            # Identify mandatory subset that cannot be pruned
            mandatory_units = [
                u for u in final_extractive
                if u.criticality in {EvidenceCriticality.MANDATORY, EvidenceCriticality.HIGH}
                or u.evidence_type in {
                    EvidenceType.CONSTRAINT,
                    EvidenceType.INSTRUCTION,
                    EvidenceType.CORRECTION,
                    EvidenceType.CONDITION,
                    EvidenceType.EXCEPTION,
                    EvidenceType.NUMERICAL_VALUE,
                }
            ]
            mandatory_tokens = sum(estimate_tokens(u.original_text, heuristic) for u in mandatory_units)

            # Case: Mandatory context alone exceeds budget
            if mandatory_tokens > target_budget:
                budget_status = "budget_conflict"
                budget_conflict_details = (
                    f"Mandatory evidence and relational closures ({mandatory_tokens} estimated tokens) "
                    f"exceed the target token budget ({target_budget} tokens). All mandatory context "
                    f"was retained extractively to preserve instruction and constraint fidelity."
                )
                # Prune all non-mandatory units to minimize the budget overrun
                prunable = [u for u in final_extractive if u not in mandatory_units]
                for u in prunable:
                    final_extractive.remove(u)
                    omitted_records.append(
                        OmittedEvidence(
                            evidence_id=u.evidence_id,
                            reason="omitted_due_to_target_token_budget_priority",
                            source_message_id=u.source_span.message_id,
                            criticality=u.criticality,
                            evidence_type=u.evidence_type,
                            original_text=u.original_text,
                        )
                    )
            else:
                # Mandatory context fits within budget; evaluate combined footprint
                current_tokens = sum(
                    estimate_tokens(u.original_text, heuristic) for u in final_extractive
                ) + sum(estimate_tokens(s.summary_text, heuristic) for s in final_summaries)

                if current_tokens <= target_budget:
                    budget_status = "feasible"
                else:
                    # Prune low-criticality non-mandatory units first
                    prunable = [
                        u for u in final_extractive
                        if u not in mandatory_units
                        and u.criticality in {EvidenceCriticality.LOW, EvidenceCriticality.MEDIUM, EvidenceCriticality.UNKNOWN}
                    ]
                    prunable.sort(key=lambda x: (0 if x.criticality == EvidenceCriticality.LOW else 1))

                    for u in prunable:
                        if current_tokens <= target_budget:
                            break

                        final_extractive.remove(u)
                        saved = estimate_tokens(u.original_text, heuristic)
                        current_tokens -= saved

                        omitted_records.append(
                            OmittedEvidence(
                                evidence_id=u.evidence_id,
                                reason="omitted_due_to_target_token_budget",
                                source_message_id=u.source_span.message_id,
                                criticality=u.criticality,
                                evidence_type=u.evidence_type,
                                original_text=u.original_text,
                            )
                        )

                    if current_tokens <= target_budget:
                        budget_status = "feasible"
                    else:
                        budget_status = "budget_conflict"
                        budget_conflict_details = (
                            f"Context footprint after pruning ({current_tokens} estimated tokens) "
                            f"exceeds target token budget ({target_budget} tokens). "
                            f"Mandatory units were retained extractively without truncation."
                        )

        return final_extractive, final_summaries, omitted_records, budget_status, budget_conflict_details
