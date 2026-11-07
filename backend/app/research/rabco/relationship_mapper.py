"""Evidence Relationship Mapping Engine for Multi-Turn Context Optimization.

Project: Argus — Risk-Aware, Counterfactual-Bounded Context Optimization (RABCO).
Phase: 3 — Evidence Relationship Mapping.

Guarantees:
1. Immutability: EvidenceUnit instances are strictly read-only and never modified.
2. Historical Preservation: Antecedent values, instructions, and contradicted facts
   are never discarded, overwritten, or dropped. Unresolved contradictions are explicitly
   linked with CONTRADICTS relationships while keeping both units intact.
3. Strict Provenance Integrity: Every generated EvidenceRelationship references valid,
   existing source and target evidence identifiers present in the input collection.
4. Non-Suppressive Relationship Topology: Permits valid directed cycles, mutual dependencies,
   and circular references without artificial DAG truncation. Cycle detection is available
   exclusively as a diagnostic and reporting capability.
5. Configurable Thresholds: Parameter thresholds (token length, overlap minimums, identifier lengths)
   are managed in a centralized, documented configuration dataclass without scattered magic numbers.
6. Zero Hardcoding: Rule definitions and evaluators are decoupled from execution logic
   via an extensible, configurable RelationshipRuleRegistry.
7. Deduplication & Idempotency: Duplicate relations are rejected, and relationship IDs
   are deterministically generated based on source, target, and relationship classification.
"""

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import logging
import re
from typing import (
    Any,
    Callable,
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

from app.core.config import settings
from app.schemas.rabco import (
    ConversationMessage,
    EvidenceRelationship,
    EvidenceType,
    EvidenceUnit,
    RelationshipType,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class RelationshipValidationError(Exception):
    """Raised when an evidence relationship violates structural or relational constraints."""

    def __init__(self, message: str, relationship_id: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.relationship_id = relationship_id


# ============================================================================
# Centralized Parameter Configuration (Thresholds)
# ============================================================================

@dataclass(frozen=True)
class RelationshipThresholds:
    """Configurable parameter thresholds for evidence relationship evaluation.

    Architectural Principle:
    These numerical values are experimental operational parameters rather than
    fundamental domain invariants. They are explicitly exposed here to avoid
    scattered magic numbers, allow systematic parameter evaluation, and facilitate
    domain-specific tuning. None of these default values are asserted to be
    empirically optimal.
    """

    min_token_length: int = 3
    """Minimum character length for raw tokens to be considered substantive keywords.
    Controls noise filtering in token extraction."""

    min_identifier_length: int = 4
    """Minimum character length for standalone code identifiers without underscores.
    Controls false-positive suppression in entity co-reference."""

    min_shared_tokens_cue_match: int = 1
    """Minimum shared substantive tokens needed when an explicit cue word is present.
    Controls relational linking for cue-bearing rules (corrects, depends_on, conditions, exceptions, references)."""

    min_shared_tokens_supersede: int = 2
    """Minimum shared substantive tokens required to link an updated decision/policy
    to a prior decision/policy when no explicit supersede cue phrase is present."""

    min_shared_tokens_contradiction: int = 2
    """Minimum shared substantive tokens required to evaluate opposing polarity or conflicting numbers
    on shared entities when exact code identifiers are absent."""

    min_shared_tokens_related_to: int = 3
    """Minimum shared substantive tokens required between two units to trigger a fallback
    topical association (RELATED_TO). Higher values restrict topical links to denser overlap."""


# ============================================================================
# Mapping Context & Analysis Helpers
# ============================================================================

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

_RE_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b|\b`([^`]+)`|\"([^\"]+)\"")
_RE_NUMERICAL = re.compile(r"(?:[₹$€£]\s*)?\b\d+(?:[.,]\d+)?(?:\s*(?:k|m|b|ms|s|sec|seconds|tokens|usd|inr|gb|mb|kb|%))?\b", re.IGNORECASE)
_RE_NEGATIVE_POLARITY = re.compile(
    r"\b(?:not|never|no|cannot|can't|must\s+not|should\s+not|shouldn't|prohibit|prohibited|"
    r"strictly\s+avoid|forbid|forbidden|disallow|disallowed|don't|without)\b",
    re.IGNORECASE,
)
_RE_POSITIVE_POLARITY = re.compile(
    r"\b(?:must|shall|always|ensure|require|required|allow|allowed|permit|permitted|"
    r"enable|enabled|mandatory)\b",
    re.IGNORECASE,
)


def _extract_canonical_tokens(text: str, min_length: int = 3) -> Set[str]:
    """Extract lowercased substantive tokens excluding common stopwords.

    Args:
        text: Input text to extract tokens from.
        min_length: Minimum token character length.
    """
    raw_tokens = re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())
    return {tok for tok in raw_tokens if len(tok) >= min_length and tok not in _STOP_WORDS}


def _extract_identifiers(text: str, min_length: int = 3) -> Set[str]:
    """Extract code identifiers, constants, and quoted terms.

    Args:
        text: Input text to extract identifiers from.
        min_length: Minimum identifier character length.
    """
    results: Set[str] = set()
    for match in _RE_IDENTIFIER.finditer(text):
        quoted1 = match.group(1)
        quoted2 = match.group(2)
        raw_id = match.group(0)
        val = quoted1 or quoted2 or raw_id
        val_clean = val.strip("`\"'").lower()
        if len(val_clean) >= min_length and val_clean not in _STOP_WORDS:
            results.add(val_clean)
    return results


def _extract_numerical_values(text: str) -> Set[str]:
    """Extract normalized numerical and monetary mentions."""
    matches = _RE_NUMERICAL.findall(text)
    return {m.strip().lower() for m in matches if m.strip()}


def _detect_polarity(text: str) -> Optional[bool]:
    """Return True if positive/mandatory, False if negative/prohibitive, None if neutral."""
    has_neg = bool(_RE_NEGATIVE_POLARITY.search(text))
    has_pos = bool(_RE_POSITIVE_POLARITY.search(text))
    if has_neg and not has_pos:
        return False
    if has_pos and not has_neg:
        return True
    return None


@dataclass(frozen=True)
class MappingContext:
    """Precomputed analytical context over the input evidence and conversation."""

    evidence_by_id: Mapping[str, EvidenceUnit]
    evidence_order: Mapping[str, int]
    message_order: Mapping[str, int]
    tokens_by_id: Mapping[str, FrozenSet[str]]
    identifiers_by_id: Mapping[str, FrozenSet[str]]
    numbers_by_id: Mapping[str, FrozenSet[str]]
    polarity_by_id: Mapping[str, Optional[bool]]
    thresholds: RelationshipThresholds = field(default_factory=RelationshipThresholds)

    @classmethod
    def build(
        cls,
        evidence_units: Sequence[EvidenceUnit],
        messages: Optional[Sequence[ConversationMessage]] = None,
        thresholds: Optional[RelationshipThresholds] = None,
    ) -> "MappingContext":
        """Construct an immutable mapping context for relation evaluation."""
        effective_thresholds = thresholds or RelationshipThresholds()
        ev_by_id = {u.evidence_id: u for u in evidence_units}
        ev_order = {u.evidence_id: idx for idx, u in enumerate(evidence_units)}

        if messages:
            msg_order = {m.message_id: idx for idx, m in enumerate(messages)}
        else:
            # Infer message sequence order from first appearance in evidence units
            msg_order = {}
            for u in evidence_units:
                mid = u.source_span.message_id
                if mid not in msg_order:
                    msg_order[mid] = len(msg_order)

        tokens_by_id = {
            u.evidence_id: frozenset(_extract_canonical_tokens(u.original_text, min_length=effective_thresholds.min_token_length))
            for u in evidence_units
        }
        identifiers_by_id = {
            u.evidence_id: frozenset(_extract_identifiers(u.original_text, min_length=effective_thresholds.min_token_length))
            for u in evidence_units
        }
        numbers_by_id = {
            u.evidence_id: frozenset(_extract_numerical_values(u.original_text))
            for u in evidence_units
        }
        polarity_by_id = {
            u.evidence_id: _detect_polarity(u.original_text)
            for u in evidence_units
        }

        return cls(
            evidence_by_id=ev_by_id,
            evidence_order=ev_order,
            message_order=msg_order,
            tokens_by_id=tokens_by_id,
            identifiers_by_id=identifiers_by_id,
            numbers_by_id=numbers_by_id,
            polarity_by_id=polarity_by_id,
            thresholds=effective_thresholds,
        )

    def is_temporally_after(self, source_id: str, target_id: str) -> bool:
        """Check if source evidence unit occurred in a subsequent message or span."""
        src_unit = self.evidence_by_id[source_id]
        tgt_unit = self.evidence_by_id[target_id]

        src_msg_idx = self.message_order.get(src_unit.source_span.message_id, 0)
        tgt_msg_idx = self.message_order.get(tgt_unit.source_span.message_id, 0)

        if src_msg_idx > tgt_msg_idx:
            return True
        if src_msg_idx == tgt_msg_idx:
            return src_unit.source_span.start_offset >= tgt_unit.source_span.end_offset
        return False


# ============================================================================
# Rule Registry & Definition Pattern
# ============================================================================

RuleEvaluator = Callable[[EvidenceUnit, EvidenceUnit, MappingContext], Optional[str]]


@dataclass(frozen=True)
class RelationshipRule:
    """Configurable rule definition for relationship classification.

    Attributes:
        name: Unique mnemonic name for the rule.
        relationship_type: Target RelationshipType enum value.
        evaluator: Pure function evaluating (source, target, context) -> Optional[description].
        description_template: Format string explaining the relationship rationale.
        allowed_source_types: Optional set of permitted EvidenceType values for source.
        allowed_target_types: Optional set of permitted EvidenceType values for target.
        temporal_requirement: Directional constraint ("source_after_target", "target_after_source", "none").
        is_symmetric: If True, candidate evaluation treats (A, B) symmetrically.
        priority: Relative evaluation priority (lower value executes first).
        parameters: Optional dictionary of rule-specific parameters and experimental thresholds.
    """

    name: str
    relationship_type: RelationshipType
    evaluator: RuleEvaluator
    description_template: str = "{source_id} {rel_type} {target_id} based on {rule_name}"
    allowed_source_types: Optional[FrozenSet[EvidenceType]] = None
    allowed_target_types: Optional[FrozenSet[EvidenceType]] = None
    temporal_requirement: str = "none"
    is_symmetric: bool = False
    priority: int = 100
    parameters: Mapping[str, Any] = field(default_factory=dict)


class RelationshipRuleRegistry:
    """Centralized, extensible registry of relationship detection rules."""

    def __init__(self) -> None:
        self._rules: List[RelationshipRule] = []

    def register(self, rule: RelationshipRule) -> None:
        """Register a new relationship rule."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority)

    def get_rules(self) -> Tuple[RelationshipRule, ...]:
        """Return registered rules sorted by execution priority."""
        return tuple(self._rules)


# ============================================================================
# Built-in Deterministic Evaluator Functions
# ============================================================================

_RE_CORRECTION_CUES = re.compile(
    r"\b(?:instead\s+of|changed?\s+(?:to|from)|updated?\s+to|correction:?|"
    r"replace\s+.*?\s+with|scratch\s+that|disregard\s+that|not\s+.*?\s+but\s+rather|"
    r"actually,?\s+(?:use|set|make))\b",
    re.IGNORECASE,
)

_RE_DEPENDENCY_CUES = re.compile(
    r"\b(?:depends\s+on|dependent\s+on|requires|prerequisite|based\s+on|"
    r"needs|after\s+completing|following\s+the|contingent\s+on)\b",
    re.IGNORECASE,
)

_RE_SUPERSEDE_CUES = re.compile(
    r"\b(?:we\s+will\s+use|new\s+policy|going\s+forward|updated\s+requirement|"
    r"moving\s+forward|now\s+use|switch\s+to|supersedes|overrides)\b",
    re.IGNORECASE,
)

_RE_EXCEPTION_CUES = re.compile(
    r"\b(?:except\s+(?:when|for|if)|with\s+the\s+exception\s+of|excluding|"
    r"unless|save\s+for|excepting)\b",
    re.IGNORECASE,
)

_RE_CONDITION_CUES = re.compile(
    r"\b(?:if|when|provided\s+that|in\s+case|as\s+long\s+as|assuming\s+that)\b",
    re.IGNORECASE,
)

_RE_APPLIES_TO_CUES = re.compile(
    r"\b(?:applies\s+to|for\s+all|in\s+the|applicable\s+to|target\s+is|scoped\s+to)\b",
    re.IGNORECASE,
)

_RE_ANAPHORA_CUES = re.compile(
    r"\b(?:this\s+(?:setting|rule|approach|limit|model|prompt|policy)|"
    r"that\s+(?:setting|rule|approach|limit|model|prompt|policy)|"
    r"the\s+aforementioned|the\s+same|these\s+parameters)\b",
    re.IGNORECASE,
)


def evaluate_corrects(src: EvidenceUnit, tgt: EvidenceUnit, ctx: MappingContext) -> Optional[str]:
    """Evaluate whether source explicitly corrects target."""
    if not ctx.is_temporally_after(src.evidence_id, tgt.evidence_id):
        return None

    src_has_correction_type = src.evidence_type == EvidenceType.CORRECTION
    src_has_correction_cue = bool(_RE_CORRECTION_CUES.search(src.original_text))

    if not (src_has_correction_type or src_has_correction_cue):
        return None

    # Check shared entities, identifiers, or numerical overlap
    shared_ids = ctx.identifiers_by_id[src.evidence_id] & ctx.identifiers_by_id[tgt.evidence_id]
    shared_tokens = ctx.tokens_by_id[src.evidence_id] & ctx.tokens_by_id[tgt.evidence_id]

    src_numbers = ctx.numbers_by_id[src.evidence_id]
    tgt_numbers = ctx.numbers_by_id[tgt.evidence_id]
    has_numerical_shift = bool(src_numbers and tgt_numbers and src_numbers != tgt_numbers)

    min_shared = ctx.thresholds.min_shared_tokens_cue_match
    if shared_ids or len(shared_tokens) >= min_shared or has_numerical_shift:
        detail = f"updated value from antecedent '{tgt.evidence_id}'" if has_numerical_shift else f"shared context {sorted(shared_ids or shared_tokens)[:2]}"
        return f"Evidence '{src.evidence_id}' explicitly corrects '{tgt.evidence_id}' ({detail})"

    return None


def evaluate_supersedes(src: EvidenceUnit, tgt: EvidenceUnit, ctx: MappingContext) -> Optional[str]:
    """Evaluate whether a newer decision/instruction supersedes an earlier one."""
    if not ctx.is_temporally_after(src.evidence_id, tgt.evidence_id):
        return None

    # Target must be an earlier decision, instruction, constraint, or preference
    if tgt.evidence_type not in {
        EvidenceType.DECISION,
        EvidenceType.INSTRUCTION,
        EvidenceType.CONSTRAINT,
        EvidenceType.PREFERENCE,
    }:
        return None

    has_supersede_cue = bool(_RE_SUPERSEDE_CUES.search(src.original_text))
    shared_ids = ctx.identifiers_by_id[src.evidence_id] & ctx.identifiers_by_id[tgt.evidence_id]
    shared_tokens = ctx.tokens_by_id[src.evidence_id] & ctx.tokens_by_id[tgt.evidence_id]

    min_shared_cue = ctx.thresholds.min_shared_tokens_cue_match
    min_shared_no_cue = ctx.thresholds.min_shared_tokens_supersede

    if (has_supersede_cue and (shared_ids or len(shared_tokens) >= min_shared_cue)) or (shared_ids and len(shared_tokens) >= min_shared_no_cue):
        return f"Evidence '{src.evidence_id}' supersedes earlier policy '{tgt.evidence_id}' on subject {sorted(shared_ids or shared_tokens)[:2]}"

    return None


def evaluate_contradicts(src: EvidenceUnit, tgt: EvidenceUnit, ctx: MappingContext) -> Optional[str]:
    """Evaluate whether two evidence units express conflicting assertions."""
    # Check polarity opposition on shared substantive topic
    src_pol = ctx.polarity_by_id[src.evidence_id]
    tgt_pol = ctx.polarity_by_id[tgt.evidence_id]

    shared_ids = ctx.identifiers_by_id[src.evidence_id] & ctx.identifiers_by_id[tgt.evidence_id]
    shared_tokens = ctx.tokens_by_id[src.evidence_id] & ctx.tokens_by_id[tgt.evidence_id]

    min_shared = ctx.thresholds.min_shared_tokens_contradiction
    if not (shared_ids or len(shared_tokens) >= min_shared):
        return None

    # Case 1: Direct polarity opposition (e.g. prohibited vs required)
    if src_pol is not None and tgt_pol is not None and src_pol != tgt_pol:
        return f"Opposing polarity on shared topic {sorted(shared_ids or shared_tokens)[:2]}"

    # Case 2: Conflicting numbers on same exact identifier without explicit correction cues
    src_numbers = ctx.numbers_by_id[src.evidence_id]
    tgt_numbers = ctx.numbers_by_id[tgt.evidence_id]
    has_conflicting_numbers = bool(src_numbers and tgt_numbers and src_numbers.isdisjoint(tgt_numbers))

    src_has_correction = bool(_RE_CORRECTION_CUES.search(src.original_text)) or src.evidence_type == EvidenceType.CORRECTION
    if has_conflicting_numbers and shared_ids and not src_has_correction:
        return f"Conflicting numerical values {sorted(src_numbers)} vs {sorted(tgt_numbers)} on entity {sorted(shared_ids)}"

    return None


def evaluate_depends_on(src: EvidenceUnit, tgt: EvidenceUnit, ctx: MappingContext) -> Optional[str]:
    """Evaluate whether source explicitly declares a dependency on target."""
    if not ctx.is_temporally_after(src.evidence_id, tgt.evidence_id):
        return None

    if not _RE_DEPENDENCY_CUES.search(src.original_text):
        return None

    shared_ids = ctx.identifiers_by_id[src.evidence_id] & ctx.identifiers_by_id[tgt.evidence_id]
    shared_tokens = ctx.tokens_by_id[src.evidence_id] & ctx.tokens_by_id[tgt.evidence_id]

    min_shared = ctx.thresholds.min_shared_tokens_cue_match
    if shared_ids or len(shared_tokens) >= min_shared:
        return f"Evidence '{src.evidence_id}' declares dependency on '{tgt.evidence_id}' for {sorted(shared_ids or shared_tokens)[:2]}"

    return None


def evaluate_conditional_on(src: EvidenceUnit, tgt: EvidenceUnit, ctx: MappingContext) -> Optional[str]:
    """Evaluate whether source rule or action is conditional on target condition."""
    is_tgt_condition = (
        tgt.evidence_type == EvidenceType.CONDITION
        or bool(_RE_CONDITION_CUES.search(tgt.original_text))
    )
    if not is_tgt_condition:
        return None

    if src.evidence_type not in {
        EvidenceType.INSTRUCTION,
        EvidenceType.DECISION,
        EvidenceType.CONSTRAINT,
        EvidenceType.GOAL,
        EvidenceType.FACT,
    }:
        return None

    shared_ids = ctx.identifiers_by_id[src.evidence_id] & ctx.identifiers_by_id[tgt.evidence_id]
    shared_tokens = ctx.tokens_by_id[src.evidence_id] & ctx.tokens_by_id[tgt.evidence_id]

    min_shared = ctx.thresholds.min_shared_tokens_cue_match
    if shared_ids or len(shared_tokens) >= min_shared:
        return f"Evidence '{src.evidence_id}' is conditional on state '{tgt.evidence_id}'"

    return None


def evaluate_exception_to(src: EvidenceUnit, tgt: EvidenceUnit, ctx: MappingContext) -> Optional[str]:
    """Evaluate whether source represents an exception to general rule/constraint target."""
    is_src_exception = (
        src.evidence_type == EvidenceType.EXCEPTION
        or bool(_RE_EXCEPTION_CUES.search(src.original_text))
    )
    if not is_src_exception:
        return None

    if tgt.evidence_type not in {
        EvidenceType.CONSTRAINT,
        EvidenceType.INSTRUCTION,
        EvidenceType.FACT,
        EvidenceType.DECISION,
    }:
        return None

    shared_ids = ctx.identifiers_by_id[src.evidence_id] & ctx.identifiers_by_id[tgt.evidence_id]
    shared_tokens = ctx.tokens_by_id[src.evidence_id] & ctx.tokens_by_id[tgt.evidence_id]

    min_shared = ctx.thresholds.min_shared_tokens_cue_match
    if shared_ids or len(shared_tokens) >= min_shared:
        return f"Exception in '{src.evidence_id}' carves out from general directive in '{tgt.evidence_id}'"

    return None


def evaluate_applies_to(src: EvidenceUnit, tgt: EvidenceUnit, ctx: MappingContext) -> Optional[str]:
    """Evaluate whether source constraint/directive specifies applicability to target."""
    if src.evidence_type not in {
        EvidenceType.CONSTRAINT,
        EvidenceType.INSTRUCTION,
        EvidenceType.PREFERENCE,
        EvidenceType.DECISION,
    }:
        return None

    has_applies_cue = bool(_RE_APPLIES_TO_CUES.search(src.original_text))
    shared_ids = ctx.identifiers_by_id[src.evidence_id] & ctx.identifiers_by_id[tgt.evidence_id]

    if has_applies_cue and shared_ids:
        return f"Directive '{src.evidence_id}' explicitly applies to entity/target '{tgt.evidence_id}'"

    return None


def evaluate_refers_to(src: EvidenceUnit, tgt: EvidenceUnit, ctx: MappingContext) -> Optional[str]:
    """Evaluate whether source contains an anaphoric reference to target."""
    if not ctx.is_temporally_after(src.evidence_id, tgt.evidence_id):
        return None

    if not _RE_ANAPHORA_CUES.search(src.original_text):
        return None

    shared_ids = ctx.identifiers_by_id[src.evidence_id] & ctx.identifiers_by_id[tgt.evidence_id]
    shared_tokens = ctx.tokens_by_id[src.evidence_id] & ctx.tokens_by_id[tgt.evidence_id]

    min_shared = ctx.thresholds.min_shared_tokens_cue_match
    if shared_ids or len(shared_tokens) >= min_shared:
        return f"Evidence '{src.evidence_id}' refers back to antecedent in '{tgt.evidence_id}'"

    return None


def evaluate_same_entity(src: EvidenceUnit, tgt: EvidenceUnit, ctx: MappingContext) -> Optional[str]:
    """Evaluate whether source and target mention an identical canonical entity/code identifier."""
    if src.evidence_id == tgt.evidence_id:
        return None

    shared_ids = ctx.identifiers_by_id[src.evidence_id] & ctx.identifiers_by_id[tgt.evidence_id]
    if shared_ids:
        min_id_len = ctx.thresholds.min_identifier_length
        substantive = {i for i in shared_ids if len(i) >= min_id_len or "_" in i}
        if substantive:
            return f"Shares canonical identifier {sorted(substantive)[:2]}"

    return None


def evaluate_related_to(src: EvidenceUnit, tgt: EvidenceUnit, ctx: MappingContext) -> Optional[str]:
    """Fallback topical association based on significant keyword overlap."""
    if src.evidence_id == tgt.evidence_id:
        return None

    shared_tokens = ctx.tokens_by_id[src.evidence_id] & ctx.tokens_by_id[tgt.evidence_id]
    min_tokens = ctx.thresholds.min_shared_tokens_related_to
    if len(shared_tokens) >= min_tokens:
        return f"Topical semantic association via shared terms {sorted(shared_tokens)[:3]}"

    return None


# ============================================================================
# Default Registry Initialization
# ============================================================================

def create_default_registry(
    thresholds: Optional[RelationshipThresholds] = None,
) -> RelationshipRuleRegistry:
    """Build and populate the default zero-hardcoded rule registry.

    Args:
        thresholds: Optional configured parameter thresholds. If omitted,
                    default thresholds are applied.
    """
    effective_thresholds = thresholds or RelationshipThresholds()
    registry = RelationshipRuleRegistry()

    # Priority 10: Strict corrections and exceptions
    registry.register(RelationshipRule(
        name="rule_corrects_explicit",
        relationship_type=RelationshipType.CORRECTS,
        evaluator=evaluate_corrects,
        temporal_requirement="source_after_target",
        priority=10,
        parameters={"min_shared_tokens": effective_thresholds.min_shared_tokens_cue_match},
    ))

    registry.register(RelationshipRule(
        name="rule_exception_carveout",
        relationship_type=RelationshipType.EXCEPTION_TO,
        evaluator=evaluate_exception_to,
        priority=15,
        parameters={"min_shared_tokens": effective_thresholds.min_shared_tokens_cue_match},
    ))

    # Priority 20: Supersedes & Contradictions
    registry.register(RelationshipRule(
        name="rule_supersedes_policy",
        relationship_type=RelationshipType.SUPERSEDES,
        evaluator=evaluate_supersedes,
        temporal_requirement="source_after_target",
        priority=20,
        parameters={
            "min_shared_tokens_cue": effective_thresholds.min_shared_tokens_cue_match,
            "min_shared_tokens_no_cue": effective_thresholds.min_shared_tokens_supersede,
        },
    ))

    registry.register(RelationshipRule(
        name="rule_contradiction_detection",
        relationship_type=RelationshipType.CONTRADICTS,
        evaluator=evaluate_contradicts,
        is_symmetric=True,
        priority=25,
        parameters={"min_shared_tokens": effective_thresholds.min_shared_tokens_contradiction},
    ))

    # Priority 30: Dependencies & Conditions
    registry.register(RelationshipRule(
        name="rule_depends_on_explicit",
        relationship_type=RelationshipType.DEPENDS_ON,
        evaluator=evaluate_depends_on,
        temporal_requirement="source_after_target",
        priority=30,
        parameters={"min_shared_tokens": effective_thresholds.min_shared_tokens_cue_match},
    ))

    registry.register(RelationshipRule(
        name="rule_conditional_applicability",
        relationship_type=RelationshipType.CONDITIONAL_ON,
        evaluator=evaluate_conditional_on,
        priority=35,
        parameters={"min_shared_tokens": effective_thresholds.min_shared_tokens_cue_match},
    ))

    # Priority 40: Scoping & References
    registry.register(RelationshipRule(
        name="rule_applies_to_scope",
        relationship_type=RelationshipType.APPLIES_TO,
        evaluator=evaluate_applies_to,
        priority=40,
    ))

    registry.register(RelationshipRule(
        name="rule_refers_to_anaphora",
        relationship_type=RelationshipType.REFERS_TO,
        evaluator=evaluate_refers_to,
        temporal_requirement="source_after_target",
        priority=45,
        parameters={"min_shared_tokens": effective_thresholds.min_shared_tokens_cue_match},
    ))

    # Priority 50: Entity Equivalence
    registry.register(RelationshipRule(
        name="rule_same_entity_identifier",
        relationship_type=RelationshipType.SAME_ENTITY,
        evaluator=evaluate_same_entity,
        is_symmetric=True,
        priority=50,
        parameters={"min_identifier_length": effective_thresholds.min_identifier_length},
    ))

    # Priority 90: Fallback Topical Overlap
    registry.register(RelationshipRule(
        name="rule_related_to_topical",
        relationship_type=RelationshipType.RELATED_TO,
        evaluator=evaluate_related_to,
        is_symmetric=True,
        priority=90,
        parameters={"min_shared_tokens": effective_thresholds.min_shared_tokens_related_to},
    ))

    return registry


# ============================================================================
# Relationship Mapper Core Engine
# ============================================================================

class RelationshipMapper:
    """Identifies and validates relationships between atomic evidence units.

    Guarantees:
    - Never suppresses valid directed relationship edges due to cycle detection.
    - Preserves circular dependencies, mutual corrections, and unresolved contradictions.
    - Emits deterministic, provenance-linked EvidenceRelationship instances.
    - Diagnostic inspection of cycles via `find_directed_cycles`.
    """

    def __init__(
        self,
        registry: Optional[RelationshipRuleRegistry] = None,
        thresholds: Optional[RelationshipThresholds] = None,
        enable_llm_assistance: bool = False,
    ) -> None:
        self.thresholds = thresholds or RelationshipThresholds()
        self.registry = registry or create_default_registry(self.thresholds)
        self.enable_llm_assistance = enable_llm_assistance

    def register_rule(self, rule: RelationshipRule) -> None:
        """Register a custom relationship rule dynamically."""
        self.registry.register(rule)

    @staticmethod
    def generate_relationship_id(source_id: str, target_id: str, rel_type: RelationshipType) -> str:
        """Derive a stable, deterministic relationship identifier."""
        raw_key = f"{source_id}:{target_id}:{rel_type.value}"
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:12]
        return f"rel_{rel_type.value[:4]}_{digest}"

    def map_relationships(
        self,
        evidence_units: Sequence[EvidenceUnit],
        messages: Optional[Sequence[ConversationMessage]] = None,
        thresholds: Optional[RelationshipThresholds] = None,
    ) -> Tuple[EvidenceRelationship, ...]:
        """Synchronously map relationships using registered deterministic rules.

        Preserves all valid directed edges, including directed cycles, mutual dependencies,
        and unresolved contradictions. Does not suppress edges based on acyclicity constraints.
        """
        if not evidence_units:
            return ()

        effective_thresholds = thresholds or self.thresholds
        context = MappingContext.build(evidence_units, messages, effective_thresholds)
        relationships: List[EvidenceRelationship] = []
        seen_edges: Set[Tuple[str, str, RelationshipType]] = set()

        rules = self.registry.get_rules()

        # Pairwise evaluation across all evidence units
        for i, src in enumerate(evidence_units):
            for j, tgt in enumerate(evidence_units):
                # Domain invariant: disallow self-referential relationships
                if i == j:
                    continue

                for rule in rules:
                    # Enforce type constraints if defined
                    if rule.allowed_source_types and src.evidence_type not in rule.allowed_source_types:
                        continue
                    if rule.allowed_target_types and tgt.evidence_type not in rule.allowed_target_types:
                        continue

                    # Canonicalize symmetric relations to avoid duplicate mirror edges
                    edge_key = (
                        (min(src.evidence_id, tgt.evidence_id), max(src.evidence_id, tgt.evidence_id), rule.relationship_type)
                        if rule.is_symmetric
                        else (src.evidence_id, tgt.evidence_id, rule.relationship_type)
                    )
                    if edge_key in seen_edges:
                        continue

                    # Execute rule evaluator
                    explanation = rule.evaluator(src, tgt, context)
                    if explanation is None:
                        continue

                    rel_id = self.generate_relationship_id(src.evidence_id, tgt.evidence_id, rule.relationship_type)

                    relationship = EvidenceRelationship(
                        relationship_id=rel_id,
                        source_evidence_id=src.evidence_id,
                        target_evidence_id=tgt.evidence_id,
                        relationship_type=rule.relationship_type,
                        description=explanation,
                    )

                    relationships.append(relationship)
                    seen_edges.add(edge_key)

                    # Once a rule of this priority matches, move to the next candidate pair
                    break

        return self.validate_relationships(tuple(relationships), evidence_units)

    async def map_relationships_async(
        self,
        evidence_units: Sequence[EvidenceUnit],
        messages: Optional[Sequence[ConversationMessage]] = None,
        thresholds: Optional[RelationshipThresholds] = None,
    ) -> Tuple[EvidenceRelationship, ...]:
        """Asynchronously map relationships with optional local Ollama assistance."""
        deterministic_rels = self.map_relationships(evidence_units, messages, thresholds)

        if not self.enable_llm_assistance or not evidence_units:
            return deterministic_rels

        llm_rels = await self._query_llm_relationships(evidence_units, messages, deterministic_rels)
        all_rels = tuple(deterministic_rels) + tuple(llm_rels)
        return self.validate_relationships(all_rels, evidence_units)

    def validate_relationships(
        self,
        relationships: Sequence[EvidenceRelationship],
        evidence_units: Sequence[EvidenceUnit],
    ) -> Tuple[EvidenceRelationship, ...]:
        """Strictly validate relationship integrity, provenance, and uniqueness.

        Invariants enforced:
        1. Existence: source_evidence_id and target_evidence_id must exist in input evidence.
        2. Non-self-referential: source_evidence_id != target_evidence_id.
        3. Unique relationship_id across the result set.
        4. Deduplication: At most one edge per (source_id, target_id, relationship_type).
        """
        valid_evidence_ids = {u.evidence_id for u in evidence_units}
        seen_rel_ids: Set[str] = set()
        seen_pairs: Set[Tuple[str, str, RelationshipType]] = set()
        validated: List[EvidenceRelationship] = []

        for rel in relationships:
            # 1. Existence check
            if rel.source_evidence_id not in valid_evidence_ids:
                raise RelationshipValidationError(
                    f"source_evidence_id '{rel.source_evidence_id}' does not exist in input evidence.",
                    relationship_id=rel.relationship_id,
                )
            if rel.target_evidence_id not in valid_evidence_ids:
                raise RelationshipValidationError(
                    f"target_evidence_id '{rel.target_evidence_id}' does not exist in input evidence.",
                    relationship_id=rel.relationship_id,
                )

            # 2. Non-self-referential check
            if rel.source_evidence_id == rel.target_evidence_id:
                raise RelationshipValidationError(
                    f"Self-referential relationship detected: '{rel.source_evidence_id}' -> '{rel.target_evidence_id}'.",
                    relationship_id=rel.relationship_id,
                )

            # 3. Unique relationship ID check
            if rel.relationship_id in seen_rel_ids:
                raise RelationshipValidationError(
                    f"Duplicate relationship_id '{rel.relationship_id}'.",
                    relationship_id=rel.relationship_id,
                )

            # 4. Duplicate pair check
            pair_key = (rel.source_evidence_id, rel.target_evidence_id, rel.relationship_type)
            if pair_key in seen_pairs:
                continue

            seen_rel_ids.add(rel.relationship_id)
            seen_pairs.add(pair_key)
            validated.append(rel)

        return tuple(validated)

    @staticmethod
    def find_directed_cycles(
        relationships: Sequence[EvidenceRelationship],
        filter_type: Optional[RelationshipType] = None,
    ) -> List[Tuple[str, ...]]:
        """Diagnostic utility to inspect directed cycles in a relationship set.

        Guarantees:
        - Read-only diagnostics: Never alters or suppresses relationships.
        - Identifies elementary cycles (e.g. A -> B -> A or A -> B -> C -> A).
        - Supports filtering to a specific RelationshipType (e.g. DEPENDS_ON).

        Args:
            relationships: Sequence of relationships to inspect.
            filter_type: Optional RelationshipType filter to isolate cycles in one category.

        Returns:
            List of cycles, each represented as a tuple of evidence identifiers.
        """
        adj: Dict[str, Set[str]] = defaultdict(set)
        for r in relationships:
            if filter_type is None or r.relationship_type == filter_type:
                adj[r.source_evidence_id].add(r.target_evidence_id)

        cycles: List[Tuple[str, ...]] = []
        visited: Set[str] = set()
        rec_stack: List[str] = []
        in_stack: Set[str] = set()

        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.append(node)
            in_stack.add(node)

            for neighbor in adj.get(node, ()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in in_stack:
                    idx = rec_stack.index(neighbor)
                    cycle_nodes = tuple(rec_stack[idx:]) + (neighbor,)
                    cycles.append(cycle_nodes)

            rec_stack.pop()
            in_stack.remove(node)

        for n in list(adj.keys()):
            if n not in visited:
                dfs(n)

        return cycles

    async def _query_llm_relationships(
        self,
        evidence_units: Sequence[EvidenceUnit],
        messages: Optional[Sequence[ConversationMessage]],
        existing_rels: Sequence[EvidenceRelationship],
    ) -> List[EvidenceRelationship]:
        """Query local Ollama instance for high-level semantic cross-turn relationships."""
        existing_pairs = {
            (r.source_evidence_id, r.target_evidence_id, r.relationship_type)
            for r in existing_rels
        }
        valid_ids = {u.evidence_id for u in evidence_units}
        units_manifest = [
            {
                "id": u.evidence_id,
                "type": u.evidence_type.value,
                "text": u.original_text,
                "msg": u.source_span.message_id,
            }
            for u in evidence_units
        ]

        allowed_types = [t.value for t in RelationshipType]
        prompt = (
            "Analyze these atomic evidence units extracted from a multi-turn conversation and identify "
            "subtle or implicit relationships between them.\n"
            f"Allowed relationship types: {allowed_types}\n"
            "Return a JSON array of objects with keys: source_id, target_id, type, explanation.\n"
            "Only reference exact IDs from the manifest. Return ONLY valid JSON.\n\n"
            f"Evidence units manifest:\n{json.dumps(units_manifest, indent=2)}"
        )

        payload = {
            "model": settings.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
            },
        }

        discovered: List[EvidenceRelationship] = []
        try:
            async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT_SECONDS) as client:
                resp = await client.post(f"{settings.OLLAMA_BASE_URL}/api/generate", json=payload)
                if resp.status_code != 200:
                    return []

                data = resp.json()
                raw_resp = data.get("response", "")
                parsed = json.loads(raw_resp)
                items = parsed if isinstance(parsed, list) else parsed.get("relationships", [])

                for item in items:
                    if not isinstance(item, dict):
                        continue

                    src_id = item.get("source_id")
                    tgt_id = item.get("target_id")
                    rel_type_str = item.get("type")
                    explanation = item.get("explanation", "Discovered via semantic LLM analysis")

                    if not src_id or not tgt_id or not rel_type_str:
                        continue
                    if src_id not in valid_ids or tgt_id not in valid_ids or src_id == tgt_id:
                        continue

                    try:
                        rel_type = RelationshipType(rel_type_str.lower())
                    except ValueError:
                        continue

                    pair_key = (src_id, tgt_id, rel_type)
                    if pair_key in existing_pairs:
                        continue

                    rel_id = self.generate_relationship_id(src_id, tgt_id, rel_type)
                    discovered.append(
                        EvidenceRelationship(
                            relationship_id=rel_id,
                            source_evidence_id=src_id,
                            target_evidence_id=tgt_id,
                            relationship_type=rel_type,
                            description=f"LLM-Assisted: {explanation}",
                        )
                    )
                    existing_pairs.add(pair_key)

        except Exception as exc:
            logger.warning("Optional LLM relationship assistance failed gracefully: %s", exc)

        return discovered
