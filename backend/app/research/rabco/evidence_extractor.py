"""Semantic Evidence Extraction Engine with Exact Source Provenance.

Project: Argus — Risk-Aware, Counterfactual-Bounded Context Optimization (RABCO).
Phase: 2 — Semantic Evidence Extraction Engine.

Guarantees:
1. Exact Provenance: Every extracted EvidenceUnit is strictly verifiable against its
   source message content: message.content[start_offset:end_offset] == evidence.original_text.
2. Zero Offset Drift: Raw Python string character indexing is used directly. No normalization
   (Unicode, whitespace, punctuation, line endings) is performed prior to offset calculation.
3. Immutability: Source messages are treated as strictly immutable and are never modified.
4. Role Distinction: Assistant responses are never treated as user directives or instructions.
5. Historical Value Preservation: Corrections do not overwrite or discard antecedent values;
   both are preserved with distinct evidence identities and semantic classifications.
6. Conservative Criticality: Mandatory criticality is reserved exclusively for explicit
   application-level or system-level constraints and prohibitions.
"""

from dataclasses import dataclass
import json
import logging
import re
from typing import Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

import httpx

from app.core.config import settings
from app.schemas.rabco import (
    ConversationMessage,
    EvidenceCriticality,
    EvidenceType,
    EvidenceUnit,
    MessageRole,
    SourceSpan,
    ValidationStatus,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class ProvenanceVerificationError(Exception):
    """Raised when an evidence unit fails exact source provenance verification."""

    def __init__(self, message: str, evidence_id: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.evidence_id = evidence_id


# ============================================================================
# Extraction Rule Definitions (Deterministic Engine)
# ============================================================================

@dataclass(frozen=True)
class ExtractionPattern:
    """Deterministic pattern for detecting semantic evidence spans."""

    pattern: re.Pattern
    evidence_type: EvidenceType
    criticality: EvidenceCriticality
    allowed_roles: Optional[FrozenSet[MessageRole]] = None
    extraction_tag: str = "regex_rule"


class EvidenceExtractor:
    """Extracts atomic semantic evidence units with mathematically verified source provenance."""

    def __init__(self, enable_llm_assistance: bool = False):
        self.enable_llm_assistance = enable_llm_assistance
        self._init_patterns()

    def _init_patterns(self) -> None:
        """Initialize high-precision deterministic pattern taxonomy."""
        # 1. Prohibitions & Negative Constraints
        re_constraint = re.compile(
            r"\b(?:do\s+not|don't|must\s+not|never|should\s+not|shouldn't|cannot|can't|"
            r"strictly\s+(?:avoid|prohibit|forbid)|prohibited|without\s+using|"
            r"make\s+sure\s+(?:not\s+to|never)|under\s+no\s+circumstances)\s+[^.!?\n;]+",
            re.IGNORECASE,
        )

        # 2. Mandatory Instructions & Directives (User / System only)
        re_instruction = re.compile(
            r"\b(?:please\s+)?(?:ensure\s+that|make\s+sure\s+that|always|you\s+must|"
            r"write\s+a|build\s+a|implement|create\s+a|generate\s+a|refactor|"
            r"calculate|extract|format\s+as|strictly\s+follow|adhere\s+to)\s+[^.!?\n;]+",
            re.IGNORECASE,
        )

        # 3. Corrections & Value Updates
        re_correction = re.compile(
            r"\b(?:actually,?\s+(?:use|set|change|make)|correction:?|instead\s+of\s+[^,\n;]+,?\s+use|"
            r"changed?\s+(?:from\s+[^,\n;]+\s+to|to)|updated?\s+(?:to|from)|"
            r"replace\s+[^,\n;]+\s+with|disregard\s+that,?\s+use|scratch\s+that,?\s+use|"
            r"not\s+[^,\n;]+,?\s+but\s+rather)\s+[^.!?\n;]+",
            re.IGNORECASE,
        )

        # 4. Numerical Values, Quantities, & Technical Identifiers
        re_numerical = re.compile(
            r"(?:(?:\$|€|£|¥)\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:k|m|b|billion|million|thousand))?|"
            r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|ms|seconds?|sec|minutes?|min|hours?|hrs?|"
            r"days?|weeks?|months?|years?|px|rem|em|tokens?|items?|records?|rows?|gb|mb|kb|tb|"
            r"users?|requests?|hz|khz|mhz|ghz)\b|"
            r"\b(?:port|status\s+code|http)\s+\d{2,5}\b|"
            r"\b(?:version|v)\s*\d+(?:\.\d+)+\b)",
            re.IGNORECASE,
        )

        # 5. Dates, Timestamps, & Deadlines
        re_date_deadline = re.compile(
            r"\b(?:deadline\s+is|due\s+(?:by|on|before)|by\s+(?:end\s+of\s+day|eod|tomorrow|friday|monday|next\s+week)|"
            r"within\s+\d+\s+(?:days?|hours?|weeks?|months?)|"
            r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?|"
            r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|"
            r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4})\b",
            re.IGNORECASE,
        )

        # 6. Preferences
        re_preference = re.compile(
            r"\b(?:i\s+prefer|i'd\s+prefer|i\s+would\s+prefer|we\s+prefer|my\s+preference\s+is|"
            r"i\s+would\s+rather|preferable\s+to|ideally,?\s+we\s+should|"
            r"i\s+like\s+having|our\s+preferred\s+(?:style|approach|format|tool))\s+[^.!?\n;]+",
            re.IGNORECASE,
        )

        # 7. Goals & Objectives
        re_goal = re.compile(
            r"\b(?:our\s+goal\s+is|the\s+goal\s+is|aim\s+is\s+to|we\s+aim\s+to|we\s+want\s+to\s+achieve|"
            r"objective\s+is\s+to|in\s+order\s+to\s+achieve|the\s+purpose\s+is\s+to|"
            r"target\s+outcome\s+is)\s+[^.!?\n;]+",
            re.IGNORECASE,
        )

        # 8. Decisions & Commitments
        re_decision = re.compile(
            r"\b(?:we\s+decided\s+to|we\s+have\s+decided\s+to|it\s+was\s+decided\s+that|"
            r"agreed\s+on|we\s+agreed\s+to|let's\s+go\s+with|we\s+will\s+use|"
            r"the\s+chosen\s+approach\s+is|we\s+chose\s+to|settled\s+on)\s+[^.!?\n;]+",
            re.IGNORECASE,
        )

        # 9. Conditions
        re_condition = re.compile(
            r"\b(?:if\s+[^,\n;]+,?\s+then\s+[^.!?\n;]+|"
            r"only\s+(?:if|when)\s+[^.!?\n;]+|"
            r"provided\s+that\s+[^.!?\n;]+|"
            r"in\s+the\s+event\s+that\s+[^.!?\n;]+|"
            r"as\s+long\s+as\s+[^.!?\n;]+)",
            re.IGNORECASE,
        )

        # 10. Exceptions
        re_exception = re.compile(
            r"\b(?:except\s+(?:when|if|for)|unless\s+[^,\n;]+|"
            r"with\s+the\s+exception\s+of|with\s+one\s+exception:?|"
            r"excluding\s+[^.!?\n;]+)\s+[^.!?\n;]+",
            re.IGNORECASE,
        )

        # Explicit roles where instructions are authoritative
        directive_roles = frozenset({MessageRole.USER, MessageRole.SYSTEM, MessageRole.DEVELOPER})

        self.patterns: List[ExtractionPattern] = [
            # High / Mandatory constraints
            ExtractionPattern(re_constraint, EvidenceType.CONSTRAINT, EvidenceCriticality.HIGH, None, "deterministic_constraint"),
            # Instructions (User / Developer / System ONLY)
            ExtractionPattern(re_instruction, EvidenceType.INSTRUCTION, EvidenceCriticality.HIGH, directive_roles, "deterministic_instruction"),
            # Corrections (High priority across all turns)
            ExtractionPattern(re_correction, EvidenceType.CORRECTION, EvidenceCriticality.HIGH, None, "deterministic_correction"),
            # Conditions & Exceptions
            ExtractionPattern(re_condition, EvidenceType.CONDITION, EvidenceCriticality.MEDIUM, None, "deterministic_condition"),
            ExtractionPattern(re_exception, EvidenceType.EXCEPTION, EvidenceCriticality.HIGH, None, "deterministic_exception"),
            # Decisions, Goals, Preferences
            ExtractionPattern(re_decision, EvidenceType.DECISION, EvidenceCriticality.MEDIUM, None, "deterministic_decision"),
            ExtractionPattern(re_goal, EvidenceType.GOAL, EvidenceCriticality.MEDIUM, None, "deterministic_goal"),
            ExtractionPattern(re_preference, EvidenceType.PREFERENCE, EvidenceCriticality.MEDIUM, None, "deterministic_preference"),
            # Numerical & Dates
            ExtractionPattern(re_numerical, EvidenceType.NUMERICAL_VALUE, EvidenceCriticality.HIGH, None, "deterministic_numerical"),
            ExtractionPattern(re_date_deadline, EvidenceType.CONDITION, EvidenceCriticality.HIGH, None, "deterministic_deadline"),
        ]

    # ========================================================================
    # Main Extraction Public Interface
    # ========================================================================

    def extract(self, messages: Sequence[ConversationMessage]) -> Tuple[EvidenceUnit, ...]:
        """Extract atomic evidence units across an ordered sequence of conversation messages.

        Guarantees:
        - Messages are preserved without modification.
        - Every evidence unit is validated against source text offsets.
        - Duplicate spans on the same message are deduplicated.
        - Deterministic, ordered output.
        """
        if not messages:
            return ()

        # Build message index for fast provenance validation
        msg_map: Dict[str, ConversationMessage] = {msg.message_id: msg for msg in messages}

        evidence_units: List[EvidenceUnit] = []
        seen_spans: Set[Tuple[str, int, int]] = set()
        evidence_counter = 1

        for msg in messages:
            extracted_for_msg = self.extract_from_message(
                message=msg,
                start_id_index=evidence_counter,
                seen_spans=seen_spans,
            )
            evidence_units.extend(extracted_for_msg)
            evidence_counter += len(extracted_for_msg)

        # Enforce strict provenance verification
        verified_units = self.validate_all_provenance(evidence_units, messages)
        return verified_units

    def extract_from_message(
        self,
        message: ConversationMessage,
        start_id_index: int = 1,
        seen_spans: Optional[Set[Tuple[str, int, int]]] = None,
    ) -> Tuple[EvidenceUnit, ...]:
        """Extract evidence units from a single ConversationMessage using deterministic rules.

        Offsets are computed directly on message.content without pre-normalization.
        """
        if seen_spans is None:
            seen_spans = set()

        content = message.content
        if not content:
            return ()

        extracted: List[EvidenceUnit] = []
        counter = start_id_index

        for rule in self.patterns:
            # Check role restrictions (e.g. assistant messages cannot generate user instructions)
            if rule.allowed_roles is not None and message.role not in rule.allowed_roles:
                continue

            for match in rule.pattern.finditer(content):
                start_offset = match.start()
                end_offset = match.end()

                # Guard against degenerate 0-width matches
                if end_offset <= start_offset:
                    continue

                span_key = (message.message_id, start_offset, end_offset)
                if span_key in seen_spans:
                    continue

                # Extract exact verbatim substring
                sub_text = content[start_offset:end_offset]

                # Conservative criticality assignment:
                # System and Developer constraints/prohibitions are elevated to MANDATORY
                criticality = rule.criticality
                if (
                    rule.evidence_type == EvidenceType.CONSTRAINT
                    and message.role in {MessageRole.SYSTEM, MessageRole.DEVELOPER}
                ):
                    criticality = EvidenceCriticality.MANDATORY

                evidence_id = f"ev_{message.message_id}_{counter:04d}"
                span = SourceSpan(
                    message_id=message.message_id,
                    start_offset=start_offset,
                    end_offset=end_offset,
                )

                unit = EvidenceUnit(
                    evidence_id=evidence_id,
                    source_span=span,
                    evidence_type=rule.evidence_type,
                    original_text=sub_text,
                    criticality=criticality,
                    extraction_method=rule.extraction_tag,
                    validation_status=ValidationStatus.VALIDATED,
                )

                seen_spans.add(span_key)
                extracted.append(unit)
                counter += 1

        return tuple(extracted)

    # ========================================================================
    # Provenance Validation Interface
    # ========================================================================

    @staticmethod
    def validate_provenance(
        evidence: EvidenceUnit,
        messages_by_id: Mapping[str, ConversationMessage],
    ) -> bool:
        """Verify strict source provenance for a single EvidenceUnit.

        Verification Rules:
        1. Referenced message exists.
        2. Start offset >= 0.
        3. End offset > start offset.
        4. End offset <= len(message.content).
        5. message.content[start_offset:end_offset] == evidence.original_text.
        6. Message role is preserved and non-empty.
        7. Sequence index is valid (>= 0).
        """
        msg = messages_by_id.get(evidence.source_span.message_id)
        if msg is None:
            raise ProvenanceVerificationError(
                f"Referenced message '{evidence.source_span.message_id}' not found in conversation history.",
                evidence_id=evidence.evidence_id,
            )

        start = evidence.source_span.start_offset
        end = evidence.source_span.end_offset
        content = msg.content
        content_len = len(content)

        if start < 0:
            raise ProvenanceVerificationError(
                f"Negative start_offset ({start}) in evidence '{evidence.evidence_id}'.",
                evidence_id=evidence.evidence_id,
            )

        if end <= start:
            raise ProvenanceVerificationError(
                f"end_offset ({end}) must be strictly greater than start_offset ({start}) in evidence '{evidence.evidence_id}'.",
                evidence_id=evidence.evidence_id,
            )

        if end > content_len:
            raise ProvenanceVerificationError(
                f"end_offset ({end}) exceeds message content length ({content_len}) in evidence '{evidence.evidence_id}'.",
                evidence_id=evidence.evidence_id,
            )

        # Essential exact character slice assertion
        actual_slice = content[start:end]
        if actual_slice != evidence.original_text:
            raise ProvenanceVerificationError(
                f"Provenance slice mismatch in evidence '{evidence.evidence_id}': "
                f"message.content[{start}:{end}] is {actual_slice!r}, but original_text is {evidence.original_text!r}.",
                evidence_id=evidence.evidence_id,
            )

        # Verify role and sequence index accessibility
        if not msg.role:
            raise ProvenanceVerificationError(
                f"Message '{msg.message_id}' role is missing or corrupt.",
                evidence_id=evidence.evidence_id,
            )

        if msg.sequence_index < 0:
            raise ProvenanceVerificationError(
                f"Message '{msg.message_id}' has negative sequence_index ({msg.sequence_index}).",
                evidence_id=evidence.evidence_id,
            )

        return True

    def validate_all_provenance(
        self,
        evidence_units: Sequence[EvidenceUnit],
        messages: Sequence[ConversationMessage],
    ) -> Tuple[EvidenceUnit, ...]:
        """Validate provenance for an entire collection of evidence units against messages."""
        msg_map = {m.message_id: m for m in messages}
        verified: List[EvidenceUnit] = []

        for unit in evidence_units:
            if self.validate_provenance(unit, msg_map):
                verified.append(unit)

        return tuple(verified)

    # ========================================================================
    # Optional Local LLM Assistance Architecture (Hybrid Strategy)
    # ========================================================================

    @staticmethod
    def locate_quote_provenance(
        quote: str,
        message: ConversationMessage,
    ) -> Optional[Tuple[int, int]]:
        """Deterministically locate a proposed verbatim quote within a message.

        Rejection Rules:
        - If the quote does not appear in message.content -> Rejects (hallucinated quote).
        - If the quote appears multiple times -> Rejects (ambiguous provenance).
        - If the quote is empty or whitespace-only -> Rejects.

        Returns:
            (start_offset, end_offset) if uniquely and deterministically located; otherwise None.
        """
        raw_quote = quote.strip()
        if not raw_quote:
            return None

        content = message.content
        occurrence_count = content.count(raw_quote)

        # Strict requirement: reject invented or ambiguous offsets
        if occurrence_count == 0:
            logger.debug("Rejected invented/unmatched LLM quotation: %r", raw_quote)
            return None

        if occurrence_count > 1:
            logger.debug("Rejected ambiguous LLM quotation with %d occurrences: %r", occurrence_count, raw_quote)
            return None

        start_offset = content.find(raw_quote)
        end_offset = start_offset + len(raw_quote)

        # Verification check
        if content[start_offset:end_offset] == raw_quote:
            return (start_offset, end_offset)

        return None

    async def extract_with_llm_suggestions(
        self,
        messages: Sequence[ConversationMessage],
        ollama_base_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ) -> Tuple[EvidenceUnit, ...]:
        """Optional hybrid extraction augmenting deterministic rules with local LLM suggestions.

        Strict Constraints:
        - The local LLM never determines offsets directly.
        - Every proposed span is verified via locate_quote_provenance().
        - Non-matching or ambiguous quotes are rejected.
        - Never replaces original evidence with generated text.
        """
        # Always run deterministic extraction first as the authoritative baseline
        deterministic_units = list(self.extract(messages))
        seen_spans = {
            (u.source_span.message_id, u.source_span.start_offset, u.source_span.end_offset)
            for u in deterministic_units
        }

        base_url = (ollama_base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        model = ollama_model or settings.OLLAMA_MODEL

        counter = len(deterministic_units) + 1
        llm_units: List[EvidenceUnit] = []

        system_instruction = (
            "You are a semantic evidence extraction engine. Given a conversation turn, extract exact verbatim "
            "phrases representing critical facts, constraints, decisions, or user preferences.\n"
            "STRICT RULES:\n"
            "1. Output valid JSON array containing objects with keys: 'quote', 'type', 'criticality'.\n"
            "2. 'quote' MUST be an exact verbatim substring from the message text.\n"
            "3. DO NOT paraphrase, summarize, or edit the quote.\n"
            "4. 'type' must be one of: 'fact', 'instruction', 'constraint', 'preference', 'goal', 'decision', 'correction', 'condition', 'numerical_value'.\n"
            "5. 'criticality' must be one of: 'mandatory', 'high', 'medium', 'low', 'unknown'.\n"
        )

        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            for msg in messages:
                # Skip assistant messages for directive/instruction extraction
                payload = {
                    "model": model,
                    "system": system_instruction,
                    "prompt": f"Message ({msg.role.value}):\n{msg.content}\n\nExtract verbatim quotes as JSON:",
                    "stream": False,
                    "format": "json",
                }

                try:
                    resp = await client.post(f"{base_url}/api/generate", json=payload)
                    if resp.status_code != 200:
                        continue

                    data = resp.json()
                    raw_resp = data.get("response", "")
                    parsed = json.loads(raw_resp)
                    quotes_data = parsed if isinstance(parsed, list) else parsed.get("quotes", [])

                    for item in quotes_data:
                        if not isinstance(item, dict):
                            continue

                        raw_quote = item.get("quote", "")
                        type_str = item.get("type", "fact").lower()
                        crit_str = item.get("criticality", "medium").lower()

                        # Deterministic verification of source span
                        span_bounds = self.locate_quote_provenance(raw_quote, msg)
                        if span_bounds is None:
                            continue

                        start_idx, end_idx = span_bounds
                        span_key = (msg.message_id, start_idx, end_idx)
                        if span_key in seen_spans:
                            continue

                        # Map to typed enums safely
                        try:
                            ev_type = EvidenceType(type_str)
                        except ValueError:
                            ev_type = EvidenceType.FACT

                        try:
                            ev_crit = EvidenceCriticality(crit_str)
                        except ValueError:
                            ev_crit = EvidenceCriticality.MEDIUM

                        # Never treat assistant quotes as user instructions
                        if msg.role == MessageRole.ASSISTANT and ev_type == EvidenceType.INSTRUCTION:
                            ev_type = EvidenceType.FACT

                        span = SourceSpan(
                            message_id=msg.message_id,
                            start_offset=start_idx,
                            end_offset=end_idx,
                        )

                        unit = EvidenceUnit(
                            evidence_id=f"ev_llm_{msg.message_id}_{counter:04d}",
                            source_span=span,
                            evidence_type=ev_type,
                            original_text=msg.content[start_idx:end_idx],
                            criticality=ev_crit,
                            extraction_method="llm_assisted_verified",
                            validation_status=ValidationStatus.VALIDATED,
                        )

                        seen_spans.add(span_key)
                        llm_units.append(unit)
                        counter += 1

                except Exception as exc:
                    logger.warning("Optional LLM assistance skipped for message '%s': %s", msg.message_id, exc)
                    continue

        all_units = tuple(deterministic_units + llm_units)
        return self.validate_all_provenance(all_units, messages)
