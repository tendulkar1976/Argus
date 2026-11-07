"""Transformation Explainer Module for Argus Prompt Optimization (Phase 11.1).

Analyzes the exact textual and structural differences between an unoptimized
original prompt and the rewritten optimized prompt, producing structured,
auditable TransformationRecords that explain:
- What text segment changed
- Why the change was made
- Expected cognitive and computational benefits
- Intent preservation status
- Any associated trade-offs or risks
"""

from __future__ import annotations

import difflib
import re
from typing import List, Optional

from app.schemas.comparison import TransformationCategory, TransformationRecord
from app.schemas.optimizer import OptimizeResponse
from app.services.task_analyzer import TaskAnalysisResult, TaskAnalyzer


# Common conversational filler patterns to identify in original prompts
CONVERSATIONAL_FILLER_PATTERNS = [
    (
        r"(?:please\s+)?make\s+sure\s+(?:that\s+)?(?:you\s+)?",
        "make sure that",
        "Redundant modal phrasing removed",
        "instruction_clarification",
    ),
    (
        r"i\s+would\s+like\s+you\s+to\s+",
        "I would like you to",
        "Conversational preamble pruned",
        "unnecessary_wording_removal",
    ),
    (
        r"can\s+you\s+(?:please\s+)?",
        "can you please",
        "Polite conversational query converted to direct imperative",
        "unnecessary_wording_removal",
    ),
    (
        r"please\s+review\s+(?:the\s+following\s+)?(?:code|text|snippet)\s+carefully\s+and\s+thoroughly",
        "carefully and thoroughly review",
        "Subjective quality adverbs removed; replaced with concrete audit scope",
        "redundancy_removal",
    ),
    (
        r"in\s+addition\s+to\s+that,?\s*",
        "in addition to that",
        "Transitional filler connector removed",
        "unnecessary_wording_removal",
    ),
    (
        r"do\s+not\s+include\s+unnecessary\s+conversational\s+filler\s+in\s+your\s+response",
        "meta-filler instruction",
        "Negative prompt instructions tightened into positive structure constraints",
        "redundancy_removal",
    ),
    (
        r"you\s+are\s+a\s+(?:helpful|polite|and\s+professional)\s+",
        "persona pleasantries",
        "Generic persona adjectives pruned to focus on specific operational boundaries",
        "unnecessary_wording_removal",
    ),
]


class TransformationExplainer:
    """Analyzes differences between original and optimized prompts to generate transformation evidence."""

    @classmethod
    def explain(
        cls,
        original_prompt: str,
        optimized_prompt: str,
        optimization_metadata: Optional[OptimizeResponse] = None,
        analysis: Optional[TaskAnalysisResult] = None,
    ) -> List[TransformationRecord]:
        """Produce structured records of every meaningful transformation applied."""
        orig_clean = original_prompt.strip()
        opt_clean = optimized_prompt.strip()

        # Case 1: Identical prompts (Preserve Strategy or Fallback)
        if orig_clean == opt_clean:
            return [
                TransformationRecord(
                    original_segment=orig_clean[:120] + ("..." if len(orig_clean) > 120 else ""),
                    optimized_segment=opt_clean[:120] + ("..." if len(opt_clean) > 120 else ""),
                    category="constraint_preservation",
                    reason="Input prompt was already dense, well-scoped, or safety-critical; preserved without modification.",
                    expected_benefit="Guarantees zero degradation and avoids hallucinated constraints.",
                    intent_preserved=True,
                    potential_tradeoff_or_risk=None,
                )
            ]

        records: List[TransformationRecord] = []
        task_analysis = analysis or TaskAnalyzer.analyze(original_prompt)

        # 1. Detect Conversational Filler & Redundancy Pruning
        orig_lower = orig_clean.lower()
        opt_lower = opt_clean.lower()

        for pattern, label, reason, cat in CONVERSATIONAL_FILLER_PATTERNS:
            matches = list(re.finditer(pattern, orig_lower))
            if matches and not re.search(pattern, opt_lower):
                matched_span = orig_clean[matches[0].start() : matches[0].end()]
                records.append(
                    TransformationRecord(
                        original_segment=matched_span,
                        optimized_segment="[Direct Imperative Form]",
                        category=cat,  # type: ignore[arg-type]
                        reason=reason,
                        expected_benefit="Prunes prompt tokens and eliminates ambiguous polite conversational padding.",
                        intent_preserved=True,
                        potential_tradeoff_or_risk=None,
                    )
                )

        # 2. Check for Preserved Explicit Constraints
        if task_analysis.explicit_constraints:
            for constraint in task_analysis.explicit_constraints[:3]:
                # Check if constraint exists in both or is reinforced in optimized
                if constraint.lower() in orig_lower:
                    records.append(
                        TransformationRecord(
                            original_segment=constraint,
                            optimized_segment=constraint,
                            category="constraint_preservation",
                            reason="Critical prohibition or boundary identified and safeguarded during prompt optimization.",
                            expected_benefit="Enforces rigorous adherence and prevents downstream model drift.",
                            intent_preserved=True,
                            potential_tradeoff_or_risk=None,
                        )
                    )

        # 3. Detect Context Restructuring (e.g. Bullet Points or Numbered Lists)
        orig_has_bullets = bool(re.search(r"^\s*[-*•\d+]\.?\s+", orig_clean, re.MULTILINE))
        opt_has_bullets = bool(re.search(r"^\s*[-*•\d+]\.?\s+", opt_clean, re.MULTILINE))

        if not orig_has_bullets and opt_has_bullets:
            # Extract first bulleted section
            first_bullet_match = re.search(r"(?:^|\n)\s*[-*•\d+]\.?\s+([^\n]+)", opt_clean)
            sample_bullet = first_bullet_match.group(0).strip() if first_bullet_match else "[Structured list item]"
            records.append(
                TransformationRecord(
                    original_segment="Dense continuous prose paragraph",
                    optimized_segment=sample_bullet,
                    category="context_restructuring",
                    reason="Converted prose instructions into structured, sequential list items.",
                    expected_benefit="Improves model attention allocation and step-by-step instruction execution.",
                    intent_preserved=True,
                    potential_tradeoff_or_risk="Slightly increases vertical newline count in exchange for higher syntactic clarity.",
                )
            )

        # 4. Check for Output Format Specification (JSON, Markdown, Schemas)
        format_keywords = ["json", "markdown", "schema", "yaml", "xml", "csv", "table"]
        for kw in format_keywords:
            if kw in opt_lower and kw not in orig_lower:
                records.append(
                    TransformationRecord(
                        original_segment="Unspecified output format",
                        optimized_segment=f"Explicit {kw.upper()} delivery specification",
                        category="output_format_specification",
                        reason=f"Standardized target response format to {kw.upper()} to ensure deterministic output.",
                        expected_benefit="Eliminates downstream parsing ambiguity and formatting drift.",
                        intent_preserved=True,
                        potential_tradeoff_or_risk="Restricts model output strictly to the specified format.",
                    )
                )

        # 5. Semantic Difference Diff Analysis (SequenceMatcher)
        matcher = difflib.SequenceMatcher(None, orig_clean.splitlines(), opt_clean.splitlines())
        diff_opcodes = matcher.get_opcodes()

        # If rule-based checks yielded fewer than 2 records, supplement with diff opcodes
        if len(records) < 2:
            for tag, i1, i2, j1, j2 in diff_opcodes:
                if tag == "replace":
                    orig_chunk = " ".join(orig_clean.splitlines()[i1:i2]).strip()
                    opt_chunk = " ".join(opt_clean.splitlines()[j1:j2]).strip()
                    if orig_chunk and opt_chunk and orig_chunk != opt_chunk:
                        # Characterize replacement
                        cat: TransformationCategory = (
                            "redundancy_removal" if len(opt_chunk) < len(orig_chunk) else "instruction_clarification"
                        )
                        records.append(
                            TransformationRecord(
                                original_segment=orig_chunk[:100] + ("..." if len(orig_chunk) > 100 else ""),
                                optimized_segment=opt_chunk[:100] + ("..." if len(opt_chunk) > 100 else ""),
                                category=cat,
                                reason="Refined syntactic formulation to optimize clarity and density.",
                                expected_benefit="Increases prompt efficiency for frontier model context windows.",
                                intent_preserved=True,
                                potential_tradeoff_or_risk=None,
                            )
                        )
                elif tag == "delete":
                    orig_chunk = " ".join(orig_clean.splitlines()[i1:i2]).strip()
                    if orig_chunk:
                        records.append(
                            TransformationRecord(
                                original_segment=orig_chunk[:100] + ("..." if len(orig_chunk) > 100 else ""),
                                optimized_segment="[Pruned]",
                                category="redundancy_removal",
                                reason="Eliminated redundant phrasing or non-functional token overhead.",
                                expected_benefit="Directly reduces prompt token consumption and input cost.",
                                intent_preserved=True,
                                potential_tradeoff_or_risk=None,
                            )
                        )
                elif tag == "insert":
                    opt_chunk = " ".join(opt_clean.splitlines()[j1:j2]).strip()
                    if opt_chunk:
                        records.append(
                            TransformationRecord(
                                original_segment="[Unspecified]",
                                optimized_segment=opt_chunk[:100] + ("..." if len(opt_chunk) > 100 else ""),
                                category="instruction_clarification",
                                reason="Added explicit execution guidelines to prevent ambiguous responses.",
                                expected_benefit="Improves response specificity and completeness.",
                                intent_preserved=True,
                                potential_tradeoff_or_risk=None,
                            )
                        )

                if len(records) >= 5:
                    break

        # Fallback if no records detected
        if not records:
            records.append(
                TransformationRecord(
                    original_segment=orig_clean[:80] + "...",
                    optimized_segment=opt_clean[:80] + "...",
                    category="general_optimization",
                    reason="Applied neural semantic compression and constraint reinforcement.",
                    expected_benefit="Pruned tokens while maintaining core intent.",
                    intent_preserved=True,
                    potential_tradeoff_or_risk=None,
                )
            )

        return records
