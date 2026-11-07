"""Task classification and complexity estimation for Intelligent Model Routing (Phase 6).

Evaluates:
- Domain categorization (coding, reasoning, math, summarization, extraction, data analysis, general).
- Multi-dimensional complexity assessment:
  - Input token volume
  - Constraint and prohibition density
  - Syntactic code markers
  - Multi-step procedural instructions
  - Reasoning markers
"""

import math
import re
from typing import List, Tuple
from app.core.config import settings
from app.services.routing.models import ComplexityScore, RoutingTaskCategory


class TaskClassifier:
    """Classifies prompts into domain task categories using structural and keyword analysis."""

    _CODING_PATTERNS = [
        re.compile(r"\b(def |class |function |async |import |return |select |from |where |insert |update |delete )\b", re.IGNORECASE),
        re.compile(r"\b(python|javascript|typescript|c\+\+|rust|golang|java|sql|html|css|bash|powershell|regex)\b", re.IGNORECASE),
        re.compile(r"\b(write a script|write code|implement|debug|refactor|compile|unit test|syntax error)\b", re.IGNORECASE),
        re.compile(r"```[a-z]*\n", re.IGNORECASE),
    ]

    _REASONING_PATTERNS = [
        re.compile(r"\b(step[- ]by[- ]step|prove that|deduce|why does|analyze the cause|derive|counterfactual|under what conditions)\b", re.IGNORECASE),
        re.compile(r"\b(logic puzzle|paradox|implication|formal proof|tradeoff analysis|root cause)\b", re.IGNORECASE),
    ]

    _MATH_PATTERNS = [
        re.compile(r"\b(calculate|compute|solve for|integral|derivative|matrix|vector|probability|standard deviation|formula|equation)\b", re.IGNORECASE),
        re.compile(r"[\+\-\*\/\^=><≈≠≤≥∑∏√∫]"),
    ]

    _DATA_ANALYSIS_PATTERNS = [
        re.compile(r"\b(csv|dataframe|pandas|numpy|aggregate|groupby|dataset|distribution|percentile|regression|variance|histogram)\b", re.IGNORECASE),
    ]

    _EXTRACTION_PATTERNS = [
        re.compile(r"\b(extract|parse|pull out|convert to json|structured data|fields:|table of|entities from)\b", re.IGNORECASE),
    ]

    _SUMMARIZATION_PATTERNS = [
        re.compile(r"\b(summarize|summary|tl;?dr|brief overview|condense|abstract|synopsis|main points)\b", re.IGNORECASE),
    ]

    _CREATIVE_PATTERNS = [
        re.compile(r"\b(write a story|poem|narrative|dialogue between|screenplay|essay|creative|metaphor)\b", re.IGNORECASE),
    ]

    @classmethod
    def classify(cls, prompt: str) -> RoutingTaskCategory:
        """Determine the primary task category for the input prompt."""
        text = prompt.strip()
        if not text:
            return RoutingTaskCategory.GENERAL_DIALOGUE

        # Match counts
        coding_matches = sum(len(p.findall(text)) for p in cls._CODING_PATTERNS)
        reasoning_matches = sum(len(p.findall(text)) for p in cls._REASONING_PATTERNS)
        math_matches = sum(len(p.findall(text)) for p in cls._MATH_PATTERNS)
        data_matches = sum(len(p.findall(text)) for p in cls._DATA_ANALYSIS_PATTERNS)
        extraction_matches = sum(len(p.findall(text)) for p in cls._EXTRACTION_PATTERNS)
        summary_matches = sum(len(p.findall(text)) for p in cls._SUMMARIZATION_PATTERNS)
        creative_matches = sum(len(p.findall(text)) for p in cls._CREATIVE_PATTERNS)

        # Prioritized classification
        if coding_matches >= 2 or (coding_matches >= 1 and "```" in text):
            return RoutingTaskCategory.CODING
        if math_matches >= 3:
            return RoutingTaskCategory.MATHEMATICS
        if data_matches >= 2:
            return RoutingTaskCategory.DATA_ANALYSIS
        if reasoning_matches >= 2:
            return RoutingTaskCategory.REASONING
        if extraction_matches >= 2:
            return RoutingTaskCategory.EXTRACTION
        if summary_matches >= 1 and len(text) > 300:
            return RoutingTaskCategory.SUMMARIZATION
        if creative_matches >= 2:
            return RoutingTaskCategory.CREATIVE_WRITING

        # Fallback to single match highest signal
        signals = [
            (coding_matches, RoutingTaskCategory.CODING),
            (reasoning_matches, RoutingTaskCategory.REASONING),
            (data_matches, RoutingTaskCategory.DATA_ANALYSIS),
            (math_matches, RoutingTaskCategory.MATHEMATICS),
            (extraction_matches, RoutingTaskCategory.EXTRACTION),
            (summary_matches, RoutingTaskCategory.SUMMARIZATION),
            (creative_matches, RoutingTaskCategory.CREATIVE_WRITING),
        ]
        signals.sort(key=lambda x: x[0], reverse=True)
        if signals[0][0] > 0:
            return signals[0][1]

        return RoutingTaskCategory.GENERAL_DIALOGUE


class ComplexityEstimator:
    """Estimates the multi-indicator complexity score of a prompt."""

    _PROHIBITION_PATTERNS = [
        re.compile(r"\b(do not|don't|never|cannot|must not|prohibit|no \w+ allowed)\b", re.IGNORECASE),
        re.compile(r"\b(strictly|mandatory|required format|only return|without using)\b", re.IGNORECASE),
    ]

    _MULTI_STEP_PATTERNS = [
        re.compile(r"\b(first|then|next|finally|step \d|phase \d|after that)\b", re.IGNORECASE),
        re.compile(r"^\s*(\d+\.|\-|\*)\s+", re.MULTILINE),
    ]

    _REASONING_MARKERS = [
        re.compile(r"\b(why|how does|compare|contrast|tradeoff|evaluate|edge case|counterfactual|proof)\b", re.IGNORECASE),
    ]

    @classmethod
    def estimate(cls, prompt: str) -> ComplexityScore:
        """Compute complexity score and diagnostic indicators."""
        text = prompt.strip()
        char_count = len(text)
        token_heuristic = settings.CHARS_PER_TOKEN_HEURISTIC or 4.0
        token_estimate = max(1, math.ceil(char_count / token_heuristic)) if char_count > 0 else 0

        # Constraints
        constraints = 0
        for pattern in cls._PROHIBITION_PATTERNS:
            constraints += len(pattern.findall(text))

        # Multi-step
        multi_step_hits = sum(len(p.findall(text)) for p in cls._MULTI_STEP_PATTERNS)
        has_multi_step = multi_step_hits >= 2

        # Code syntax
        has_code = "```" in text or bool(re.search(r"\b(def |class |function |import )\b", text))

        # Reasoning density
        reasoning_hits = sum(len(p.findall(text)) for p in cls._REASONING_MARKERS)
        reasoning_density = min(1.0, reasoning_hits / max(1, token_estimate / 25))

        # Weighted complexity composition
        # Tokens: up to 1000 tokens contributes up to 0.35
        token_component = min(0.35, (token_estimate / 1000.0) * 0.35)
        # Constraints: up to 5 constraints contributes up to 0.25
        constraint_component = min(0.25, (constraints / 5.0) * 0.25)
        # Multi-step contributes 0.15
        step_component = 0.15 if has_multi_step else 0.0
        # Code syntax contributes 0.10
        code_component = 0.10 if has_code else 0.0
        # Reasoning density contributes up to 0.15
        reasoning_component = reasoning_density * 0.15

        raw_score = token_component + constraint_component + step_component + code_component + reasoning_component
        score = max(0.05, min(1.0, round(raw_score, 3)))

        justification = (
            f"Estimated {token_estimate} tokens, {constraints} constraints, "
            f"multi_step={has_multi_step}, code={has_code}, reasoning_density={reasoning_density:.2f}"
        )

        return ComplexityScore(
            score=score,
            token_count_estimate=token_estimate,
            constraint_count=constraints,
            has_code_syntax=has_code,
            has_multi_step_logic=has_multi_step,
            reasoning_density=round(reasoning_density, 3),
            justification=justification,
        )
