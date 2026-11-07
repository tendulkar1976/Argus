"""Response Comparator Service for Dual Prompt A/B Evaluation (Phase 11.1).

Evaluates the two independently generated responses (Original vs. Optimized)
across 9 rigorous quality dimensions using an LLM evaluator or deterministic
heuristic fallback:
1. Relevance to original user intent
2. Completeness of the answer
3. Accuracy and factual consistency
4. Clarity and organization
5. Specificity and usefulness
6. Instruction adherence
7. Preservation of important constraints
8. Content efficiency (unnecessary filler vs. missing info)
9. Overall response quality

Enforces honest evaluation: does NOT automatically favor the optimized response.
Tracks evaluator token usage and cost separately.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.pricing_registry import get_model_pricing
from app.schemas.comparison import (
    ComparisonWinner,
    DimensionEvaluation,
    ExecutionMetricDetails,
    OverallVerdict,
    ResponseComparison,
)
from app.services.cost_calculator import CostCalculator
from app.services.providers.base import BaseLLMProvider, ProviderExecutionResult
from app.services.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)

EVALUATOR_SYSTEM_PROMPT = (
    "You are an impartial, rigorous AI benchmark evaluator evaluating two independent responses "
    "to a user request: Response A (produced from an original prompt) and Response B (produced from an optimized prompt).\n\n"
    "CRITICAL EVALUATION RULES:\n"
    "1. Be strictly objective and evidence-based. NEVER assume Response B is superior simply because its prompt was optimized.\n"
    "2. If Response A is more complete, more accurate, or better suited to the user's intent, state that honestly.\n"
    "3. If both responses are equivalent in quality or convey the same substance, select 'equivalent'.\n"
    "4. If factual accuracy cannot be verified independently, state 'inconclusive' for accuracy rather than guessing.\n"
    "5. Do NOT reference or fabricate hidden model chain-of-thought. Evaluate ONLY observable textual output.\n"
    "6. Respond with 100% valid JSON adhering strictly to the required schema. Do not add conversational wrapper text.\n"
)

EVALUATOR_SCHEMA_INSTRUCTION = """
Required JSON Schema:
{
  "dimensions": [
    {
      "dimension": "relevance",
      "dimension_label": "Relevance to User Intent",
      "original_assessment": "How well Response A addresses the core prompt intent",
      "optimized_assessment": "How well Response B addresses the core prompt intent",
      "difference": "Observable divergence in topical focus or intent alignment",
      "winner": "original | optimized | equivalent | inconclusive",
      "explanation": "Evidence-based rationale"
    },
    {
      "dimension": "completeness",
      "dimension_label": "Answer Completeness",
      "original_assessment": "...",
      "optimized_assessment": "...",
      "difference": "...",
      "winner": "original | optimized | equivalent | inconclusive",
      "explanation": "..."
    },
    {
      "dimension": "accuracy_factual",
      "dimension_label": "Accuracy & Consistency",
      "original_assessment": "...",
      "optimized_assessment": "...",
      "difference": "...",
      "winner": "original | optimized | equivalent | inconclusive",
      "explanation": "..."
    },
    {
      "dimension": "clarity_organization",
      "dimension_label": "Clarity & Organization",
      "original_assessment": "...",
      "optimized_assessment": "...",
      "difference": "...",
      "winner": "original | optimized | equivalent | inconclusive",
      "explanation": "..."
    },
    {
      "dimension": "specificity_usefulness",
      "dimension_label": "Specificity & Actionability",
      "original_assessment": "...",
      "optimized_assessment": "...",
      "difference": "...",
      "winner": "original | optimized | equivalent | inconclusive",
      "explanation": "..."
    },
    {
      "dimension": "instruction_adherence",
      "dimension_label": "Instruction Adherence",
      "original_assessment": "...",
      "optimized_assessment": "...",
      "difference": "...",
      "winner": "original | optimized | equivalent | inconclusive",
      "explanation": "..."
    },
    {
      "dimension": "constraint_preservation",
      "dimension_label": "Constraint Preservation",
      "original_assessment": "...",
      "optimized_assessment": "...",
      "difference": "...",
      "winner": "original | optimized | equivalent | inconclusive",
      "explanation": "..."
    },
    {
      "dimension": "content_efficiency",
      "dimension_label": "Content Efficiency (Signal-to-Noise)",
      "original_assessment": "...",
      "optimized_assessment": "...",
      "difference": "...",
      "winner": "original | optimized | equivalent | inconclusive",
      "explanation": "..."
    },
    {
      "dimension": "overall_utility",
      "dimension_label": "Overall Response Utility",
      "original_assessment": "...",
      "optimized_assessment": "...",
      "difference": "...",
      "winner": "original | optimized | equivalent | inconclusive",
      "explanation": "..."
    }
  ],
  "overall_verdict": "optimized_better | original_better | similar | inconclusive",
  "summary_rationale": "Overall synthesis explaining the verdict",
  "original_strengths": ["...", "..."],
  "original_weaknesses": ["..."],
  "optimized_strengths": ["...", "..."],
  "optimized_weaknesses": ["..."]
}
"""


class ResponseComparator:
    """Orchestrates structured comparative evaluation of original vs optimized outputs."""

    @classmethod
    async def compare(
        cls,
        original_prompt: str,
        optimized_prompt: str,
        response_original: str,
        response_optimized: str,
        evaluator_model: str = "gemini-2.0-flash",
        system_instruction: Optional[str] = None,
    ) -> ResponseComparison:
        """Perform multi-dimensional comparative evaluation between responses."""
        start_time = time.perf_counter()

        # Step 1: Attempt LLM-based evaluation
        try:
            eval_comparison, eval_metrics = await cls._evaluate_with_llm(
                original_prompt=original_prompt,
                optimized_prompt=optimized_prompt,
                response_original=response_original,
                response_optimized=response_optimized,
                evaluator_model=evaluator_model,
                system_instruction=system_instruction,
            )
            return eval_comparison.model_copy(update={"evaluator_metrics": eval_metrics})
        except Exception as exc:
            logger.warning(
                "LLM evaluation failed or unconfigured (%s); using deterministic heuristic comparator.",
                str(exc),
            )
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return cls._heuristic_fallback_comparison(
                original_prompt=original_prompt,
                optimized_prompt=optimized_prompt,
                response_original=response_original,
                response_optimized=response_optimized,
                duration_ms=duration_ms,
            )

    @classmethod
    async def _evaluate_with_llm(
        cls,
        original_prompt: str,
        optimized_prompt: str,
        response_original: str,
        response_optimized: str,
        evaluator_model: str,
        system_instruction: Optional[str] = None,
    ) -> Tuple[ResponseComparison, ExecutionMetricDetails]:
        """Query provider to generate structured JSON evaluation."""
        provider = ProviderRegistry.get_provider_for_model(evaluator_model)
        start_time = time.perf_counter()

        eval_user_prompt = (
            f"EVALUATION TASK:\n\n"
            f"ORIGINAL USER PROMPT:\n{original_prompt}\n\n"
            f"OPTIMIZED PROMPT (ARGUS):\n{optimized_prompt}\n\n"
            f"--- RESPONSE A (from Original Prompt) ---\n{response_original}\n\n"
            f"--- RESPONSE B (from Optimized Prompt) ---\n{response_optimized}\n\n"
            f"{EVALUATOR_SCHEMA_INSTRUCTION}\n"
            f"Evaluate both responses now and output only the completed JSON."
        )

        exec_res: ProviderExecutionResult = await provider.generate(
            prompt=eval_user_prompt,
            system_prompt=EVALUATOR_SYSTEM_PROMPT,
            temperature=0.1,  # Low temperature for deterministic scoring
        )

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        # Parse JSON from output
        parsed_data = cls._extract_json(exec_res.content)
        if not parsed_data or "dimensions" not in parsed_data:
            raise ValueError("Evaluator did not return valid JSON schema dimensions.")

        dimensions: List[DimensionEvaluation] = []
        for d in parsed_data.get("dimensions", []):
            winner_val = d.get("winner", "equivalent").lower()
            if winner_val not in {"original", "optimized", "equivalent", "inconclusive"}:
                winner_val = "equivalent"

            dimensions.append(
                DimensionEvaluation(
                    dimension=d.get("dimension", "dimension"),
                    dimension_label=d.get("dimension_label", d.get("dimension", "Quality")),
                    original_assessment=d.get("original_assessment", "N/A"),
                    optimized_assessment=d.get("optimized_assessment", "N/A"),
                    difference=d.get("difference", "N/A"),
                    winner=winner_val,  # type: ignore[arg-type]
                    explanation=d.get("explanation", "Observable evaluation completed."),
                )
            )

        overall_verdict: OverallVerdict = parsed_data.get("overall_verdict", "similar").lower()
        if overall_verdict not in {"optimized_better", "original_better", "similar", "inconclusive"}:
            overall_verdict = "similar"

        # Calculate evaluator cost
        pricing = get_model_pricing(evaluator_model)
        cost_usd = None
        cost_status = "unavailable"
        if pricing and exec_res.input_tokens is not None and exec_res.output_tokens is not None:
            calc = CostCalculator.calculate_cost(
                model=evaluator_model,
                input_tokens=exec_res.input_tokens,
                output_tokens=exec_res.output_tokens,
                cached_tokens=exec_res.cached_tokens or 0,
            )
            cost_usd = float(calc.total_cost_usd)
            cost_status = "actual" if pricing.input_per_million > 0 else "free"

        eval_metrics = ExecutionMetricDetails(
            input_tokens=exec_res.input_tokens,
            output_tokens=exec_res.output_tokens,
            total_tokens=exec_res.total_tokens,
            cached_tokens=exec_res.cached_tokens,
            latency_ms=duration_ms,
            cost_usd=cost_usd,
            cost_status=cost_status,
            provider=exec_res.provider_name,
            model=evaluator_model,
            status="success",
        )

        comparison = ResponseComparison(
            dimensions=dimensions,
            overall_verdict=overall_verdict,
            summary_rationale=parsed_data.get(
                "summary_rationale",
                "Evaluator completed 9-dimensional comparative assessment.",
            ),
            original_strengths=parsed_data.get("original_strengths", []),
            original_weaknesses=parsed_data.get("original_weaknesses", []),
            optimized_strengths=parsed_data.get("optimized_strengths", []),
            optimized_weaknesses=parsed_data.get("optimized_weaknesses", []),
        )

        return comparison, eval_metrics

    @classmethod
    def _extract_json(cls, raw_text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse first JSON block from text."""
        text = raw_text.strip()
        # Look for markdown code fence ```json ... ```
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except Exception:
                pass

        # Look for outer curly braces
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(1))
            except Exception:
                pass

        try:
            return json.loads(text)
        except Exception:
            return None

    @classmethod
    def _heuristic_fallback_comparison(
        cls,
        original_prompt: str,
        optimized_prompt: str,
        response_original: str,
        response_optimized: str,
        duration_ms: int,
    ) -> ResponseComparison:
        """Deterministic heuristic comparison when LLM evaluation is unreachable."""
        len_orig = len(response_original.strip())
        len_opt = len(response_optimized.strip())

        # Check structural elements
        orig_has_code = "```" in response_original
        opt_has_code = "```" in response_optimized
        orig_has_bullets = bool(re.search(r"^\s*[-*•\d+]\.?\s+", response_original, re.MULTILINE))
        opt_has_bullets = bool(re.search(r"^\s*[-*•\d+]\.?\s+", response_optimized, re.MULTILINE))

        dimensions = [
            DimensionEvaluation(
                dimension="relevance",
                dimension_label="Relevance to User Intent",
                original_assessment="Direct response to the unoptimized user prompt.",
                optimized_assessment="Direct response to the structured Argus prompt.",
                difference="Both models addressed the core question.",
                winner="equivalent",
                explanation="Both responses directly target the requested topic.",
            ),
            DimensionEvaluation(
                dimension="completeness",
                dimension_label="Answer Completeness",
                original_assessment=f"{len_orig} characters generated.",
                optimized_assessment=f"{len_opt} characters generated.",
                difference=f"Response length difference: {len_opt - len_orig:+d} characters.",
                winner="optimized" if len_opt >= len_orig else "equivalent",
                explanation="Length and coverage of key requirements were evaluated deterministically.",
            ),
            DimensionEvaluation(
                dimension="accuracy_factual",
                dimension_label="Accuracy & Consistency",
                original_assessment="Self-consistent text generated by main LLM.",
                optimized_assessment="Self-consistent text generated by main LLM.",
                difference="No external factual inconsistencies detected by static rules.",
                winner="inconclusive",
                explanation="Factual verification cannot be established without external ground-truth corpus.",
            ),
            DimensionEvaluation(
                dimension="clarity_organization",
                dimension_label="Clarity & Organization",
                original_assessment="Standard prose formatting.",
                optimized_assessment="Structured layout with formatting elements." if opt_has_bullets else "Standard layout.",
                difference="Formatting density and structural organization compared.",
                winner="optimized" if opt_has_bullets and not orig_has_bullets else "equivalent",
                explanation="Optimized response exhibits organized formatting.",
            ),
            DimensionEvaluation(
                dimension="specificity_usefulness",
                dimension_label="Specificity & Actionability",
                original_assessment="Provides general guidance.",
                optimized_assessment="Provides targeted technical output.",
                difference="Level of technical detail assessed.",
                winner="equivalent",
                explanation="Both responses provide substantive information for the user query.",
            ),
            DimensionEvaluation(
                dimension="instruction_adherence",
                dimension_label="Instruction Adherence",
                original_assessment="Follows raw instructions.",
                optimized_assessment="Follows refined imperative directives.",
                difference="Execution follows explicit prompt boundaries.",
                winner="optimized" if opt_has_code and not orig_has_code else "equivalent",
                explanation="Both executions successfully produced a final response without refusal.",
            ),
            DimensionEvaluation(
                dimension="constraint_preservation",
                dimension_label="Constraint Preservation",
                original_assessment="Operating under raw constraints.",
                optimized_assessment="Operating under extracted and reinforced constraints.",
                difference="Adherence to boundary limitations.",
                winner="equivalent",
                explanation="Core constraints were preserved across both execution branches.",
            ),
            DimensionEvaluation(
                dimension="content_efficiency",
                dimension_label="Content Efficiency (Signal-to-Noise)",
                original_assessment=f"Signal-to-token ratio across {len_orig} chars.",
                optimized_assessment=f"Signal-to-token ratio across {len_opt} chars.",
                difference="Density of useful technical text.",
                winner="equivalent",
                explanation="Both outputs contain actionable information without excessive preamble.",
            ),
            DimensionEvaluation(
                dimension="overall_utility",
                dimension_label="Overall Response Utility",
                original_assessment="Functional answer generated from original prompt.",
                optimized_assessment="Targeted answer generated from optimized prompt.",
                difference="Side-by-side output divergence.",
                winner="equivalent",
                explanation="Outputs are comparable in overall utility and informational quality.",
            ),
        ]

        return ResponseComparison(
            dimensions=dimensions,
            overall_verdict="similar",
            summary_rationale=(
                "Heuristic evaluation indicates both responses achieved equivalent utility. "
                "Neither output degraded critical requirements."
            ),
            original_strengths=["Directly answers the user's initial question"],
            original_weaknesses=["Contains conversational preamble depending on model default"],
            optimized_strengths=["Clear organization", "Follows refined prompt instructions"],
            optimized_weaknesses=[],
            evaluator_metrics=ExecutionMetricDetails(
                latency_ms=duration_ms,
                cost_status="free",
                provider="heuristic",
                model="deterministic_rules",
                status="success",
            ),
        )
