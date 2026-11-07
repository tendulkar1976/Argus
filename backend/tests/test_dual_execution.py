"""Unit tests for Dual Prompt Execution and Explainable Comparison (Phase 11.1).

Tests:
1. TransformationExplainer:
   - Conversational filler and redundancy pruning extraction.
   - Constraint preservation tracking.
   - Identical prompt edge case (no transformations).
2. ResponseComparator:
   - LLM structured JSON response parsing into 9 quality dimensions.
   - Deterministic heuristic fallback comparator.
   - Separate evaluator metric and cost isolation.
3. DualExecutionService:
   - End-to-end dual execution with identical model conditions.
   - Independent error handling and failure isolation (one branch fails).
   - Resource accounting and performance delta metrics.
4. FastAPI Endpoint:
   - POST /api/compare contract validation.
"""

import json
from unittest.mock import AsyncMock, patch
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.schemas.comparison import (
    DualExecutionRequest,
    DualExecutionResponse,
    ResponseComparison,
    TransformationRecord,
)
from app.schemas.optimizer import OptimizeResponse, TokenEstimation
from app.services.dual_execution_service import DualExecutionService
from app.services.providers.base import ProviderError, ProviderExecutionResult
from app.services.response_comparator import ResponseComparator
from app.services.transformation_explainer import TransformationExplainer


# ============================================================================
# 1. Transformation Explainer Unit Tests
# ============================================================================

def test_transformation_explainer_filler_pruning():
    """Verify conversational filler phrases are detected and mapped to redundancy removal."""
    orig = "Please make sure to review this Python function carefully and thoroughly. In addition to that, check for bugs."
    opt = "Review the Python function for bugs."

    records = TransformationExplainer.explain(orig, opt)
    assert len(records) > 0
    categories = [r.category for r in records]
    assert "instruction_clarification" in categories or "redundancy_removal" in categories
    assert all(r.intent_preserved for r in records)


def test_transformation_explainer_constraint_preservation():
    """Verify explicit constraints are recorded as preserved."""
    orig = "Review code. Do not disclose internal IP addresses under any circumstances."
    opt = "Review code. Do not disclose internal IP addresses under any circumstances."

    records = TransformationExplainer.explain(orig, opt)
    assert len(records) >= 1
    assert records[0].category == "constraint_preservation"
    assert records[0].intent_preserved is True


def test_transformation_explainer_identical_prompt():
    """Verify identical prompts produce a safe preservation record."""
    prompt = "Write a quicksort algorithm in C++."
    records = TransformationExplainer.explain(prompt, prompt)
    assert len(records) == 1
    assert records[0].category == "constraint_preservation"
    assert "preserved" in records[0].reason.lower()


# ============================================================================
# 2. Response Comparator Unit Tests
# ============================================================================

@pytest.mark.asyncio
async def test_response_comparator_llm_parsing():
    """Verify LLM evaluation JSON is parsed into 9 dimensions and verdict."""
    mock_json_payload = {
        "dimensions": [
            {
                "dimension": "relevance",
                "dimension_label": "Relevance to User Intent",
                "original_assessment": "Addresses query directly",
                "optimized_assessment": "Addresses query with specific technical focus",
                "difference": "Optimized output has higher precision",
                "winner": "optimized",
                "explanation": "Focused response without conversational pleasantries",
            },
            {
                "dimension": "completeness",
                "dimension_label": "Answer Completeness",
                "original_assessment": "Complete",
                "optimized_assessment": "Complete with edge cases",
                "difference": "Optimized covers edge cases",
                "winner": "optimized",
                "explanation": "Edge case handling observed",
            },
            {
                "dimension": "accuracy_factual",
                "dimension_label": "Accuracy & Consistency",
                "original_assessment": "Accurate",
                "optimized_assessment": "Accurate",
                "difference": "No discrepancies",
                "winner": "equivalent",
                "explanation": "Both answers are factually sound",
            },
            {
                "dimension": "clarity_organization",
                "dimension_label": "Clarity & Organization",
                "original_assessment": "Prose paragraph",
                "optimized_assessment": "Structured code and bullet points",
                "difference": "Format structure",
                "winner": "optimized",
                "explanation": "Clearer markdown structure",
            },
            {
                "dimension": "specificity_usefulness",
                "dimension_label": "Specificity & Actionability",
                "original_assessment": "General instructions",
                "optimized_assessment": "Actionable code snippet",
                "difference": "Implementation specifics",
                "winner": "optimized",
                "explanation": "Directly runnable code provided",
            },
            {
                "dimension": "instruction_adherence",
                "dimension_label": "Instruction Adherence",
                "original_assessment": "Followed instructions",
                "optimized_assessment": "Followed instructions",
                "difference": "Equivalent adherence",
                "winner": "equivalent",
                "explanation": "Both adhered to system boundaries",
            },
            {
                "dimension": "constraint_preservation",
                "dimension_label": "Constraint Preservation",
                "original_assessment": "Preserved",
                "optimized_assessment": "Preserved",
                "difference": "None",
                "winner": "equivalent",
                "explanation": "Constraints respected across both",
            },
            {
                "dimension": "content_efficiency",
                "dimension_label": "Content Efficiency",
                "original_assessment": "Contained pleasantries",
                "optimized_assessment": "High signal-to-noise ratio",
                "difference": "Filler reduction",
                "winner": "optimized",
                "explanation": "Direct to point",
            },
            {
                "dimension": "overall_utility",
                "dimension_label": "Overall Response Utility",
                "original_assessment": "Good utility",
                "optimized_assessment": "High technical utility",
                "difference": "Depth of solution",
                "winner": "optimized",
                "explanation": "Higher developer productivity outcome",
            },
        ],
        "overall_verdict": "optimized_better",
        "summary_rationale": "Optimized prompt yielded significantly more structured and actionable code.",
        "original_strengths": ["Polite tone", "Correct concept"],
        "original_weaknesses": ["Preamble filler", "Lacked edge cases"],
        "optimized_strengths": ["Clear formatting", "Edge case coverage"],
        "optimized_weaknesses": [],
    }

    mock_exec_res = ProviderExecutionResult(
        provider_name="gemini",
        model="gemini-2.0-flash",
        content=f"```json\n{json.dumps(mock_json_payload)}\n```",
        input_tokens=420,
        output_tokens=310,
        total_tokens=730,
        latency_seconds=1.2,
    )

    with patch("app.services.providers.registry.ProviderRegistry.get_provider_for_model") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.provider_name = "gemini"
        mock_provider.generate = AsyncMock(return_value=mock_exec_res)
        mock_get_provider.return_value = mock_provider

        comparison = await ResponseComparator.compare(
            original_prompt="Write quicksort",
            optimized_prompt="Implement quicksort in Python with edge cases",
            response_original="Here is quicksort...",
            response_optimized="def quicksort(arr): ...",
            evaluator_model="gemini-2.0-flash",
        )

        assert comparison.overall_verdict == "optimized_better"
        assert len(comparison.dimensions) == 9
        assert comparison.evaluator_metrics is not None
        assert comparison.evaluator_metrics.total_tokens == 730
        assert comparison.evaluator_metrics.status == "success"


@pytest.mark.asyncio
async def test_response_comparator_heuristic_fallback():
    """Verify fallback comparator activates cleanly when provider raises an error."""
    with patch("app.services.providers.registry.ProviderRegistry.get_provider_for_model") as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(side_effect=ProviderError("Rate limit reached"))
        mock_get_provider.return_value = mock_provider

        comparison = await ResponseComparator.compare(
            original_prompt="Explain Docker",
            optimized_prompt="Explain containerization and Docker architecture",
            response_original="Docker is a platform for containers.",
            response_optimized="Docker isolates apps in containers.\n- Daemon\n- Client",
            evaluator_model="gemini-2.0-flash",
        )

        assert comparison is not None
        assert len(comparison.dimensions) == 9
        assert comparison.overall_verdict in {"similar", "optimized_better", "original_better", "inconclusive"}
        assert comparison.evaluator_metrics is not None
        assert comparison.evaluator_metrics.provider == "heuristic"


# ============================================================================
# 3. Dual Execution Service Unit Tests
# ============================================================================

@pytest.mark.asyncio
async def test_dual_execution_service_success():
    """Verify end-to-end dual execution dispatches both prompts and computes performance metrics."""
    mock_opt = OptimizeResponse(
        original_prompt="Please make sure to review this code carefully",
        optimized_prompt="Review code for security vulnerabilities",
        mode="balanced",
        target_model="gemini-2.0-flash",
        task_category="coding",
        strategy="compress",
        applied_strategy="compress",
        outcome="validated_rewrite",
        length_result="compressed",
        validation_status="passed",
        is_preserved=False,
        original_metrics=TokenEstimation(char_count=48, estimated_tokens=12),
        optimized_metrics=TokenEstimation(char_count=39, estimated_tokens=10),
        token_reduction_percent=16.6,
        token_change_percent=-16.6,
        token_savings=2,
        char_delta=-9,
        preserved_constraints=["security"],
        execution_time_ms=120,
        model="qwen2.5:1.5b",
        timestamp="2026-11-06T12:00:00Z",
    )

    res_a = ProviderExecutionResult(
        provider_name="gemini",
        model="gemini-2.0-flash",
        content="Original review: code looks fine.",
        input_tokens=15,
        output_tokens=25,
        total_tokens=40,
        latency_seconds=0.6,
    )

    res_b = ProviderExecutionResult(
        provider_name="gemini",
        model="gemini-2.0-flash",
        content="Optimized review: identified 2 security items:\n1. Input sanitization",
        input_tokens=12,
        output_tokens=35,
        total_tokens=47,
        latency_seconds=0.7,
    )

    with patch("app.services.ollama_service.OllamaService.optimize_prompt", AsyncMock(return_value=mock_opt)), \
         patch("app.services.providers.registry.ProviderRegistry.get_provider_for_model") as mock_get_provider, \
         patch("app.services.response_comparator.ResponseComparator.compare", AsyncMock(return_value=ResponseComparison(
             dimensions=[],
             overall_verdict="optimized_better",
             summary_rationale="Optimized prompt produced specific security items.",
         ))):

        mock_provider = AsyncMock()
        mock_provider.provider_name = "gemini"
        mock_provider.generate = AsyncMock(side_effect=[res_a, res_b])
        mock_get_provider.return_value = mock_provider

        req = DualExecutionRequest(
            prompt="Please make sure to review this code carefully",
            mode="balanced",
            target_model="gemini-2.0-flash",
            run_comparison=True,
        )

        resp: DualExecutionResponse = await DualExecutionService.execute(req)

        assert resp.target_model == "gemini-2.0-flash"
        assert resp.is_equivalent_conditions is True
        assert resp.original_execution.response_text == "Original review: code looks fine."
        assert resp.optimized_execution.response_text.startswith("Optimized review")
        assert resp.performance.prompt_tokens_saved == 3  # 15 - 12
        assert resp.performance.prompt_token_reduction_percent == 20.0
        assert resp.comparison is not None
        assert resp.comparison.overall_verdict == "optimized_better"


@pytest.mark.asyncio
async def test_dual_execution_service_error_isolation():
    """Verify that a failure in one execution branch does not crash the response."""
    mock_opt = OptimizeResponse(
        original_prompt="hello",
        optimized_prompt="Hello, assist me.",
        mode="balanced",
        target_model="gemini-2.0-flash",
        task_category="general",
        strategy="clarify",
        applied_strategy="clarify",
        outcome="validated_rewrite",
        length_result="expanded",
        validation_status="passed",
        is_preserved=False,
        original_metrics=TokenEstimation(char_count=5, estimated_tokens=2),
        optimized_metrics=TokenEstimation(char_count=18, estimated_tokens=5),
        token_reduction_percent=0,
        token_change_percent=150,
        token_savings=0,
        char_delta=13,
        preserved_constraints=[],
        execution_time_ms=100,
        model="qwen2.5:1.5b",
        timestamp="2026-11-06T12:00:00Z",
    )

    res_a = ProviderExecutionResult(
        provider_name="gemini",
        model="gemini-2.0-flash",
        content="Hello! How can I help?",
        input_tokens=5,
        output_tokens=10,
        total_tokens=15,
        latency_seconds=0.4,
    )

    with patch("app.services.ollama_service.OllamaService.optimize_prompt", AsyncMock(return_value=mock_opt)), \
         patch("app.services.providers.registry.ProviderRegistry.get_provider_for_model") as mock_get_provider:

        mock_provider = AsyncMock()
        mock_provider.provider_name = "gemini"
        # Branch A succeeds, Branch B throws ProviderError
        mock_provider.generate = AsyncMock(side_effect=[res_a, ProviderError("Rate limit exceeded")])
        mock_get_provider.return_value = mock_provider

        req = DualExecutionRequest(
            prompt="hello",
            target_model="gemini-2.0-flash",
            run_comparison=True,
        )

        resp: DualExecutionResponse = await DualExecutionService.execute(req)

        assert resp.original_execution.metrics.status == "success"
        assert resp.optimized_execution.metrics.status == "error"
        assert "Rate limit" in (resp.optimized_execution.metrics.error_message or "")
        # Comparison is skipped cleanly when one branch fails
        assert resp.comparison is None


# ============================================================================
# 4. FastAPI Route Unit Tests
# ============================================================================

@pytest.mark.asyncio
async def test_fastapi_compare_endpoint_contract():
    """Verify POST /api/compare endpoint contract."""
    mock_dual_response = DualExecutionResponse(
        request_id="ab_test123",
        timestamp="2026-11-06T12:00:00Z",
        target_model="gemini-2.0-flash",
        target_provider="gemini",
        is_equivalent_conditions=True,
        optimization=OptimizeResponse(
            original_prompt="hello",
            optimized_prompt="hello",
            mode="balanced",
            target_model="gemini-2.0-flash",
            task_category="general",
            strategy="preserve",
            applied_strategy="preserve",
            outcome="unchanged_original",
            length_result="unchanged",
            validation_status="passed",
            is_preserved=True,
            original_metrics=TokenEstimation(char_count=5, estimated_tokens=2),
            optimized_metrics=TokenEstimation(char_count=5, estimated_tokens=2),
            token_reduction_percent=0,
            token_change_percent=0,
            token_savings=0,
            char_delta=0,
            preserved_constraints=[],
            execution_time_ms=10,
            model="qwen2.5:1.5b",
            timestamp="2026-11-06T12:00:00Z",
        ),
        transformations=[],
        original_execution={
            "prompt": "hello",
            "response_text": "Hi!",
            "metrics": {
                "latency_ms": 150,
                "cost_status": "free",
                "provider": "gemini",
                "model": "gemini-2.0-flash",
                "status": "success",
            },
        },
        optimized_execution={
            "prompt": "hello",
            "response_text": "Hi!",
            "metrics": {
                "latency_ms": 150,
                "cost_status": "free",
                "provider": "gemini",
                "model": "gemini-2.0-flash",
                "status": "success",
            },
        },
        comparison=None,
        performance={
            "prompt_tokens_original": 2,
            "prompt_tokens_optimized": 2,
            "prompt_tokens_saved": 0,
            "prompt_token_reduction_percent": 0.0,
            "latency_original_ms": 150,
            "latency_optimized_ms": 150,
            "latency_difference_ms": 0,
        },
    )

    with patch("app.services.dual_execution_service.DualExecutionService.execute", AsyncMock(return_value=mock_dual_response)):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/compare",
                json={
                    "prompt": "hello",
                    "mode": "balanced",
                    "target_model": "gemini-2.0-flash",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["request_id"] == "ab_test123"
            assert data["target_model"] == "gemini-2.0-flash"
            assert "performance" in data
            assert "original_execution" in data
            assert "optimized_execution" in data
