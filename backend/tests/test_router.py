"""Unit tests for Phase 6 Intelligent Model Routing Engine.

Covers:
1. Task classification across domains.
2. Complexity estimation and indicators.
3. Model capability registry inspection for credentialed providers (Gemini, DeepSeek, Ollama).
4. Multidimensional scoring (quality, cost, latency).
5. Strategy adaptation (balanced, cost_optimized, performance, quality).
6. Context window restrictions and filtering.
7. User constraint enforcement (coding, reasoning, allowed providers).
8. Fallback chain generation and absence of routing loops.
9. Diagnostic rejection logging.
"""

from unittest.mock import patch
import pytest

from app.services.routing.classifier import ComplexityEstimator, TaskClassifier
from app.services.routing.engine import ModelRoutingEngine
from app.services.routing.models import (
    ComplexityScore,
    CostTier,
    LatencyTier,
    RoutingConstraints,
    RoutingDecision,
    RoutingPolicy,
    RoutingTaskCategory,
)
from app.services.routing.registry import ModelCapabilityRegistry
from app.services.routing.scorer import ModelScorer


# ============================================================================
# 1. Task Classification Tests
# ============================================================================

def test_task_classification_coding():
    prompt = "Write a Python script with def calculate_fibonacci(n: int) -> int: that computes numbers."
    cat = TaskClassifier.classify(prompt)
    assert cat == RoutingTaskCategory.CODING


def test_task_classification_reasoning():
    prompt = "Explain step-by-step why the Byzantine Generals Problem cannot be solved with 3 generals and 1 traitor."
    cat = TaskClassifier.classify(prompt)
    assert cat == RoutingTaskCategory.REASONING


def test_task_classification_data_analysis():
    prompt = "Given a sales CSV with columns revenue and cost, calculate the monthly average and standard deviation."
    cat = TaskClassifier.classify(prompt)
    assert cat == RoutingTaskCategory.DATA_ANALYSIS


def test_task_classification_general():
    prompt = "Hello, can you help me brainstorm some naming ideas for a productivity software startup?"
    cat = TaskClassifier.classify(prompt)
    assert cat == RoutingTaskCategory.GENERAL_DIALOGUE


# ============================================================================
# 2. Complexity Estimation Tests
# ============================================================================

def test_complexity_simple_prompt():
    prompt = "What is the capital of France?"
    comp = ComplexityEstimator.estimate(prompt)
    assert comp.score < 0.35
    assert not comp.has_code_syntax
    assert not comp.has_multi_step_logic


def test_complexity_complex_prompt():
    prompt = (
        "Implement a distributed consensus algorithm in Python. "
        "Strict constraints: do not use external libraries, must not fail under network partitions, "
        "first validate node timestamps, then broadcast votes, and finally commit the state. "
        "Why does this guarantee safety under asynchronous network conditions?"
    )
    comp = ComplexityEstimator.estimate(prompt)
    assert comp.score >= 0.50
    assert comp.constraint_count >= 2
    assert comp.has_multi_step_logic


# ============================================================================
# 3. Model Capability Registry Tests
# ============================================================================

def test_capability_registry_lookups():
    gemini = ModelCapabilityRegistry.get_model("gemini-2.0-flash")
    assert gemini is not None
    assert gemini.provider_name == "gemini"
    assert gemini.context_window == 1_048_576

    deepseek = ModelCapabilityRegistry.get_model("deepseek-chat")
    assert deepseek is not None
    assert deepseek.provider_name == "deepseek"

    qwen = ModelCapabilityRegistry.get_model("qwen2.5:1.5b")
    assert qwen is not None
    assert qwen.cost_tier == CostTier.FREE
    assert qwen.provider_name == "ollama"

    gemini_models = ModelCapabilityRegistry.get_models_for_provider("gemini")
    assert len(gemini_models) >= 2


# ============================================================================
# 4. Multi-Criteria Scoring & Strategy Adaptation Tests
# ============================================================================

def test_cost_score_prefers_free_local_model():
    qwen = ModelCapabilityRegistry.get_model("qwen2.5:1.5b")
    deepseek_reasoner = ModelCapabilityRegistry.get_model("deepseek-reasoner")

    cost_qwen, _ = ModelScorer.compute_cost_score(qwen, 1000)
    cost_ds, _ = ModelScorer.compute_cost_score(deepseek_reasoner, 1000)

    assert cost_qwen == 1.0  # free local model gets maximum cost score
    assert cost_ds < cost_qwen


def test_latency_score_ranking():
    flash = ModelCapabilityRegistry.get_model("gemini-2.0-flash")
    slow_reasoner = ModelCapabilityRegistry.get_model("deepseek-reasoner")

    lat_flash, _ = ModelScorer.compute_latency_score(flash)
    lat_slow, _ = ModelScorer.compute_latency_score(slow_reasoner)

    assert lat_flash > lat_slow
    assert lat_flash == 0.85
    assert lat_slow == 0.30


def test_strategy_weight_adjustment():
    policy_cost = RoutingPolicy(strategy="cost_optimized")
    weights_cost = ModelScorer.resolve_effective_weights(policy_cost)
    assert weights_cost["cost"] > weights_cost["quality"]

    policy_perf = RoutingPolicy(strategy="performance")
    weights_perf = ModelScorer.resolve_effective_weights(policy_perf)
    assert weights_perf["latency"] > weights_perf["cost"]

    policy_qual = RoutingPolicy(strategy="quality")
    weights_qual = ModelScorer.resolve_effective_weights(policy_qual)
    assert weights_qual["quality"] > weights_qual["cost"]


# ============================================================================
# 5. Routing Engine Decisions & Constraints
# ============================================================================

def test_routing_engine_selects_best_candidate():
    engine = ModelRoutingEngine()
    available = {
        "gemini": True,
        "deepseek": True,
        "ollama": True,
    }

    decision = engine.route(
        prompt="Write a quick Python script to reverse a string.",
        configured_providers=available,
    )

    assert isinstance(decision, RoutingDecision)
    assert decision.selected_model is not None
    assert decision.selected_provider in available
    assert len(decision.candidate_evaluations) > 0
    assert len(decision.fallback_chain) > 0
    assert decision.selected_model not in decision.fallback_chain


def test_routing_engine_enforces_context_window_filter():
    engine = ModelRoutingEngine()
    available = {"gemini": True, "deepseek": True, "ollama": True}

    # Simulate giant prompt exceeding 32k context of qwen2.5:1.5b
    fake_prompt = "word " * 40_000

    decision = engine.route(
        prompt=fake_prompt,
        configured_providers=available,
    )

    # Models with <= 32k context should be marked ineligible
    rejections = decision.rejections
    assert "qwen2.5:1.5b" in rejections
    assert "exceeds model limit" in rejections["qwen2.5:1.5b"].lower()
    # Selected model must have high context capacity (e.g. Gemini 1M/2M)
    sel = ModelCapabilityRegistry.get_model(decision.selected_model)
    assert sel.context_window > 32_768


def test_routing_engine_enforces_user_constraints():
    engine = ModelRoutingEngine()
    available = {"gemini": True, "deepseek": True, "ollama": True}

    constraints = RoutingConstraints(
        allowed_providers=["gemini"],
        requires_coding=True,
    )

    decision = engine.route(
        prompt="Create a REST API endpoint in FastAPI with token authentication.",
        constraints=constraints,
        configured_providers=available,
    )

    assert decision.selected_provider == "gemini"
    for cand in decision.candidate_evaluations:
        if cand.provider_name != "gemini":
            assert not cand.eligible
            assert "allowed_providers" in cand.ineligibility_reason
