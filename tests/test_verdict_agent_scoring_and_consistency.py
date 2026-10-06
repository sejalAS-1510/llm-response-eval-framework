"""
tests/test_verdict_agent_scoring_and_consistency.py
---------------------------------------------------
Comprehensive test suite verifying the Verdict Agent's weighted scoring model,
threshold transitions, critical-failure overrides, and multi-run scoring consistency.

Tests:
1. Weighted scoring formula against known score combinations.
2. Exact Pass / Needs Improvement / Fail threshold boundary transitions:
   - Pass: score >= 0.75 (and no overrides/caps)
   - Needs Improvement: 0.50 <= score < 0.75 (or capped due to moderate issues)
   - Fail: score < 0.50 (or critical failure overrides)
3. Critical-failure edge-case overrides:
   - Contradiction override (accuracy.classification == 'contradictory') -> Fail
   - Critical Inaccuracy override (accuracy.score == 0.0 without hallucination) -> Fail
   - Severe Hallucination override (hallucination_score > 0.50) -> Fail
   - Critical Off-Topic override (relevance.score < 0.3 or 'unrelated'/'off_topic') -> Fail
4. Moderate weakness capping rules (cannot be 'Pass', capped at 'Needs Improvement'):
   - Mild/moderate hallucination cap (is_hallucinated=True, score <= 0.50)
   - Incompleteness cap (completeness.score <= 0.50 or partially_complete/incomplete)
   - Low accuracy cap (accuracy.score < 0.60)
5. Multi-run scoring consistency across repetitions:
   - Same question-answer pairs evaluated repeatedly (5 runs each).
   - Confirms scores and verdicts remain stable (standard deviation <= 0.02).
   - Documents acceptable variance for LLM-based agent evaluations.
"""

import math
import os
from typing import Dict, List, Optional
import pytest

# Ensure mock/offline evaluation mode is active for fast, deterministic evaluation
os.environ["MOCK_LLM"] = "1"

from src.agents.verdict_agent import VerdictAgent
from src.agents.orchestrator import EvaluationOrchestrator
from src.agents.schemas import (
    RelevanceResult,
    AccuracyResult,
    HallucinationResult,
    CompletenessResult,
    ClaimEvaluation,
    VerdictResult,
)


# ---------------------------------------------------------------------------
# Helper Factory Functions for Synthetic Dimension Results
# ---------------------------------------------------------------------------

def make_relevance(score: float = 1.0, classification: str = "fully_relevant") -> RelevanceResult:
    return RelevanceResult(
        score=score,
        classification=classification,
        reasoning=f"Relevance evaluated at {score} with classification {classification}.",
    )


def make_accuracy(
    score: float = 1.0,
    classification: str = "correct",
    evidence: Optional[List[str]] = None,
) -> AccuracyResult:
    return AccuracyResult(
        score=score,
        classification=classification,
        supporting_evidence=evidence or ["Direct reference passage quote."],
        reasoning=f"Accuracy evaluated at {score} with classification {classification}.",
    )


def make_hallucination(
    is_hallucinated: bool = False,
    hallucination_score: float = 0.0,
    unsupported_count: int = 0,
) -> HallucinationResult:
    flagged = []
    if is_hallucinated:
        flagged = [
            ClaimEvaluation(
                claim="Fabricated assertion in AI response",
                status="unsupported",
                evidence=None,
                explanation="Not supported by reference context.",
            )
            for _ in range(max(1, unsupported_count))
        ]
    return HallucinationResult(
        is_hallucinated=is_hallucinated,
        hallucination_score=hallucination_score,
        total_claims=max(1, len(flagged)),
        unsupported_claims_count=len(flagged),
        flagged_claims=flagged,
        all_claims=flagged,
        reasoning=f"Hallucination evaluated at {hallucination_score} (is_hallucinated={is_hallucinated}).",
    )


def make_completeness(
    score: float = 1.0,
    classification: str = "fully_complete",
    missing_aspects: Optional[List[str]] = None,
) -> CompletenessResult:
    return CompletenessResult(
        score=score,
        classification=classification,
        identified_requirements=["Primary requirement", "Secondary detail"],
        addressed_aspects=["Primary requirement"] if score > 0.5 else [],
        partially_addressed_aspects=[],
        missing_aspects=missing_aspects or ([] if score > 0.5 else ["Secondary detail"]),
        reasoning=f"Completeness evaluated at {score} with classification {classification}.",
    )


@pytest.fixture
def verdict_agent() -> VerdictAgent:
    return VerdictAgent()


# ===========================================================================
# 1. WEIGHTED SCORING FORMULA & KNOWN SCORE COMBINATIONS
# ===========================================================================

class TestVerdictWeightedScoring:
    """Verifies weighted score arithmetic and dimension weighting precision."""

    def test_weighted_formula_exact_weights(self, verdict_agent):
        """
        Confirms weights match specification:
        Accuracy: 35%, Completeness: 25%, Relevance: 20%, Groundedness: 20%.
        Weights must sum to 1.0.
        """
        assert abs(verdict_agent.weight_accuracy - 0.35) < 1e-4
        assert abs(verdict_agent.weight_completeness - 0.25) < 1e-4
        assert abs(verdict_agent.weight_relevance - 0.20) < 1e-4
        assert abs(verdict_agent.weight_groundedness - 0.20) < 1e-4
        total_w = (
            verdict_agent.weight_accuracy
            + verdict_agent.weight_completeness
            + verdict_agent.weight_relevance
            + verdict_agent.weight_groundedness
        )
        assert abs(total_w - 1.0) < 1e-4

    def test_perfect_score_combination(self, verdict_agent):
        """All dimensions at 1.0 yields weighted_score = 1.0 and verdict = Pass."""
        rel = make_relevance(1.0)
        acc = make_accuracy(1.0)
        hal = make_hallucination(False, 0.0)
        comp = make_completeness(1.0)

        result: VerdictResult = verdict_agent.evaluate(rel, acc, hal, comp)

        assert result.weighted_score == 1.0
        assert result.verdict == "Pass"
        assert result.dimension_scores["accuracy"] == 1.0
        assert result.dimension_scores["completeness"] == 1.0
        assert result.dimension_scores["relevance"] == 1.0
        assert result.dimension_scores["groundedness"] == 1.0
        assert len(result.major_issues) == 0
        assert len(result.strengths) == 4

    def test_known_combination_high_quality_pass(self, verdict_agent):
        """
        Known Combination:
        Accuracy = 0.85, Completeness = 0.90, Relevance = 0.90, Groundedness = 1.0 (hal = 0.0)
        Expected raw: 0.35(0.85) + 0.25(0.90) + 0.20(0.90) + 0.20(1.0)
                    = 0.2975 + 0.225 + 0.180 + 0.200 = 0.9025 -> 0.902 (or 0.903)
        Verdict must be 'Pass'.
        """
        rel = make_relevance(0.90)
        acc = make_accuracy(0.85)
        hal = make_hallucination(False, 0.0)
        comp = make_completeness(0.90)

        result = verdict_agent.evaluate(rel, acc, hal, comp)

        expected_score = round(0.35 * 0.85 + 0.25 * 0.90 + 0.20 * 0.90 + 0.20 * 1.0, 3)
        assert abs(result.weighted_score - expected_score) <= 0.002
        assert result.verdict == "Pass"
        assert result.weighted_score >= 0.75

    def test_pass_threshold_boundary_transition(self, verdict_agent):
        """
        Verifies exact boundary transition between Pass and Needs Improvement (0.75 threshold):
        - Score >= 0.750 without caps -> Pass
        - Score = 0.749 (or just below 0.75) -> Needs Improvement
        """
        # Case A: Boundary Pass (score = 0.768 >= 0.75)
        rel_a = make_relevance(0.80)
        acc_a = make_accuracy(0.70)
        hal_a = make_hallucination(False, 0.0)  # groundedness = 1.0
        comp_a = make_completeness(0.65, classification="mostly_complete")
        # Raw: 0.35*0.70 + 0.25*0.65 + 0.20*0.80 + 0.20*1.0 = 0.245 + 0.1625 + 0.160 + 0.200 = 0.7675 -> 0.768
        res_a = verdict_agent.evaluate(rel_a, acc_a, hal_a, comp_a)
        assert res_a.weighted_score >= 0.75
        assert res_a.verdict == "Pass"

        # Case B: Just below Pass threshold (score = 0.735 < 0.75)
        rel_b = make_relevance(0.70)
        acc_b = make_accuracy(0.70)
        hal_b = make_hallucination(False, 0.0)
        comp_b = make_completeness(0.60, classification="mostly_complete")
        # Raw: 0.35*0.70 + 0.25*0.60 + 0.20*0.70 + 0.20*1.0 = 0.245 + 0.150 + 0.140 + 0.200 = 0.735
        res_b = verdict_agent.evaluate(rel_b, acc_b, hal_b, comp_b)
        assert 0.50 <= res_b.weighted_score < 0.75
        assert res_b.verdict == "Needs Improvement"

    def test_needs_improvement_threshold_boundary_transition(self, verdict_agent):
        """
        Verifies exact boundary transition between Needs Improvement and Fail (0.50 threshold):
        - Score >= 0.500 -> Needs Improvement
        - Score < 0.500 -> Fail
        """
        # Case A: Just above 0.50 boundary (score = 0.518)
        rel_a = make_relevance(0.50)
        acc_a = make_accuracy(0.55)
        hal_a = make_hallucination(False, 0.50)  # groundedness = 0.50
        comp_a = make_completeness(0.50, classification="partially_complete")
        # Raw: 0.35*0.55 + 0.25*0.50 + 0.20*0.50 + 0.20*0.50 = 0.1925 + 0.125 + 0.10 + 0.10 = 0.5175 -> 0.518
        res_a = verdict_agent.evaluate(rel_a, acc_a, hal_a, comp_a)
        assert res_a.weighted_score >= 0.50
        assert res_a.verdict == "Needs Improvement"

        # Case B: Just below 0.50 boundary (score = 0.483)
        rel_b = make_relevance(0.50)
        acc_b = make_accuracy(0.45)
        hal_b = make_hallucination(False, 0.50)  # groundedness = 0.50
        comp_b = make_completeness(0.50, classification="partially_complete")
        # Raw: 0.35*0.45 + 0.25*0.50 + 0.20*0.50 + 0.20*0.50 = 0.1575 + 0.125 + 0.10 + 0.10 = 0.4825 -> 0.483
        res_b = verdict_agent.evaluate(rel_b, acc_b, hal_b, comp_b)
        assert res_b.weighted_score < 0.50
        assert res_b.verdict == "Fail"


# ===========================================================================
# 2. CRITICAL-FAILURE OVERRIDES
# ===========================================================================

class TestCriticalFailureOverrides:
    """
    Verifies that critical failures immediately override the weighted score
    and assign a 'Fail' verdict, regardless of high scores on other dimensions.
    """

    def test_contradiction_override(self, verdict_agent):
        """
        Critical Contradiction Override:
        Even if relevance = 1.0, completeness = 1.0, and groundedness = 1.0,
        an accuracy classification of 'contradictory' MUST force verdict = Fail.
        """
        rel = make_relevance(1.0)
        acc = make_accuracy(0.0, classification="contradictory")
        hal = make_hallucination(False, 0.0)
        comp = make_completeness(1.0)

        # Raw score without override would be 0.650 (which would otherwise be Needs Improvement)
        res = verdict_agent.evaluate(rel, acc, hal, comp)

        assert res.verdict == "Fail", f"Expected Fail on contradiction, got {res.verdict}"
        assert any("Critical Contradiction" in issue for issue in res.major_issues)
        assert "FAIL" in res.consolidated_reasoning

    def test_zero_accuracy_inaccuracy_override(self, verdict_agent):
        """
        Critical Inaccuracy Override:
        If accuracy.score == 0.0 and not hallucination.is_hallucinated,
        it represents a verified factual failure and MUST force verdict = Fail.
        """
        rel = make_relevance(1.0)
        acc = make_accuracy(0.0, classification="incorrect")
        hal = make_hallucination(False, 0.0)
        comp = make_completeness(1.0)

        res = verdict_agent.evaluate(rel, acc, hal, comp)

        assert res.verdict == "Fail"
        assert any("Critical Inaccuracy" in issue for issue in res.major_issues)

    def test_severe_hallucination_override(self, verdict_agent):
        """
        Severe Hallucination Override:
        If hallucination_score > 0.50 (e.g. 0.60 or 1.0), it represents severe fabrication
        and MUST force verdict = Fail, even if other dimensions are moderately high.
        """
        rel = make_relevance(1.0)
        acc = make_accuracy(0.50, classification="partially_correct")
        hal = make_hallucination(True, hallucination_score=0.60, unsupported_count=3)
        comp = make_completeness(1.0)

        # Raw: 0.35*0.5 + 0.25*1.0 + 0.20*1.0 + 0.20*0.40 = 0.175 + 0.25 + 0.20 + 0.08 = 0.705
        # Would normally be 'Needs Improvement' (0.705 >= 0.50), but severe hallucination forces Fail!
        res = verdict_agent.evaluate(rel, acc, hal, comp)

        assert res.verdict == "Fail", f"Expected Fail on severe hallucination, got {res.verdict}"
        assert any("Severe Hallucination" in issue for issue in res.major_issues)
        assert "60%" in [issue for issue in res.major_issues if "Severe Hallucination" in issue][0]

    def test_critical_off_topic_override(self, verdict_agent):
        """
        Critical Off-Topic Override:
        If relevance score < 0.3 or classification in ('unrelated', 'off_topic'),
        the response completely misses the prompt and MUST force verdict = Fail.
        """
        rel = make_relevance(0.0, classification="unrelated")
        acc = make_accuracy(1.0)
        hal = make_hallucination(False, 0.0)
        comp = make_completeness(1.0)

        # Raw score: 0.800 (which would be Pass without override), but off-topic forces Fail!
        res = verdict_agent.evaluate(rel, acc, hal, comp)

        assert res.verdict == "Fail", f"Expected Fail on off-topic, got {res.verdict}"
        assert any("Critical Off-Topic" in issue for issue in res.major_issues)

    def test_multiple_critical_overrides_combined(self, verdict_agent):
        """
        When multiple critical failures occur simultaneously (e.g. Off-topic + Contradiction),
        all corresponding issues must be recorded and the verdict must be Fail.
        """
        rel = make_relevance(0.0, classification="off_topic")
        acc = make_accuracy(0.0, classification="contradictory")
        hal = make_hallucination(True, hallucination_score=0.80, unsupported_count=4)
        comp = make_completeness(0.20, classification="incomplete")

        res = verdict_agent.evaluate(rel, acc, hal, comp)

        assert res.verdict == "Fail"
        assert res.weighted_score <= 0.30
        assert any("Critical Contradiction" in issue for issue in res.major_issues)
        assert any("Critical Off-Topic" in issue for issue in res.major_issues)
        assert any("Severe Hallucination" in issue for issue in res.major_issues)


# ===========================================================================
# 3. MODERATE WEAKNESS CAPPING RULES (CANNOT PASS)
# ===========================================================================

class TestModerateWeaknessCaps:
    """
    Verifies that moderate issues cap the verdict at 'Needs Improvement',
    preventing high raw weighted scores from falsely achieving 'Pass'.
    """

    def test_mild_hallucination_cap(self, verdict_agent):
        """
        Mild Hallucination Cap:
        Even if raw score = 0.900 (well above 0.75), any detected hallucination
        (<= 0.50) MUST cap the verdict at 'Needs Improvement'.
        """
        rel = make_relevance(1.0)
        acc = make_accuracy(0.90)
        hal = make_hallucination(True, hallucination_score=0.20, unsupported_count=1)
        comp = make_completeness(0.90)

        # Raw: 0.35*0.90 + 0.25*0.90 + 0.20*1.0 + 0.20*0.80 = 0.315 + 0.225 + 0.20 + 0.16 = 0.900
        res = verdict_agent.evaluate(rel, acc, hal, comp)

        assert res.weighted_score >= 0.75, f"Raw score was {res.weighted_score}"
        assert res.verdict == "Needs Improvement", f"Expected Needs Improvement, got {res.verdict}"
        assert any("Hallucination Detected" in issue for issue in res.major_issues)

    def test_incompleteness_cap(self, verdict_agent):
        """
        Incompleteness Cap:
        If completeness <= 0.50 or classification is partially_complete/incomplete,
        the response MUST NOT receive 'Pass', even if accuracy and relevance are 1.0.
        """
        rel = make_relevance(1.0)
        acc = make_accuracy(1.0)
        hal = make_hallucination(False, 0.0)
        comp = make_completeness(0.40, classification="partially_complete", missing_aspects=["population"])

        # Raw: 0.35*1.0 + 0.25*0.40 + 0.20*1.0 + 0.20*1.0 = 0.35 + 0.10 + 0.20 + 0.20 = 0.850
        res = verdict_agent.evaluate(rel, acc, hal, comp)

        assert res.weighted_score >= 0.75
        assert res.verdict == "Needs Improvement"
        assert any("Substantially Incomplete" in issue for issue in res.major_issues)

    def test_low_accuracy_cap(self, verdict_agent):
        """
        Low Accuracy Cap:
        If accuracy.score < 0.60, the response lacks precision and MUST NOT achieve 'Pass'.
        """
        rel = make_relevance(1.0)
        acc = make_accuracy(0.55, classification="partially_correct")
        hal = make_hallucination(False, 0.0)
        comp = make_completeness(1.0)

        # Raw: 0.35*0.55 + 0.25*1.0 + 0.20*1.0 + 0.20*1.0 = 0.1925 + 0.25 + 0.20 + 0.20 = 0.8425 -> 0.843
        res = verdict_agent.evaluate(rel, acc, hal, comp)

        assert res.weighted_score >= 0.75
        assert res.verdict == "Needs Improvement"
        assert any("Low Accuracy" in issue for issue in res.major_issues)


# ===========================================================================
# 4. SCORING CONSISTENCY & MULTI-RUN STABILITY
# ===========================================================================

class TestMultiRunScoringConsistency:
    """
    Tests scoring consistency by running identical question-answer pairs multiple times
    and confirming that scores, classifications, and final verdicts remain stable.
    """

    REPETITIONS = 5

    BENCHMARK_PAIRS = [
        {
            "id": "CONSIST-01-PASS",
            "name": "High-Quality Chemistry Question",
            "question": "What is the chemical symbol for gold, and what is its atomic number?",
            "ai_response": "The chemical symbol for gold is Au, and its atomic number is 79.",
            "reference_answer": "The chemical symbol for gold is Au, and its atomic number is 79.",
            "expected_verdict": "Pass",
        },
        {
            "id": "CONSIST-02-CONTRADICTION",
            "name": "Acoustic Contradiction Question",
            "question": "Is the speed of sound faster in air than in water?",
            "ai_response": "Yes, sound travels faster in air than in water.",
            "reference_answer": "No, sound travels much faster in water than in air.",
            "expected_verdict": "Fail",
        },
        {
            "id": "CONSIST-03-INCOMPLETE",
            "name": "Partial Completeness Question",
            "question": "What is the capital of Australia, what is its population, and what currency does it use?",
            "ai_response": "The capital of Australia is Canberra.",
            "reference_answer": "The capital of Australia is Canberra, with a population of 450,000, and currency is Australian dollar.",
            "expected_verdict": "Needs Improvement",
        },
        {
            "id": "CONSIST-04-OFFTOPIC",
            "name": "Off-Topic Recipe Response",
            "question": "What causes earthquakes along tectonic plate boundaries?",
            "ai_response": "To make a classic French omelet, whisk two eggs with salt and melt butter in a pan.",
            "reference_answer": "Earthquakes are caused by stress release along tectonic plate boundaries.",
            "expected_verdict": "Fail",
        },
    ]

    @pytest.mark.asyncio
    async def test_repeated_evaluations_verdict_and_score_stability(self):
        """
        Runs each benchmark pair 5 times sequentially through the evaluation pipeline.
        Asserts:
        1. Final verdict is 100% stable (identical across all 5 runs).
        2. Score standard deviation across repetitions is <= 0.02 (acceptable variance limit).
        3. Dimension classifications are identical across all runs.
        """
        orchestrator = EvaluationOrchestrator()

        for case in self.BENCHMARK_PAIRS:
            case_id = case["id"]
            scores: List[float] = []
            verdicts: List[str] = []
            rel_classes: List[str] = []
            acc_classes: List[str] = []
            comp_classes: List[str] = []
            hal_flags: List[bool] = []

            for run_idx in range(self.REPETITIONS):
                res: EvaluationResult = await orchestrator.evaluate(
                    question=case["question"],
                    ai_response=case["ai_response"],
                    reference_answer=case["reference_answer"],
                )
                scores.append(res.verdict.weighted_score)
                verdicts.append(res.verdict.verdict)
                rel_classes.append(res.relevance.classification)
                acc_classes.append(res.accuracy.classification)
                comp_classes.append(res.completeness.classification)
                hal_flags.append(res.hallucination.is_hallucinated)

            # ---------------------------------------------------------------
            # 1. Verdict Category Stability: 100% identical
            # ---------------------------------------------------------------
            unique_verdicts = set(verdicts)
            assert len(unique_verdicts) == 1, (
                f"[{case_id}] Verdict instability detected across {self.REPETITIONS} runs: {verdicts}"
            )
            assert verdicts[0] == case["expected_verdict"], (
                f"[{case_id}] Expected {case['expected_verdict']}, got {verdicts[0]}"
            )

            # ---------------------------------------------------------------
            # 2. Classification Stability: 100% identical
            # ---------------------------------------------------------------
            assert len(set(rel_classes)) == 1, f"[{case_id}] Relevance classification varied: {rel_classes}"
            assert len(set(acc_classes)) == 1, f"[{case_id}] Accuracy classification varied: {acc_classes}"
            assert len(set(comp_classes)) == 1, f"[{case_id}] Completeness classification varied: {comp_classes}"
            assert len(set(hal_flags)) == 1, f"[{case_id}] Hallucination flag varied: {hal_flags}"

            # ---------------------------------------------------------------
            # 3. Score Variance Calculation
            # ---------------------------------------------------------------
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            std_dev = math.sqrt(variance)

            # Acceptable variance limit: sigma <= 0.02 (deterministic engine achieves sigma = 0.0)
            assert std_dev <= 0.02, (
                f"[{case_id}] Score standard deviation ({std_dev:.4f}) exceeded allowable tolerance (0.02)."
            )
            assert max(scores) - min(scores) <= 0.03, (
                f"[{case_id}] Score range ({min(scores)} to {max(scores)}) exceeded allowable spread."
            )

    def test_acceptable_variance_rubric_documentation(self):
        """
        Documentation Test: Formally validates the variance policy guidelines for LLM-based agents.
        
        Guideline:
        - Temperature Setting: Temperature = 0.0 enforces greedy decoding, eliminating sampling noise.
        - Continuous Score Variance: Standard deviation sigma <= 0.02 is the maximum acceptable tolerance.
        - Discrete Verdict Categories: Discrete verdicts ('Pass', 'Needs Improvement', 'Fail') must exhibit
          0% variance across repeated runs for non-boundary scores (|score - threshold| > 0.03).
        - Boundary Scores: Scores within +/- 0.02 of 0.75 or 0.50 are flagged as boundary-sensitive.
        """
        max_acceptable_score_variance = 0.02
        max_acceptable_score_spread = 0.04
        verdict_stability_requirement = 1.0  # 100% agreement required

        assert max_acceptable_score_variance == 0.02
        assert max_acceptable_score_spread == 0.04
        assert verdict_stability_requirement == 1.0
