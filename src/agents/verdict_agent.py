"""
M3.2: Verdict Agent & Weighted Evaluation.
Combines Relevance, Accuracy, Hallucination Detection, and Completeness scores into
a single overall evaluation using a weighted scoring model and critical failure overrides.

NOTE: As per design constraints, the Verdict Agent aggregates already-computed evaluations
and does NOT re-evaluate dimensions or trigger redundant LLM calls.
"""

from typing import Dict, List, Optional
from .schemas import (
    RelevanceResult,
    AccuracyResult,
    HallucinationResult,
    CompletenessResult,
    VerdictResult,
    VerdictCategory,
)


class VerdictAgent:
    """
    Verdict Agent aggregates all 4 evaluation dimensions using a weighted scoring
    model, enforces safety guardrails/overrides, and produces a consolidated verdict.
    """
    def __init__(
        self,
        weight_accuracy: float = 0.35,
        weight_completeness: float = 0.25,
        weight_relevance: float = 0.20,
        weight_groundedness: float = 0.20,
        pass_threshold: float = 0.75,
        needs_improvement_threshold: float = 0.50,
    ):
        self.weight_accuracy = weight_accuracy
        self.weight_completeness = weight_completeness
        self.weight_relevance = weight_relevance
        self.weight_groundedness = weight_groundedness
        self.pass_threshold = pass_threshold
        self.needs_improvement_threshold = needs_improvement_threshold

        # Ensure weights sum to 1.0
        total_w = weight_accuracy + weight_completeness + weight_relevance + weight_groundedness
        if abs(total_w - 1.0) > 1e-4:
            self.weight_accuracy /= total_w
            self.weight_completeness /= total_w
            self.weight_relevance /= total_w
            self.weight_groundedness /= total_w

    def evaluate(
        self,
        relevance: RelevanceResult,
        accuracy: AccuracyResult,
        hallucination: HallucinationResult,
        completeness: CompletenessResult,
    ) -> VerdictResult:
        """
        Calculates the weighted evaluation score, applies critical failure overrides,
        and constructs the consolidated executive reasoning.
        """
        # Groundedness is the complement of hallucination score
        groundedness_score = max(0.0, min(1.0, 1.0 - hallucination.hallucination_score))

        # Normalized dimension scores
        dimension_scores: Dict[str, float] = {
            "relevance": round(float(relevance.score), 3),
            "accuracy": round(float(accuracy.score), 3),
            "completeness": round(float(completeness.score), 3),
            "groundedness": round(float(groundedness_score), 3),
        }

        # Calculate base weighted score
        raw_weighted_score = (
            self.weight_accuracy * accuracy.score
            + self.weight_completeness * completeness.score
            + self.weight_relevance * relevance.score
            + self.weight_groundedness * groundedness_score
        )
        weighted_score = round(max(0.0, min(1.0, raw_weighted_score)), 3)

        major_issues: List[str] = []
        strengths: List[str] = []
        critical_fail = False
        cap_needs_improvement = False

        # -------------------------------------------------------------
        # 1. Critical Failure Overrides
        # -------------------------------------------------------------
        # Contradiction override
        if accuracy.classification == "contradictory":
            critical_fail = True
            major_issues.append("Critical Contradiction: Response contains statements directly contradicted by ground-truth reference.")
        elif accuracy.score == 0.0 and not hallucination.is_hallucinated:
            critical_fail = True
            major_issues.append("Critical Inaccuracy: Response failed factual verification.")

        # Severe hallucination override (> 50% claims fabricated)
        if hallucination.is_hallucinated and hallucination.hallucination_score > 0.50:
            critical_fail = True
            major_issues.append(f"Severe Hallucination: {int(hallucination.hallucination_score * 100)}% of extracted factual claims are ungrounded or fabricated.")

        # Completely off-topic or unrelated
        if relevance.score < 0.3 or relevance.classification in ("unrelated", "off_topic"):
            critical_fail = True
            major_issues.append("Critical Off-Topic: Response fails to address the user's question or intent.")

        # -------------------------------------------------------------
        # 2. Moderate Issue Caps (Cannot be 'Pass')
        # -------------------------------------------------------------
        if hallucination.is_hallucinated and not critical_fail:
            cap_needs_improvement = True
            major_issues.append(f"Hallucination Detected: Contains {hallucination.unsupported_claims_count} claim(s) unsupported by the reference knowledge base.")

        if (completeness.score <= 0.50 or completeness.classification in ("partially_complete", "incomplete")) and not critical_fail:
            cap_needs_improvement = True
            missing_info = ", ".join(completeness.missing_aspects[:2]) if completeness.missing_aspects else "key sub-questions"
            major_issues.append(f"Substantially Incomplete: Omitted critical aspects ({missing_info}).")

        if accuracy.score < 0.60 and not critical_fail:
            cap_needs_improvement = True
            major_issues.append("Low Accuracy: Multiple factual assertions lack precision or support.")

        if completeness.missing_aspects and completeness.score > 0.50:
            major_issues.append(f"Omission: Left out {len(completeness.missing_aspects)} sub-aspect(s): {', '.join(completeness.missing_aspects[:2])}.")

        # -------------------------------------------------------------
        # 3. Identify Strengths
        # -------------------------------------------------------------
        if relevance.score >= 0.85:
            strengths.append("High Relevance: Directly addresses the user's question and prompt parameters.")
        if accuracy.score >= 0.85:
            strengths.append("High Accuracy: Factual statements align tightly with ground truth evidence.")
        if not hallucination.is_hallucinated and groundedness_score >= 0.95:
            strengths.append("Grounded Content: Zero hallucinations or ungrounded assertions detected.")
        if completeness.score >= 0.85:
            strengths.append("Comprehensive Coverage: Answers all sub-questions and key requirements.")

        # -------------------------------------------------------------
        # 4. Final Verdict Determination
        # -------------------------------------------------------------
        if critical_fail or weighted_score < self.needs_improvement_threshold:
            verdict: VerdictCategory = "Fail"
        elif cap_needs_improvement or weighted_score < self.pass_threshold:
            verdict: VerdictCategory = "Needs Improvement"
        else:
            verdict: VerdictCategory = "Pass"

        # -------------------------------------------------------------
        # 5. Consolidated Executive Reasoning
        # -------------------------------------------------------------
        reasoning_paragraphs = []

        if verdict == "Pass":
            reasoning_paragraphs.append(
                f"OVERALL VERDICT: PASS (Weighted Score: {weighted_score * 100:.1f}%). "
                "The response meets quality standards across all evaluation dimensions."
            )
        elif verdict == "Needs Improvement":
            reasoning_paragraphs.append(
                f"OVERALL VERDICT: NEEDS IMPROVEMENT (Weighted Score: {weighted_score * 100:.1f}%). "
                "The response is functional but requires revision to address highlighted quality or coverage issues."
            )
        else:
            reasoning_paragraphs.append(
                f"OVERALL VERDICT: FAIL (Weighted Score: {weighted_score * 100:.1f}%). "
                "The response failed critical quality standards due to factual inaccuracies, severe omissions, or ungrounded content."
            )

        # Dimension breakdown narrative
        breakdown_text = (
            f"Dimension Breakdown: Accuracy={dimension_scores['accuracy'] * 100:.0f}% (weight: {self.weight_accuracy:.0%}), "
            f"Completeness={dimension_scores['completeness'] * 100:.0f}% (weight: {self.weight_completeness:.0%}), "
            f"Relevance={dimension_scores['relevance'] * 100:.0f}% (weight: {self.weight_relevance:.0%}), "
            f"Groundedness={dimension_scores['groundedness'] * 100:.0f}% (weight: {self.weight_groundedness:.0%})."
        )
        reasoning_paragraphs.append(breakdown_text)

        if strengths:
            reasoning_paragraphs.append("Key Strengths: " + "; ".join(strengths))

        if major_issues:
            reasoning_paragraphs.append("Issues Identified: " + "; ".join(major_issues))

        consolidated_reasoning = "\n\n".join(reasoning_paragraphs)

        return VerdictResult(
            weighted_score=weighted_score,
            verdict=verdict,
            dimension_scores=dimension_scores,
            major_issues=major_issues,
            strengths=strengths,
            consolidated_reasoning=consolidated_reasoning,
        )
