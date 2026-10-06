"""
src/reporting/recommendations.py
---------------------------------
Recommendation module for LLM response evaluation.
Scans aggregated batch report data and generates 2-5 actionable, plain-English
improvement recommendations based on recurring weaknesses and empirical frequency thresholds.
Avoids hardcoded text by computing dynamic metrics (exact percentages, counts, and samples)
from stored evaluation results.
"""

import re
import logging
from typing import List, Optional, Dict, Any, Union

from src.reporting.schemas import (
    BatchReportData,
    RecommendationItem,
    PerResponseReportDetail,
    OverallStats,
    HallucinationFrequency,
)

logger = logging.getLogger(__name__)

# Regex pattern for identifying numeric, statistical, date, or financial claims
NUMERIC_STATISTICAL_PATTERN = re.compile(
    r"\b\d+([.,]\d+)?%?|\$|€|£|¥|\b(percent|percentage|average|ratio|year|century|million|billion|thousand)\b",
    re.IGNORECASE,
)


class RecommendationEngine:
    """
    Diagnostic recommendation engine that analyzes BatchReportData across
    dimensions (accuracy, completeness, relevance, grounding, pipeline)
    and produces 2 to 5 prioritized, plain-English improvement recommendations.
    """

    def __init__(
        self,
        hallucination_threshold_pct: float = 15.0,
        numeric_claim_threshold_pct: float = 10.0,
        completeness_threshold_pct: float = 15.0,
        accuracy_threshold_pct: float = 15.0,
        relevance_threshold_pct: float = 15.0,
        non_pass_threshold_pct: float = 35.0,
        pipeline_fail_threshold_pct: float = 5.0,
    ):
        self.hallucination_threshold_pct = hallucination_threshold_pct
        self.numeric_claim_threshold_pct = numeric_claim_threshold_pct
        self.completeness_threshold_pct = completeness_threshold_pct
        self.accuracy_threshold_pct = accuracy_threshold_pct
        self.relevance_threshold_pct = relevance_threshold_pct
        self.non_pass_threshold_pct = non_pass_threshold_pct
        self.pipeline_fail_threshold_pct = pipeline_fail_threshold_pct

    def generate(self, report_data: BatchReportData) -> List[RecommendationItem]:
        """
        Main entrypoint: scans report_data and returns 2 to 5 prioritized RecommendationItem objects.
        Guarantees that length is between 2 and 5.
        """
        candidates: List[RecommendationItem] = []
        succ_items = [r for r in report_data.per_response_details if r.status == "success"]
        succ_count = len(succ_items)
        total_records = report_data.metadata.total_records or len(report_data.per_response_details)
        stats = report_data.overall_stats

        # -------------------------------------------------------------------
        # Rule 1: Unsupported Numeric & Statistical Claims
        # -------------------------------------------------------------------
        if succ_count > 0:
            numeric_responses = []
            numeric_sample_claims = []
            for r in succ_items:
                has_num_issue = False
                for claim_item in r.flagged_claims_breakdown:
                    text_to_check = f"{claim_item.claim} {claim_item.explanation or ''}"
                    if NUMERIC_STATISTICAL_PATTERN.search(text_to_check):
                        has_num_issue = True
                        if claim_item.claim and claim_item.claim not in numeric_sample_claims:
                            numeric_sample_claims.append(claim_item.claim)
                # Also check plain flagged claims list if breakdown is empty
                if not has_num_issue and r.flagged_hallucinated_claims:
                    for c_str in r.flagged_hallucinated_claims:
                        if NUMERIC_STATISTICAL_PATTERN.search(c_str):
                            has_num_issue = True
                            if c_str not in numeric_sample_claims:
                                numeric_sample_claims.append(c_str)
                if has_num_issue:
                    numeric_responses.append(r)

            num_count = len(numeric_responses)
            num_pct = round(num_count / succ_count * 100, 1)

            # Trigger if threshold met or if small batch with at least 1 instance
            if (num_pct >= self.numeric_claim_threshold_pct) or (succ_count <= 5 and num_count >= 1):
                priority = "high" if num_pct >= 25.0 else "medium"
                candidates.append(
                    RecommendationItem(
                        id="rec-grounding-numeric",
                        category="grounding",
                        priority=priority,
                        headline="Unsupported Numeric & Statistical Claims",
                        plain_english=(
                            f"{num_pct}% of responses had unsupported numeric claims — "
                            f"consider stricter grounding for statistics and quantitative data."
                        ),
                        metric_trigger=f"{num_pct}% of responses ({num_count}/{succ_count}) had unsupported numeric or statistical claims",
                        remediation_advice=(
                            "Consider stricter grounding for statistics, verify quantitative facts against source tables, "
                            "and enforce numeric constraint validation before returning outputs."
                        ),
                        affected_count=num_count,
                        affected_percent=num_pct,
                        dimension="groundedness",
                        sample_issues=numeric_sample_claims[:3],
                    )
                )

        # -------------------------------------------------------------------
        # Rule 2: General Ungrounded & Fabricated Claims (Hallucination Rate)
        # -------------------------------------------------------------------
        if succ_count > 0:
            hal_freq = report_data.hallucination_frequency
            hal_count = hal_freq.flagged_responses_count
            hal_pct = hal_freq.rate_percent

            if (hal_pct >= self.hallucination_threshold_pct) or (succ_count <= 5 and hal_count >= 1):
                priority = "high" if hal_pct >= 25.0 else "medium"
                candidates.append(
                    RecommendationItem(
                        id="rec-grounding-hallucinations",
                        category="grounding",
                        priority=priority,
                        headline="High Ungrounded Hallucination Frequency",
                        plain_english=(
                            f"{hal_pct}% of responses contained ungrounded or contradicted claims — "
                            f"enforce stricter RAG retrieval grounding and reduce sampling temperature."
                        ),
                        metric_trigger=f"{hal_pct}% of responses ({hal_count}/{succ_count}) contained ungrounded or contradicted claims",
                        remediation_advice=(
                            "Enforce strict retrieval-augmented generation (RAG) context grounding, require explicit verbatim citations, "
                            "and lower model temperature to reduce generative extrapolation."
                        ),
                        affected_count=hal_count,
                        affected_percent=hal_pct,
                        dimension="groundedness",
                        sample_issues=hal_freq.sample_flagged_claims[:3],
                    )
                )

        # -------------------------------------------------------------------
        # Rule 3: Missing Required Aspects & Incomplete Responses
        # -------------------------------------------------------------------
        if succ_count > 0:
            comp_responses = [
                r for r in succ_items
                if (r.missing_aspects and len(r.missing_aspects) > 0)
                or (r.completeness and r.completeness.score < 0.70)
            ]
            comp_count = len(comp_responses)
            comp_pct = round(comp_count / succ_count * 100, 1)

            sample_missing: List[str] = []
            for r in comp_responses:
                for ma in r.missing_aspects:
                    if ma and ma not in sample_missing:
                        sample_missing.append(ma)
                    if len(sample_missing) >= 3:
                        break
                if len(sample_missing) >= 3:
                    break

            if (comp_pct >= self.completeness_threshold_pct) or (succ_count <= 5 and comp_count >= 1):
                priority = "high" if comp_pct >= 30.0 else "medium"
                candidates.append(
                    RecommendationItem(
                        id="rec-completeness-missing-aspects",
                        category="completeness",
                        priority=priority,
                        headline="Omitted Prompt Constraints & Sub-Questions",
                        plain_english=(
                            f"{comp_pct}% of responses missed required criteria or sub-questions — "
                            f"implement checklist-based prompt templates to ensure full question coverage."
                        ),
                        metric_trigger=f"{comp_pct}% of responses ({comp_count}/{succ_count}) omitted required criteria or sub-questions",
                        remediation_advice=(
                            "Incorporate structured sub-question decomposition or chain-of-thought checklist prompts "
                            "to ensure all query aspects and constraints are exhaustively addressed."
                        ),
                        affected_count=comp_count,
                        affected_percent=comp_pct,
                        dimension="completeness",
                        sample_issues=sample_missing,
                    )
                )

        # -------------------------------------------------------------------
        # Rule 4: Low Factual Consistency / Reference Divergence
        # -------------------------------------------------------------------
        if succ_count > 0:
            acc_responses = [
                r for r in succ_items
                if r.accuracy and r.accuracy.score < 0.70
            ]
            acc_count = len(acc_responses)
            acc_pct = round(acc_count / succ_count * 100, 1)

            sample_acc_issues: List[str] = []
            for r in acc_responses:
                if r.accuracy and r.accuracy.reasoning:
                    sample_acc_issues.append(r.accuracy.reasoning)
                    if len(sample_acc_issues) >= 3:
                        break

            if (acc_pct >= self.accuracy_threshold_pct) or (succ_count <= 5 and acc_count >= 1):
                priority = "high" if acc_pct >= 25.0 else "medium"
                candidates.append(
                    RecommendationItem(
                        id="rec-accuracy-factual-consistency",
                        category="accuracy",
                        priority=priority,
                        headline="Factual Accuracy & Knowledge Inconsistencies",
                        plain_english=(
                            f"{acc_pct}% of responses exhibited factual inconsistencies — "
                            f"add an automated fact-verification critique step against trusted reference documentation."
                        ),
                        metric_trigger=f"{acc_pct}% of responses ({acc_count}/{succ_count}) scored below 70% in factual accuracy",
                        remediation_advice=(
                            "Introduce an automated fact-verification critique step comparing generated statements directly "
                            "against trusted reference answers before finalizing responses."
                        ),
                        affected_count=acc_count,
                        affected_percent=acc_pct,
                        dimension="accuracy",
                        sample_issues=sample_acc_issues,
                    )
                )

        # -------------------------------------------------------------------
        # Rule 5: Instruction Drift & Low Relevance
        # -------------------------------------------------------------------
        if succ_count > 0:
            rel_responses = [
                r for r in succ_items
                if r.relevance and r.relevance.score < 0.70
            ]
            rel_count = len(rel_responses)
            rel_pct = round(rel_count / succ_count * 100, 1)

            if (rel_pct >= self.relevance_threshold_pct) or (succ_count <= 5 and rel_count >= 1):
                priority = "high" if rel_pct >= 30.0 else "medium"
                candidates.append(
                    RecommendationItem(
                        id="rec-relevance-instruction-drift",
                        category="relevance",
                        priority=priority,
                        headline="Instruction Drift & Off-Topic Elaboration",
                        plain_english=(
                            f"{rel_pct}% of responses suffered from instruction drift or off-topic elaboration — "
                            f"tighten system prompt constraints and enforce concise response bounds."
                        ),
                        metric_trigger=f"{rel_pct}% of responses ({rel_count}/{succ_count}) exhibited relevance drift or failed to adhere to prompt constraints",
                        remediation_advice=(
                            "Tighten system prompt instructions with explicit negative constraints and enforce "
                            "concise response bounds to eliminate verbose tangential responses."
                        ),
                        affected_count=rel_count,
                        affected_percent=rel_pct,
                        dimension="relevance",
                        sample_issues=[],
                    )
                )

        # -------------------------------------------------------------------
        # Rule 6: High Overall Defect Rate (Verdict Needs Improvement + Fail)
        # -------------------------------------------------------------------
        if succ_count > 0:
            non_pass_count = stats.needs_improvement_count + stats.fail_count
            non_pass_pct = round(non_pass_count / succ_count * 100, 1)

            if non_pass_pct >= self.non_pass_threshold_pct:
                priority = "high" if non_pass_pct >= 50.0 else "medium"
                candidates.append(
                    RecommendationItem(
                        id="rec-quality-non-pass-rate",
                        category="general_quality",
                        priority=priority,
                        headline="Elevated Batch Failure & Needs-Improvement Rate",
                        plain_english=(
                            f"{non_pass_pct}% of responses failed to meet passing quality thresholds — "
                            f"prioritize few-shot prompt adjustments on the lowest-scoring failure clusters."
                        ),
                        metric_trigger=f"{non_pass_pct}% of responses ({non_pass_count}/{succ_count}) failed to receive a clean Pass verdict",
                        remediation_advice=(
                            "Target few-shot demonstration examples and prompt revisions toward the lowest-performing "
                            "prompt clusters identified in the batch failure breakdown."
                        ),
                        affected_count=non_pass_count,
                        affected_percent=non_pass_pct,
                        dimension="composite",
                        sample_issues=[],
                    )
                )

        # -------------------------------------------------------------------
        # Rule 7: Pipeline Processing Failures or Ingestion Errors
        # -------------------------------------------------------------------
        failed_records = report_data.metadata.failed_records
        if failed_records > 0 and total_records > 0:
            fail_pct = round(failed_records / total_records * 100, 1)
            if (fail_pct >= self.pipeline_fail_threshold_pct) or (failed_records >= 1):
                priority = "high" if fail_pct >= 10.0 else "medium"
                candidates.append(
                    RecommendationItem(
                        id="rec-pipeline-ingestion-failures",
                        category="pipeline",
                        priority=priority,
                        headline="Input Data Ingestion & Evaluation Failures",
                        plain_english=(
                            f"{fail_pct}% of batch rows failed during evaluation — "
                            f"sanitize input dataset headers and remove empty question rows."
                        ),
                        metric_trigger=f"{fail_pct}% of submission rows ({failed_records}/{total_records}) encountered processing errors or missing fields",
                        remediation_advice=(
                            "Sanitize input CSV files before submission, verify question column headers, "
                            "and filter out empty query rows to ensure 100% evaluation completion."
                        ),
                        affected_count=failed_records,
                        affected_percent=fail_pct,
                        dimension="pipeline",
                        sample_issues=[],
                    )
                )

        # -------------------------------------------------------------------
        # Guarantee 2 to 5 Recommendations: Add Fine-tuning Opportunities if needed
        # -------------------------------------------------------------------
        if len(candidates) < 2 and succ_count > 0:
            # Opportunity A: Identify lowest scoring dimension even if acceptable
            dim_scores = [
                ("relevance", stats.average_relevance_percent, stats.average_relevance_score),
                ("accuracy", stats.average_accuracy_percent, stats.average_accuracy_score),
                ("completeness", stats.average_completeness_percent, stats.average_completeness_score),
                ("groundedness", stats.average_groundedness_percent, stats.average_groundedness_score),
            ]
            dim_scores.sort(key=lambda x: x[1])
            lowest_dim_name, lowest_dim_pct, _ = dim_scores[0]

            already_covered_categories = {c.dimension for c in candidates}
            if lowest_dim_name not in already_covered_categories:
                candidates.append(
                    RecommendationItem(
                        id=f"rec-refinement-{lowest_dim_name}",
                        category=lowest_dim_name,
                        priority="low",
                        headline=f"{lowest_dim_name.capitalize()} Optimization Opportunity",
                        plain_english=(
                            f"{lowest_dim_name.capitalize()} averaged {lowest_dim_pct}% across the batch — "
                            f"consider targeted prompt tuning to elevate this lowest-scoring dimension."
                        ),
                        metric_trigger=f"{lowest_dim_name.capitalize()} was the lowest scoring dimension averaging {lowest_dim_pct}%",
                        remediation_advice=(
                            f"Focus prompt optimization and contextual reference tuning on {lowest_dim_name} "
                            f"to raise average batch scores above current {lowest_dim_pct}% baseline."
                        ),
                        affected_count=succ_count,
                        affected_percent=round(100.0 - lowest_dim_pct, 1),
                        dimension=lowest_dim_name,
                        sample_issues=[],
                    )
                )

        if len(candidates) < 2:
            # Opportunity B: Benchmark expansion & adversarial stress-testing
            pass_pct = stats.pass_percent
            candidates.append(
                RecommendationItem(
                    id="rec-hardening-adversarial",
                    category="general_quality",
                    priority="low",
                    headline="Adversarial Robustness & Benchmark Hardening",
                    plain_english=(
                        f"{pass_pct}% of responses achieved Pass verdict — "
                        f"expand the evaluation set with adversarial edge cases to stress-test boundary conditions."
                    ),
                    metric_trigger=f"{pass_pct}% of evaluated responses achieved a Pass verdict",
                    remediation_advice=(
                        "Expand evaluation benchmarks with multi-turn adversarial edge cases, ambiguous phrasing, "
                        "and complex multi-hop queries to stress-test model robustness."
                    ),
                    affected_count=stats.pass_count,
                    affected_percent=pass_pct,
                    dimension="composite",
                    sample_issues=[],
                )
            )

        if len(candidates) < 2:
            # Opportunity C: Reference answer richness
            records_without_ref = sum(1 for r in succ_items if not r.reference_answer)
            if records_without_ref > 0:
                unref_pct = round(records_without_ref / succ_count * 100, 1)
                candidates.append(
                    RecommendationItem(
                        id="rec-dataset-reference-enrichment",
                        category="dataset_quality",
                        priority="low",
                        headline="Ground Truth Reference Answer Enrichment",
                        plain_english=(
                            f"{unref_pct}% of records lacked ground-truth reference answers — "
                            f"provide reference answers to enable deterministic factual consistency scoring."
                        ),
                        metric_trigger=f"{unref_pct}% of records ({records_without_ref}/{succ_count}) lacked ground-truth reference answers",
                        remediation_advice=(
                            "Provide canonical reference answers for all prompts to enable deterministic "
                            "factual consistency verification and reliable hallucination attribution."
                        ),
                        affected_count=records_without_ref,
                        affected_percent=unref_pct,
                        dimension="reference",
                        sample_issues=[],
                    )
                )

        # -------------------------------------------------------------------
        # Prioritization & Selection: Pick Top 2 to 5
        # -------------------------------------------------------------------
        priority_map = {"high": 3, "medium": 2, "low": 1}

        # Deduplicate candidates by id
        seen_ids = set()
        unique_candidates: List[RecommendationItem] = []
        for c in candidates:
            if c.id not in seen_ids:
                seen_ids.add(c.id)
                unique_candidates.append(c)

        # Sort by priority desc, then by affected_percent desc
        unique_candidates.sort(
            key=lambda x: (priority_map.get(x.priority, 1), x.affected_percent),
            reverse=True,
        )

        # Slice: guarantee between 2 and 5 recommendations
        target_count = min(5, max(2, len(unique_candidates)))
        final_recommendations = unique_candidates[:target_count]

        logger.info(
            f"Generated {len(final_recommendations)} recommendations for batch {report_data.metadata.batch_id} "
            f"(candidates found: {len(unique_candidates)})"
        )
        return final_recommendations


# Reusable singleton instance and functional interface
default_recommendation_engine = RecommendationEngine()


def generate_recommendations(report_data: BatchReportData) -> List[RecommendationItem]:
    """
    Reusable functional interface to generate recommendations from BatchReportData.
    """
    return default_recommendation_engine.generate(report_data)
