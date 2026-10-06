"""
tests/test_recommendations.py
------------------------------
Unit and integration tests for the recommendations module.
Verifies that 2 to 5 actionable, plain-English improvement recommendations are generated
based on frequency thresholds across evaluation dimensions and flagged issues,
without hardcoded static text.
"""

import pytest
from fastapi.testclient import TestClient

from src.input_module.main import app
from src.reporting.schemas import (
    BatchReportData,
    ReportMetadata,
    OverallStats,
    HallucinationFrequency,
    PerResponseReportDetail,
    DimensionReportDetail,
    FlaggedClaimItem,
    RecommendationItem,
)
from src.reporting.recommendations import (
    RecommendationEngine,
    generate_recommendations,
)
from src.reporting.report_service import generate_batch_report_data

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures for synthetic batches
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_hallucination_report() -> BatchReportData:
    """Creates a report where 42% of responses have unsupported numeric claims."""
    total = 50
    items = []
    # 21 items (42%) have unsupported numeric claims
    for i in range(total):
        if i < 21:
            claims = [
                FlaggedClaimItem(
                    claim=f"Revenue grew by {10 + i}% in 202{i % 4}",
                    status="unsupported",
                    evidence=None,
                    explanation="Numeric claim not present in source context",
                )
            ]
            flagged_strs = [c.claim for c in claims]
            hal_detail = DimensionReportDetail(
                score=0.4, score_percent=40.0, classification="hallucinated",
                reasoning="Contains ungrounded numeric metrics.", evidence=None
            )
        else:
            claims = []
            flagged_strs = []
            hal_detail = DimensionReportDetail(
                score=1.0, score_percent=100.0, classification="grounded",
                reasoning="Fully grounded.", evidence=None
            )

        items.append(
            PerResponseReportDetail(
                index=i + 1,
                question=f"Question {i + 1}?",
                response=f"Response {i + 1}",
                reference_answer=f"Reference {i + 1}",
                status="success",
                hallucination=hal_detail,
                flagged_hallucinated_claims=flagged_strs,
                flagged_claims_breakdown=claims,
                missing_aspects=[],
                weighted_composite_score=0.6 if i < 21 else 0.95,
                final_verdict="Needs Improvement" if i < 21 else "Pass",
                verdict_explanation="Evaluation completed",
            )
        )

    return BatchReportData(
        metadata=ReportMetadata(
            batch_id="batch-numeric-test",
            filename="numeric_test.csv",
            timestamp="2026-09-26T20:00:00Z",
            report_generated_at="2026-09-26T20:00:00Z",
            total_records=total,
            successful_records=total,
            failed_records=0,
        ),
        overall_stats=OverallStats(
            pass_count=29, pass_percent=58.0,
            needs_improvement_count=21, needs_improvement_percent=42.0,
            fail_count=0, fail_percent=0.0,
            average_relevance_score=0.90, average_relevance_percent=90.0,
            average_accuracy_score=0.85, average_accuracy_percent=85.0,
            average_completeness_score=0.88, average_completeness_percent=88.0,
            average_groundedness_score=0.75, average_groundedness_percent=75.0,
            average_weighted_score=0.80, average_weighted_percent=80.0,
        ),
        hallucination_frequency=HallucinationFrequency(
            flagged_responses_count=21,
            total_evaluated_responses=total,
            rate_percent=42.0,
            total_claims_extracted=100,
            total_unsupported_claims=21,
            sample_flagged_claims=["Revenue grew by 10% in 2020", "Profit margin was 15%"],
        ),
        per_response_details=items,
    )


@pytest.fixture
def perfect_batch_report() -> BatchReportData:
    """Creates a report where all responses pass with high scores."""
    total = 20
    items = [
        PerResponseReportDetail(
            index=i + 1,
            question=f"What is item {i + 1}?",
            response=f"Item {i + 1} explanation.",
            reference_answer=f"Item {i + 1} explanation.",
            status="success",
            relevance=DimensionReportDetail(score=1.0, score_percent=100.0, reasoning="Direct match"),
            accuracy=DimensionReportDetail(score=1.0, score_percent=100.0, reasoning="Accurate"),
            completeness=DimensionReportDetail(score=0.92, score_percent=92.0, reasoning="Substantially complete"),
            hallucination=DimensionReportDetail(score=1.0, score_percent=100.0, reasoning="Grounded"),
            flagged_hallucinated_claims=[],
            flagged_claims_breakdown=[],
            missing_aspects=[],
            weighted_composite_score=0.98,
            final_verdict="Pass",
            verdict_explanation="High quality response.",
        )
        for i in range(total)
    ]

    return BatchReportData(
        metadata=ReportMetadata(
            batch_id="batch-perfect-test",
            filename="perfect.csv",
            timestamp="2026-09-26T20:00:00Z",
            report_generated_at="2026-09-26T20:00:00Z",
            total_records=total,
            successful_records=total,
            failed_records=0,
        ),
        overall_stats=OverallStats(
            pass_count=total, pass_percent=100.0,
            needs_improvement_count=0, needs_improvement_percent=0.0,
            fail_count=0, fail_percent=0.0,
            average_relevance_score=1.0, average_relevance_percent=100.0,
            average_accuracy_score=1.0, average_accuracy_percent=100.0,
            average_completeness_score=0.92, average_completeness_percent=92.0,
            average_groundedness_score=1.0, average_groundedness_percent=100.0,
            average_weighted_score=0.98, average_weighted_percent=98.0,
        ),
        hallucination_frequency=HallucinationFrequency(
            flagged_responses_count=0,
            total_evaluated_responses=total,
            rate_percent=0.0,
            total_claims_extracted=40,
            total_unsupported_claims=0,
            sample_flagged_claims=[],
        ),
        per_response_details=items,
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_numeric_hallucination_recommendation_trigger(numeric_hallucination_report):
    """
    Verifies that a batch with 42% unsupported numeric claims triggers
    the numeric grounding recommendation with exact percentage and count.
    """
    recs = generate_recommendations(numeric_hallucination_report)

    # Must be bounded between 2 and 5
    assert 2 <= len(recs) <= 5

    # Check that numeric grounding recommendation was generated
    numeric_rec = next((r for r in recs if r.id == "rec-grounding-numeric"), None)
    assert numeric_rec is not None, "Expected rec-grounding-numeric recommendation to be triggered"
    assert numeric_rec.affected_count == 21
    assert numeric_rec.affected_percent == 42.0

    # Ensure dynamic plain-English content contains the exact percentage and grounding advice
    assert "42.0%" in numeric_rec.plain_english or "42%" in numeric_rec.plain_english
    assert "unsupported numeric claims" in numeric_rec.plain_english.lower()
    assert "statistics" in numeric_rec.plain_english.lower() or "quantitative" in numeric_rec.plain_english.lower()
    assert numeric_rec.priority in ("high", "medium")


def test_perfect_batch_returns_bounded_recommendations(perfect_batch_report):
    """
    Verifies that even a flawless batch produces between 2 and 5 recommendations
    focusing on benchmark hardening and lowest-dimension optimization.
    """
    recs = generate_recommendations(perfect_batch_report)

    assert 2 <= len(recs) <= 5
    # Should contain adversarial hardening or lowest-dimension refinement
    assert any("adversarial" in r.plain_english.lower() or "optimiz" in r.plain_english.lower() for r in recs)


def test_stored_105_batch_recommendations():
    """
    Verifies recommendations generated from the actual stored 105-row batch (batch-1d00a7b7).
    """
    report = generate_batch_report_data(batch_id="batch-1d00a7b7")
    recs = report.recommendations

    assert 2 <= len(recs) <= 5, f"Expected 2-5 recommendations, got {len(recs)}"

    for r in recs:
        assert isinstance(r, RecommendationItem)
        assert r.id
        assert r.headline
        assert r.plain_english
        assert r.metric_trigger
        assert r.remediation_advice
        assert r.priority in ("high", "medium", "low")
        assert r.affected_count >= 0
        assert 0.0 <= r.affected_percent <= 100.0

    # In 1d00a7b7, missing aspects and ungrounded claims are dominant weaknesses
    categories = [r.category for r in recs]
    assert any(c in categories for c in ("completeness", "grounding", "accuracy"))


def test_not_hardcoded_text_dynamic_metrics():
    """
    Verifies that recommendations dynamically embed the exact counts and percentages
    from the input batch rather than static placeholders.
    """
    report_a = generate_batch_report_data(batch_id="batch-81a61d79")
    recs_a = report_a.recommendations

    # batch-81a61d79 has 1 record that failed with 100% hallucination rate
    assert len(recs_a) >= 2
    hal_rec = next((r for r in recs_a if "hallucination" in r.id or r.category == "grounding"), None)
    assert hal_rec is not None
    assert hal_rec.affected_percent == 100.0
    assert "100.0%" in hal_rec.plain_english or "100%" in hal_rec.plain_english


def test_report_data_api_endpoint_includes_recommendations():
    """
    Verifies that GET /api/v1/batch/{batch_id}/report-data returns populated recommendations.
    """
    response = client.get("/api/v1/batch/batch-1d00a7b7/report-data")
    assert response.status_code == 200
    data = response.json()

    assert "recommendations" in data
    recs = data["recommendations"]
    assert isinstance(recs, list)
    assert 2 <= len(recs) <= 5

    first = recs[0]
    assert "id" in first
    assert "headline" in first
    assert "plain_english" in first
    assert "metric_trigger" in first
    assert "remediation_advice" in first
    assert "affected_count" in first
    assert "affected_percent" in first


def test_dedicated_recommendations_endpoint():
    """
    Verifies that GET /api/v1/batch/{batch_id}/recommendations returns 2-5 recommendations.
    """
    response = client.get("/api/v1/batch/batch-1d00a7b7/recommendations")
    assert response.status_code == 200
    recs = response.json()

    assert isinstance(recs, list)
    assert 2 <= len(recs) <= 5

    # 404 for non-existent batch
    non_existent = client.get("/api/v1/batch/non-existent-batch-id-999/recommendations")
    assert non_existent.status_code == 404


def test_custom_thresholds_configuration():
    """
    Verifies that custom thresholds in RecommendationEngine work as configured.
    """
    report = generate_batch_report_data(batch_id="batch-1d00a7b7")

    # Engine with very strict 90% threshold for everything
    strict_engine = RecommendationEngine(
        hallucination_threshold_pct=90.0,
        completeness_threshold_pct=90.0,
        accuracy_threshold_pct=90.0,
    )
    recs = strict_engine.generate(report)
    # Bounded guarantee must still hold
    assert 2 <= len(recs) <= 5
