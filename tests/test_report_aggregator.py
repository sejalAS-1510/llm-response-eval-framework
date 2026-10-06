"""
tests/test_report_aggregator.py
-------------------------------
Unit and integration tests for the ReportDataAggregator and report data models.
Verifies that the data aggregation layer pulls exclusively from stored evaluation records,
properly computing metadata, overall stats, dimensional averages, hallucination frequency,
and granular per-response details feeding the PDF report.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.input_module.main import app, BATCH_DIR
from src.reporting import (
    ReportDataAggregator,
    generate_batch_report_data,
    BatchReportData,
    ReportMetadata,
    OverallStats,
    HallucinationFrequency,
    PerResponseReportDetail,
)
from src.agents.schemas import (
    BatchEvaluationSummary,
    BatchEvaluationItem,
    EvaluationResult,
    RelevanceResult,
    AccuracyResult,
    CompletenessResult,
    HallucinationResult,
    ClaimEvaluation,
    VerdictResult,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def aggregator():
    return ReportDataAggregator(storage_dir=BATCH_DIR)


def test_aggregator_pulls_from_stored_batch(aggregator):
    """
    Confirms that ReportDataAggregator pulls directly from stored evaluation records
    on disk and outputs a valid BatchReportData object.
    """
    stored_batches = aggregator.list_stored_batch_ids()
    assert len(stored_batches) > 0, "No stored batches found on disk."

    test_batch_id = stored_batches[0]
    report_data = aggregator.get_report_data_by_batch_id(test_batch_id)

    # 1. Validate top-level object
    assert isinstance(report_data, BatchReportData)
    assert report_data.metadata.batch_id == test_batch_id

    # 2. Validate metadata
    meta = report_data.metadata
    assert isinstance(meta, ReportMetadata)
    assert meta.total_records > 0
    assert meta.successful_records + meta.failed_records == meta.total_records
    assert meta.timestamp != ""
    assert meta.report_generated_at != ""

    # 3. Validate overall stats
    stats = report_data.overall_stats
    assert isinstance(stats, OverallStats)
    assert stats.pass_count + stats.needs_improvement_count + stats.fail_count == meta.successful_records
    assert 0.0 <= stats.pass_percent <= 100.0
    assert 0.0 <= stats.needs_improvement_percent <= 100.0
    assert 0.0 <= stats.fail_percent <= 100.0

    # Dimension averages
    assert 0.0 <= stats.average_relevance_score <= 1.0
    assert 0.0 <= stats.average_relevance_percent <= 100.0
    assert 0.0 <= stats.average_accuracy_score <= 1.0
    assert 0.0 <= stats.average_accuracy_percent <= 100.0
    assert 0.0 <= stats.average_completeness_score <= 1.0
    assert 0.0 <= stats.average_completeness_percent <= 100.0
    assert 0.0 <= stats.average_groundedness_score <= 1.0
    assert 0.0 <= stats.average_groundedness_percent <= 100.0
    assert 0.0 <= stats.average_weighted_score <= 1.0
    assert 0.0 <= stats.average_weighted_percent <= 100.0

    # 4. Validate hallucination frequency
    hal_freq = report_data.hallucination_frequency
    assert isinstance(hal_freq, HallucinationFrequency)
    assert hal_freq.total_evaluated_responses == meta.successful_records
    assert hal_freq.flagged_responses_count <= meta.successful_records
    assert 0.0 <= hal_freq.rate_percent <= 100.0
    assert hal_freq.total_claims_extracted >= hal_freq.total_unsupported_claims

    # 5. Validate per-response details list
    details = report_data.per_response_details
    assert len(details) == meta.total_records
    for detail in details:
        assert isinstance(detail, PerResponseReportDetail)
        assert detail.index >= 1
        assert detail.question != ""
        assert detail.response != ""
        assert detail.final_verdict in ("Pass", "Needs Improvement", "Fail", "Failed")

        if detail.status == "success":
            assert detail.relevance is not None
            assert 0.0 <= detail.relevance.score <= 1.0
            assert detail.relevance.reasoning != ""

            assert detail.accuracy is not None
            assert 0.0 <= detail.accuracy.score <= 1.0
            assert detail.accuracy.reasoning != ""

            assert detail.completeness is not None
            assert 0.0 <= detail.completeness.score <= 1.0
            assert detail.completeness.reasoning != ""

            assert detail.hallucination is not None
            assert 0.0 <= detail.hallucination.score <= 1.0
            assert detail.hallucination.reasoning != ""

            assert isinstance(detail.flagged_hallucinated_claims, list)
            assert isinstance(detail.missing_aspects, list)
            assert 0.0 <= detail.weighted_composite_score <= 1.0
            assert detail.verdict_explanation != ""


def test_reusable_function_interface():
    """
    Confirms generate_batch_report_data operates as a reusable function
    capable of taking either batch_id or pre-loaded summary.
    """
    aggregator = ReportDataAggregator(storage_dir=BATCH_DIR)
    stored_batches = aggregator.list_stored_batch_ids()
    assert len(stored_batches) > 0

    batch_id = stored_batches[0]

    # Test via batch_id
    r1 = generate_batch_report_data(batch_id=batch_id, storage_dir=BATCH_DIR)
    assert isinstance(r1, BatchReportData)
    assert r1.metadata.batch_id == batch_id

    # Test via pre-loaded summary
    summary = aggregator.load_stored_batch_summary(batch_id)
    r2 = generate_batch_report_data(summary=summary)
    assert isinstance(r2, BatchReportData)
    assert r2.metadata.batch_id == batch_id
    assert len(r2.per_response_details) == len(r1.per_response_details)

    # Test error when neither is passed
    with pytest.raises(ValueError):
        generate_batch_report_data()


def test_api_report_data_endpoint(client):
    """
    Confirms GET /api/v1/batch/{batch_id}/report-data serves the aggregated report data
    conforming strictly to the BatchReportData schema.
    """
    aggregator = ReportDataAggregator(storage_dir=BATCH_DIR)
    stored_batches = aggregator.list_stored_batch_ids()
    assert len(stored_batches) > 0

    batch_id = stored_batches[0]
    resp = client.get(f"/api/v1/batch/{batch_id}/report-data")
    assert resp.status_code == 200

    data = resp.json()
    assert data["metadata"]["batch_id"] == batch_id
    assert "overall_stats" in data
    assert "hallucination_frequency" in data
    assert "per_response_details" in data
    assert len(data["per_response_details"]) == data["metadata"]["total_records"]


def test_combined_report_data_aggregation(client, aggregator):
    """
    Confirms that loading combined report data aggregates across all historical batches.
    """
    report_data = aggregator.get_report_data_by_batch_id("all-batches-combined")
    assert isinstance(report_data, BatchReportData)
    assert report_data.metadata.batch_id == "all-batches-combined"
    assert report_data.metadata.total_records >= len(report_data.per_response_details)

    # Test via API
    resp = client.get("/api/v1/batch/all-batches-combined/report-data")
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["batch_id"] == "all-batches-combined"


def test_nonexistent_batch_raises_404(client, aggregator):
    """
    Confirms requesting a non-existent batch ID properly raises FileNotFoundError
    and maps to 404 in FastAPI.
    """
    fake_id = "non-existent-batch-999"
    with pytest.raises(FileNotFoundError):
        aggregator.get_report_data_by_batch_id(fake_id)

    resp = client.get(f"/api/v1/batch/{fake_id}/report-data")
    assert resp.status_code == 404


def test_synthetic_batch_with_failed_and_empty_rows(aggregator):
    """
    Validates report data aggregation when a batch contains mixed success, failed,
    and hallucinated records.
    """
    item_success = BatchEvaluationItem(
        index=1,
        question="What is the boiling point of water?",
        ai_response="Water boils at 100C.",
        reference_answer="100C",
        status="success",
        result=EvaluationResult(
            question="What is the boiling point of water?",
            ai_response="Water boils at 100C.",
            context_used="Water boils at 100C at sea level.",
            relevance=RelevanceResult(score=1.0, classification="fully_relevant", reasoning="Directly answers."),
            accuracy=AccuracyResult(score=1.0, classification="correct", supporting_evidence=["Water boils at 100C"], reasoning="Factual."),
            completeness=CompletenessResult(score=1.0, classification="fully_complete", missing_aspects=[], reasoning="Complete."),
            hallucination=HallucinationResult(is_hallucinated=False, hallucination_score=0.0, total_claims=1, unsupported_claims_count=0, flagged_claims=[], all_claims=[], reasoning="Grounded."),
            verdict=VerdictResult(weighted_score=1.0, verdict="Pass", consolidated_reasoning="Excellent answer.", dimension_scores={"groundedness": 1.0}),
            evaluated_at="2026-09-26T12:00:00Z"
        )
    )

    item_failed = BatchEvaluationItem(
        index=2,
        question="[No Question Provided]",
        ai_response="",
        status="failed",
        error_message="API connection timeout during evaluation"
    )

    item_hal = BatchEvaluationItem(
        index=3,
        question="Who was the first king of Mars?",
        ai_response="King Marvin was the first king of Mars in 1842.",
        status="success",
        result=EvaluationResult(
            question="Who was the first king of Mars?",
            ai_response="King Marvin was the first king of Mars in 1842.",
            context_used="Mars is an uninhabited planet with no monarchy.",
            relevance=RelevanceResult(score=0.8, classification="partially_relevant", reasoning="Answers query."),
            accuracy=AccuracyResult(score=0.1, classification="incorrect", supporting_evidence=[], reasoning="Untrue."),
            completeness=CompletenessResult(score=0.5, classification="partially_complete", missing_aspects=["Mars has no king"], reasoning="Omitted reality."),
            hallucination=HallucinationResult(
                is_hallucinated=True,
                hallucination_score=1.0,
                total_claims=1,
                unsupported_claims_count=1,
                flagged_claims=[
                    ClaimEvaluation(claim="King Marvin was first king of Mars", status="unsupported", explanation="Fabricated claim")
                ],
                all_claims=[],
                reasoning="Severe fabrication."
            ),
            verdict=VerdictResult(weighted_score=0.35, verdict="Fail", consolidated_reasoning="Severe fabrication.", dimension_scores={"groundedness": 0.0}),
            evaluated_at="2026-09-26T12:00:00Z"
        )
    )

    from src.agents.batch_evaluator import BatchEvaluator
    test_evaluator = BatchEvaluator()
    synth_items = [item_success, item_failed, item_hal]

    summary = BatchEvaluationSummary(
        batch_id="test-synthetic-batch",
        filename="synthetic.csv",
        created_at="2026-09-26T12:00:00Z",
        statistics=test_evaluator.calculate_statistics(synth_items),
        items=synth_items,
    )

    report_data = aggregator.build_report_data(summary)

    # Metadata
    assert report_data.metadata.total_records == 3
    assert report_data.metadata.successful_records == 2
    assert report_data.metadata.failed_records == 1

    # Overall stats
    assert report_data.overall_stats.pass_count == 1
    assert report_data.overall_stats.fail_count == 1
    assert report_data.overall_stats.needs_improvement_count == 0

    # Hallucination frequency
    assert report_data.hallucination_frequency.flagged_responses_count == 1
    assert report_data.hallucination_frequency.total_evaluated_responses == 2
    assert report_data.hallucination_frequency.rate_percent == 50.0
    assert "King Marvin was first king of Mars" in report_data.hallucination_frequency.sample_flagged_claims

    # Per-response details
    assert len(report_data.per_response_details) == 3

    # Check failed item
    failed_detail = report_data.per_response_details[1]
    assert failed_detail.status == "failed"
    assert failed_detail.final_verdict == "Failed"
    assert "timeout" in failed_detail.error_message
    assert failed_detail.relevance is None

    # Check hallucinated item
    hal_detail = report_data.per_response_details[2]
    assert hal_detail.status == "success"
    assert hal_detail.final_verdict == "Fail"
    assert "King Marvin was first king of Mars" in hal_detail.flagged_hallucinated_claims
    assert "Mars has no king" in hal_detail.missing_aspects
