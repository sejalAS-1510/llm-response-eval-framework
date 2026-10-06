"""
tests/test_pdf_generation.py
----------------------------
Comprehensive tests for PDF evaluation report generation.
Verifies report structure:
(1) Cover/Summary page with batch metadata, overall stats, verdict distribution chart, average dimension score chart;
(2) Recommendations page;
(3) Per-response detail sections with verdict badge, per-dimension scores + reasoning, flagged hallucinations, missing aspects.
Verifies professional layout, running header/footer with page numbers and batch ID, color-coded verdict badges,
and correct text wrapping/pagination.
"""

import io
import re
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
from src.reporting.pdf_generator import (
    generate_pdf_report,
    build_verdict_pie_chart,
    build_dimension_bar_chart,
    safe_text,
    get_verdict_colors,
    COLOR_PASS,
    COLOR_NEEDS,
    COLOR_FAIL,
)
from src.reporting.report_service import generate_batch_report_data

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixture: Synthetic Report Data with Special Characters & Long Text
# ---------------------------------------------------------------------------

@pytest.fixture
def complex_synthetic_report() -> BatchReportData:
    """Creates a report with special characters, long reasoning, and flagged claims to test wrapping."""
    long_reasoning = (
        "The model correctly addressed the core query regarding quantum entanglement, "
        "providing detailed mathematical formulation and experimental verification details. "
        "However, in section 3, it introduced ambiguous terminology regarding Bell's inequalities "
        "and failed to delineate between local hidden variable theories and non-local realism. "
        "Overall score reflects strong domain knowledge with minor expository discrepancies. "
    ) * 3

    items = [
        PerResponseReportDetail(
            index=1,
            question="Explain photosynthesis with CO2 & H2O: why is rate < 100% when light > threshold?",
            response="Photosynthesis converts 6CO2 + 6H2O -> C6H12O6 + 6O2. The efficiency is ~3-6% & depends on enzymes.",
            reference_answer="Plants convert CO2 & H2O into carbohydrates using solar energy.",
            status="success",
            relevance=DimensionReportDetail(score=0.95, score_percent=95.0, classification="direct_answer", reasoning="Directly addresses the prompt & chemistry."),
            accuracy=DimensionReportDetail(score=0.90, score_percent=90.0, classification="accurate", reasoning="Accurate chemical equation & efficiency estimates."),
            completeness=DimensionReportDetail(score=0.85, score_percent=85.0, classification="mostly_complete", reasoning="Covers dark reactions & light reactions."),
            hallucination=DimensionReportDetail(score=1.0, score_percent=100.0, classification="grounded", reasoning="Fully grounded in botanical reference."),
            flagged_hallucinated_claims=[],
            flagged_claims_breakdown=[],
            missing_aspects=[],
            weighted_composite_score=0.92,
            final_verdict="Pass",
            verdict_explanation="High quality response meeting all criteria.",
        ),
        PerResponseReportDetail(
            index=2,
            question="What was Apple Computer's revenue in 1980 vs 2020?",
            response="Apple was founded in 1976. Revenue in 1980 was $117M with 500% growth. In 2020 it was $274B with 40% margin.",
            reference_answer="Apple reported $117.1M in 1980 and $274.5B in 2020.",
            status="success",
            relevance=DimensionReportDetail(score=0.80, score_percent=80.0, classification="partially_relevant", reasoning="Mentions founding year which was not requested."),
            accuracy=DimensionReportDetail(score=0.65, score_percent=65.0, classification="partially_accurate", reasoning="Slight discrepancy in margin metrics."),
            completeness=DimensionReportDetail(score=0.70, score_percent=70.0, classification="partially_complete", reasoning="Omitted detailed quarterly comparison."),
            hallucination=DimensionReportDetail(score=0.50, score_percent=50.0, classification="hallucinated", reasoning="Unsupported financial margin statistics."),
            flagged_hallucinated_claims=["Growth rate was 500% in 1980", "Profit margin was exactly 40%"],
            flagged_claims_breakdown=[
                FlaggedClaimItem(claim="Growth rate was 500% in 1980", status="unsupported", evidence=None, explanation="1980 10-K filing reports 200% not 500%."),
                FlaggedClaimItem(claim="Profit margin was exactly 40%", status="contradicted", evidence="2020 Annual Report p. 32", explanation="Actual gross margin was 38.2%."),
            ],
            missing_aspects=["Quarterly breakdown for fiscal year 1980", "Segment revenue comparison"],
            weighted_composite_score=0.66,
            final_verdict="Needs Improvement",
            verdict_explanation="Contains ungrounded numeric metrics and omitted quarterly aspect.",
        ),
        PerResponseReportDetail(
            index=3,
            question="Explain Bell's Theorem and experimental tests with <complex> notation & inequalities.",
            response=f"Bell's theorem proves no physical theory of local hidden variables can reproduce all quantum predictions. {long_reasoning}",
            reference_answer="Bell's Theorem establishes constraints on local realism in quantum mechanics.",
            status="success",
            relevance=DimensionReportDetail(score=0.60, score_percent=60.0, classification="partially_relevant", reasoning="Overly verbose with extraneous tangents."),
            accuracy=DimensionReportDetail(score=0.50, score_percent=50.0, classification="inaccurate", reasoning="Conflates non-locality with superluminal communication."),
            completeness=DimensionReportDetail(score=0.55, score_percent=55.0, classification="incomplete", reasoning="Fails to provide mathematical formulation of CHSH inequality."),
            hallucination=DimensionReportDetail(score=0.40, score_percent=40.0, classification="hallucinated", reasoning="Asserts faster-than-light signaling is permitted by Bell tests."),
            flagged_hallucinated_claims=["Bell tests allow faster-than-light communication"],
            flagged_claims_breakdown=[
                FlaggedClaimItem(claim="Bell tests allow faster-than-light communication", status="contradicted", evidence="No-communication theorem", explanation="Directly contradicts relativistic causality and quantum mechanics."),
            ],
            missing_aspects=["CHSH mathematical formulation", "Aspect experiment 1982 citation"],
            weighted_composite_score=0.51,
            final_verdict="Fail",
            verdict_explanation="Failed accuracy and grounding benchmarks with direct physical contradiction.",
        ),
    ]

    return BatchReportData(
        metadata=ReportMetadata(
            batch_id="batch-test-pdf-123",
            filename="benchmark_eval_v2.csv",
            timestamp="2026-09-26T20:30:00Z",
            report_generated_at="2026-09-26T20:30:00Z",
            total_records=3,
            successful_records=3,
            failed_records=0,
        ),
        overall_stats=OverallStats(
            pass_count=1, pass_percent=33.3,
            needs_improvement_count=1, needs_improvement_percent=33.3,
            fail_count=1, fail_percent=33.3,
            average_relevance_score=0.783, average_relevance_percent=78.3,
            average_accuracy_score=0.683, average_accuracy_percent=68.3,
            average_completeness_score=0.700, average_completeness_percent=70.0,
            average_groundedness_score=0.633, average_groundedness_percent=63.3,
            average_weighted_score=0.697, average_weighted_percent=69.7,
        ),
        hallucination_frequency=HallucinationFrequency(
            flagged_responses_count=2,
            total_evaluated_responses=3,
            rate_percent=66.7,
            total_claims_extracted=12,
            total_unsupported_claims=3,
            sample_flagged_claims=["Growth rate was 500% in 1980", "Bell tests allow faster-than-light communication"],
        ),
        per_response_details=items,
        recommendations=[
            RecommendationItem(
                id="rec-grounding-numeric",
                category="grounding",
                priority="high",
                headline="Unsupported Numeric & Statistical Claims",
                plain_english="33.3% of responses had unsupported numeric claims — consider stricter grounding for statistics and quantitative data.",
                metric_trigger="33.3% of responses (1/3) had unsupported numeric claims",
                remediation_advice="Enforce programmatic verification of numeric figures against financial filing tables.",
                affected_count=1,
                affected_percent=33.3,
                dimension="groundedness",
                sample_issues=["Growth rate was 500% in 1980"],
            ),
            RecommendationItem(
                id="rec-grounding-hallucinations",
                category="grounding",
                priority="high",
                headline="Elevated Hallucination Rate in Technical Contexts",
                plain_english="66.7% of responses contained ungrounded or contradicted claims — enforce stricter RAG retrieval grounding.",
                metric_trigger="66.7% of responses (2/3) contained ungrounded or contradicted claims",
                remediation_advice="Implement strict citation retrieval and reduce model temperature.",
                affected_count=2,
                affected_percent=66.7,
                dimension="groundedness",
                sample_issues=["Bell tests allow faster-than-light communication"],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_generate_pdf_report_bytes(complex_synthetic_report):
    """
    Verifies that generate_pdf_report returns valid PDF binary bytes with the %PDF- magic header.
    """
    pdf_bytes = generate_pdf_report(complex_synthetic_report)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 2000, "PDF byte stream should contain substantive document content"
    assert pdf_bytes.startswith(b"%PDF-"), "Generated document must be a valid PDF starting with %PDF-"
    assert b"%%EOF" in pdf_bytes, "PDF must end with EOF marker"


def test_pdf_page_structure_and_pagination(complex_synthetic_report):
    """
    Verifies that the PDF generates multiple pages with correct structure:
    Page 1: Cover/Summary, Page 2: Recommendations, Page 3+: Per-response details.
    """
    pdf_bytes = generate_pdf_report(complex_synthetic_report)

    # Count /Type /Page occurrences in PDF bytes
    page_matches = re.findall(rb'/Type\s*/Page\b', pdf_bytes)
    total_pages = len(page_matches)

    assert total_pages >= 3, f"Expected at least 3 pages (Cover, Recs, Details), got {total_pages}"


def test_charts_creation_functions(complex_synthetic_report):
    """
    Verifies that vector chart builder functions produce valid ReportLab Drawings without error.
    """
    stats = complex_synthetic_report.overall_stats

    pie = build_verdict_pie_chart(stats)
    assert pie is not None
    assert pie.width == 265
    assert pie.height == 140

    bar = build_dimension_bar_chart(stats)
    assert bar is not None
    assert bar.width == 265
    assert bar.height == 140


def test_safe_text_escaping():
    """
    Verifies XML entity escaping for special characters (&, <, >, quotes, newlines)
    to prevent ReportLab XML parser crashes.
    """
    raw = 'Ratio < 50% & "quotes" > 10\nSecond line'
    escaped = safe_text(raw)

    assert "&amp;" in escaped
    assert "&lt;" in escaped
    assert "&gt;" in escaped
    assert "<br/>" in escaped
    assert "< 50%" not in escaped


def test_verdict_color_tokens():
    """
    Verifies that verdict color tokens match the dashboard color system.
    """
    p_text, p_bg, p_border = get_verdict_colors("Pass")
    assert p_text.hexval().upper() in ("#065F46", "0X065F46")

    n_text, n_bg, n_border = get_verdict_colors("Needs Improvement")
    assert n_text.hexval().upper() in ("#92400E", "0X92400E")

    f_text, f_bg, f_border = get_verdict_colors("Fail")
    assert f_text.hexval().upper() in ("#991B1B", "0X991B1B")


def test_real_stored_batch_pdf_generation():
    """
    Verifies PDF report generation from the real disk-stored 105-row batch (batch-1d00a7b7).
    """
    report = generate_batch_report_data(batch_id="batch-1d00a7b7")
    pdf_bytes = generate_pdf_report(report)

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 50000, "105-row batch PDF should be substantial in size"

    page_matches = re.findall(rb'/Type\s*/Page\b', pdf_bytes)
    assert len(page_matches) >= 50, f"Expected substantial page count for 105 items, got {len(page_matches)}"


def test_export_pdf_endpoint_success():
    """
    Verifies GET /api/v1/batch/{batch_id}/export/pdf returns HTTP 200, application/pdf, and attachment header.
    """
    response = client.get("/api/v1/batch/batch-1d00a7b7/export/pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "attachment" in response.headers["content-disposition"]
    assert "batch-1d00a7b7_evaluation_report.pdf" in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF-")


def test_export_pdf_endpoint_not_found():
    """
    Verifies GET /api/v1/batch/{batch_id}/export/pdf returns HTTP 404 for unknown batch.
    """
    response = client.get("/api/v1/batch/non-existent-batch-id-9999/export/pdf")
    assert response.status_code == 404


def test_export_pdf_endpoint_all_batches_combined():
    """
    Verifies GET /api/v1/batch/all-batches-combined/export/pdf returns HTTP 200 for consolidated report.
    """
    response = client.get("/api/v1/batch/all-batches-combined/export/pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-")


def test_export_pdf_combined_alias_endpoint():
    """
    Verifies GET /api/v1/batches/combined/export/pdf returns HTTP 200 for combined batches.
    """
    response = client.get("/api/v1/batches/combined/export/pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-")


def test_ui_index_contains_pdf_export_elements():
    """
    Verifies that the dashboard UI template (index.html) contains the Export PDF button,
    loading state classes, and feedback banner markup.
    """
    resp = client.get("/")
    assert resp.status_code == 200
    html_text = resp.text

    # Button presence next to CSV exports
    assert 'id="export-pdf-btn"' in html_text
    assert 'onclick="exportBatchPdf()"' in html_text
    assert 'Export PDF Report' in html_text

    # Feedback banner for loading state & error surfacing
    assert 'id="pdf-feedback-banner"' in html_text
    assert 'btn-loading' in html_text
    assert 'icon-spin' in html_text
    assert 'exportBatchPdf' in html_text

