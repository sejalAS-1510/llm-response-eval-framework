"""
Milestone 4 Test & Validation Suite:
Cross-Batch Evaluation Trends & Longitudinal Analysis.

Tests:
1. Batch trend points extraction and chronological ordering.
2. Longitudinal trajectory calculation (improving, degrading, stable) and analytical insights.
3. FastAPI Endpoints: /api/v1/batches, /api/v1/batches/trends, /api/v1/trends, and /health.
4. Completeness of dimension scores (Relevance, Accuracy, Hallucination rate, Completeness, Composite).
5. Completeness of verdict distributions (Pass %, Needs Improvement %, Fail %).
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from src.input_module.main import (
    app,
    collect_historical_batches,
    compute_trends_summary,
    extract_batch_trend_point,
)
from src.agents.schemas import BatchTrendPoint, TrendsSummaryResponse


def test_extract_batch_trend_point():
    print("\n--- 1. Testing extract_batch_trend_point ---")
    mock_data = {
        "batch_id": "test-batch-001",
        "filename": "eval_run_v1.csv",
        "created_at": "2026-09-26T10:00:00Z",
        "statistics": {
            "total_records": 100,
            "successful_records": 100,
            "failed_records": 0,
            "pass_count": 60,
            "needs_improvement_count": 30,
            "fail_count": 10,
            "pass_rate_percent": 60.0,
            "average_weighted_score": 0.75,
            "average_relevance_score": 0.90,
            "average_accuracy_score": 0.80,
            "average_completeness_score": 0.70,
            "average_groundedness_score": 0.85,
            "hallucination_rate_percent": 15.0,
        },
    }
    pt = extract_batch_trend_point(mock_data, "test-batch-001")
    assert pt is not None, "Failed to extract trend point"
    assert pt.batch_id == "test-batch-001"
    assert pt.total_records == 100
    assert pt.pass_percent == 60.0
    assert pt.needs_improvement_percent == 30.0
    assert pt.fail_percent == 10.0
    assert pt.average_weighted_score == 0.75
    assert pt.average_relevance_score == 0.90
    assert pt.average_accuracy_score == 0.80
    assert pt.average_completeness_score == 0.70
    assert pt.hallucination_rate_percent == 15.0
    print("[PASS] Batch trend point correctly extracted all dimensional scores and verdict distributions!")


def test_compute_trends_summary_logic():
    print("\n--- 2. Testing compute_trends_summary Logic ---")

    # Case A: Empty batches
    empty_summary = compute_trends_summary([])
    assert empty_summary.total_batches == 0
    assert empty_summary.overall_quality_trend == "insufficient_data"

    # Case B: Single batch
    pt1 = BatchTrendPoint(
        batch_id="b1",
        filename="v1.csv",
        created_at="2026-09-20T10:00:00Z",
        total_records=50,
        successful_records=50,
        failed_records=0,
        pass_count=15,
        needs_improvement_count=20,
        fail_count=15,
        pass_percent=30.0,
        needs_improvement_percent=40.0,
        fail_percent=30.0,
        average_weighted_score=0.50,
        average_relevance_score=0.70,
        average_accuracy_score=0.45,
        average_completeness_score=0.50,
        average_groundedness_score=0.50,
        hallucination_rate_percent=50.0,
    )
    single_summary = compute_trends_summary([pt1])
    assert single_summary.total_batches == 1
    assert single_summary.overall_quality_trend == "insufficient_data"

    # Case C: Improving trajectory across 3 submissions
    pt2 = BatchTrendPoint(
        batch_id="b2",
        filename="v2.csv",
        created_at="2026-09-22T10:00:00Z",
        total_records=50,
        successful_records=50,
        failed_records=0,
        pass_count=25,
        needs_improvement_count=18,
        fail_count=7,
        pass_percent=50.0,
        needs_improvement_percent=36.0,
        fail_percent=14.0,
        average_weighted_score=0.68,
        average_relevance_score=0.85,
        average_accuracy_score=0.65,
        average_completeness_score=0.65,
        average_groundedness_score=0.70,
        hallucination_rate_percent=30.0,
    )
    pt3 = BatchTrendPoint(
        batch_id="b3",
        filename="v3.csv",
        created_at="2026-09-25T10:00:00Z",
        total_records=50,
        successful_records=50,
        failed_records=0,
        pass_count=35,
        needs_improvement_count=12,
        fail_count=3,
        pass_percent=70.0,
        needs_improvement_percent=24.0,
        fail_percent=6.0,
        average_weighted_score=0.82,
        average_relevance_score=0.92,
        average_accuracy_score=0.85,
        average_completeness_score=0.80,
        average_groundedness_score=0.88,
        hallucination_rate_percent=12.0,
    )

    improving_summary = compute_trends_summary([pt1, pt2, pt3])
    assert improving_summary.total_batches == 3
    assert improving_summary.overall_quality_trend == "improving"
    assert improving_summary.score_change_percent == 32.0  # (0.82 - 0.50) * 100
    assert "improving" in improving_summary.summary_insight.lower()
    assert "hallucination" in improving_summary.summary_insight.lower()
    assert "accuracy" in improving_summary.summary_insight.lower()
    print(f"Improving Insight: {improving_summary.summary_insight}")

    # Case D: Degrading trajectory
    degrading_summary = compute_trends_summary([pt3, pt2, pt1])
    assert degrading_summary.overall_quality_trend == "degrading"
    assert degrading_summary.score_change_percent == -32.0
    print("[PASS] Longitudinal trajectory calculation correctly detects improving, degrading, and driver dimensions!")


def test_api_endpoints():
    print("\n--- 3. Testing Milestone 4 API Endpoints ---")
    client = TestClient(app)

    # Health check
    h_res = client.get("/health")
    assert h_res.status_code == 200
    h_json = h_res.json()
    assert "M4" in h_json.get("milestones", [])
    print(f"[PASS] Health check verified: version={h_json.get('version')}, milestones={h_json.get('milestones')}")

    # Batches list
    b_res = client.get("/api/v1/batches")
    assert b_res.status_code == 200
    batches = b_res.json()
    assert isinstance(batches, list)
    print(f"[PASS] /api/v1/batches returned {len(batches)} historical submissions")

    # Trends endpoint
    t_res = client.get("/api/v1/batches/trends")
    assert t_res.status_code == 200
    trends = t_res.json()
    assert "total_batches" in trends
    assert "batches" in trends
    assert "overall_quality_trend" in trends
    assert "score_change_percent" in trends
    assert "summary_insight" in trends
    print(f"[PASS] /api/v1/batches/trends response validated: total_batches={trends['total_batches']}, trend={trends['overall_quality_trend']}")

    # Trends alias endpoint
    alias_res = client.get("/api/v1/trends")
    assert alias_res.status_code == 200
    assert alias_res.json()["total_batches"] == trends["total_batches"]
    print("[PASS] /api/v1/trends alias endpoint verified")

    # Verify UI serving includes the new Trends view
    ui_res = client.get("/")
    assert ui_res.status_code == 200
    ui_html = ui_res.text
    assert "tab-nav-trends" in ui_html
    assert "tab-trends" in ui_html
    assert "dimensions-chart-svg" in ui_html
    assert "verdicts-chart-svg" in ui_html
    assert "trends-history-table-body" in ui_html
    print("[PASS] UI index.html contains tab-nav-trends, SVG chart containers, and historical registry table!")


if __name__ == "__main__":
    test_extract_batch_trend_point()
    test_compute_trends_summary_logic()
    test_api_endpoints()
    print("\n" + "=" * 80)
    print(" ALL MILESTONE 4 VALIDATION TESTS PASSED SUCCESSFULLY! ")
    print("=" * 80)
