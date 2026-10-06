"""
tests/test_data_integrity.py
----------------------------
Automated data-integrity validation suite.
Confirms that every single metric, score, count, and rate served by the batch APIs
and displayed on the dashboard is computed dynamically from stored structured evaluation
records (BatchEvaluationItem) at query time, not cached or manually entered.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.input_module.main import app, BATCH_DIR, BATCH_STORE
from src.agents.batch_evaluator import BatchEvaluator
from src.agents.schemas import BatchEvaluationItem, BatchEvaluationSummary, EvaluationResult, VerdictResult, AccuracyResult, RelevanceResult, CompletenessResult, HallucinationResult


@pytest.fixture
def client():
    return TestClient(app)


def test_stored_batches_query_time_exact_computation(client):
    """
    Validates that for all persisted batch files in data/batches/*.json,
    the numbers returned by GET /api/v1/batch/{id} exactly match independent
    manual aggregations of the raw items array.
    """
    json_files = list(BATCH_DIR.glob("*.json"))
    assert len(json_files) > 0, "No persisted batch files found in data/batches"

    for jpath in json_files:
        raw_data = json.loads(jpath.read_text(encoding="utf-8"))
        batch_id = raw_data.get("batch_id", jpath.stem)
        raw_items = raw_data.get("items", [])
        
        # 1. Independent manual calculation directly from structured raw_items
        total = len(raw_items)
        success_items = [it for it in raw_items if it.get("status") == "success" and it.get("result")]
        failed_items = [it for it in raw_items if it.get("status") != "success" or not it.get("result")]
        
        succ_count = len(success_items)
        failed_count = len(failed_items)
        
        pass_count = sum(1 for it in success_items if it["result"]["verdict"]["verdict"] == "Pass")
        needs_count = sum(1 for it in success_items if it["result"]["verdict"]["verdict"] == "Needs Improvement")
        fail_count = sum(1 for it in success_items if it["result"]["verdict"]["verdict"] == "Fail")
        hal_count = sum(1 for it in success_items if it["result"].get("hallucination", {}).get("is_hallucinated") is True)
        
        expected_pass_rate = round((pass_count / total * 100), 1) if total > 0 else 0.0
        expected_hal_rate = round((hal_count / succ_count * 100), 1) if succ_count > 0 else 0.0
        
        expected_avg_weight = (sum(it["result"]["verdict"]["weighted_score"] for it in success_items) / succ_count) if succ_count > 0 else 0.0
        expected_avg_rel = (sum(it["result"]["relevance"]["score"] for it in success_items) / succ_count) if succ_count > 0 else 0.0
        expected_avg_acc = (sum(it["result"]["accuracy"]["score"] for it in success_items) / succ_count) if succ_count > 0 else 0.0
        expected_avg_comp = (sum(it["result"]["completeness"]["score"] for it in success_items) / succ_count) if succ_count > 0 else 0.0

        # 2. Query the API endpoint at runtime
        resp = client.get(f"/api/v1/batch/{batch_id}")
        assert resp.status_code == 200, f"Failed to get batch {batch_id}: {resp.text}"
        data = resp.json()
        stats = data["statistics"]

        # 3. Assert mathematical equality between stored records and API-returned numbers
        assert stats["total_records"] == total
        assert stats["successful_records"] == succ_count
        assert stats["failed_records"] == failed_count
        assert stats["pass_count"] == pass_count
        assert stats["needs_improvement_count"] == needs_count
        assert stats["fail_count"] == fail_count
        assert abs(stats["pass_rate_percent"] - expected_pass_rate) < 0.15
        assert abs(stats["hallucination_rate_percent"] - expected_hal_rate) < 0.15
        assert abs(stats["average_weighted_score"] - expected_avg_weight) < 1e-3
        assert abs(stats["average_relevance_score"] - expected_avg_rel) < 1e-3
        assert abs(stats["average_accuracy_score"] - expected_avg_acc) < 1e-3
        assert abs(stats["average_completeness_score"] - expected_avg_comp) < 1e-3


def test_combined_batches_query_time_exact_computation(client):
    """
    Validates that GET /api/v1/batches/combined dynamically aggregates every
    historical batch record across disk without caching or static hardcoding.
    """
    json_files = list(BATCH_DIR.glob("*.json"))
    all_raw_items = []
    for jpath in json_files:
        raw_data = json.loads(jpath.read_text(encoding="utf-8"))
        all_raw_items.extend(raw_data.get("items", []))

    total = len(all_raw_items)
    success_items = [it for it in all_raw_items if it.get("status") == "success" and it.get("result")]
    succ_count = len(success_items)

    expected_pass = sum(1 for it in success_items if it["result"]["verdict"]["verdict"] == "Pass")
    expected_needs = sum(1 for it in success_items if it["result"]["verdict"]["verdict"] == "Needs Improvement")
    expected_fail = sum(1 for it in success_items if it["result"]["verdict"]["verdict"] == "Fail")

    resp = client.get("/api/v1/batches/combined")
    assert resp.status_code == 200
    data = resp.json()
    stats = data["statistics"]

    assert stats["total_records"] == total
    assert stats["successful_records"] == succ_count
    assert stats["pass_count"] == expected_pass
    assert stats["needs_improvement_count"] == expected_needs
    assert stats["fail_count"] == expected_fail
    assert len(data["items"]) == total


def test_dynamic_recomputation_upon_record_mutation(client):
    """
    Proves that statistics are computed dynamically at query time from structured items
    by injecting an in-memory batch, mutating its underlying structured records, and confirming
    the subsequent query reflects the change without server restart or manual intervention.
    """
    mock_batch_id = "test-dynamic-integrity-check"
    
    import copy
    # Load a valid item from an existing batch on disk
    sample_json = list(BATCH_DIR.glob("*.json"))[0]
    raw_data = json.loads(sample_json.read_text(encoding="utf-8"))
    base_item_dict = raw_data["items"][0]

    dict1 = copy.deepcopy(base_item_dict)
    dict1["index"] = 1
    dict1["batch_id"] = mock_batch_id
    dict1["result"]["verdict"]["verdict"] = "Pass"
    dict1["result"]["verdict"]["weighted_score"] = 0.90
    dict1["result"]["accuracy"]["score"] = 0.90
    dict1["result"]["hallucination"]["is_hallucinated"] = False

    dict2 = copy.deepcopy(base_item_dict)
    dict2["index"] = 2
    dict2["batch_id"] = mock_batch_id
    dict2["result"]["verdict"]["verdict"] = "Fail"
    dict2["result"]["verdict"]["weighted_score"] = 0.30
    dict2["result"]["accuracy"]["score"] = 0.20
    dict2["result"]["hallucination"]["is_hallucinated"] = True

    item1 = BatchEvaluationItem(**dict1)
    item2 = BatchEvaluationItem(**dict2)

    evaluator = BatchEvaluator()
    summary = BatchEvaluationSummary(
        batch_id=mock_batch_id,
        filename="test.csv",
        created_at="2026-09-26T12:00:00Z",
        statistics=evaluator.calculate_statistics([item1, item2]),
        items=[item1, item2],
    )

    BATCH_STORE[mock_batch_id] = summary

    try:
        # First query: 2 items, 1 Pass, 1 Fail -> Pass rate = 50%
        r1 = client.get(f"/api/v1/batch/{mock_batch_id}")
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["statistics"]["total_records"] == 2
        assert d1["statistics"]["pass_count"] == 1
        assert d1["statistics"]["fail_count"] == 1
        assert d1["statistics"]["pass_rate_percent"] == 50.0

        # Now MUTATE item2 to be a Pass with 100% scores in the structured item store
        item2.result.verdict.verdict = "Pass"
        item2.result.verdict.weighted_score = 0.95
        item2.result.accuracy.score = 0.95
        item2.result.hallucination.is_hallucinated = False

        # Query again at runtime - dynamic calculation MUST produce 2 Passes, 0 Fails, 100% pass rate
        r2 = client.get(f"/api/v1/batch/{mock_batch_id}")
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["statistics"]["pass_count"] == 2, "Statistics were not computed at query time from structured items!"
        assert d2["statistics"]["fail_count"] == 0
        assert d2["statistics"]["pass_rate_percent"] == 100.0
        assert d2["statistics"]["average_weighted_score"] == pytest.approx((0.90 + 0.95) / 2, rel=1e-3)

    finally:
        # Cleanup mock
        BATCH_STORE.pop(mock_batch_id, None)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
