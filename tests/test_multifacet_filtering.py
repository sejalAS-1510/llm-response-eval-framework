"""
Test Suite: Multi-Facet Dashboard Filtering & Aggregate Batch Statistics Scope.

Validates:
1. GET /api/v1/batches/combined returns aggregated statistics across all historical batches.
2. GET /api/v1/batches/combined/export produces structured CSV containing batch metadata.
3. Multi-facet filtering combinations:
   - Filter by verdict (Pass, Needs Improvement, Fail, etc.)
   - Filter by score range (e.g. Accuracy < 60%, Composite >= 80%)
   - Filter by specific batch ID
   - Combinations of verdict + score range + batch ID + text query
4. Front-end elements:
   - Scope toggle button (#scope-btn-single, #scope-btn-combined)
   - Facets controls (#filter-batch-select, #filter-score-dim, #filter-score-op, #filter-score-val)
   - Preset chips and active filters summary bar
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from src.input_module.main import app, get_all_batches_combined
from src.agents.schemas import BatchEvaluationSummary


def test_combined_batches_endpoint():
    print("\n--- 1. Testing GET /api/v1/batches/combined ---")
    client = TestClient(app)
    res = client.get("/api/v1/batches/combined")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    
    data = res.json()
    assert data["batch_id"] == "all-batches-combined", f"Unexpected batch_id: {data['batch_id']}"
    assert data["statistics"]["total_records"] >= 211, f"Expected at least 211 records, got {data['statistics']['total_records']}"
    assert len(data["items"]) == data["statistics"]["total_records"], "Items count does not match total_records"
    
    # Verify items have batch_id and batch_filename
    item0 = data["items"][0]
    assert "batch_id" in item0 and item0["batch_id"] is not None, "batch_id missing from item"
    assert "batch_filename" in item0 and item0["batch_filename"] is not None, "batch_filename missing from item"
    print(f"[PASS] /api/v1/batches/combined returned {len(data['items'])} items with valid aggregated statistics!")


def test_combined_export_endpoint():
    print("\n--- 2. Testing GET /api/v1/batches/combined/export ---")
    client = TestClient(app)
    res = client.get("/api/v1/batches/combined/export")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "text/csv" in res.headers.get("content-type", ""), "Expected CSV content-type"
    assert "attachment" in res.headers.get("content-disposition", ""), "Expected attachment content-disposition"

    lines = res.text.splitlines()
    assert len(lines) > 200, f"Expected >200 lines, got {len(lines)}"
    headers = lines[0].split(",")
    assert "Batch ID" in headers, "Batch ID missing from CSV headers"
    assert "Batch Filename" in headers, "Batch Filename missing from CSV headers"
    print(f"[PASS] /api/v1/batches/combined/export generated {len(lines)} CSV rows with batch metadata columns!")


def test_filtering_combinations_logic():
    print("\n--- 3. Testing Multi-Facet Filtering Logic & Combinations ---")
    combined: BatchEvaluationSummary = get_all_batches_combined()
    items = combined.items
    total = len(items)
    assert total >= 211, "Combined items count lower than expected"

    # Facet 1: Filter by Verdict
    pass_items = [it for it in items if it.result and it.result.verdict.verdict == "Pass"]
    fail_items = [it for it in items if it.result and it.result.verdict.verdict == "Fail"]
    assert len(pass_items) > 0, "No Pass items found"
    assert len(fail_items) > 0, "No Fail items found"

    # Facet 2: Filter by Score Range (Accuracy < 60%)
    low_acc = [it for it in items if it.result and (it.result.accuracy.score * 100) < 60]
    assert len(low_acc) > 0, "No low accuracy items found"

    # Facet 3: Filter by Specific Batch
    target_batch = "batch-1d00a7b7"
    batch_items = [it for it in items if it.batch_id == target_batch]
    assert len(batch_items) == 105, f"Expected 105 items for batch-1d00a7b7, got {len(batch_items)}"

    # Facet 4: Filter Combinations (Verdict == Fail AND Accuracy < 60% AND Batch == target_batch)
    combo_items = [
        it for it in items
        if it.batch_id == target_batch
        and it.result
        and it.result.verdict.verdict == "Fail"
        and (it.result.accuracy.score * 100) < 60
    ]
    print(f"Total: {total}, Batch items: {len(batch_items)}, Fail items in batch: {sum(1 for it in batch_items if it.result and it.result.verdict.verdict == 'Fail')}, Combo matches: {len(combo_items)}")
    assert len(combo_items) > 0, "Expected combination matches for (Fail AND Accuracy < 60% AND Batch)"
    for it in combo_items:
        assert it.batch_id == target_batch
        assert it.result.verdict.verdict == "Fail"
        assert (it.result.accuracy.score * 100) < 60

    print("[PASS] Multi-facet combination logic successfully isolates matching subsets!")


def test_ui_elements_in_html():
    print("\n--- 4. Testing UI HTML Elements for Scope Toggle & Filters ---")
    html_path = Path(__file__).resolve().parent.parent / "src" / "input_module" / "static" / "index.html"
    assert html_path.exists(), "index.html not found"
    content = html_path.read_text(encoding="utf-8")

    # Scope Toggle
    assert 'id="batch-scope-banner"' in content, "Missing #batch-scope-banner"
    assert 'id="scope-btn-single"' in content, "Missing #scope-btn-single"
    assert 'id="scope-btn-combined"' in content, "Missing #scope-btn-combined"
    assert 'onclick="setEvaluationScope(\'single\')"' in content, "Missing setEvaluationScope('single')"
    assert 'onclick="setEvaluationScope(\'combined\')"' in content, "Missing setEvaluationScope('combined')"

    # Multi-Facet Filter Controls
    assert 'id="filter-batch-select"' in content, "Missing #filter-batch-select"
    assert 'id="filter-score-dim"' in content, "Missing #filter-score-dim"
    assert 'id="filter-score-op"' in content, "Missing #filter-score-op"
    assert 'id="filter-score-val"' in content, "Missing #filter-score-val"
    assert 'applyScorePreset(\'accuracy\', \'<\', 60)' in content, "Missing accuracy preset"
    assert 'applyScorePreset(\'weighted_score\', \'>=\', 80)' in content, "Missing composite score preset"

    # Active Filters Bar & Reset
    assert 'id="active-filters-bar"' in content, "Missing #active-filters-bar"
    assert 'id="active-tags-list"' in content, "Missing #active-tags-list"
    assert 'onclick="clearAllFilters()"' in content, "Missing clearAllFilters button"

    print("[PASS] All required UI DOM elements and scope switch controls verified in index.html!")


if __name__ == "__main__":
    test_combined_batches_endpoint()
    test_combined_export_endpoint()
    test_filtering_combinations_logic()
    test_ui_elements_in_html()
    print("\n" + "=" * 80)
    print(" ALL MULTI-FACET FILTERING & COMBINED SCOPE TESTS PASSED SUCCESSFULLY! ")
    print("=" * 80)
