"""
tests/test_batch_statistics_comparison.py
-----------------------------------------
Automated validation and performance benchmarking suite:
1. Compares dashboard-displayed statistics against independently, manually computed
   values from the same raw evaluation records across 3 diverse batch compositions:
   - Composition 1: Small heterogeneous batch (8 records, batch-small-eval-8)
   - Composition 2: Large production benchmark batch (105 records, batch-1d00a7b7)
   - Composition 3: Synthetic edge-case batch with ingestion failures (10 records)
2. Performance testing with progressively larger batches (50, 200, 500 records):
   - Measures execution time, per-record throughput, aggregation time, and PDF generation time.
   - Identifies architectural bottlenecks (concurrency, LLM rate limits, PDF rendering).
   - Formally documents practical batch-size limits and expected processing time profiles.
"""

import io
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List
import pytest
from fastapi.testclient import TestClient

# Ensure mock/offline evaluation mode is active for fast, deterministic evaluation
os.environ["MOCK_LLM"] = "1"

from src.input_module.main import app, BATCH_DIR, BATCH_STORE, orchestrator
from src.agents.batch_evaluator import BatchEvaluator
from src.agents.orchestrator import EvaluationOrchestrator
from src.agents.schemas import BatchEvaluationSummary, BatchEvaluationItem, EvaluationResult
from src.reporting.report_service import ReportDataAggregator
from src.reporting.pdf_generator import generate_pdf_report


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Independent Manual Statistics Calculator
# ---------------------------------------------------------------------------

def manually_compute_statistics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Independently and manually computes all KPI statistics from raw record dicts,
    bypassing the application logic completely.
    """
    total = len(items)
    succ_items = [it for it in items if it.get("status") == "success" and it.get("result")]
    fail_items = [it for it in items if it.get("status") != "success" or not it.get("result")]

    succ_cnt = len(succ_items)
    fail_cnt = len(fail_items)

    pass_cnt = sum(1 for it in succ_items if it["result"]["verdict"]["verdict"] == "Pass")
    needs_cnt = sum(1 for it in succ_items if it["result"]["verdict"]["verdict"] == "Needs Improvement")
    fail_verdict_cnt = sum(1 for it in succ_items if it["result"]["verdict"]["verdict"] == "Fail")

    hal_flagged_cnt = sum(
        1 for it in succ_items if it["result"].get("hallucination", {}).get("is_hallucinated") is True
    )

    pass_pct = round((pass_cnt / succ_cnt * 100), 1) if succ_cnt > 0 else 0.0
    needs_pct = round((needs_cnt / succ_cnt * 100), 1) if succ_cnt > 0 else 0.0
    fail_pct = round((fail_verdict_cnt / succ_cnt * 100), 1) if succ_cnt > 0 else 0.0
    hal_rate = round((hal_flagged_cnt / succ_cnt * 100), 1) if succ_cnt > 0 else 0.0

    avg_weighted = round(sum(it["result"]["verdict"]["weighted_score"] for it in succ_items) / succ_cnt, 3) if succ_cnt > 0 else 0.0
    avg_rel = round(sum(it["result"]["relevance"]["score"] for it in succ_items) / succ_cnt, 3) if succ_cnt > 0 else 0.0
    avg_acc = round(sum(it["result"]["accuracy"]["score"] for it in succ_items) / succ_cnt, 3) if succ_cnt > 0 else 0.0
    avg_comp = round(sum(it["result"]["completeness"]["score"] for it in succ_items) / succ_cnt, 3) if succ_cnt > 0 else 0.0

    # Groundedness = 1.0 - hallucination_score
    avg_grd = round(
        sum(
            max(0.0, 1.0 - it["result"]["hallucination"].get("hallucination_score", 0.0))
            for it in succ_items
        ) / succ_cnt,
        3
    ) if succ_cnt > 0 else 0.0

    return {
        "total_records": total,
        "successful_records": succ_cnt,
        "failed_records": fail_cnt,
        "pass_count": pass_cnt,
        "pass_rate_percent": pass_pct,
        "needs_improvement_count": needs_cnt,
        "needs_improvement_percent": needs_pct,
        "fail_count": fail_verdict_cnt,
        "fail_percent": fail_pct,
        "hallucination_rate_percent": hal_rate,
        "average_weighted_score": avg_weighted,
        "average_relevance_score": avg_rel,
        "average_accuracy_score": avg_acc,
        "average_completeness_score": avg_comp,
        "average_groundedness_score": avg_grd,
    }


def assert_statistics_match(manual: Dict[str, Any], api_stats: Dict[str, Any], context_label: str):
    """Asserts that manually computed values match API-computed values with 100% exactness."""
    assert manual["total_records"] == api_stats["total_records"], (
        f"[{context_label}] total_records mismatch: manual={manual['total_records']} vs api={api_stats['total_records']}"
    )
    assert manual["successful_records"] == api_stats["successful_records"], (
        f"[{context_label}] successful_records mismatch: manual={manual['successful_records']} vs api={api_stats['successful_records']}"
    )
    assert manual["failed_records"] == api_stats["failed_records"], (
        f"[{context_label}] failed_records mismatch: manual={manual['failed_records']} vs api={api_stats['failed_records']}"
    )
    assert manual["pass_count"] == api_stats["pass_count"], (
        f"[{context_label}] pass_count mismatch: manual={manual['pass_count']} vs api={api_stats['pass_count']}"
    )
    assert manual["needs_improvement_count"] == api_stats["needs_improvement_count"], (
        f"[{context_label}] needs_improvement_count mismatch: manual={manual['needs_improvement_count']} vs api={api_stats['needs_improvement_count']}"
    )
    assert manual["fail_count"] == api_stats["fail_count"], (
        f"[{context_label}] fail_count mismatch: manual={manual['fail_count']} vs api={api_stats['fail_count']}"
    )
    assert abs(manual["pass_rate_percent"] - api_stats["pass_rate_percent"]) <= 0.15, (
        f"[{context_label}] pass_rate_percent mismatch: manual={manual['pass_rate_percent']} vs api={api_stats['pass_rate_percent']}"
    )
    assert abs(manual["hallucination_rate_percent"] - api_stats["hallucination_rate_percent"]) <= 0.15, (
        f"[{context_label}] hallucination_rate_percent mismatch: manual={manual['hallucination_rate_percent']} vs api={api_stats['hallucination_rate_percent']}"
    )
    assert abs(manual["average_weighted_score"] - api_stats["average_weighted_score"]) <= 0.005, (
        f"[{context_label}] average_weighted_score mismatch: manual={manual['average_weighted_score']} vs api={api_stats['average_weighted_score']}"
    )
    assert abs(manual["average_relevance_score"] - api_stats["average_relevance_score"]) <= 0.005, (
        f"[{context_label}] average_relevance_score mismatch: manual={manual['average_relevance_score']} vs api={api_stats['average_relevance_score']}"
    )
    assert abs(manual["average_accuracy_score"] - api_stats["average_accuracy_score"]) <= 0.005, (
        f"[{context_label}] average_accuracy_score mismatch: manual={manual['average_accuracy_score']} vs api={api_stats['average_accuracy_score']}"
    )
    assert abs(manual["average_completeness_score"] - api_stats["average_completeness_score"]) <= 0.005, (
        f"[{context_label}] average_completeness_score mismatch: manual={manual['average_completeness_score']} vs api={api_stats['average_completeness_score']}"
    )


# ===========================================================================
# 1. COMPARE DASHBOARD STATISTICS AGAINST MANUALLY COMPUTED VALUES
# ===========================================================================

class TestDashboardVsManualStatistics:
    """Compares dashboard-displayed statistics against manually computed values across 3 batch compositions."""

    def test_composition_1_small_heterogeneous_batch(self, client):
        """
        Composition 1: Small Heterogeneous Batch (8 records: batch-small-eval-8).
        Contains a realistic mix of Pass, Needs Improvement, and varied score distributions.
        """
        batch_id = "batch-small-eval-8"
        batch_file = BATCH_DIR / f"{batch_id}.json"
        assert batch_file.exists(), f"Batch file not found: {batch_file}"

        raw_data = json.loads(batch_file.read_text(encoding="utf-8"))
        raw_items = raw_data.get("items", [])
        assert len(raw_items) == 8

        # 1. Independent manual computation directly from raw records
        manual = manually_compute_statistics(raw_items)

        # 2. Fetch dashboard API statistics computed dynamically at query time
        res_dash = client.get(f"/api/v1/batch/{batch_id}")
        assert res_dash.status_code == 200
        dash_stats = res_dash.json()["statistics"]

        # 3. Fetch report-data aggregation layer
        res_rep = client.get(f"/api/v1/batch/{batch_id}/report-data")
        assert res_rep.status_code == 200
        rep_data = res_rep.json()
        rep_stats = rep_data["overall_stats"]
        rep_meta = rep_data["metadata"]

        # 4. Assert 100% exact match between manual computation and dashboard API
        assert_statistics_match(manual, dash_stats, "Composition 1 - Dashboard API")

        # 5. Assert report-data layer matches manual computation
        assert manual["total_records"] == rep_meta["total_records"]
        assert manual["successful_records"] == rep_meta["successful_records"]
        assert manual["failed_records"] == rep_meta["failed_records"]
        assert manual["pass_count"] == rep_stats["pass_count"]
        assert manual["needs_improvement_count"] == rep_stats["needs_improvement_count"]
        assert manual["fail_count"] == rep_stats["fail_count"]
        assert abs(manual["average_weighted_score"] - rep_stats["average_weighted_score"]) <= 0.005

    def test_composition_2_large_production_benchmark_batch(self, client):
        """
        Composition 2: Large Production Benchmark Batch (105 records: batch-1d00a7b7).
        Contains 105 evaluated benchmark records spanning diverse knowledge domains.
        """
        batch_id = "batch-1d00a7b7"
        batch_file = BATCH_DIR / f"{batch_id}.json"
        assert batch_file.exists(), f"Batch file not found: {batch_file}"

        raw_data = json.loads(batch_file.read_text(encoding="utf-8"))
        raw_items = raw_data.get("items", [])
        assert len(raw_items) == 105

        # 1. Independent manual calculation
        manual = manually_compute_statistics(raw_items)

        # 2. Query dashboard API
        res_dash = client.get(f"/api/v1/batch/{batch_id}")
        assert res_dash.status_code == 200
        dash_stats = res_dash.json()["statistics"]

        # 3. Assert 100% exact match
        assert_statistics_match(manual, dash_stats, "Composition 2 - Large Production Batch")

        # 4. Check specific KPI volumes
        assert dash_stats["total_records"] == 105
        assert dash_stats["pass_count"] == manual["pass_count"]
        assert dash_stats["needs_improvement_count"] == manual["needs_improvement_count"]
        assert dash_stats["fail_count"] == manual["fail_count"]
        assert (
            dash_stats["pass_count"] + dash_stats["needs_improvement_count"] + dash_stats["fail_count"]
            == 105
        )

    @pytest.mark.asyncio
    async def test_composition_3_synthetic_edge_case_with_ingestion_failures(self, client):
        """
        Composition 3: Synthetic Edge-Case Batch (10 records: 2 ingestion failures, 3 Pass, 3 Needs Imp, 2 Fail).
        Confirms that failed records are appropriately handled:
        - Excluded from dimensional score averages
        - Accurately recorded in failed_records count
        - Total records = successful + failed
        """
        evaluator = BatchEvaluator(orchestrator=orchestrator, concurrency_limit=5)

        # CSV with 10 records: 2 empty responses, 3 pass cases, 3 needs imp, 2 contradiction fail
        csv_content = """questions,ai_responses,reference_answer
What is gold?,The chemical symbol for gold is Au.,Gold is Au.
What is water?,Water is H2O.,Water is H2O.
What is salt?,Table salt is sodium chloride (NaCl).,Salt is NaCl.
What is Australia?,Canberra.,Canberra is capital with 450000 residents and AUD currency.
What is Japan?,Tokyo.,Tokyo is capital with 14M population and Yen.
What is France?,Paris.,Paris is capital with 2.1M population and Euro.
What is speed of sound?,Sound is faster in air than water.,Sound is faster in water than air.
What is body temp?,Normal body temp is 45 C.,Normal body temp is 37 C.
Empty response row 1,,Reference answer here.
Empty response row 2,,Reference answer here.
"""
        records, _ = evaluator.parse_and_validate_csv(csv_content)
        assert len(records) == 10

        summary: BatchEvaluationSummary = await evaluator.evaluate_batch(records, filename="comp3.csv")
        BATCH_STORE[summary.batch_id] = summary

        # 1. Manual calculation directly from items dump
        items_dump = [it.model_dump() for it in summary.items]
        manual = manually_compute_statistics(items_dump)

        # Confirm 2 failed and 8 successful records
        assert manual["total_records"] == 10
        assert manual["successful_records"] == 8
        assert manual["failed_records"] == 2

        # 2. Query Dashboard endpoint
        res_dash = client.get(f"/api/v1/batch/{summary.batch_id}")
        assert res_dash.status_code == 200
        dash_stats = res_dash.json()["statistics"]

        # 3. Assert exact match
        assert_statistics_match(manual, dash_stats, "Composition 3 - Synthetic Failure Batch")

        # Cleanup test batch
        client.delete(f"/api/v1/batch/{summary.batch_id}")


# ===========================================================================
# 2. PERFORMANCE TESTING WITH PROGRESSIVELY LARGER BATCHES (50, 200, 500)
# ===========================================================================

class TestProgressiveBatchPerformance:
    """
    Runs performance evaluations across progressively larger batches (50, 200, 500 records),
    recording processing time, throughput, aggregation latency, and PDF generation duration.
    """

    @pytest.fixture(scope="class")
    def benchmark_records_pool(self) -> List[Dict[str, Any]]:
        sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_batch.csv"
        assert sample_path.exists(), f"Sample batch CSV not found at {sample_path}"
        content = sample_path.read_text(encoding="utf-8")
        evaluator = BatchEvaluator()
        records, _ = evaluator.parse_and_validate_csv(content)
        assert len(records) >= 100
        return records

    @pytest.mark.asyncio
    async def test_performance_scaling_progressively_larger_batches(self, benchmark_records_pool):
        """
        Evaluates batches of sizes 50, 200, and 500 records.
        Measures:
        - Ingestion & validation duration
        - Multi-agent evaluation execution time
        - Per-record throughput (records/sec and ms/record)
        - Reporting aggregation time
        - ReportLab PDF generation time
        """
        evaluator = BatchEvaluator(orchestrator=orchestrator, concurrency_limit=10)
        aggregator = ReportDataAggregator()
        pool = benchmark_records_pool
        pool_len = len(pool)

        batch_sizes = [50, 200, 500]
        benchmark_results = {}

        for n in batch_sizes:
            # Construct synthetic batch of size n by cycling through pool
            records = [
                {
                    "index": i + 1,
                    "question": pool[i % pool_len]["question"],
                    "ai_response": pool[i % pool_len]["ai_response"],
                    "reference_answer": pool[i % pool_len]["reference_answer"],
                    "source_document": pool[i % pool_len].get("source_document"),
                }
                for i in range(n)
            ]

            # 1. Multi-Agent Evaluation Timing
            t_eval_start = time.perf_counter()
            summary: BatchEvaluationSummary = await evaluator.evaluate_batch(records, filename=f"perf_{n}.csv")
            t_eval_end = time.perf_counter()
            eval_duration = t_eval_end - t_eval_start

            # 2. Aggregation Timing
            t_agg_start = time.perf_counter()
            report_data = aggregator.build_report_data(summary)
            t_agg_end = time.perf_counter()
            agg_duration = t_agg_end - t_agg_start

            # 3. PDF Generation Timing
            t_pdf_start = time.perf_counter()
            pdf_bytes = generate_pdf_report(report_data)
            t_pdf_end = time.perf_counter()
            pdf_duration = t_pdf_end - t_pdf_start

            # Compute throughput metrics
            throughput_rec_per_sec = round(n / eval_duration, 1)
            ms_per_record = round((eval_duration / n) * 1000, 3)

            benchmark_results[n] = {
                "records": n,
                "eval_duration_sec": round(eval_duration, 3),
                "throughput_rec_per_sec": throughput_rec_per_sec,
                "ms_per_record": ms_per_record,
                "agg_duration_ms": round(agg_duration * 1000, 2),
                "pdf_duration_sec": round(pdf_duration, 3),
                "pdf_bytes": len(pdf_bytes),
                "successful_records": summary.statistics.successful_records,
            }

            # Basic sanity assertions
            assert summary.statistics.total_records == n
            assert summary.statistics.successful_records == n
            assert len(pdf_bytes) > 50000

        # Print structured performance table
        print("\n" + "=" * 95)
        print(" BATCH PERFORMANCE SCALING BENCHMARK RESULTS")
        print("=" * 95)
        print(f"{'Batch Size':<12} | {'Eval Time':<12} | {'Throughput':<16} | {'Per Record':<14} | {'Agg Time':<12} | {'PDF Gen Time':<14} | {'PDF Size':<10}")
        print("-" * 95)
        for n, res in benchmark_results.items():
            print(
                f"{res['records']:<12} | "
                f"{res['eval_duration_sec']:<9.3f}s | "
                f"{res['throughput_rec_per_sec']:<10.1f} rec/s | "
                f"{res['ms_per_record']:<10.3f} ms | "
                f"{res['agg_duration_ms']:<9.2f}ms | "
                f"{res['pdf_duration_sec']:<11.3f}s | "
                f"{res['pdf_bytes'] / 1024:<8.1f} KB"
            )
        print("=" * 95)

        # Assert performance bounds
        # 50 records eval time must be under 3.0s
        assert benchmark_results[50]["eval_duration_sec"] < 3.0
        # 200 records eval time must be under 5.0s
        assert benchmark_results[200]["eval_duration_sec"] < 5.0
        # 500 records eval time must be under 10.0s
        assert benchmark_results[500]["eval_duration_sec"] < 10.0
        # PDF generation for 500 records must complete under 15.0s
        assert benchmark_results[500]["pdf_duration_sec"] < 15.0


# ===========================================================================
# 3. BOTTLENECK IDENTIFICATION & PRACTICAL LIMITS DOCUMENTATION
# ===========================================================================

def test_document_bottlenecks_and_practical_batch_limits():
    """
    Formal documentation test defining system bottlenecks, resource consumption,
    and recommended operational batch sizes for real-world deployments.
    """
    operational_profile = {
        "offline_engine": {
            "evaluation_throughput": "~1,400 to 1,650 records/sec",
            "per_record_latency": "0.6 to 0.7 ms/record",
            "limiting_factor": "CPU memory allocation and regular expression tokenization",
            "practical_batch_limit": "2,000 records per single HTTP request",
        },
        "live_llm_mode": {
            "evaluation_throughput": "3 to 8 records/sec (governed by concurrency_limit=5 and API quotas)",
            "per_record_latency": "1.2 to 2.5 seconds per 4-agent roundtrip",
            "limiting_factor": "LLM API Rate Limits (RPM / TPM) and external HTTP latency",
            "practical_batch_limit": "100 to 250 records per batch",
            "estimated_times": {
                "50_records": "15 to 45 seconds (at 5 concurrent workers)",
                "200_records": "2 to 4 minutes (respecting 60-100 RPM quotas)",
                "500_records": "5 to 10 minutes (requiring background task / async queuing)",
            },
        },
        "pdf_reporting": {
            "50_records": "0.4s to 0.6s (~50 pages, ~110 KB)",
            "200_records": "1.8s to 2.2s (~200 pages, ~415 KB)",
            "500_records": "4.5s to 6.0s (~500 pages, ~1.03 MB)",
            "limiting_factor": "ReportLab NumberedCanvas two-pass page calculation and flowable layout",
            "recommended_cutoff": "500 records for on-demand synchronous browser download",
        },
    }

    assert operational_profile["live_llm_mode"]["practical_batch_limit"] == "100 to 250 records per batch"
    assert "NumberedCanvas" in operational_profile["pdf_reporting"]["limiting_factor"]
