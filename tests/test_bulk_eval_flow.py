"""
Validation suite for the updated CSV Bulk Upload and Evaluation Flow.
Tests:
1. Missing required columns validation (rejects invalid files, returns clear error).
2. Empty questions handling (allowed, not rejected, evaluated, displayed as '—' / 'Empty').
3. Individual failed rows (one failed row does not halt the batch; remaining rows evaluate).
4. Bulk evaluation of 100+ rows (105 records from sample CSV).
5. Verdict Agent reuse verification (agent is NOT recreated per row).
6. Final output structure and CSV export integrity.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io
import asyncio
from fastapi.testclient import TestClient
from src.input_module.main import app, orchestrator, batch_evaluator
from src.agents.batch_evaluator import BatchEvaluator
from src.agents.verdict_agent import VerdictAgent


def test_missing_required_columns():
    print("\n--- 1. Testing Missing Required Columns Validation ---")
    client = TestClient(app)

    # Case A: Missing 'questions' column
    csv_missing_q = "col_x,ai_responses\nfoo,This is an AI response.\n"
    res_a = client.post(
        "/api/v1/batch/upload-csv",
        files={"file": ("missing_q.csv", io.BytesIO(csv_missing_q.encode("utf-8")), "text/csv")},
    )
    print(f"Case A (Missing 'questions'): status={res_a.status_code}, error={res_a.json().get('detail')}")
    assert res_a.status_code == 400
    assert "Missing required column(s)" in res_a.json()["detail"]
    assert "'questions'" in res_a.json()["detail"]

    # Case B: Missing 'ai_responses' column
    csv_missing_r = "questions,other_col\nWhat is Python?,A programming language.\n"
    res_b = client.post(
        "/api/v1/batch/upload-csv",
        files={"file": ("missing_r.csv", io.BytesIO(csv_missing_r.encode("utf-8")), "text/csv")},
    )
    print(f"Case B (Missing 'ai_responses'): status={res_b.status_code}, error={res_b.json().get('detail')}")
    assert res_b.status_code == 400
    assert "Missing required column(s)" in res_b.json()["detail"]
    assert "'ai_responses'" in res_b.json()["detail"]

    # Case C: Both required columns missing
    csv_random = "name,email,age\nAlice,alice@example.com,30\n"
    res_c = client.post(
        "/api/v1/batch/upload-csv",
        files={"file": ("random.csv", io.BytesIO(csv_random.encode("utf-8")), "text/csv")},
    )
    print(f"Case C (Both missing): status={res_c.status_code}, error={res_c.json().get('detail')}")
    assert res_c.status_code == 400
    assert "'questions'" in res_c.json()["detail"] and "'ai_responses'" in res_c.json()["detail"]

    # Case D: Empty file
    res_d = client.post(
        "/api/v1/batch/upload-csv",
        files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
    )
    assert res_d.status_code == 400

    print("[PASS] Missing required columns correctly rejected with clear validation errors!")


async def test_empty_questions_and_failed_rows():
    print("\n--- 2. Testing Empty Questions & Resilient Failed Row Handling ---")
    evaluator = BatchEvaluator(orchestrator=orchestrator, concurrency_limit=5)

    # CSV with:
    # Row 1: Valid Q & A
    # Row 2: Empty Question (should be allowed and evaluated)
    # Row 3: Empty AI Response (should fail individually without stopping Row 1, 2, 4)
    # Row 4: Valid Q & A
    mixed_csv = """questions,ai_responses,reference_answer
What is the chemical symbol for gold?,The chemical symbol for gold is Au.,Gold is Au.
,Water boils at 100 degrees Celsius under standard atmospheric pressure.,Water boils at 100 C.
What is the capital of Japan?,,Tokyo is the capital of Japan.
What is the boiling point of ethanol?,Ethanol boils at approximately 78.37 degrees Celsius.,Ethanol boils at 78.4 C.
"""
    records, warnings = evaluator.parse_and_validate_csv(mixed_csv)
    print(f"Parsed {len(records)} records, {len(warnings)} warning(s).")
    assert len(records) == 4
    # Empty question row must be preserved
    assert records[1]["question"] in ("[No Question Provided]", "", "—")

    # Evaluate batch
    summary = await evaluator.evaluate_batch(records, filename="mixed_batch.csv")

    print(f"Batch Summary: Total={summary.statistics.total_records}, Success={summary.statistics.successful_records}, Failed={summary.statistics.failed_records}")
    assert summary.statistics.total_records == 4
    assert summary.statistics.successful_records == 3, "Rows 1, 2, and 4 should succeed"
    assert summary.statistics.failed_records == 1, "Row 3 should be marked as failed"

    # Verify Row 2 (Empty question)
    row2 = summary.items[1]
    assert row2.status == "success"
    assert row2.result is not None
    assert row2.result.verdict.verdict in ("Pass", "Needs Improvement", "Fail")
    print(f"Row 2 (Empty Q): Successfully evaluated! Verdict={row2.result.verdict.verdict}, Score={row2.result.verdict.weighted_score}")

    # Verify Row 3 (Failed row due to empty AI response)
    row3 = summary.items[2]
    assert row3.status == "failed"
    assert row3.result is None
    assert "empty" in row3.error_message.lower()
    print(f"Row 3 (Empty Response): Failed row isolated with clear reason: '{row3.error_message}'")

    # Verify Row 4 (Evaluated despite Row 3 failing)
    row4 = summary.items[3]
    assert row4.status == "success"
    assert row4.result is not None
    print(f"Row 4: Successfully evaluated despite previous row failure! Verdict={row4.result.verdict.verdict}")

    # Verify CSV export handles empty questions and failed rows
    csv_export = evaluator.export_summary_to_csv(summary)
    assert "FAILED" in csv_export
    assert "AI Response is empty" in csv_export
    print("[PASS] Empty questions evaluated seamlessly and individual failures isolated without halting batch!")


async def test_verdict_agent_not_recreated():
    print("\n--- 3. Verifying Verdict Agent Re-use (No Per-Row Re-instantiation) ---")
    initial_verdict_agent = orchestrator.verdict_agent
    assert isinstance(initial_verdict_agent, VerdictAgent)
    agent_id = id(initial_verdict_agent)

    # Confirm batch_evaluator uses orchestrator's verdict agent
    assert id(batch_evaluator.orchestrator.verdict_agent) == agent_id

    # Run evaluation
    res = await orchestrator.evaluate(
        question="What is the speed of sound?",
        ai_response="The speed of sound in air is approximately 343 meters per second.",
        reference_answer="Sound travels at ~343 m/s in air.",
    )
    assert id(orchestrator.verdict_agent) == agent_id
    print(f"VerdictAgent instance ID retained: {agent_id} (Reused across evaluations, not recreated per row).")
    print("[PASS] Verdict Agent verified to be reused and not recreated per row!")


async def test_100_plus_csv_bulk_evaluation():
    print("\n--- 4. Testing 100+ Records Bulk Evaluation Flow ---")
    sample_file = Path(__file__).resolve().parent.parent / "data" / "sample_batch.csv"
    assert sample_file.exists(), f"Sample batch file not found at {sample_file}"

    csv_content = sample_file.read_text(encoding="utf-8")
    records, warnings = batch_evaluator.parse_and_validate_csv(csv_content)
    print(f"Ingested {len(records)} benchmark records from {sample_file.name}.")
    assert len(records) >= 100, f"Expected 100+ records, got {len(records)}"

    # Evaluate all 100+ records
    summary = await batch_evaluator.evaluate_batch(records, filename=sample_file.name)
    stats = summary.statistics

    print(f"Bulk Evaluation Summary (100+ records):")
    print(f"  Total Rows: {stats.total_records}")
    print(f"  Successfully Evaluated: {stats.successful_records}")
    print(f"  Skipped/Failed: {stats.failed_records}")
    print(f"  Pass: {stats.pass_count} ({stats.pass_rate_percent}%)")
    print(f"  Needs Improvement: {stats.needs_improvement_count}")
    print(f"  Fail: {stats.fail_count}")
    print(f"  Average Weighted Score: {stats.average_weighted_score}")
    print(f"  Hallucination Rate: {stats.hallucination_rate_percent}%")

    assert stats.total_records == len(records)
    assert stats.successful_records == len(records)
    assert stats.failed_records == 0
    assert len(summary.items) == len(records)
    assert stats.pass_rate_percent > 0

    # Verify CSV export
    exported_csv = batch_evaluator.export_summary_to_csv(summary)
    lines = exported_csv.strip().split("\n")
    print(f"Exported CSV line count: {len(lines)} (Header + {len(lines)-1} rows)")
    assert len(lines) == stats.total_records + 1

    print("[PASS] 100+ records bulk evaluation executed successfully with complete statistics and export!")


async def main():
    print("=" * 80)
    print(" BULK CSV EVALUATION FLOW VALIDATION SUITE ")
    print("=" * 80)
    test_missing_required_columns()
    await test_empty_questions_and_failed_rows()
    await test_verdict_agent_not_recreated()
    await test_100_plus_csv_bulk_evaluation()
    print("\n" + "=" * 80)
    print(" ALL BULK CSV FLOW VALIDATION TESTS PASSED SUCCESSFULLY! ")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
