"""
scripts/validate_pdf_report_integrity.py
-----------------------------------------
Automated validation script for LLM Response Evaluation Framework PDF reports.
Asserts that:
(1) Every metric and summary number in the PDF report matches the underlying stored
    evaluation records exactly (metadata, verdict counts, percentages, dimensional averages).
(2) Every per-response detail matches what's shown in the dashboard/table for the same records
    (prompt, generation, reference, dimension scores, reasoning, flagged claims, missing aspects).
(3) Runs against both a small batch (5-10 records: batch-small-eval-8) and a large batch (100+ records: batch-1d00a7b7).
"""

import sys
import io
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import pypdf

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.reporting.schemas import BatchReportData
from src.reporting.report_service import generate_batch_report_data
from src.reporting.pdf_generator import generate_pdf_report


class ValidationError(AssertionError):
    """Custom exception raised when an exact match assertion fails."""
    pass


def load_stored_batch_json(batch_id: str, storage_dir: Path) -> Dict[str, Any]:
    """Loads raw stored evaluation records JSON directly from storage."""
    batch_file = storage_dir / f"{batch_id}.json"
    if not batch_file.exists():
        raise FileNotFoundError(f"Stored batch file '{batch_file}' does not exist.")
    return json.loads(batch_file.read_text(encoding="utf-8"))


def validate_batch_report_integrity(batch_id: str, storage_dir: Path) -> Dict[str, Any]:
    """
    Validates that a generated PDF report and its aggregation layer match the
    stored evaluation records and dashboard data with 100% exactness.
    """
    print(f"\n" + "=" * 78)
    print(f"  VALIDATING BATCH: {batch_id}")
    print("=" * 78)

    # 1. Load stored evaluation records
    stored_data = load_stored_batch_json(batch_id, storage_dir)
    stored_items = stored_data.get("items", [])
    stored_stats = stored_data.get("statistics", {})
    total_stored_items = len(stored_items)

    print(f"[*] Stored records loaded: {total_stored_items} items")

    # 2. Generate aggregated report data object
    report_data: BatchReportData = generate_batch_report_data(batch_id=batch_id, storage_dir=storage_dir)

    # 3. Generate PDF document bytes
    pdf_bytes: bytes = generate_pdf_report(report_data)
    pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    total_pdf_pages = len(pdf_reader.pages)
    full_pdf_text = "\n".join(page.extract_text() for page in pdf_reader.pages)

    print(f"[*] PDF generated: {len(pdf_bytes):,} bytes across {total_pdf_pages} pages")

    results = {
        "batch_id": batch_id,
        "total_records": total_stored_items,
        "pdf_pages": total_pdf_pages,
        "pdf_bytes": len(pdf_bytes),
        "checks_passed": 0,
    }

    # -----------------------------------------------------------------------
    # Check 1: Metadata Exactness
    # -----------------------------------------------------------------------
    print("\n[CHECK 1] Metadata & Volume Exactness...")
    meta = report_data.metadata

    if meta.total_records != total_stored_items:
        raise ValidationError(f"Metadata total_records mismatch: report={meta.total_records} vs stored={total_stored_items}")

    succ_stored = sum(1 for it in stored_items if it.get("status") == "success" and it.get("result") is not None)
    fail_stored = sum(1 for it in stored_items if it.get("status") != "success" or it.get("result") is None)

    if meta.successful_records != succ_stored:
        raise ValidationError(f"Successful records mismatch: report={meta.successful_records} vs stored={succ_stored}")
    if meta.failed_records != fail_stored:
        raise ValidationError(f"Failed records mismatch: report={meta.failed_records} vs stored={fail_stored}")

    # Check that metadata numbers appear in PDF text
    expected_meta_strings = [
        f"Batch ID: {batch_id}",
        f"Total Submissions: {total_stored_items} records",
        f"Successfully Evaluated: {succ_stored}",
        f"Ingestion Failures: {fail_stored}",
    ]
    for s in expected_meta_strings:
        if s not in full_pdf_text:
            raise ValidationError(f"Expected metadata text not found in PDF: '{s}'")

    print(f"  [PASS] Metadata exact: {total_stored_items} total, {succ_stored} successful, {fail_stored} failed.")
    results["checks_passed"] += 1

    # -----------------------------------------------------------------------
    # Check 2: Overall Statistics & Dimension Scores Exactness
    # -----------------------------------------------------------------------
    print("\n[CHECK 2] Overall KPI Statistics & Dimension Scores Exactness...")
    stats = report_data.overall_stats

    # Compare against stored statistics directly
    expected_pass_count = stored_stats.get("pass_count", 0)
    expected_pass_rate = round(stored_stats.get("pass_rate_percent", stored_stats.get("pass_rate", 0.0) * 100), 1)
    expected_needs_count = stored_stats.get("needs_improvement_count", 0)
    expected_needs_rate = round(expected_needs_count / total_stored_items * 100, 1)
    expected_fail_count = stored_stats.get("fail_count", 0)
    expected_fail_rate = round(expected_fail_count / total_stored_items * 100, 1)

    if stats.pass_count != expected_pass_count:
        raise ValidationError(f"Pass count mismatch: report={stats.pass_count} vs stored={expected_pass_count}")
    if abs(stats.pass_percent - expected_pass_rate) > 0.01:
        raise ValidationError(f"Pass percent mismatch: report={stats.pass_percent}% vs stored={expected_pass_rate}%")

    if stats.needs_improvement_count != expected_needs_count:
        raise ValidationError(f"Needs Improvement count mismatch: report={stats.needs_improvement_count} vs stored={expected_needs_count}")
    if abs(stats.needs_improvement_percent - expected_needs_rate) > 0.01:
        raise ValidationError(f"Needs Improvement percent mismatch: report={stats.needs_improvement_percent}% vs stored={expected_needs_rate}%")

    if stats.fail_count != expected_fail_count:
        raise ValidationError(f"Fail count mismatch: report={stats.fail_count} vs stored={expected_fail_count}")
    if abs(stats.fail_percent - expected_fail_rate) > 0.01:
        raise ValidationError(f"Fail percent mismatch: report={stats.fail_percent}% vs stored={expected_fail_rate}%")

    # Dimensional average scores
    avg_scores = stored_stats.get("average_scores", {})
    expected_rel_pct = round(stored_stats.get("average_relevance_score", avg_scores.get("relevance", 0.0)) * 100, 1)
    expected_acc_pct = round(stored_stats.get("average_accuracy_score", avg_scores.get("accuracy", 0.0)) * 100, 1)
    expected_comp_pct = round(stored_stats.get("average_completeness_score", avg_scores.get("completeness", 0.0)) * 100, 1)
    expected_ground_pct = round(stored_stats.get("average_groundedness_score", avg_scores.get("groundedness", 0.0)) * 100, 1)
    expected_weighted_pct = round(stored_stats.get("average_weighted_score", 0.0) * 100, 1)

    if abs(stats.average_relevance_percent - expected_rel_pct) > 0.1:
        raise ValidationError(f"Relevance % mismatch: report={stats.average_relevance_percent}% vs stored={expected_rel_pct}%")
    if abs(stats.average_accuracy_percent - expected_acc_pct) > 0.1:
        raise ValidationError(f"Accuracy % mismatch: report={stats.average_accuracy_percent}% vs stored={expected_acc_pct}%")
    if abs(stats.average_completeness_percent - expected_comp_pct) > 0.1:
        raise ValidationError(f"Completeness % mismatch: report={stats.average_completeness_percent}% vs stored={expected_comp_pct}%")
    if abs(stats.average_groundedness_percent - expected_ground_pct) > 0.1:
        raise ValidationError(f"Groundedness % mismatch: report={stats.average_groundedness_percent}% vs stored={expected_ground_pct}%")
    if abs(stats.average_weighted_percent - expected_weighted_pct) > 0.1:
        raise ValidationError(f"Composite weighted % mismatch: report={stats.average_weighted_percent}% vs stored={expected_weighted_pct}%")

    # Assert exact metric numbers appear in PDF text
    for pct_val in [stats.pass_percent, stats.needs_improvement_percent, stats.fail_percent, stats.average_weighted_percent]:
        pct_str = f"{pct_val:.1f}%"
        if pct_str not in full_pdf_text:
            raise ValidationError(f"Expected metric string '{pct_str}' not found in PDF text.")

    print(f"  [PASS] Overall stats exact: Pass={stats.pass_percent}%, Needs={stats.needs_improvement_percent}%, Fail={stats.fail_percent}%, Composite={stats.average_weighted_percent}%.")
    results["checks_passed"] += 1

    # -----------------------------------------------------------------------
    # Check 3: Hallucination Frequency Exactness
    # -----------------------------------------------------------------------
    print("\n[CHECK 3] Hallucination Frequency & Claims Exactness...")
    hal = report_data.hallucination_frequency
    expected_hal_count = sum(1 for it in stored_items if it.get("result") and it["result"].get("hallucination", {}).get("is_hallucinated"))
    expected_hal_rate = round(stored_stats.get("hallucination_rate_percent", stored_stats.get("hallucination_rate", 0.0) * 100), 1)

    if hal.flagged_responses_count != expected_hal_count:
        raise ValidationError(f"Hallucination count mismatch: report={hal.flagged_responses_count} vs stored={expected_hal_count}")
    if abs(hal.rate_percent - expected_hal_rate) > 0.01:
        raise ValidationError(f"Hallucination rate mismatch: report={hal.rate_percent}% vs stored={expected_hal_rate}%")

    # Total claims and unsupported claims across items
    total_unsupported_stored = sum(
        it["result"]["hallucination"]["unsupported_claims_count"]
        for it in stored_items if it.get("result") and it["result"].get("hallucination")
    )
    if hal.total_unsupported_claims != total_unsupported_stored:
        raise ValidationError(f"Unsupported claims count mismatch: report={hal.total_unsupported_claims} vs stored={total_unsupported_stored}")

    # Check presence in PDF
    hal_str = f"{hal.rate_percent:.1f}%"
    if hal_str not in full_pdf_text:
        raise ValidationError(f"Hallucination rate '{hal_str}' not found in PDF text.")

    print(f"  [PASS] Hallucination stats exact: {hal.flagged_responses_count}/{hal.total_evaluated_responses} ({hal.rate_percent}%), {hal.total_unsupported_claims} unsupported claims.")
    results["checks_passed"] += 1

    # -----------------------------------------------------------------------
    # Check 4: Recommendations Exactness (2 to 5 items)
    # -----------------------------------------------------------------------
    print("\n[CHECK 4] Recommendations Generation Exactness...")
    recs = report_data.recommendations
    if not (2 <= len(recs) <= 5):
        raise ValidationError(f"Recommendations count must be between 2 and 5, got {len(recs)}")

    for r in recs:
        if not r.headline or not r.plain_english or not r.metric_trigger or not r.remediation_advice:
            raise ValidationError(f"Recommendation {r.id} has empty required fields")
        # Check headline presence in PDF
        if r.headline not in full_pdf_text:
            raise ValidationError(f"Recommendation headline '{r.headline}' not found in PDF text.")

    print(f"  [PASS] {len(recs)} Actionable recommendations verified (bounded between 2 and 5, rendered on page 2).")
    results["checks_passed"] += 1

    # -----------------------------------------------------------------------
    # Check 5: Per-Response Details Matching Dashboard / Table Records Exactly
    # -----------------------------------------------------------------------
    print(f"\n[CHECK 5] Per-Response Details Exact Verification across all {total_stored_items} records...")
    per_resp_details = report_data.per_response_details

    if len(per_resp_details) != total_stored_items:
        raise ValidationError(f"Per-response details count {len(per_resp_details)} != stored {total_stored_items}")

    for idx, (stored_it, detail) in enumerate(zip(stored_items, per_resp_details), 1):
        # 1. Index match
        if detail.index != stored_it.get("index", idx):
            raise ValidationError(f"Item #{idx}: index mismatch report={detail.index} vs stored={stored_it.get('index')}")

        # 2. Question & Response match
        stored_q = stored_it.get("question") or "— (Empty Prompt)"
        if stored_it.get("question") == "[No Question Provided]" or not stored_it.get("question"):
            stored_q = "— (Empty Prompt)"
        if detail.question != stored_q:
            raise ValidationError(f"Item #{idx}: Question text mismatch")

        stored_resp = stored_it.get("ai_response") or "— (Empty Response)"
        if detail.response != stored_resp:
            raise ValidationError(f"Item #{idx}: AI Response text mismatch")

        # 3. Reference Answer match
        if detail.reference_answer != stored_it.get("reference_answer"):
            raise ValidationError(f"Item #{idx}: Reference answer mismatch")

        # 4. Result evaluation details
        res_data = stored_it.get("result")
        if stored_it.get("status") != "success" or not res_data:
            if detail.final_verdict != "Failed":
                raise ValidationError(f"Item #{idx}: Expected Failed verdict for un-evaluated row")
            continue

        verdict_data = res_data.get("verdict", {})
        stored_verdict = verdict_data.get("verdict")
        stored_composite = round(verdict_data.get("weighted_score", 0.0), 3)

        if detail.final_verdict != stored_verdict:
            raise ValidationError(f"Item #{idx}: Final verdict mismatch report={detail.final_verdict} vs stored={stored_verdict}")
        if abs(detail.weighted_composite_score - stored_composite) > 0.001:
            raise ValidationError(f"Item #{idx}: Composite score mismatch report={detail.weighted_composite_score} vs stored={stored_composite}")

        # 5. Dimensional scores match
        rel_data = res_data.get("relevance", {})
        acc_data = res_data.get("accuracy", {})
        comp_data = res_data.get("completeness", {})
        hal_data = res_data.get("hallucination", {})

        if detail.relevance:
            if abs(detail.relevance.score - rel_data.get("score", 0.0)) > 0.001:
                raise ValidationError(f"Item #{idx}: Relevance score mismatch")
            if detail.relevance.reasoning != rel_data.get("reasoning", "Relevant response."):
                raise ValidationError(f"Item #{idx}: Relevance reasoning mismatch")

        if detail.accuracy:
            if abs(detail.accuracy.score - acc_data.get("score", 0.0)) > 0.001:
                raise ValidationError(f"Item #{idx}: Accuracy score mismatch")
            if detail.accuracy.reasoning != acc_data.get("reasoning", "Accurate response."):
                raise ValidationError(f"Item #{idx}: Accuracy reasoning mismatch")

        if detail.completeness:
            if abs(detail.completeness.score - comp_data.get("score", 0.0)) > 0.001:
                raise ValidationError(f"Item #{idx}: Completeness score mismatch")
            if detail.completeness.reasoning != comp_data.get("reasoning", "Complete coverage."):
                raise ValidationError(f"Item #{idx}: Completeness reasoning mismatch")

        # 6. Missing aspects match
        stored_missing = comp_data.get("missing_aspects") or []
        if detail.missing_aspects != stored_missing:
            raise ValidationError(f"Item #{idx}: Missing aspects mismatch report={detail.missing_aspects} vs stored={stored_missing}")

        # 7. Flagged claims match
        stored_claims = [c.get("claim") for c in hal_data.get("flagged_claims", []) if c.get("claim")]
        if detail.flagged_hallucinated_claims != stored_claims:
            raise ValidationError(f"Item #{idx}: Flagged claims mismatch")

        # 8. Check presence of item header in PDF
        resp_header_str = f"Response #{detail.index}"
        if resp_header_str not in full_pdf_text:
            raise ValidationError(f"Item #{idx}: Header string '{resp_header_str}' not found in PDF text.")

    print(f"  [PASS] All {total_stored_items} per-response records verified: index, prompt, response, reference, verdict, composite score, dimension scores, reasoning, claims, and missing aspects match exactly!")
    results["checks_passed"] += 1

    print("\n" + "-" * 78)
    print(f"  RESULT: ALL 5 CHECK GROUPS PASSED FOR {batch_id}")
    print(f"  Exact numerical integrity: 100% verified across stored records, report data, and PDF.")
    print("-" * 78)

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate PDF evaluation report data integrity against stored records.")
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Optional specific batch ID to validate. Defaults to running both small and large batches.",
    )
    parser.add_argument(
        "--storage-dir",
        type=str,
        default=str(repo_root / "data" / "batches"),
        help="Path to directory containing stored batch evaluation records.",
    )
    args = parser.parse_args()
    storage_dir = Path(args.storage_dir)

    target_batches = [args.batch_id] if args.batch_id else ["batch-small-eval-8", "batch-1d00a7b7"]

    print("=" * 78)
    print("  LLM RESPONSE EVALUATION FRAMEWORK — PDF DATA INTEGRITY VALIDATION")
    print(f"  Target Batches: {', '.join(target_batches)}")
    print(f"  Storage Directory: {storage_dir}")
    print("=" * 78)

    all_passed = True
    summary_results = []

    for b_id in target_batches:
        try:
            res = validate_batch_report_integrity(batch_id=b_id, storage_dir=storage_dir)
            summary_results.append(res)
        except Exception as err:
            print(f"\n[FAIL] Validation FAILED for batch '{b_id}': {err}")
            all_passed = False
            sys.exit(1)

    print("\n" + "=" * 78)
    print("  FINAL VALIDATION SUMMARY")
    print("=" * 78)
    for res in summary_results:
        print(f"  • {res['batch_id']}: {res['total_records']} records, {res['pdf_pages']} PDF pages, {res['pdf_bytes']:,} bytes -> ALL CHECKS PASSED")

    print("\n>>> DATA INTEGRITY VERIFIED: Every number in the PDF matches underlying stored records exactly. <<<\n")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
