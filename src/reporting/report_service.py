"""
src/reporting/report_service.py
-------------------------------
Reusable data aggregation service that feeds structured data to PDF reports.
Pulls only from stored evaluation records (disk storage or in-memory batch store),
producing a comprehensive, validated BatchReportData object per batch.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from src.agents.schemas import (
    BatchEvaluationItem,
    BatchEvaluationSummary,
    BatchStatistics,
    EvaluationResult,
)
from src.reporting.schemas import (
    BatchReportData,
    ReportMetadata,
    OverallStats,
    HallucinationFrequency,
    DimensionReportDetail,
    FlaggedClaimItem,
    PerResponseReportDetail,
    RecommendationItem,
)
from src.reporting.recommendations import generate_recommendations

logger = logging.getLogger(__name__)

DEFAULT_BATCH_STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "batches"


class ReportDataAggregator:
    """
    Service responsible for loading stored evaluation records and transforming them
    into a structured BatchReportData object suitable for PDF report generation.
    """

    def __init__(self, storage_dir: Optional[Union[str, Path]] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else DEFAULT_BATCH_STORAGE_DIR

    def list_stored_batch_ids(self) -> List[str]:
        """Returns a list of all batch IDs stored on disk."""
        if not self.storage_dir.exists():
            return []
        batch_ids = []
        for p in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                batch_ids.append(data.get("batch_id", p.stem))
            except Exception as e:
                logger.warning(f"Error reading batch file {p.name}: {e}")
        return batch_ids

    def load_stored_batch_summary(self, batch_id: str) -> BatchEvaluationSummary:
        """
        Loads a BatchEvaluationSummary from disk or active in-memory store.
        If batch_id is 'all-batches-combined' or 'combined', aggregates across all stored batches.
        """
        if batch_id in ("all-batches-combined", "combined"):
            return self._load_all_batches_combined()

        # 1. Try disk file
        batch_file = self.storage_dir / f"{batch_id}.json"
        if batch_file.exists():
            try:
                raw_data = json.loads(batch_file.read_text(encoding="utf-8"))
                return BatchEvaluationSummary(**raw_data)
            except Exception as e:
                logger.error(f"Error parsing batch file {batch_file}: {e}")
                raise ValueError(f"Corrupted or invalid batch file '{batch_file.name}': {e}")

        # 2. Try in-memory BATCH_STORE from input_module.main if accessible
        try:
            from src.input_module.main import BATCH_STORE
            if batch_id in BATCH_STORE:
                return BATCH_STORE[batch_id]
        except ImportError:
            pass

        # 3. Check for matching batch_id inside any json file in storage_dir
        if self.storage_dir.exists():
            for p in self.storage_dir.glob("*.json"):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if data.get("batch_id") == batch_id:
                        return BatchEvaluationSummary(**data)
                except Exception:
                    continue

        raise FileNotFoundError(f"Stored evaluation batch '{batch_id}' was not found in {self.storage_dir}.")

    def _load_all_batches_combined(self) -> BatchEvaluationSummary:
        """Loads and consolidates all persisted batch files into one combined summary."""
        all_items: List[BatchEvaluationItem] = []
        global_idx = 1
        latest_time = datetime.now(timezone.utc).isoformat()
        total_batches = 0

        if self.storage_dir.exists():
            json_paths = sorted(self.storage_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
            for jpath in json_paths:
                try:
                    raw = json.loads(jpath.read_text(encoding="utf-8"))
                    b_id = raw.get("batch_id", jpath.stem)
                    b_fn = raw.get("filename") or "batch.csv"
                    if raw.get("created_at"):
                        latest_time = raw.get("created_at")
                    for it_raw in raw.get("items", []):
                        try:
                            it = BatchEvaluationItem(**it_raw)
                            it.batch_id = b_id
                            it.batch_filename = b_fn
                            it.index = global_idx
                            global_idx += 1
                            all_items.append(it)
                        except Exception:
                            continue
                    total_batches += 1
                except Exception as e:
                    logger.warning(f"Error loading {jpath.name} for combined report: {e}")

        # Calculate statistics directly from items
        from src.agents.batch_evaluator import BatchEvaluator
        evaluator = BatchEvaluator()
        stats = evaluator.calculate_statistics(all_items)

        return BatchEvaluationSummary(
            batch_id="all-batches-combined",
            filename=f"All Batches Combined ({total_batches} batches)",
            created_at=latest_time,
            statistics=stats,
            items=all_items,
        )

    def build_report_data(self, summary: BatchEvaluationSummary) -> BatchReportData:
        """
        Pure aggregation function: transforms a BatchEvaluationSummary into a validated
        BatchReportData object containing metadata, overall stats, hallucination frequency,
        and per-response details list.
        """
        items = summary.items
        total_records = len(items)
        successful_items = [it for it in items if it.status == "success" and it.result is not None]
        failed_items = [it for it in items if it.status != "success" or it.result is None]

        succ_count = len(successful_items)
        failed_count = len(failed_items)

        # 1. Metadata
        metadata = ReportMetadata(
            batch_id=summary.batch_id,
            filename=summary.filename or f"{summary.batch_id}.csv",
            timestamp=summary.created_at,
            report_generated_at=datetime.now(timezone.utc).isoformat(),
            total_records=total_records,
            successful_records=succ_count,
            failed_records=failed_count,
        )

        # 2. Overall Stats
        if succ_count == 0:
            overall_stats = OverallStats(
                pass_count=0,
                pass_percent=0.0,
                needs_improvement_count=0,
                needs_improvement_percent=0.0,
                fail_count=0,
                fail_percent=0.0,
                average_relevance_score=0.0,
                average_relevance_percent=0.0,
                average_accuracy_score=0.0,
                average_accuracy_percent=0.0,
                average_completeness_score=0.0,
                average_completeness_percent=0.0,
                average_groundedness_score=0.0,
                average_groundedness_percent=0.0,
                average_weighted_score=0.0,
                average_weighted_percent=0.0,
            )
            hallucination_freq = HallucinationFrequency(
                flagged_responses_count=0,
                total_evaluated_responses=0,
                rate_percent=0.0,
                total_claims_extracted=0,
                total_unsupported_claims=0,
                sample_flagged_claims=[],
            )
        else:
            pass_count = sum(1 for it in successful_items if it.result.verdict.verdict == "Pass")
            needs_count = sum(1 for it in successful_items if it.result.verdict.verdict == "Needs Improvement")
            fail_count = sum(1 for it in successful_items if it.result.verdict.verdict == "Fail")

            pass_pct = round((pass_count / total_records * 100), 1)
            needs_pct = round((needs_count / total_records * 100), 1)
            fail_pct = round((fail_count / total_records * 100), 1)

            avg_rel = round(sum(it.result.relevance.score for it in successful_items) / succ_count, 3)
            avg_acc = round(sum(it.result.accuracy.score for it in successful_items) / succ_count, 3)
            avg_comp = round(sum(it.result.completeness.score for it in successful_items) / succ_count, 3)

            # Groundedness: 1.0 - hallucination_score (or from dimension_scores)
            def _get_groundedness(it: BatchEvaluationItem) -> float:
                if it.result.verdict.dimension_scores and "groundedness" in it.result.verdict.dimension_scores:
                    return float(it.result.verdict.dimension_scores["groundedness"])
                return max(0.0, 1.0 - it.result.hallucination.hallucination_score)

            avg_ground = round(sum(_get_groundedness(it) for it in successful_items) / succ_count, 3)
            avg_weighted = round(sum(it.result.verdict.weighted_score for it in successful_items) / succ_count, 3)

            overall_stats = OverallStats(
                pass_count=pass_count,
                pass_percent=pass_pct,
                needs_improvement_count=needs_count,
                needs_improvement_percent=needs_pct,
                fail_count=fail_count,
                fail_percent=fail_pct,
                average_relevance_score=avg_rel,
                average_relevance_percent=round(avg_rel * 100, 1),
                average_accuracy_score=avg_acc,
                average_accuracy_percent=round(avg_acc * 100, 1),
                average_completeness_score=avg_comp,
                average_completeness_percent=round(avg_comp * 100, 1),
                average_groundedness_score=avg_ground,
                average_groundedness_percent=round(avg_ground * 100, 1),
                average_weighted_score=avg_weighted,
                average_weighted_percent=round(avg_weighted * 100, 1),
            )

            # 3. Hallucination Frequency
            flagged_resp_count = sum(1 for it in successful_items if it.result.hallucination.is_hallucinated)
            hal_rate_pct = round((flagged_resp_count / succ_count * 100), 1)
            total_claims = sum(it.result.hallucination.total_claims for it in successful_items)
            total_unsupported = sum(it.result.hallucination.unsupported_claims_count for it in successful_items)

            # Extract distinct sample of flagged claims
            sample_claims: List[str] = []
            for it in successful_items:
                for c in it.result.hallucination.flagged_claims:
                    if c.claim and c.claim not in sample_claims:
                        sample_claims.append(c.claim)
                        if len(sample_claims) >= 10:
                            break
                if len(sample_claims) >= 10:
                    break

            hallucination_freq = HallucinationFrequency(
                flagged_responses_count=flagged_resp_count,
                total_evaluated_responses=succ_count,
                rate_percent=hal_rate_pct,
                total_claims_extracted=total_claims,
                total_unsupported_claims=total_unsupported,
                sample_flagged_claims=sample_claims,
            )

        # 4. Per-Response Detail List
        details: List[PerResponseReportDetail] = []
        for item in items:
            q_text = item.question if item.question and item.question != "[No Question Provided]" else "— (Empty Prompt)"
            a_text = item.ai_response if item.ai_response else "— (Empty Response)"

            if item.status != "success" or not item.result:
                err_msg = item.error_message or "Evaluation failed or was skipped during pipeline processing"
                details.append(
                    PerResponseReportDetail(
                        index=item.index,
                        question=q_text,
                        response=a_text,
                        reference_answer=item.reference_answer,
                        source_document=item.source_document,
                        status=item.status,
                        error_message=err_msg,
                        relevance=None,
                        accuracy=None,
                        completeness=None,
                        hallucination=None,
                        flagged_hallucinated_claims=[],
                        flagged_claims_breakdown=[],
                        missing_aspects=[],
                        weighted_composite_score=0.0,
                        final_verdict="Failed",
                        verdict_explanation=err_msg,
                    )
                )
                continue

            res: EvaluationResult = item.result

            # Relevance Detail
            rel_detail = DimensionReportDetail(
                score=round(res.relevance.score, 3),
                score_percent=round(res.relevance.score * 100, 1),
                classification=getattr(res.relevance, "classification", None),
                reasoning=res.relevance.reasoning or "Relevant response.",
                evidence=None,
            )

            # Accuracy Detail
            acc_detail = DimensionReportDetail(
                score=round(res.accuracy.score, 3),
                score_percent=round(res.accuracy.score * 100, 1),
                classification=getattr(res.accuracy, "classification", None),
                reasoning=res.accuracy.reasoning or "Accurate response.",
                evidence=res.accuracy.supporting_evidence if res.accuracy.supporting_evidence else None,
            )

            # Completeness Detail
            comp_detail = DimensionReportDetail(
                score=round(res.completeness.score, 3),
                score_percent=round(res.completeness.score * 100, 1),
                classification=getattr(res.completeness, "classification", None),
                reasoning=res.completeness.reasoning or "Complete coverage.",
                evidence=res.completeness.addressed_aspects if res.completeness.addressed_aspects else None,
            )

            # Hallucination / Groundedness Detail
            hal = res.hallucination
            grounded_score = round(max(0.0, 1.0 - hal.hallucination_score), 3)
            hal_evidence = [c.explanation for c in hal.flagged_claims if c.explanation]
            hal_detail = DimensionReportDetail(
                score=grounded_score,
                score_percent=round(grounded_score * 100, 1),
                classification="grounded" if not hal.is_hallucinated else "hallucinated",
                reasoning=hal.reasoning or ("Fully grounded in context." if not hal.is_hallucinated else "Contains ungrounded claims."),
                evidence=hal_evidence if hal_evidence else None,
            )

            # Flagged claims
            flagged_claims_list = [c.claim for c in hal.flagged_claims if c.claim]
            flagged_breakdown = [
                FlaggedClaimItem(
                    claim=c.claim,
                    status=c.status,
                    evidence=c.evidence,
                    explanation=c.explanation,
                )
                for c in hal.flagged_claims
            ]

            # Missing aspects
            missing_aspects = res.completeness.missing_aspects if res.completeness.missing_aspects else []

            # Composite Score and Verdict
            comp_score = round(res.verdict.weighted_score, 3)
            final_v = res.verdict.verdict
            v_explanation = res.verdict.consolidated_reasoning or f"Final Verdict: {final_v} (Composite Score: {round(comp_score * 100, 1)}%)"

            details.append(
                PerResponseReportDetail(
                    index=item.index,
                    question=q_text,
                    response=a_text,
                    reference_answer=item.reference_answer,
                    source_document=item.source_document,
                    status="success",
                    error_message=None,
                    relevance=rel_detail,
                    accuracy=acc_detail,
                    completeness=comp_detail,
                    hallucination=hal_detail,
                    flagged_hallucinated_claims=flagged_claims_list,
                    flagged_claims_breakdown=flagged_breakdown,
                    missing_aspects=missing_aspects,
                    weighted_composite_score=comp_score,
                    final_verdict=final_v,
                    verdict_explanation=v_explanation,
                )
            )

        report_obj = BatchReportData(
            metadata=metadata,
            overall_stats=overall_stats,
            hallucination_frequency=hallucination_freq,
            per_response_details=details,
        )
        report_obj.recommendations = generate_recommendations(report_obj)
        return report_obj

    def get_report_data_by_batch_id(self, batch_id: str) -> BatchReportData:
        """
        High-level service API: loads stored records for batch_id and aggregates the report-data object.
        """
        summary = self.load_stored_batch_summary(batch_id)
        return self.build_report_data(summary)


# ---------------------------------------------------------------------------
# Reusable helper functions
# ---------------------------------------------------------------------------

def generate_batch_report_data(
    batch_id: Optional[str] = None,
    summary: Optional[BatchEvaluationSummary] = None,
    storage_dir: Optional[Union[str, Path]] = None,
) -> BatchReportData:
    """
    Reusable functional interface to generate the report-data object.
    Accepts either an explicit batch_id (to pull from storage) or a pre-loaded BatchEvaluationSummary.
    """
    aggregator = ReportDataAggregator(storage_dir=storage_dir)
    if summary is not None:
        return aggregator.build_report_data(summary)
    if batch_id is not None:
        return aggregator.get_report_data_by_batch_id(batch_id)
    raise ValueError("Either 'batch_id' or 'summary' must be supplied to generate_batch_report_data.")
