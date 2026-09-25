"""
M3.4: Batch Evaluation Module.
Enables parsing, validating, and evaluating multiple question-answer pairs (including 100+ records)
from CSV files, with graceful empty question handling, concurrency controls, and aggregated statistics.
"""

import io
import csv
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from .schemas import (
    BatchEvaluationItem,
    BatchEvaluationSummary,
    BatchStatistics,
    EvaluationResult,
)
from .orchestrator import EvaluationOrchestrator

logger = logging.getLogger(__name__)


class BatchEvaluator:
    """
    Handles CSV ingestion, data validation, concurrent orchestrator execution,
    and batch statistics calculation for high-volume evaluation workloads.
    """
    def __init__(
        self,
        orchestrator: Optional[EvaluationOrchestrator] = None,
        concurrency_limit: int = 5,
    ):
        self.orchestrator = orchestrator or EvaluationOrchestrator()
        self.concurrency_limit = concurrency_limit

    @staticmethod
    def _normalize_header(header: str) -> str:
        """Standardizes column headers by stripping BOM, whitespace, symbols, and lowercasing."""
        return header.lower().strip().replace("-", "_").replace(" ", "_").lstrip("\ufeff")

    def parse_and_validate_csv(
        self,
        csv_content: str,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Parses CSV content, matches flexible column names, and validates required columns.
        Returns a list of parsed record dicts and a list of non-fatal warnings/errors.
        """
        if not csv_content or not csv_content.strip():
            raise ValueError("Uploaded CSV file is empty.")

        # Detect delimiter automatically (comma, tab, semicolon)
        sample = csv_content[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        except Exception:
            dialect = csv.excel

        reader = csv.reader(io.StringIO(csv_content), dialect=dialect)
        try:
            raw_headers = next(reader)
        except StopIteration:
            raise ValueError("CSV file contains no rows.")

        if not raw_headers:
            raise ValueError("CSV file is missing a header row.")

        norm_headers = [self._normalize_header(h) for h in raw_headers]

        # Flexible column mapping dictionaries
        QUESTION_VARIANTS = {"question", "questions", "query", "prompt", "q", "user_question", "input"}
        RESPONSE_VARIANTS = {
            "ai_response", "ai_responses", "response", "answer", "a", "ai_answer", "output",
            "model_response", "completion", "generated_response", "llm_response", "model_answer"
        }
        REFERENCE_VARIANTS = {
            "reference_answer", "reference", "ref_answer", "ground_truth", "expected_answer",
            "correct_answer", "target", "reference_response", "ideal_answer"
        }
        SOURCE_VARIANTS = {
            "source_document", "source", "context", "document", "doc", "retrieved_context", "sources"
        }

        col_q = next((i for i, h in enumerate(norm_headers) if h in QUESTION_VARIANTS), None)
        col_r = next((i for i, h in enumerate(norm_headers) if h in RESPONSE_VARIANTS), None)
        col_ref = next((i for i, h in enumerate(norm_headers) if h in REFERENCE_VARIANTS), None)
        col_src = next((i for i, h in enumerate(norm_headers) if h in SOURCE_VARIANTS), None)

        missing_cols = []
        if col_q is None:
            missing_cols.append("'questions'")
        if col_r is None:
            missing_cols.append("'ai_responses'")

        if missing_cols:
            raise ValueError(
                f"Missing required column(s): {' and '.join(missing_cols)}. "
                f"The CSV must contain both 'questions' and 'ai_responses' columns. "
                f"Detected columns: {raw_headers}"
            )

        parsed_rows = []
        warnings = []
        row_idx = 0

        for row in reader:
            if not row or not any(field.strip() for field in row):
                continue  # skip completely blank rows
            row_idx += 1

            # Extract fields safely
            q_val = row[col_q].strip() if (col_q is not None and col_q < len(row)) else ""
            r_val = row[col_r].strip() if (col_r < len(row)) else ""
            ref_val = row[col_ref].strip() if (col_ref is not None and col_ref < len(row)) else ""
            src_val = row[col_src].strip() if (col_src is not None and col_src < len(row)) else ""

            # Extra note requirement: Question may be empty
            if not q_val:
                warnings.append(f"Row {row_idx}: Question is empty. Evaluated as zero-question response.")
                q_val = "[No Question Provided]"

            parsed_rows.append({
                "index": row_idx,
                "question": q_val,
                "ai_response": r_val,
                "reference_answer": ref_val or None,
                "source_document": src_val or None,
            })

        if not parsed_rows:
            raise ValueError("CSV contains no valid data rows after the header.")

        return parsed_rows, warnings

    async def evaluate_batch(
        self,
        records: List[Dict[str, Any]],
        filename: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchEvaluationSummary:
        """
        Executes evaluation for all items in the batch using an async worker semaphore pool.
        """
        batch_id = f"batch-{uuid.uuid4().hex[:8]}"
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        total_records = len(records)
        completed_count = 0
        lock = asyncio.Lock()

        evaluated_items: List[BatchEvaluationItem] = []

        async def _eval_single(record: Dict[str, Any]) -> BatchEvaluationItem:
            nonlocal completed_count
            idx = record["index"]
            q = record["question"]
            resp = record["ai_response"]
            ref = record.get("reference_answer")
            src = record.get("source_document")

            if not resp or not resp.strip():
                async with lock:
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count, total_records)
                return BatchEvaluationItem(
                    index=idx,
                    question=q,
                    ai_response="",
                    reference_answer=ref,
                    source_document=src,
                    result=None,
                    status="failed",
                    error_message="AI Response is empty; cannot evaluate.",
                )

            async with semaphore:
                try:
                    eval_result: EvaluationResult = await self.orchestrator.evaluate(
                        question=q,
                        ai_response=resp,
                        reference_answer=ref,
                        source_document=src,
                        submission_id=idx,
                    )
                    item = BatchEvaluationItem(
                        index=idx,
                        question=q,
                        ai_response=resp,
                        reference_answer=ref,
                        source_document=src,
                        result=eval_result,
                        status="success",
                    )
                except Exception as exc:
                    logger.error(f"Error evaluating batch row {idx}: {exc}")
                    item = BatchEvaluationItem(
                        index=idx,
                        question=q,
                        ai_response=resp,
                        reference_answer=ref,
                        source_document=src,
                        result=None,
                        status="failed",
                        error_message=str(exc),
                    )

                async with lock:
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count, total_records)

                return item

        # Run all records concurrently up to concurrency limit
        tasks = [_eval_single(rec) for rec in records]
        evaluated_items = await asyncio.gather(*tasks)

        # Sort items by original row index
        evaluated_items.sort(key=lambda it: it.index)

        # Calculate aggregated statistics
        stats = self.calculate_statistics(evaluated_items)

        return BatchEvaluationSummary(
            batch_id=batch_id,
            filename=filename,
            created_at=datetime.now(timezone.utc).isoformat(),
            statistics=stats,
            items=evaluated_items,
        )

    @staticmethod
    def calculate_statistics(items: List[BatchEvaluationItem]) -> BatchStatistics:
        """Aggregates batch-level metrics across all evaluated records."""
        total = len(items)
        successful = [it for it in items if it.status == "success" and it.result is not None]
        failed = [it for it in items if it.status != "success" or it.result is None]

        succ_count = len(successful)
        failed_count = len(failed)

        if succ_count == 0:
            return BatchStatistics(
                total_records=total,
                successful_records=0,
                failed_records=failed_count,
                pass_count=0,
                needs_improvement_count=0,
                fail_count=0,
                pass_rate_percent=0.0,
                average_weighted_score=0.0,
                average_relevance_score=0.0,
                average_accuracy_score=0.0,
                average_completeness_score=0.0,
                average_groundedness_score=0.0,
                hallucination_rate_percent=0.0,
            )

        pass_count = sum(1 for it in successful if it.result.verdict.verdict == "Pass")
        needs_imp_count = sum(1 for it in successful if it.result.verdict.verdict == "Needs Improvement")
        fail_count = sum(1 for it in successful if it.result.verdict.verdict == "Fail")

        avg_weighted = sum(it.result.verdict.weighted_score for it in successful) / succ_count
        avg_rel = sum(it.result.relevance.score for it in successful) / succ_count
        avg_acc = sum(it.result.accuracy.score for it in successful) / succ_count
        avg_comp = sum(it.result.completeness.score for it in successful) / succ_count
        avg_ground = sum(it.result.verdict.dimension_scores.get("groundedness", 0.0) for it in successful) / succ_count

        hallucinated_cases = sum(1 for it in successful if it.result.hallucination.is_hallucinated)
        hallucination_rate = (hallucinated_cases / succ_count) * 100.0

        return BatchStatistics(
            total_records=total,
            successful_records=succ_count,
            failed_records=failed_count,
            pass_count=pass_count,
            needs_improvement_count=needs_imp_count,
            fail_count=fail_count,
            pass_rate_percent=round((pass_count / succ_count) * 100.0, 1),
            average_weighted_score=round(avg_weighted, 3),
            average_relevance_score=round(avg_rel, 3),
            average_accuracy_score=round(avg_acc, 3),
            average_completeness_score=round(avg_comp, 3),
            average_groundedness_score=round(avg_ground, 3),
            hallucination_rate_percent=round(hallucination_rate, 1),
        )

    @staticmethod
    def export_summary_to_csv(summary: BatchEvaluationSummary) -> str:
        """Generates a structured CSV report from evaluated batch items."""
        output = io.StringIO()
        writer = csv.writer(output)

        headers = [
            "Row",
            "Question",
            "AI Response",
            "Reference Answer",
            "Verdict",
            "Overall Weighted Score",
            "Relevance Score",
            "Accuracy Score",
            "Completeness Score",
            "Groundedness Score",
            "Hallucination Detected",
            "Missing Aspects",
            "Major Issues",
            "Strengths",
            "Consolidated Reasoning",
        ]
        writer.writerow(headers)

        for item in summary.items:
            res = item.result
            q_text = "—" if (not item.question or item.question in ("[No Question Provided]", "[Empty]", "Empty", "—")) else item.question
            if res is None:
                writer.writerow([
                    item.index,
                    q_text,
                    item.ai_response,
                    item.reference_answer or "",
                    "FAILED",
                    "0.0",
                    "0.0",
                    "0.0",
                    "0.0",
                    "0.0",
                    "N/A",
                    "",
                    item.error_message or "Evaluation error",
                    "",
                    item.error_message or "Failed",
                ])
                continue

            writer.writerow([
                item.index,
                q_text,
                item.ai_response,
                item.reference_answer or "",
                res.verdict.verdict,
                f"{res.verdict.weighted_score:.3f}",
                f"{res.relevance.score:.3f}",
                f"{res.accuracy.score:.3f}",
                f"{res.completeness.score:.3f}",
                f"{res.verdict.dimension_scores.get('groundedness', 0.0):.3f}",
                "YES" if res.hallucination.is_hallucinated else "NO",
                "; ".join(res.completeness.missing_aspects),
                "; ".join(res.verdict.major_issues),
                "; ".join(res.verdict.strengths),
                res.verdict.consolidated_reasoning.replace("\n", " "),
            ])

        return output.getvalue()
