"""
Evaluation Input Module (M1.3).

Single endpoint that accepts a question + AI response (plus optional
reference answer / source document), validates it, and stores it.
Run with: uvicorn src.input_module.main:app --reload
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .schemas import EvaluationSubmission, SubmissionResponse
from .storage import get_submission, insert_submission
from ..agents.orchestrator import EvaluationOrchestrator
from ..agents.schemas import (
    EvaluationResult,
    BatchEvaluationItem,
    BatchEvaluationSummary,
    BatchTrendPoint,
    TrendsSummaryResponse,
)
from ..agents.batch_evaluator import BatchEvaluator

logger = logging.getLogger(__name__)

orchestrator = EvaluationOrchestrator()
batch_evaluator = BatchEvaluator(orchestrator=orchestrator, concurrency_limit=5)

# Store for completed batches (in-memory + disk cache persistence)
BATCH_STORE: dict[str, BatchEvaluationSummary] = {}
BATCH_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "batches"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

from ..reporting.schemas import BatchReportData, RecommendationItem
from ..reporting.report_service import ReportDataAggregator, generate_batch_report_data
from ..reporting.pdf_generator import generate_pdf_report
report_aggregator = ReportDataAggregator(storage_dir=BATCH_DIR)

app = FastAPI(title="LLM Response Eval - Input Module", version="0.2.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


from fastapi.encoders import jsonable_encoder

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"error": "invalid submission", "details": jsonable_encoder(exc.errors())},
    )


@app.post("/api/v1/evaluations", response_model=SubmissionResponse, status_code=201)
def submit_evaluation(payload: EvaluationSubmission):
    row = insert_submission(
        question=payload.question,
        ai_response=payload.ai_response,
        reference_answer=payload.reference_answer,
        source_document=payload.source_document,
    )
    return SubmissionResponse(
        id=row["id"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


@app.get("/api/v1/evaluations/{submission_id}")
def read_evaluation(submission_id: int):
    row = get_submission(submission_id)
    if row is None:
        raise HTTPException(status_code=404, detail="submission not found")
    return dict(row)


@app.post("/api/v1/evaluations/{submission_id}/evaluate", response_model=EvaluationResult)
async def run_evaluation_for_submission(submission_id: int):
    row = get_submission(submission_id)
    if row is None:
        raise HTTPException(status_code=404, detail="submission not found")

    return await orchestrator.evaluate(
        question=row["question"],
        ai_response=row["ai_response"],
        reference_answer=row["reference_answer"],
        source_document=row["source_document"],
        submission_id=submission_id,
    )


@app.post("/api/v1/evaluate", response_model=EvaluationResult)
async def evaluate_direct(payload: EvaluationSubmission):
    row = insert_submission(
        question=payload.question,
        ai_response=payload.ai_response,
        reference_answer=payload.reference_answer,
        source_document=payload.source_document,
    )
    return await orchestrator.evaluate(
        question=payload.question,
        ai_response=payload.ai_response,
        reference_answer=payload.reference_answer,
        source_document=payload.source_document,
        submission_id=row["id"],
    )


# -------------------------------------------------------------
# Milestone 3.4: Batch Evaluation Endpoints
# -------------------------------------------------------------

@app.post("/api/v1/batch/upload-csv", response_model=BatchEvaluationSummary)
async def batch_upload_csv(file: UploadFile = File(...)):
    """
    Accepts CSV upload containing 100+ question-answer pairs, validates headers,
    evaluates all records through the multi-agent pipeline, and returns full results + statistics.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV format.")

    try:
        raw_bytes = await file.read()
        try:
            csv_content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            csv_content = raw_bytes.decode("latin-1")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {e}")

    try:
        records, warnings = batch_evaluator.parse_and_validate_csv(csv_content)
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=f"CSV Validation Error: {val_err}")

    summary = await batch_evaluator.evaluate_batch(records=records, filename=file.filename)
    BATCH_STORE[summary.batch_id] = summary
    try:
        batch_file = BATCH_DIR / f"{summary.batch_id}.json"
        batch_file.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        (BATCH_DIR / "latest.txt").write_text(summary.batch_id, encoding="utf-8")
    except Exception as persist_err:
        logger.warning(f"Could not persist batch to disk: {persist_err}")
    return summary


@app.get("/api/v1/batch/latest", response_model=Optional[BatchEvaluationSummary])
@app.get("/api/v1/batches/latest", response_model=Optional[BatchEvaluationSummary])
def get_latest_batch():
    """Returns the most recently evaluated batch summary."""
    latest_file = BATCH_DIR / "latest.txt"
    if latest_file.exists():
        latest_id = latest_file.read_text(encoding="utf-8").strip()
        if latest_id:
            try:
                return get_batch_results(latest_id)
            except HTTPException:
                pass
    if BATCH_STORE:
        return list(BATCH_STORE.values())[-1]
    return None


def get_all_batches_combined() -> BatchEvaluationSummary:
    """Aggregates all evaluated records across all stored batches into a unified summary."""
    batches_map: dict[str, dict] = {}

    # 1. Read persisted batches from disk
    if BATCH_DIR.exists():
        for json_path in BATCH_DIR.glob("*.json"):
            try:
                content = json_path.read_text(encoding="utf-8")
                raw = json.loads(content)
                b_id = raw.get("batch_id", json_path.stem)
                batches_map[b_id] = raw
            except Exception as e:
                logger.warning(f"Could not load batch {json_path.name} for combined view: {e}")

    # 2. Check in-memory BATCH_STORE
    for b_id, summary in BATCH_STORE.items():
        if b_id not in batches_map:
            try:
                batches_map[b_id] = summary.model_dump()
            except Exception:
                pass

    all_items: List[BatchEvaluationItem] = []
    latest_time = datetime.utcnow().isoformat()
    sorted_raw_batches = sorted(batches_map.values(), key=lambda b: b.get("created_at", ""))

    global_idx = 1
    for raw in sorted_raw_batches:
        b_id = raw.get("batch_id", "batch")
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
            except Exception as err:
                logger.warning(f"Error parsing item in batch {b_id}: {err}")

    stats = batch_evaluator.calculate_statistics(all_items)
    return BatchEvaluationSummary(
        batch_id="all-batches-combined",
        filename=f"All Batches Combined ({len(sorted_raw_batches)} batches)",
        created_at=latest_time,
        statistics=stats,
        items=all_items,
    )


@app.get("/api/v1/batches/combined", response_model=BatchEvaluationSummary)
def get_combined_batches_summary():
    """Returns aggregate summary statistics and records across all evaluations ever run."""
    return get_all_batches_combined()


@app.get("/api/v1/batches/combined/export")
def export_combined_batches_csv():
    """Exports all historical evaluation records combined as a single CSV spreadsheet."""
    combined = get_all_batches_combined()
    csv_data = batch_evaluator.export_summary_to_csv(combined)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=all_batches_combined_evaluated.csv"
        },
    )


@app.get("/api/v1/batch/{batch_id}", response_model=BatchEvaluationSummary)
def get_batch_results(batch_id: str):
    if batch_id in ("all-batches-combined", "combined"):
        return get_all_batches_combined()

    summary = BATCH_STORE.get(batch_id)
    if not summary:
        batch_file = BATCH_DIR / f"{batch_id}.json"
        if batch_file.exists():
            try:
                data = json.loads(batch_file.read_text(encoding="utf-8"))
                summary = BatchEvaluationSummary(**data)
                BATCH_STORE[batch_id] = summary
            except Exception as e:
                logger.error(f"Error restoring batch {batch_id} from disk: {e}")
    if not summary:
        raise HTTPException(status_code=404, detail=f"Batch ID '{batch_id}' not found.")
    
    # Dynamic Query-Time Computation: always recompute statistics directly from structured evaluation items
    summary.statistics = batch_evaluator.calculate_statistics(summary.items)
    return summary


@app.get("/api/v1/batch/{batch_id}/export")
def export_batch_csv(batch_id: str):
    """Exports evaluated batch records as a formatted CSV spreadsheet."""
    if batch_id in ("all-batches-combined", "combined"):
        return export_combined_batches_csv()

    summary = get_batch_results(batch_id)
    csv_data = batch_evaluator.export_summary_to_csv(summary)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={batch_id}_evaluated.csv"
        },
    )


@app.get("/api/v1/batch/{batch_id}/report-data", response_model=BatchReportData)
def get_batch_report_data_endpoint(batch_id: str):
    """
    Returns structured aggregation data feeding the PDF report for a given batch.
    Pulls exclusively from stored structured evaluation records.
    """
    try:
        return report_aggregator.get_report_data_by_batch_id(batch_id)
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err))
    except Exception as err:
        logger.error(f"Error producing report-data for batch {batch_id}: {err}")
        raise HTTPException(status_code=500, detail=f"Failed to aggregate report data: {err}")


@app.get("/api/v1/batch/{batch_id}/recommendations", response_model=List[RecommendationItem])
def get_batch_recommendations_endpoint(batch_id: str):
    """
    Scans a batch's aggregated report data and returns 2-5 actionable improvement recommendations
    based on frequency thresholds across dimension scores and flagged issues.
    """
    try:
        report = report_aggregator.get_report_data_by_batch_id(batch_id)
        return report.recommendations
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err))
    except Exception as err:
        logger.error(f"Error producing recommendations for batch {batch_id}: {err}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {err}")


@app.get("/api/v1/batch/{batch_id}/export/pdf")
def export_batch_pdf_report_endpoint(batch_id: str):
    """
    Generates and downloads a complete, professional PDF evaluation report for a given batch.
    Includes:
    (1) Cover / Executive Summary page with metadata, KPI stats, verdict pie chart, and dimension score bar chart;
    (2) Recommendations page with empirical metric triggers and actionable remediation advice;
    (3) Per-response detail sections with color-coded verdict badges, dimension breakdown, flagged hallucinations, and missing aspects.
    """
    try:
        report_data = report_aggregator.get_report_data_by_batch_id(batch_id)
        pdf_bytes = generate_pdf_report(report_data)
        filename = f"{batch_id}_evaluation_report.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Type": "application/pdf",
            },
        )
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err))
    except Exception as err:
        logger.error(f"Error generating PDF report for batch {batch_id}: {err}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF report: {err}")


@app.get("/api/v1/batches/combined/export/pdf")
def export_combined_batches_pdf_endpoint():
    """Generates and downloads a consolidated PDF report covering all historical batches combined."""
    return export_batch_pdf_report_endpoint("all-batches-combined")





@app.get("/api/v1/batch/sample/template.csv")
def download_sample_csv():
    """Provides a sample CSV template for testing batch evaluations (105 benchmark records)."""
    sample_path = Path(__file__).resolve().parent.parent.parent / "data" / "sample_batch.csv"
    if sample_path.exists():
        sample_csv = sample_path.read_text(encoding="utf-8")
    else:
        sample_csv = (
            "question,ai_response,reference_answer,source_document\n"
            "What is the capital of France?,Paris is the capital of France and its largest city.,Paris is the capital of France.,\n"
            "Explain photosynthesis.,Photosynthesis is the process where plants convert light into chemical energy.,Plants use photosynthesis to make glucose from CO2 and water using sunlight.,\n"
            "What is the boiling point of water?,Water boils at 212 degrees Fahrenheit at sea level.,The boiling point of water is 100 degrees Celsius (212 F).,\n"
            "Who founded Apple Computer?,Steve Jobs and Steve Wozniak co-founded Apple Computer in 1976.,Steve Jobs and Steve Wozniak founded Apple.,\n"
        )
    return Response(
        content=sample_csv,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_batch_100.csv"},
    )


# -------------------------------------------------------------
# Milestone 4: Longitudinal Cross-Batch Evaluation Trends
# -------------------------------------------------------------

def extract_batch_trend_point(data: dict, batch_id: str, filename: Optional[str] = None) -> Optional[BatchTrendPoint]:
    """Extracts summary KPIs and dimension metrics from a raw batch summary dictionary."""
    stats = data.get("statistics")
    if "items" in data and data["items"]:
        try:
            parsed_items = [BatchEvaluationItem(**it_raw) for it_raw in data["items"]]
            stats = batch_evaluator.calculate_statistics(parsed_items).model_dump()
        except Exception as e:
            logger.warning(f"Could not dynamically recompute statistics for trend point: {e}")
    if not stats:
        return None
    total = stats.get("total_records", 0)
    pass_cnt = stats.get("pass_count", 0)
    needs_cnt = stats.get("needs_improvement_count", 0)
    fail_cnt = stats.get("fail_count", 0)
    pass_pct = round((pass_cnt / total * 100), 1) if total > 0 else round(stats.get("pass_rate_percent", 0.0), 1)
    needs_pct = round((needs_cnt / total * 100), 1) if total > 0 else 0.0
    fail_pct = round((fail_cnt / total * 100), 1) if total > 0 else 0.0

    return BatchTrendPoint(
        batch_id=data.get("batch_id", batch_id),
        filename=data.get("filename") or filename or "batch.csv",
        created_at=data.get("created_at") or datetime.utcnow().isoformat(),
        total_records=total,
        successful_records=stats.get("successful_records", 0),
        failed_records=stats.get("failed_records", 0),
        pass_count=pass_cnt,
        needs_improvement_count=needs_cnt,
        fail_count=fail_cnt,
        pass_percent=pass_pct,
        needs_improvement_percent=needs_pct,
        fail_percent=fail_pct,
        average_weighted_score=round(stats.get("average_weighted_score", 0.0), 3),
        average_relevance_score=round(stats.get("average_relevance_score", 0.0), 3),
        average_accuracy_score=round(stats.get("average_accuracy_score", 0.0), 3),
        average_completeness_score=round(stats.get("average_completeness_score", 0.0), 3),
        average_groundedness_score=round(stats.get("average_groundedness_score", 0.0), 3),
        hallucination_rate_percent=round(stats.get("hallucination_rate_percent", 0.0), 1),
    )


def collect_historical_batches() -> list[BatchTrendPoint]:
    """Scans persisted batch JSON files on disk and active memory, returning sorted chronological batches."""
    batches_map: dict[str, BatchTrendPoint] = {}

    # 1. Read persisted batches from disk
    if BATCH_DIR.exists():
        for json_path in BATCH_DIR.glob("*.json"):
            try:
                content = json_path.read_text(encoding="utf-8")
                raw = json.loads(content)
                b_id = raw.get("batch_id", json_path.stem)
                pt = extract_batch_trend_point(raw, b_id, raw.get("filename"))
                if pt:
                    batches_map[b_id] = pt
            except Exception as e:
                logger.warning(f"Could not load batch file {json_path.name} for trends: {e}")

    # 2. Check in-memory BATCH_STORE
    for b_id, summary in BATCH_STORE.items():
        if b_id not in batches_map:
            try:
                raw = summary.model_dump()
                pt = extract_batch_trend_point(raw, b_id, summary.filename)
                if pt:
                    batches_map[b_id] = pt
            except Exception as e:
                logger.warning(f"Could not convert in-memory batch {b_id} for trends: {e}")

    # 3. Sort chronologically (oldest to newest)
    sorted_batches = sorted(batches_map.values(), key=lambda b: b.created_at)
    return sorted_batches


def compute_trends_summary(batches: list[BatchTrendPoint]) -> TrendsSummaryResponse:
    """Computes trajectory direction, dimensional deltas, and plain-English executive insight."""
    total = len(batches)
    if total == 0:
        return TrendsSummaryResponse(
            total_batches=0,
            batches=[],
            overall_quality_trend="insufficient_data",
            score_change_percent=0.0,
            latest_batch_id=None,
            summary_insight="No historical batch evaluations found. Upload and evaluate CSV batches to analyze performance trends over time.",
        )
    if total == 1:
        b = batches[0]
        return TrendsSummaryResponse(
            total_batches=1,
            batches=batches,
            overall_quality_trend="insufficient_data",
            score_change_percent=0.0,
            latest_batch_id=b.batch_id,
            summary_insight=f"Single batch evaluation recorded ({b.total_records} rows, {round(b.average_weighted_score * 100, 1)}% composite quality). Run additional batch submissions over time to track multi-run progression.",
        )

    first_b = batches[0]
    latest_b = batches[-1]
    score_delta = latest_b.average_weighted_score - first_b.average_weighted_score
    score_change_pct = round(score_delta * 100, 1)

    if score_delta > 0.02:
        overall_trend = "improving"
    elif score_delta < -0.02:
        overall_trend = "degrading"
    else:
        overall_trend = "stable"

    acc_diff = round((latest_b.average_accuracy_score - first_b.average_accuracy_score) * 100, 1)
    hal_diff = round(latest_b.hallucination_rate_percent - first_b.hallucination_rate_percent, 1)
    comp_diff = round((latest_b.average_completeness_score - first_b.average_completeness_score) * 100, 1)

    drivers = []
    if abs(hal_diff) >= 2.0:
        drivers.append(f"{'reduction' if hal_diff < 0 else 'increase'} of {abs(hal_diff)}% in hallucination rate")
    if abs(acc_diff) >= 2.0:
        drivers.append(f"{'+' if acc_diff > 0 else ''}{acc_diff}% in factual accuracy")
    if abs(comp_diff) >= 2.0:
        drivers.append(f"{'+' if comp_diff > 0 else ''}{comp_diff}% in completeness coverage")

    driver_str = f" Driven primarily by a {', and '.join(drivers)}." if drivers else ""

    if overall_trend == "improving":
        summary_insight = f"Overall LLM response quality is improving (+{score_change_pct}% across {total} submissions, reaching {round(latest_b.average_weighted_score * 100, 1)}% composite score).{driver_str}"
    elif overall_trend == "degrading":
        summary_insight = f"Overall LLM response quality has degraded ({score_change_pct}% across {total} submissions down to {round(latest_b.average_weighted_score * 100, 1)}% composite score).{driver_str}"
    else:
        summary_insight = f"Overall LLM response quality has remained stable across {total} submissions (~{round(latest_b.average_weighted_score * 100, 1)}% composite score).{driver_str}"

    return TrendsSummaryResponse(
        total_batches=total,
        batches=batches,
        overall_quality_trend=overall_trend,
        score_change_percent=score_change_pct,
        latest_batch_id=latest_b.batch_id,
        summary_insight=summary_insight,
    )


@app.get("/api/v1/batches", response_model=List[BatchTrendPoint])
def list_batches():
    """Returns chronological list of all stored historical batch evaluations."""
    return collect_historical_batches()


@app.get("/api/v1/batches/trends", response_model=TrendsSummaryResponse)
@app.get("/api/v1/trends", response_model=TrendsSummaryResponse)
def get_evaluation_trends():
    """Returns aggregated evaluation trends and chronological trajectory across multiple batch submissions."""
    batches = collect_historical_batches()
    return compute_trends_summary(batches)


@app.delete("/api/v1/batch/{batch_id}")
def delete_batch(batch_id: str):
    """Deletes a historical batch record from storage."""
    deleted = False
    if batch_id in BATCH_STORE:
        del BATCH_STORE[batch_id]
        deleted = True
    batch_file = BATCH_DIR / f"{batch_id}.json"
    if batch_file.exists():
        try:
            batch_file.unlink()
            deleted = True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not delete batch file: {e}")

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Batch ID '{batch_id}' not found.")

    # Refresh latest.txt if necessary
    latest_file = BATCH_DIR / "latest.txt"
    if latest_file.exists() and latest_file.read_text(encoding="utf-8").strip() == batch_id:
        remaining = collect_historical_batches()
        if remaining:
            latest_file.write_text(remaining[-1].batch_id, encoding="utf-8")
        else:
            latest_file.unlink(missing_ok=True)

    return {"status": "deleted", "batch_id": batch_id}


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.3.0", "milestones": ["M1", "M2", "M3", "M4"]}