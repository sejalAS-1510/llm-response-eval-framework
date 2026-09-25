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
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .schemas import EvaluationSubmission, SubmissionResponse
from .storage import get_submission, insert_submission
from ..agents.orchestrator import EvaluationOrchestrator
from ..agents.schemas import EvaluationResult, BatchEvaluationSummary
from ..agents.batch_evaluator import BatchEvaluator

logger = logging.getLogger(__name__)

orchestrator = EvaluationOrchestrator()
batch_evaluator = BatchEvaluator(orchestrator=orchestrator, concurrency_limit=5)

# Store for completed batches (in-memory + disk cache persistence)
BATCH_STORE: dict[str, BatchEvaluationSummary] = {}
BATCH_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "batches"
BATCH_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LLM Response Eval - Input Module", version="0.2.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"error": "invalid submission", "details": exc.errors()},
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


@app.get("/api/v1/batch/{batch_id}", response_model=BatchEvaluationSummary)
def get_batch_results(batch_id: str):
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
    return summary


@app.get("/api/v1/batch/{batch_id}/export")
def export_batch_csv(batch_id: str):
    """Exports evaluated batch records as a formatted CSV spreadsheet."""
    summary = get_batch_results(batch_id)
    csv_data = batch_evaluator.export_summary_to_csv(summary)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={batch_id}_evaluated.csv"
        },
    )


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


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0", "milestones": ["M1", "M2", "M3"]}