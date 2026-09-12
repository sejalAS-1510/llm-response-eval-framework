"""
Evaluation Input Module (M1.3).

Single endpoint that accepts a question + AI response (plus optional
reference answer / source document), validates it, and stores it.
Run with: uvicorn src.input_module.main:app --reload
"""

from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .schemas import EvaluationSubmission, SubmissionResponse
from .storage import get_submission, insert_submission
from ..agents.orchestrator import EvaluationOrchestrator
from ..agents.schemas import EvaluationResult

orchestrator = EvaluationOrchestrator()

app = FastAPI(title="LLM Response Eval - Input Module", version="0.1.0")

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


@app.get("/health")
def health():
    return {"status": "ok"}