"""
Evaluation Input Module (M1.3).

Single endpoint that accepts a question + AI response (plus optional
reference answer / source document), validates it, and stores it.
Run with: uvicorn src.input_module.main:app --reload
"""

from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .schemas import EvaluationSubmission, SubmissionResponse
from .storage import get_submission, insert_submission

app = FastAPI(title="LLM Response Eval - Input Module", version="0.1.0")


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


@app.get("/health")
def health():
    return {"status": "ok"}
