"""
Evaluation Input Module (M1.3)
Single submission interface: question, AI response, optional reference
answer / source document. Validates input and persists it for the
downstream orchestrator (M2) to pick up.
"""
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import init_db, get_db, EvaluationSubmission
from app.schemas import EvaluationSubmissionRequest, EvaluationSubmissionResponse

app = FastAPI(
    title="LLM Response Evaluation Framework — Input Module",
    description="M1.3: Accepts a single evaluation submission (question, AI response, "
                "optional reference answer / source document).",
    version="0.1.0",
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/evaluate/submit", response_model=EvaluationSubmissionResponse, status_code=201)
def submit_evaluation(payload: EvaluationSubmissionRequest, db: Session = Depends(get_db)):
    record = EvaluationSubmission(
        question=payload.question,
        ai_response=payload.ai_response,
        reference_answer=payload.reference_answer,
        source_document=payload.source_document,
        status="received",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@app.get("/evaluate/{submission_id}", response_model=EvaluationSubmissionResponse)
def get_submission(submission_id: int, db: Session = Depends(get_db)):
    record = db.query(EvaluationSubmission).filter(EvaluationSubmission.id == submission_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Submission not found.")
    return record
