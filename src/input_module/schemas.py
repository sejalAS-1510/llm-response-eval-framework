"""
Request/response models for the evaluation submission endpoint.

Primary inputs are the question and the AI response. Reference answer
and source document are optional - if neither is given, later retrieval
(M1.4) is used to pull supporting context instead.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class EvaluationSubmission(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    ai_response: str = Field(..., min_length=1, max_length=8000)
    reference_answer: Optional[str] = Field(default=None, max_length=4000)
    source_document: Optional[str] = Field(default=None, max_length=20000)

    @field_validator("question", "ai_response")
    @classmethod
    def not_just_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field cannot be empty or whitespace-only")
        return v.strip()


class EvaluationSubmissionRecord(EvaluationSubmission):
    id: int
    created_at: datetime


class SubmissionResponse(BaseModel):
    id: int
    status: str = "stored"
    created_at: datetime
