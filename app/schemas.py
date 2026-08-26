from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class EvaluationSubmissionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    ai_response: str = Field(..., min_length=1, max_length=20000)
    reference_answer: Optional[str] = Field(default=None, max_length=20000)
    source_document: Optional[str] = Field(default=None, max_length=50000)

    @field_validator("question", "ai_response")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty or whitespace only.")
        return v.strip()


class EvaluationSubmissionResponse(BaseModel):
    id: int
    question: str
    ai_response: str
    reference_answer: Optional[str]
    source_document: Optional[str]
    status: str
    created_at: datetime

    class Config:
        from_attributes = True
