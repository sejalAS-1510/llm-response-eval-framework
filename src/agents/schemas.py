"""
Pydantic data models and schemas for Milestone 2 evaluation agents.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


RelevanceClassification = Literal[
    "fully_relevant",
    "partially_relevant",
    "unrelated",
    "off_topic",
]

AccuracyClassification = Literal[
    "correct",
    "partially_correct",
    "incorrect",
    "contradictory",
]

ClaimStatus = Literal[
    "supported",
    "unsupported",
    "contradicted",
]


class RelevanceResult(BaseModel):
    score: float = Field(..., description="Relevance score from 0.0 to 1.0")
    classification: RelevanceClassification = Field(..., description="Category of relevance")
    reasoning: str = Field(..., description="Detailed explanation justifying the relevance score")


class AccuracyResult(BaseModel):
    score: float = Field(..., description="Accuracy score from 0.0 to 1.0")
    classification: AccuracyClassification = Field(..., description="Category of factual correctness")
    supporting_evidence: List[str] = Field(
        description="Direct quotes or references from the reference/retrieved context",
    )
    reasoning: str = Field(..., description="Detailed explanation of the accuracy evaluation")


class ClaimEvaluation(BaseModel):
    claim: str = Field(..., description="Discrete atomic factual claim extracted from the AI response")
    status: ClaimStatus = Field(..., description="Status of the claim against context: supported, unsupported, or contradicted")
    evidence: Optional[str] = Field(
        None,
        description="Quoted passage or snippet from retrieved context if supported/contradicted",
    )
    explanation: str = Field(..., description="Reason why the claim is supported, unsupported, or contradicted")


class HallucinationResult(BaseModel):
    is_hallucinated: bool = Field(..., description="True if any claim is unsupported or contradicted")
    hallucination_score: float = Field(
        ...,
        description="Fraction of claims that are unsupported or contradicted (0.0 = completely grounded, 1.0 = completely hallucinated)",
    )
    total_claims: int = Field(..., description="Total number of factual claims extracted")
    unsupported_claims_count: int = Field(..., description="Number of unsupported or contradicted claims")
    flagged_claims: List[ClaimEvaluation] = Field(
        description="List of specific claims identified as unsupported or fabricated",
    )
    all_claims: List[ClaimEvaluation] = Field(
        description="Full breakdown of all evaluated claims",
    )
    reasoning: str = Field(..., description="Overall reasoning for the hallucination assessment")


class EvaluationResult(BaseModel):
    submission_id: Optional[int] = None
    question: str
    ai_response: str
    context_used: str
    relevance: RelevanceResult
    accuracy: AccuracyResult
    hallucination: HallucinationResult
    evaluated_at: str
