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
    completeness: "CompletenessResult"
    verdict: "VerdictResult"
    evaluated_at: str


CompletenessClassification = Literal[
    "fully_complete",
    "mostly_complete",
    "partially_complete",
    "incomplete",
]


class CompletenessResult(BaseModel):
    score: float = Field(..., description="Completeness score from 0.0 to 1.0")
    classification: CompletenessClassification = Field(..., description="Category of completeness")
    identified_requirements: List[str] = Field(
        default_factory=list,
        description="Key requirements, sub-questions, or aspects expected from the prompt/reference context",
    )
    addressed_aspects: List[str] = Field(
        default_factory=list,
        description="Aspects or sub-questions sufficiently answered by the AI response",
    )
    partially_addressed_aspects: List[str] = Field(
        default_factory=list,
        description="Aspects only partially, vaguely, or incompletely covered",
    )
    missing_aspects: List[str] = Field(
        default_factory=list,
        description="Specific omissions, unanswered sub-questions, or missing explanations",
    )
    reasoning: str = Field(..., description="Detailed justification explaining why information is complete, partial, or missing")


VerdictCategory = Literal[
    "Pass",
    "Needs Improvement",
    "Fail",
]


class VerdictResult(BaseModel):
    weighted_score: float = Field(..., description="Overall weighted score from 0.0 to 1.0")
    verdict: VerdictCategory = Field(..., description="Final overall quality verdict: Pass, Needs Improvement, or Fail")
    dimension_scores: dict = Field(
        default_factory=dict,
        description="Normalized scores for relevance, accuracy, completeness, and groundedness",
    )
    major_issues: List[str] = Field(
        default_factory=list,
        description="Critical issues, contradictions, severe hallucinations, or major omissions",
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Observed strengths across evaluated dimensions",
    )
    consolidated_reasoning: str = Field(
        ...,
        description="Consolidated executive evaluation summary explaining strengths, weaknesses, and final verdict",
    )


class BatchEvaluationItem(BaseModel):
    index: int = Field(..., description="1-indexed row number from the uploaded batch")
    question: str
    ai_response: str
    reference_answer: Optional[str] = None
    source_document: Optional[str] = None
    result: Optional[EvaluationResult] = None
    status: Literal["success", "failed", "skipped"] = "success"
    error_message: Optional[str] = None
    batch_id: Optional[str] = Field(default=None, description="Identifier of the batch this record belongs to")
    batch_filename: Optional[str] = Field(default=None, description="Original uploaded CSV filename")


class BatchStatistics(BaseModel):
    total_records: int
    successful_records: int
    failed_records: int
    pass_count: int
    needs_improvement_count: int
    fail_count: int
    pass_rate_percent: float
    average_weighted_score: float
    average_relevance_score: float
    average_accuracy_score: float
    average_completeness_score: float
    average_groundedness_score: float
    hallucination_rate_percent: float


class BatchEvaluationSummary(BaseModel):
    batch_id: str
    filename: Optional[str] = None
    created_at: str
    statistics: BatchStatistics
    items: List[BatchEvaluationItem]


class BatchTrendPoint(BaseModel):
    batch_id: str
    filename: Optional[str] = None
    created_at: str
    total_records: int
    successful_records: int
    failed_records: int
    pass_count: int
    needs_improvement_count: int
    fail_count: int
    pass_percent: float
    needs_improvement_percent: float
    fail_percent: float
    average_weighted_score: float
    average_relevance_score: float
    average_accuracy_score: float
    average_completeness_score: float
    average_groundedness_score: float
    hallucination_rate_percent: float


class TrendsSummaryResponse(BaseModel):
    total_batches: int
    batches: List[BatchTrendPoint]
    overall_quality_trend: Literal["improving", "degrading", "stable", "insufficient_data"]
    score_change_percent: float
    latest_batch_id: Optional[str] = None
    summary_insight: str


# Rebuild EvaluationResult with full forward ref resolution
EvaluationResult.model_rebuild()


