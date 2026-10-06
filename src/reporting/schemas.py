"""
src/reporting/schemas.py
------------------------
Data models defining the structured report-data object for PDF report generation.
Adheres strictly to the structured evaluation results stored on disk/database.
"""

from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field


class ReportMetadata(BaseModel):
    batch_id: str = Field(..., description="Unique batch identifier")
    filename: Optional[str] = Field(None, description="Original uploaded CSV filename")
    timestamp: str = Field(..., description="Batch creation ISO timestamp")
    report_generated_at: str = Field(..., description="Timestamp when report data was aggregated")
    total_records: int = Field(..., description="Total records in batch")
    successful_records: int = Field(..., description="Count of successfully evaluated records")
    failed_records: int = Field(..., description="Count of failed/skipped records")


class OverallStats(BaseModel):
    pass_count: int = Field(..., description="Number of responses receiving Pass verdict")
    pass_percent: float = Field(..., description="Pass percentage (0-100%)")
    needs_improvement_count: int = Field(..., description="Number of responses receiving Needs Improvement verdict")
    needs_improvement_percent: float = Field(..., description="Needs Improvement percentage (0-100%)")
    fail_count: int = Field(..., description="Number of responses receiving Fail verdict")
    fail_percent: float = Field(..., description="Fail percentage (0-100%)")
    
    # Average scores per dimension (normalized 0.0-1.0 and percentage 0-100%)
    average_relevance_score: float = Field(..., description="Mean relevance score (0.0-1.0)")
    average_relevance_percent: float = Field(..., description="Mean relevance score as percentage")
    average_accuracy_score: float = Field(..., description="Mean factual accuracy score (0.0-1.0)")
    average_accuracy_percent: float = Field(..., description="Mean factual accuracy score as percentage")
    average_completeness_score: float = Field(..., description="Mean completeness score (0.0-1.0)")
    average_completeness_percent: float = Field(..., description="Mean completeness score as percentage")
    average_groundedness_score: float = Field(..., description="Mean groundedness score (0.0-1.0)")
    average_groundedness_percent: float = Field(..., description="Mean groundedness score as percentage")
    average_weighted_score: float = Field(..., description="Mean overall composite quality score (0.0-1.0)")
    average_weighted_percent: float = Field(..., description="Mean overall composite quality score as percentage")


class FlaggedClaimItem(BaseModel):
    claim: str = Field(..., description="Atomic claim identified as ungrounded or contradicted")
    status: str = Field(..., description="Status: unsupported or contradicted")
    evidence: Optional[str] = Field(None, description="Conflicting context evidence if contradicted")
    explanation: str = Field(..., description="Reasoning justifying why claim was flagged")


class HallucinationFrequency(BaseModel):
    flagged_responses_count: int = Field(..., description="Number of responses flagged with hallucinated claims")
    total_evaluated_responses: int = Field(..., description="Total successfully evaluated responses")
    rate_percent: float = Field(..., description="Hallucination rate percentage (0-100%)")
    total_claims_extracted: int = Field(..., description="Total factual statements extracted across batch")
    total_unsupported_claims: int = Field(..., description="Total statements flagged as ungrounded/contradicted")
    sample_flagged_claims: List[str] = Field(default_factory=list, description="Representative sample of flagged claims")


class DimensionReportDetail(BaseModel):
    score: float = Field(..., description="Dimension score from 0.0 to 1.0")
    score_percent: float = Field(..., description="Dimension score as percentage (0-100%)")
    classification: Optional[str] = Field(None, description="Category classification label")
    reasoning: str = Field(..., description="Detailed diagnostic reasoning from evaluator agent")
    evidence: Optional[Union[str, List[str]]] = Field(
        None, description="Supporting evidence quotes, citations, or references"
    )


class PerResponseReportDetail(BaseModel):
    index: int = Field(..., description="Row index from CSV batch")
    question: str = Field(..., description="Prompt/question asked to the LLM")
    response: str = Field(..., description="AI generation evaluated")
    reference_answer: Optional[str] = Field(None, description="Ground truth or reference target answer")
    source_document: Optional[str] = Field(None, description="Source context document, if provided")
    status: str = Field(..., description="Status of evaluation: 'success', 'failed', or 'skipped'")
    error_message: Optional[str] = Field(None, description="Error details if row evaluation failed")

    # Dimensional scores + reasoning + evidence
    relevance: Optional[DimensionReportDetail] = Field(None, description="Relevance dimension breakdown")
    accuracy: Optional[DimensionReportDetail] = Field(None, description="Factual accuracy dimension breakdown")
    completeness: Optional[DimensionReportDetail] = Field(None, description="Completeness dimension breakdown")
    hallucination: Optional[DimensionReportDetail] = Field(None, description="Hallucination/groundedness breakdown")

    # Specific issues identified for fast reporting display
    flagged_hallucinated_claims: List[str] = Field(
        default_factory=list, description="Specific factual claims flagged as fabricated or contradicted"
    )
    flagged_claims_breakdown: List[FlaggedClaimItem] = Field(
        default_factory=list, description="Structured claim-level verification details"
    )
    missing_aspects: List[str] = Field(
        default_factory=list, description="Omitted criteria, sub-questions, or missing context"
    )
    weighted_composite_score: float = Field(..., description="Composite quality score (0.0-1.0)")
    final_verdict: str = Field(..., description="Final overall verdict: Pass, Needs Improvement, Fail, or Failed")
    verdict_explanation: str = Field(..., description="Consolidated verdict rationale and dimension weighting summary")


class RecommendationItem(BaseModel):
    """
    An actionable plain-English improvement recommendation based on batch weaknesses.
    Dynamically generated from frequency thresholds across evaluation dimensions.
    """
    id: str = Field(..., description="Unique recommendation identifier")
    category: str = Field(..., description="Weakness category: grounding, accuracy, completeness, relevance, pipeline, or general_quality")
    priority: str = Field(..., description="Priority ranking: high, medium, or low")
    headline: str = Field(..., description="Concise diagnostic summary headline")
    plain_english: str = Field(
        ...,
        description="Full plain-English recommendation statement combining metric trigger and actionable remediation advice",
    )
    metric_trigger: str = Field(..., description="Empirical evidence triggering this recommendation with exact count and percentage")
    remediation_advice: str = Field(..., description="Specific, actionable engineering advice to fix the issue")
    affected_count: int = Field(..., description="Count of responses exhibiting this weakness")
    affected_percent: float = Field(..., description="Percentage of evaluated items affected (0-100%)")
    dimension: Optional[str] = Field(None, description="Primary evaluation dimension affected")
    sample_issues: List[str] = Field(default_factory=list, description="Sample of specific flagged claims or missing aspects")


class BatchReportData(BaseModel):
    """
    Complete structured report-data object for a batch, ready to feed the PDF report generator.
    """
    metadata: ReportMetadata = Field(..., description="Batch identification and generation metadata")
    overall_stats: OverallStats = Field(..., description="Aggregated KPI metrics and dimensional averages")
    hallucination_frequency: HallucinationFrequency = Field(..., description="Hallucination frequency and claim stats")
    per_response_details: List[PerResponseReportDetail] = Field(
        ..., description="Granular response-level evaluation records"
    )
    recommendations: List[RecommendationItem] = Field(
        default_factory=list,
        description="Actionable plain-English recommendations dynamically generated from batch weaknesses",
    )

