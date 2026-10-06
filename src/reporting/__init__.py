"""
src/reporting package
---------------------
Data aggregation layer and report generation services for LLM response evaluation.
"""

from src.reporting.schemas import (
    BatchReportData,
    ReportMetadata,
    OverallStats,
    HallucinationFrequency,
    DimensionReportDetail,
    FlaggedClaimItem,
    PerResponseReportDetail,
    RecommendationItem,
)
from src.reporting.report_service import (
    ReportDataAggregator,
    generate_batch_report_data,
)
from src.reporting.recommendations import (
    RecommendationEngine,
    generate_recommendations,
)
from src.reporting.pdf_generator import (
    generate_pdf_report,
)

__all__ = [
    "BatchReportData",
    "ReportMetadata",
    "OverallStats",
    "HallucinationFrequency",
    "DimensionReportDetail",
    "FlaggedClaimItem",
    "PerResponseReportDetail",
    "RecommendationItem",
    "ReportDataAggregator",
    "generate_batch_report_data",
    "RecommendationEngine",
    "generate_recommendations",
    "generate_pdf_report",
]


