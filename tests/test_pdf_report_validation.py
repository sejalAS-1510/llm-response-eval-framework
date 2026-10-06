"""
tests/test_pdf_report_validation.py
------------------------------------
Automated data integrity tests asserting that every number in generated PDF reports
matches underlying stored evaluation records and dashboard details with 100% exactness.
Runs against:
- Small batch (8 records: batch-small-eval-8)
- Large batch (105 records: batch-1d00a7b7)
"""

from pathlib import Path
from scripts.validate_pdf_report_integrity import validate_batch_report_integrity

storage_dir = Path(__file__).resolve().parent.parent / "data" / "batches"


def test_small_batch_pdf_data_integrity():
    """
    Validates that every single number in the small batch PDF (8 records)
    matches the underlying stored records and dashboard table exactly.
    """
    result = validate_batch_report_integrity(
        batch_id="batch-small-eval-8",
        storage_dir=storage_dir,
    )
    assert result["batch_id"] == "batch-small-eval-8"
    assert result["total_records"] == 8
    assert result["pdf_pages"] >= 3
    assert result["checks_passed"] == 5


def test_large_batch_pdf_data_integrity():
    """
    Validates that every single number in the large batch PDF (105 records)
    matches the underlying stored records and dashboard table exactly.
    """
    result = validate_batch_report_integrity(
        batch_id="batch-1d00a7b7",
        storage_dir=storage_dir,
    )
    assert result["batch_id"] == "batch-1d00a7b7"
    assert result["total_records"] == 105
    assert result["pdf_pages"] >= 50
    assert result["checks_passed"] == 5
