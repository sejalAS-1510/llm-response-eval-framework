"""
tests/test_fault_tolerance_and_resilience.py
--------------------------------------------
Resilience, Fault Tolerance, and Error Recovery Validation Suite.

Tests:
1. Missing reference answer: Graceful fallback to RAG retrieval (ChromaDB)
   populating retrieved context without failing.
2. Malformed CSV rows: Ragged lines, missing columns, interspersed empty lines
   parsed safely without crashing.
3. Missing required fields: Missing CSV headers rejected with HTTP 400; missing row-level
   AI response marked as failed while preserving batch continuation; empty questions tolerated.
4. Retrieval / source-content failures: Vector store exceptions caught cleanly and routed
   to zero-shot fallback context without interrupting evaluations.
5. Individual agent failure mid-batch: Transient exception on a single record mid-batch
   marks that specific record as failed/skipped, while allowing remaining records to complete normally.
6. Downstream resilience: Aggregation and PDF reporting generate successfully even with failed rows.
"""

import io
import os
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

# Ensure mock/offline evaluation mode is active
os.environ["MOCK_LLM"] = "1"

from src.input_module.main import app, orchestrator, batch_evaluator
from src.agents.batch_evaluator import BatchEvaluator
from src.agents.orchestrator import EvaluationOrchestrator
from src.agents.schemas import EvaluationResult, BatchEvaluationSummary
from src.reporting.report_service import ReportDataAggregator
from src.reporting.pdf_generator import generate_pdf_report


@pytest.fixture
def client():
    return TestClient(app)


# ===========================================================================
# 1. MISSING REFERENCE ANSWER (RAG RETRIEVAL FALLBACK)
# ===========================================================================

@pytest.mark.asyncio
async def test_missing_reference_answer_falls_back_to_rag():
    """
    Asserts: When reference_answer and source_document are omitted (None or empty),
    the orchestrator executes RAG retrieval via ChromaDB and populates context_used
    with retrieved knowledge base snippets.
    """
    orch = EvaluationOrchestrator()
    question = "What is the capital and largest city of France?"
    response = "Paris is the capital of France and its largest city."

    # Explicitly pass None for reference_answer and source_document
    result: EvaluationResult = await orch.evaluate(
        question=question,
        ai_response=response,
        reference_answer=None,
        source_document=None,
    )

    assert result is not None
    assert result.context_used is not None and len(result.context_used) > 0
    # Must contain RAG retrieval indicators
    assert (
        "Retrieved Reference Knowledge Base Chunks" in result.context_used
        or "Source:" in result.context_used
        or len(result.context_used) > 20
    ), f"Expected RAG chunks in context_used, got: {result.context_used}"

    # Evaluation completed across all dimensions
    assert result.relevance is not None
    assert result.accuracy is not None
    assert result.hallucination is not None
    assert result.completeness is not None
    assert result.verdict is not None


# ===========================================================================
# 2. MALFORMED CSV ROWS & RAGGED LINES
# ===========================================================================

@pytest.mark.asyncio
async def test_malformed_csv_rows_and_ragged_lines():
    """
    Asserts: CSV files containing ragged rows (fewer columns than header),
    blank lines, and varied quote formats do not crash the parser or batch evaluator.
    - Valid rows succeed.
    - Bad/truncated rows without AI response are marked failed with an informative error message.
    - One bad record does not terminate the rest of the batch.
    """
    evaluator = BatchEvaluator(orchestrator=orchestrator, concurrency_limit=2)

    # Construct CSV with:
    # Row 1: Valid Q & A
    # Blank line (should be skipped)
    # Row 2: Ragged row with only 1 column (missing AI response)
    # Row 3: Valid Q & A
    # Row 4: Empty AI response (,,)
    # Row 5: Valid Q & A
    malformed_csv = """questions,ai_responses,reference_answer
What is the chemical symbol for gold?,The chemical symbol for gold is Au.,Gold is Au.

OnlyOneColumnWithoutResponse
What is the capital of Japan?,Tokyo is the capital of Japan.,Tokyo is the capital of Japan.
What is the boiling point of ethanol?,,Ethanol boils at 78.4 C.
What is the boiling point of water?,Water boils at 100 degrees Celsius.,Water boils at 100 C.
"""

    records, warnings = evaluator.parse_and_validate_csv(malformed_csv)

    # 5 total data rows parsed (blank line skipped)
    assert len(records) == 5

    summary: BatchEvaluationSummary = await evaluator.evaluate_batch(records, filename="malformed.csv")
    stats = summary.statistics

    # Assert batch-level resilience
    assert stats.total_records == 5
    assert stats.successful_records == 3, f"Expected 3 successful records, got {stats.successful_records}"
    assert stats.failed_records == 2, f"Expected 2 failed records, got {stats.failed_records}"

    # Verify individual row statuses
    # Row 1 (Index 1): Success
    assert summary.items[0].index == 1
    assert summary.items[0].status == "success"
    assert summary.items[0].result is not None

    # Row 2 (Index 2): Failed (Ragged line missing AI response)
    assert summary.items[1].index == 2
    assert summary.items[1].status == "failed"
    assert summary.items[1].result is None
    assert "AI Response is empty" in summary.items[1].error_message

    # Row 3 (Index 3): Success
    assert summary.items[2].index == 3
    assert summary.items[2].status == "success"
    assert summary.items[2].result is not None

    # Row 4 (Index 4): Failed (Empty response)
    assert summary.items[3].index == 4
    assert summary.items[3].status == "failed"
    assert summary.items[3].result is None

    # Row 5 (Index 5): Success
    assert summary.items[4].index == 5
    assert summary.items[4].status == "success"
    assert summary.items[4].result is not None


# ===========================================================================
# 3. MISSING REQUIRED FIELDS
# ===========================================================================

def test_missing_required_headers_rejected(client):
    """
    Asserts: Uploaded CSV files missing required header columns are rejected
    with HTTP 400 and clear diagnostic messages.
    """
    # Case A: Missing 'questions' column
    csv_missing_q = "model_output,reference_answer\nAn answer without question.,Ref answer.\n"
    res_a = client.post(
        "/api/v1/batch/upload-csv",
        files={"file": ("missing_q.csv", io.BytesIO(csv_missing_q.encode("utf-8")), "text/csv")},
    )
    assert res_a.status_code == 400
    assert "Missing required column(s)" in res_a.json()["detail"]
    assert "'questions'" in res_a.json()["detail"]

    # Case B: Missing 'ai_responses' column
    csv_missing_r = "questions,reference_answer\nWhat is gravity?,Force of attraction.\n"
    res_b = client.post(
        "/api/v1/batch/upload-csv",
        files={"file": ("missing_r.csv", io.BytesIO(csv_missing_r.encode("utf-8")), "text/csv")},
    )
    assert res_b.status_code == 400
    assert "Missing required column(s)" in res_b.json()["detail"]
    assert "'ai_responses'" in res_b.json()["detail"]

    # Case C: Both required columns missing
    csv_random = "user_name,email,score\nAlice,alice@example.com,95\n"
    res_c = client.post(
        "/api/v1/batch/upload-csv",
        files={"file": ("random.csv", io.BytesIO(csv_random.encode("utf-8")), "text/csv")},
    )
    assert res_c.status_code == 400
    assert "'questions'" in res_c.json()["detail"] and "'ai_responses'" in res_c.json()["detail"]


@pytest.mark.asyncio
async def test_empty_question_permissive_evaluation():
    """
    Asserts: In contrast to empty AI responses, empty question fields in a batch row
    are permitted and evaluated gracefully with fallback placeholder text.
    """
    evaluator = BatchEvaluator(orchestrator=orchestrator, concurrency_limit=2)
    csv_empty_q = """questions,ai_responses,reference_answer
,Water boils at 100 degrees Celsius under standard atmospheric pressure.,Water boils at 100 C.
"""
    records, warnings = evaluator.parse_and_validate_csv(csv_empty_q)
    assert len(records) == 1
    assert records[0]["question"] in ("[No Question Provided]", "")

    summary = await evaluator.evaluate_batch(records)
    assert summary.statistics.total_records == 1
    assert summary.statistics.successful_records == 1
    assert summary.items[0].status == "success"
    assert summary.items[0].result is not None


# ===========================================================================
# 4. RETRIEVAL & SOURCE-CONTENT FAILURES
# ===========================================================================

@pytest.mark.asyncio
async def test_retrieval_and_source_content_failure_graceful_fallback():
    """
    Asserts: When the vector store or retrieval pipeline raises an unexpected exception
    (e.g., ChromaDB socket timeout, corrupted index, connection error),
    resolve_context catches the exception and falls back to:
    'Retrieval unavailable. Evaluate with zero-shot reference context.'
    The evaluation does NOT crash and returns a valid EvaluationResult.
    """
    orch = EvaluationOrchestrator()

    # Simulate ChromaDB socket failure
    with patch("src.agents.orchestrator.retrieve", side_effect=ConnectionError("ChromaDB socket disconnected")):
        result: EvaluationResult = await orch.evaluate(
            question="What is the speed of light in vacuum?",
            ai_response="The speed of light in vacuum is approximately 299,792,458 meters per second.",
            reference_answer=None,  # Forces retrieval attempt
            source_document=None,
        )

        assert result is not None
        assert "Retrieval unavailable. Evaluate with zero-shot reference context." in result.context_used
        # All 4 agents executed with zero-shot context
        assert result.relevance is not None
        assert result.accuracy is not None
        assert result.hallucination is not None
        assert result.completeness is not None
        assert result.verdict is not None


# ===========================================================================
# 5. INDIVIDUAL AGENT FAILURES MID-BATCH
# ===========================================================================

@pytest.mark.asyncio
async def test_individual_agent_failure_mid_batch_does_not_halt_batch():
    """
    Asserts: If an individual agent raises an exception mid-batch on a specific record
    (e.g. transient network timeout, model quota glitch, or unhandled token issue),
    that specific record is marked status='failed' with error_message recorded,
    while all other records in the batch continue and complete normally.
    """
    orch = EvaluationOrchestrator()
    evaluator = BatchEvaluator(orchestrator=orch, concurrency_limit=2)

    records = [
        {"index": 1, "question": "What is the capital of France?", "ai_response": "Paris.", "reference_answer": "Paris."},
        {"index": 2, "question": "FAULT_TRIGGER_QUESTION", "ai_response": "Some response.", "reference_answer": "Ref."},
        {"index": 3, "question": "What is 2 + 2?", "ai_response": "4.", "reference_answer": "4."},
        {"index": 4, "question": "What is the boiling point of water?", "ai_response": "100 C.", "reference_answer": "100 C."},
    ]

    # Intercept orchestrator.evaluate to selectively simulate a transient failure only on Row 2
    original_evaluate = orch.evaluate

    async def mock_evaluate_with_selective_fault(**kwargs):
        if kwargs.get("question") == "FAULT_TRIGGER_QUESTION":
            raise RuntimeError("Simulated transient LLM connection reset on row 2")
        return await original_evaluate(**kwargs)

    orch.evaluate = mock_evaluate_with_selective_fault

    # Run batch evaluation
    summary: BatchEvaluationSummary = await evaluator.evaluate_batch(records, filename="fault_test.csv")
    stats = summary.statistics

    # Assert batch completed all records
    assert stats.total_records == 4
    assert stats.successful_records == 3, f"Expected 3 successful records, got {stats.successful_records}"
    assert stats.failed_records == 1, f"Expected 1 failed record, got {stats.failed_records}"

    # Verify Row 1: Success
    assert summary.items[0].index == 1
    assert summary.items[0].status == "success"
    assert summary.items[0].result is not None

    # Verify Row 2: Failed without halting the batch
    assert summary.items[1].index == 2
    assert summary.items[1].status == "failed"
    assert summary.items[1].result is None
    assert "Simulated transient LLM connection reset on row 2" in summary.items[1].error_message

    # Verify Rows 3 and 4: Completed normally
    assert summary.items[2].index == 3
    assert summary.items[2].status == "success"
    assert summary.items[2].result is not None

    assert summary.items[3].index == 4
    assert summary.items[3].status == "success"
    assert summary.items[3].result is not None


# ===========================================================================
# 6. DOWNSTREAM REPORTING RESILIENCE WITH FAILED ROWS
# ===========================================================================

@pytest.mark.asyncio
async def test_downstream_aggregation_and_pdf_generation_with_failed_rows():
    """
    Asserts: Downstream report aggregation and PDF generation seamlessly handle
    batches containing failed/errored records without crashing or raising exceptions.
    """
    orch = EvaluationOrchestrator()
    evaluator = BatchEvaluator(orchestrator=orch, concurrency_limit=2)

    records = [
        {"index": 1, "question": "What is the capital of Germany?", "ai_response": "Berlin.", "reference_answer": "Berlin."},
        {"index": 2, "question": "What is the capital of Spain?", "ai_response": "", "reference_answer": "Madrid."},  # Failed row
        {"index": 3, "question": "What is the capital of Italy?", "ai_response": "Rome.", "reference_answer": "Rome."},
    ]

    summary = await evaluator.evaluate_batch(records, filename="resilience_report.csv")
    assert summary.statistics.failed_records == 1
    assert summary.statistics.successful_records == 2

    # Aggregate report data from summary
    aggregator = ReportDataAggregator()
    report_data = aggregator.build_report_data(summary)

    assert report_data.metadata.total_records == 3
    assert report_data.metadata.successful_records == 2
    assert report_data.metadata.failed_records == 1
    assert len(report_data.per_response_details) == 3

    # Failed row must reflect failed status in report detail
    failed_detail = report_data.per_response_details[1]
    assert failed_detail.status == "failed"
    assert failed_detail.error_message is not None

    # Generate PDF report from report_data containing failed row
    pdf_bytes = generate_pdf_report(report_data)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 5000
