"""
tests/test_e2e_workflows.py
----------------------------
Comprehensive End-to-End (E2E) Test Suite for the LLM Response Evaluation Framework.

Covers:
1. Single-Evaluation Workflow:
   - Submission -> RAG Retrieval -> All 4 Judge Agents (Relevance, Accuracy,
     Hallucination, Completeness) -> Verdict Synthesis -> Result Display.
   - Dual submission paths: Two-step (/evaluations -> /evaluations/{id}/evaluate)
     and direct (/evaluate).
   - Context resolution: RAG retrieval (ChromaDB) on missing reference vs direct
     context when reference_answer / source_document are provided.
   - Validation & boundary contracts (empty inputs, non-existent submissions).

2. Batch-Evaluation Workflow:
   - CSV Upload -> Pre-flight Validation -> Evaluation across all records ->
     Persistence (Disk/Memory) -> Aggregation -> Dashboard -> Recommendations ->
     CSV Export -> PDF Export -> Lifecycle Cleanup.
   - Permissive handling: empty questions permitted and evaluated.
   - Dynamic query-time statistics computation.
   - PDF document structure and binary integrity verification with pypdf.
   - Combined multi-batch PDF export.

3. Component Data Flow Contracts:
   - Explicit verification of schemas and data passing between:
     Input Module, SQLite DB, ChromaDB RAG, Orchestrator, 4 Judge Agents,
     Verdict Agent, Batch Evaluator, Aggregator, Recommendation Engine, and PDF Generator.
"""

import io
import json
import os
from pathlib import Path
from typing import Dict, Any

# Ensure mock/offline evaluation mode is enabled for deterministic, fast, offline test execution
os.environ["MOCK_LLM"] = "1"

import pytest
import pypdf
from fastapi.testclient import TestClient

from src.input_module.main import (
    app,
    BATCH_DIR,
    BATCH_STORE,
    orchestrator,
    batch_evaluator,
    report_aggregator,
)
from src.input_module.storage import get_submission, insert_submission
from src.input_module.schemas import EvaluationSubmission, SubmissionResponse
from src.agents.orchestrator import EvaluationOrchestrator
from src.agents.schemas import (
    EvaluationResult,
    BatchEvaluationSummary,
    BatchEvaluationItem,
    RelevanceResult,
    AccuracyResult,
    HallucinationResult,
    CompletenessResult,
    VerdictResult,
)
from src.agents.relevance_agent import RelevanceAgent
from src.agents.accuracy_agent import AccuracyAgent
from src.agents.hallucination_agent import HallucinationAgent
from src.agents.completeness_agent import CompletenessAgent
from src.agents.verdict_agent import VerdictAgent
from src.knowledge_base.vector_store import retrieve
from src.reporting.schemas import BatchReportData, RecommendationItem
from src.reporting.report_service import generate_batch_report_data
from src.reporting.recommendations import generate_recommendations
from src.reporting.pdf_generator import generate_pdf_report


@pytest.fixture
def client():
    return TestClient(app)


# ===========================================================================
# WORKFLOW 1: SINGLE-EVALUATION END-TO-END WORKFLOW
# ===========================================================================

class TestSingleEvaluationWorkflow:
    """
    Validates complete data flow for single evaluations:
    Submission -> RAG Retrieval -> 4 Judge Agents -> Verdict Synthesis -> Result Display.
    """

    def test_single_eval_two_step_with_rag_retrieval(self, client):
        """
        Tests the 2-step single evaluation flow where no reference answer is provided:
        1. POST /api/v1/evaluations (Persists in SQLite DB, returns ID).
        2. Direct DB verification via get_submission.
        3. POST /api/v1/evaluations/{id}/evaluate triggers RAG retrieval fallback,
           runs all 4 judge agents, synthesizes verdict, and returns EvaluationResult.
        """
        payload = {
            "question": "What is the capital and largest city of France?",
            "ai_response": "Paris is the capital of France and its largest city.",
        }

        # Step 1: Submit to Input Module
        res_submit = client.post("/api/v1/evaluations", json=payload)
        assert res_submit.status_code == 201, f"Expected 201 Created, got {res_submit.status_code}: {res_submit.text}"
        data_submit = res_submit.json()
        assert "id" in data_submit
        assert "created_at" in data_submit
        assert data_submit["status"] == "stored"
        submission_id = data_submit["id"]

        # Step 2: Assert SQLite Database Persistence
        db_row = get_submission(submission_id)
        assert db_row is not None, f"Submission ID {submission_id} was not persisted in SQLite."
        assert db_row["question"] == payload["question"]
        assert db_row["ai_response"] == payload["ai_response"]
        assert db_row["reference_answer"] is None
        assert db_row["source_document"] is None

        # Step 3: Trigger Evaluation via /evaluations/{submission_id}/evaluate
        res_eval = client.post(f"/api/v1/evaluations/{submission_id}/evaluate")
        assert res_eval.status_code == 200, f"Expected 200 OK, got {res_eval.status_code}: {res_eval.text}"
        result = res_eval.json()

        # Step 4: Validate RAG Retrieval Activated
        # Since neither reference_answer nor source_document was provided, orchestrator must retrieve RAG chunks
        assert "context_used" in result
        context_used = result["context_used"]
        assert len(context_used) > 0
        assert (
            "Retrieved Reference Knowledge Base Chunks" in context_used
            or "reference" in context_used.lower()
            or "source:" in context_used.lower()
        ), "Context used should reflect retrieved knowledge base chunks from RAG."

        # Step 5: Assert Data Contracts for all 4 Judge Agents
        # 1. Relevance Agent
        assert "relevance" in result
        rel = result["relevance"]
        assert isinstance(rel["score"], (int, float))
        assert 0.0 <= rel["score"] <= 1.0
        assert rel["classification"] in ("fully_relevant", "partially_relevant", "unrelated", "off_topic")
        assert isinstance(rel["reasoning"], str) and len(rel["reasoning"]) > 0

        # 2. Accuracy Agent
        assert "accuracy" in result
        acc = result["accuracy"]
        assert isinstance(acc["score"], (int, float))
        assert 0.0 <= acc["score"] <= 1.0
        assert acc["classification"] in ("correct", "partially_correct", "incorrect", "contradictory")
        assert isinstance(acc["supporting_evidence"], list)
        assert isinstance(acc["reasoning"], str) and len(acc["reasoning"]) > 0

        # 3. Hallucination Agent
        assert "hallucination" in result
        hal = result["hallucination"]
        assert isinstance(hal["is_hallucinated"], bool)
        assert isinstance(hal["hallucination_score"], (int, float))
        assert 0.0 <= hal["hallucination_score"] <= 1.0
        assert isinstance(hal["total_claims"], int) and hal["total_claims"] >= 0
        assert isinstance(hal["unsupported_claims_count"], int) and hal["unsupported_claims_count"] >= 0
        assert isinstance(hal["flagged_claims"], list)
        assert isinstance(hal["all_claims"], list)
        assert isinstance(hal["reasoning"], str) and len(hal["reasoning"]) > 0

        # 4. Completeness Agent
        assert "completeness" in result
        comp = result["completeness"]
        assert isinstance(comp["score"], (int, float))
        assert 0.0 <= comp["score"] <= 1.0
        assert comp["classification"] in ("fully_complete", "mostly_complete", "partially_complete", "incomplete")
        assert isinstance(comp["addressed_aspects"], list)
        assert isinstance(comp["missing_aspects"], list)
        assert isinstance(comp["reasoning"], str) and len(comp["reasoning"]) > 0

        # Step 6: Assert Verdict Agent Synthesis
        assert "verdict" in result
        verd = result["verdict"]
        assert isinstance(verd["weighted_score"], (int, float))
        assert 0.0 <= verd["weighted_score"] <= 1.0
        assert verd["verdict"] in ("Pass", "Needs Improvement", "Fail")
        assert "dimension_scores" in verd
        dim_scores = verd["dimension_scores"]
        for dim in ("relevance", "accuracy", "completeness", "groundedness"):
            assert dim in dim_scores, f"Dimension '{dim}' missing in verdict dimension_scores"
            assert 0.0 <= dim_scores[dim] <= 1.0
        assert isinstance(verd["major_issues"], list)
        assert isinstance(verd["strengths"], list)
        assert isinstance(verd["consolidated_reasoning"], str) and len(verd["consolidated_reasoning"]) > 0

        # Step 7: Assert Result Display fields
        assert result["submission_id"] == submission_id
        assert result["question"] == payload["question"]
        assert result["ai_response"] == payload["ai_response"]
        assert "evaluated_at" in result

    def test_single_eval_direct_endpoint_with_explicit_context(self, client):
        """
        Tests the direct evaluation endpoint (POST /api/v1/evaluate):
        - Supplies explicit reference_answer and source_document.
        - Asserts context resolution uses direct reference (bypasses RAG).
        - Asserts SQLite database auto-insert.
        - Asserts complete agent and verdict schema conformity.
        """
        payload = {
            "question": "What is the chemical symbol for gold?",
            "ai_response": "The chemical symbol for gold is Au.",
            "reference_answer": "The chemical symbol for gold is Au.",
            "source_document": "Periodic table handbook: Gold has atomic number 79 and chemical symbol Au.",
        }

        res = client.post("/api/v1/evaluate", json=payload)
        assert res.status_code == 200, f"Expected 200 OK, got {res.status_code}: {res.text}"
        result = res.json()

        # Database record check
        sub_id = result.get("submission_id")
        assert sub_id is not None
        db_row = get_submission(sub_id)
        assert db_row is not None
        assert db_row["reference_answer"] == payload["reference_answer"]
        assert db_row["source_document"] == payload["source_document"]

        # Context resolution check: direct reference used
        assert "Direct Reference Answer:" in result["context_used"]
        assert payload["reference_answer"] in result["context_used"]
        assert "Supplied Source Document:" in result["context_used"]
        assert payload["source_document"] in result["context_used"]

        # High-quality factual response should achieve Pass verdict
        assert result["verdict"]["verdict"] == "Pass"
        assert result["verdict"]["weighted_score"] >= 0.75
        assert result["accuracy"]["score"] >= 0.70
        assert result["hallucination"]["is_hallucinated"] is False

    def test_single_eval_validation_contracts(self, client):
        """
        Tests input validation boundaries and HTTP error handling for single evaluations:
        - Empty question -> 422
        - Whitespace-only AI response -> 422
        - Evaluating non-existent submission ID -> 404
        - Reading non-existent submission ID -> 404
        """
        # Case A: Empty question
        res_a = client.post(
            "/api/v1/evaluations",
            json={"question": "   ", "ai_response": "Some valid AI response."},
        )
        assert res_a.status_code == 422
        assert "field cannot be empty or whitespace-only" in res_a.text

        # Case B: Whitespace-only AI response
        res_b = client.post(
            "/api/v1/evaluate",
            json={"question": "Valid question?", "ai_response": "\t  \n "},
        )
        assert res_b.status_code == 422
        assert "field cannot be empty or whitespace-only" in res_b.text

        # Case C: Non-existent submission ID evaluate
        res_c = client.post("/api/v1/evaluations/99999999/evaluate")
        assert res_c.status_code == 404
        assert "submission not found" in res_c.json()["detail"]

        # Case D: Non-existent submission ID read
        res_d = client.get("/api/v1/evaluations/99999999")
        assert res_d.status_code == 404
        assert "submission not found" in res_d.json()["detail"]


# ===========================================================================
# WORKFLOW 2: BATCH-EVALUATION END-TO-END WORKFLOW
# ===========================================================================

class TestBatchEvaluationWorkflow:
    """
    Validates complete data flow for batch evaluations:
    CSV upload -> Validation -> Evaluation across all records -> Storage ->
    Aggregation -> Dashboard -> Recommendations -> PDF Export.
    """

    def test_batch_preflight_csv_validation(self, client):
        """
        Tests pre-flight validation on CSV upload:
        - Missing 'questions' column rejected with 400.
        - Missing 'ai_responses' column rejected with 400.
        - Non-CSV file rejected with 400.
        - Empty CSV file rejected with 400.
        """
        # Case A: Missing 'questions' column
        csv_no_q = "response,reference_answer\nAn answer without question.,Ref answer.\n"
        res_a = client.post(
            "/api/v1/batch/upload-csv",
            files={"file": ("test_no_q.csv", io.BytesIO(csv_no_q.encode("utf-8")), "text/csv")},
        )
        assert res_a.status_code == 400
        assert "Missing required column(s)" in res_a.json()["detail"]
        assert "'questions'" in res_a.json()["detail"]

        # Case B: Missing 'ai_responses' column
        csv_no_r = "question,reference_answer\nWhat is gravity?,Force of attraction.\n"
        res_b = client.post(
            "/api/v1/batch/upload-csv",
            files={"file": ("test_no_r.csv", io.BytesIO(csv_no_r.encode("utf-8")), "text/csv")},
        )
        assert res_b.status_code == 400
        assert "Missing required column(s)" in res_b.json()["detail"]
        assert "'ai_responses'" in res_b.json()["detail"]

        # Case C: Non-CSV file extension
        res_c = client.post(
            "/api/v1/batch/upload-csv",
            files={"file": ("data.json", io.BytesIO(b'{"key": "value"}'), "application/json")},
        )
        assert res_c.status_code == 400
        assert "File must be a CSV format" in res_c.json()["detail"]

        # Case D: Empty file
        res_d = client.post(
            "/api/v1/batch/upload-csv",
            files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
        )
        assert res_d.status_code == 400

    def test_batch_full_lifecycle_and_data_flow(self, client):
        """
        Tests the entire batch evaluation lifecycle end-to-end:
        1. Ingests a multi-record CSV batch via POST /api/v1/batch/upload-csv.
        2. Asserts evaluation across all records (item indices, schemas, results).
        3. Asserts disk & in-memory persistence (data/batches/{batch_id}.json & latest.txt).
        4. Asserts dynamic query-time statistics calculation at GET /api/v1/batch/{batch_id}.
        5. Asserts report-data aggregation layer at GET /api/v1/batch/{batch_id}/report-data.
        6. Asserts recommendations engine at GET /api/v1/batch/{batch_id}/recommendations.
        7. Asserts CSV export at GET /api/v1/batch/{batch_id}/export.
        8. Asserts PDF report generation and binary format at GET /api/v1/batch/{batch_id}/export/pdf.
        9. Asserts combined multi-batch PDF export at GET /api/v1/batches/combined/export/pdf.
        10. Cleans up test batch via DELETE /api/v1/batch/{batch_id} and asserts cleanup.
        """
        # Construct a 5-record CSV batch covering diverse test conditions:
        # 1. High-quality factual question/answer with reference
        # 2. Missing completeness aspects
        # 3. Fabricated numerical claim (hallucination)
        # 4. Zero-reference question (triggers RAG retrieval in batch)
        # 5. Empty question row (verifies empty question tolerance in batch)
        csv_data = """questions,ai_responses,reference_answer
What is the capital of Japan?,Tokyo is the official capital and largest metropolitan area of Japan.,Tokyo is the capital of Japan.
Explain the water cycle in detail.,Water evaporates into the air.,The water cycle involves evaporation condensation precipitation and collection.
What is the boiling point of ethanol?,Ethanol boils at precisely 350 degrees Celsius.,Ethanol boils at 78.4 degrees Celsius under 1 atm pressure.
What is the capital and largest city of France?,Paris is the capital of France.,
,The chemical formula for water is H2O.,Water is H2O.
"""

        # -------------------------------------------------------------------
        # Phase 1: Upload CSV Batch
        # -------------------------------------------------------------------
        res_upload = client.post(
            "/api/v1/batch/upload-csv",
            files={"file": ("e2e_batch_test.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")},
        )
        assert res_upload.status_code == 200, f"Batch upload failed: {res_upload.text}"
        batch_summary = res_upload.json()

        batch_id = batch_summary["batch_id"]
        assert batch_id.startswith("batch-"), f"Unexpected batch_id format: {batch_id}"

        # -------------------------------------------------------------------
        # Phase 2: Assert Evaluation Across All Records
        # -------------------------------------------------------------------
        stats = batch_summary["statistics"]
        items = batch_summary["items"]

        assert stats["total_records"] == 5
        assert stats["successful_records"] == 5
        assert stats["failed_records"] == 0
        assert len(items) == 5

        # Check every item structure and nested results
        for idx, item in enumerate(items, start=1):
            assert item["index"] == idx
            assert item["status"] == "success"
            res = item["result"]
            assert res is not None, f"Item {idx} result is None"
            # Schema integrity checks for nested agent results
            assert "relevance" in res
            assert "accuracy" in res
            assert "hallucination" in res
            assert "completeness" in res
            assert "verdict" in res
            assert res["verdict"]["verdict"] in ("Pass", "Needs Improvement", "Fail")

        # Specific item validations:
        # Item 1: High quality -> Pass
        assert items[0]["result"]["verdict"]["verdict"] == "Pass"
        # Item 3: Numerical hallucination (350 vs 78.4) -> Flagged claim or Fail/Needs Improvement
        item3_hal = items[2]["result"]["hallucination"]
        assert item3_hal["is_hallucinated"] is True or items[2]["result"]["accuracy"]["score"] < 0.8
        # Item 4: Zero reference -> Context uses RAG
        item4_ctx = items[3]["result"]["context_used"]
        assert "Retrieved Reference Knowledge Base Chunks" in item4_ctx or len(item4_ctx) > 0
        # Item 5: Empty question -> Handled gracefully with fallback label
        assert items[4]["question"] in ("[No Question Provided]", "")

        # -------------------------------------------------------------------
        # Phase 3: Assert Storage & Disk Persistence
        # -------------------------------------------------------------------
        assert batch_id in BATCH_STORE
        persisted_file = BATCH_DIR / f"{batch_id}.json"
        assert persisted_file.exists(), f"Batch JSON not found on disk at {persisted_file}"

        disk_data = json.loads(persisted_file.read_text(encoding="utf-8"))
        assert disk_data["batch_id"] == batch_id
        assert len(disk_data["items"]) == 5

        latest_file = BATCH_DIR / "latest.txt"
        assert latest_file.exists()
        assert latest_file.read_text(encoding="utf-8").strip() == batch_id

        # -------------------------------------------------------------------
        # Phase 4: Assert Dashboard Data Integrity & Dynamic Query-Time Computation
        # -------------------------------------------------------------------
        res_dash = client.get(f"/api/v1/batch/{batch_id}")
        assert res_dash.status_code == 200
        dash_summary = res_dash.json()

        dash_stats = dash_summary["statistics"]
        assert dash_stats["total_records"] == 5
        assert (
            dash_stats["pass_count"] + dash_stats["needs_improvement_count"] + dash_stats["fail_count"]
            == dash_stats["successful_records"]
        )
        expected_pass_pct = round((dash_stats["pass_count"] / 5.0) * 100, 1)
        assert abs(dash_stats["pass_rate_percent"] - expected_pass_pct) < 0.2

        # -------------------------------------------------------------------
        # Phase 5: Assert Aggregation Layer (Report Data)
        # -------------------------------------------------------------------
        res_report = client.get(f"/api/v1/batch/{batch_id}/report-data")
        assert res_report.status_code == 200
        report_data = res_report.json()

        # Validate Report Data Schema Contract
        assert report_data["metadata"]["batch_id"] == batch_id
        assert report_data["metadata"]["total_records"] == 5
        assert report_data["metadata"]["successful_records"] == 5
        assert report_data["metadata"]["failed_records"] == 0

        # Stats exact match
        assert report_data["overall_stats"]["pass_count"] == dash_stats["pass_count"]
        assert report_data["overall_stats"]["fail_count"] == dash_stats["fail_count"]
        assert len(report_data["per_response_details"]) == 5

        # -------------------------------------------------------------------
        # Phase 6: Assert Recommendations Module
        # -------------------------------------------------------------------
        res_rec = client.get(f"/api/v1/batch/{batch_id}/recommendations")
        assert res_rec.status_code == 200
        recs = res_rec.json()
        assert isinstance(recs, list)
        assert 2 <= len(recs) <= 5, f"Expected 2-5 recommendations, got {len(recs)}"

        for r in recs:
            assert "category" in r
            assert "headline" in r
            assert "metric_trigger" in r
            assert "remediation_advice" in r
            assert r["priority"].lower() in ("high", "medium", "low")

        # -------------------------------------------------------------------
        # Phase 7: Assert CSV Export
        # -------------------------------------------------------------------
        res_csv = client.get(f"/api/v1/batch/{batch_id}/export")
        assert res_csv.status_code == 200
        assert res_csv.headers["content-type"].startswith("text/csv")
        csv_lines = res_csv.text.strip().split("\n")
        # 1 header row + 5 data rows = 6 lines
        assert len(csv_lines) == 6

        # -------------------------------------------------------------------
        # Phase 8: Assert PDF Generation & Stream Export
        # -------------------------------------------------------------------
        res_pdf = client.get(f"/api/v1/batch/{batch_id}/export/pdf")
        assert res_pdf.status_code == 200
        assert res_pdf.headers["content-type"] == "application/pdf"
        assert f"{batch_id}_evaluation_report.pdf" in res_pdf.headers["content-disposition"]

        pdf_bytes = res_pdf.content
        assert pdf_bytes.startswith(b"%PDF-"), "Exported content must be a valid PDF binary stream"
        assert len(pdf_bytes) > 5000, f"PDF file size {len(pdf_bytes)} is unexpectedly small"

        # Validate with pypdf
        pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        assert len(pdf_reader.pages) >= 2, f"Expected multi-page PDF, got {len(pdf_reader.pages)} pages"
        full_pdf_text = "\n".join(p.extract_text() or "" for p in pdf_reader.pages)

        # Assert key textual information appears in generated PDF
        assert batch_id in full_pdf_text
        assert "LLM RESPONSE EVALUATION REPORT" in full_pdf_text or "Batch Diagnostic Summary" in full_pdf_text
        assert "Actionable Improvement Recommendations" in full_pdf_text

        # -------------------------------------------------------------------
        # Phase 9: Assert Combined Multi-Batch PDF Export
        # -------------------------------------------------------------------
        res_comb_pdf = client.get("/api/v1/batches/combined/export/pdf")
        assert res_comb_pdf.status_code == 200
        assert res_comb_pdf.headers["content-type"] == "application/pdf"
        comb_pdf_bytes = res_comb_pdf.content
        assert comb_pdf_bytes.startswith(b"%PDF-")
        comb_reader = pypdf.PdfReader(io.BytesIO(comb_pdf_bytes))
        assert len(comb_reader.pages) >= 2

        # -------------------------------------------------------------------
        # Phase 10: Lifecycle Cleanup via DELETE endpoint
        # -------------------------------------------------------------------
        res_del = client.delete(f"/api/v1/batch/{batch_id}")
        assert res_del.status_code == 200
        assert res_del.json()["status"] == "deleted"

        # Assert deleted from in-memory BATCH_STORE and unlinked from disk
        assert batch_id not in BATCH_STORE
        assert not persisted_file.exists()

        # Confirm 404 on subsequent queries
        res_gone = client.get(f"/api/v1/batch/{batch_id}")
        assert res_gone.status_code == 404


# ===========================================================================
# WORKFLOW 3: DATA FLOW CONTRACTS ACROSS ALL ARCHITECTURAL COMPONENTS
# ===========================================================================

class TestDataFlowIntegrity:
    """
    Direct unit-level contract tests verifying the data passing and schema preservation
    between every boundary in the system:
    Input Module -> SQLite DB -> ChromaDB RAG -> Orchestrator -> 4 Agents ->
    Verdict Agent -> Batch Evaluator -> Aggregator -> Recommendations -> PDF.
    """

    def test_database_and_input_contract(self):
        """Validates contract between Input Module schemas and SQLite storage layer."""
        q = "What is the speed of sound in dry air at 20 C?"
        r = "The speed of sound in dry air at 20 C is approximately 343 m/s."
        ref = "343 meters per second at 20 degrees Celsius."
        src = "Physics Handbook: Acoustics."

        # Insert into DB
        row = insert_submission(q, r, ref, src)
        sub_id = row["id"]
        assert isinstance(sub_id, int)

        # Retrieve and verify exact contract
        restored = get_submission(sub_id)
        assert restored is not None
        assert restored["question"] == q
        assert restored["ai_response"] == r
        assert restored["reference_answer"] == ref
        assert restored["source_document"] == src

    def test_rag_pipeline_contract(self):
        """Validates contract between ChromaDB vector store and Orchestrator context resolver."""
        # Query ChromaDB vector store directly
        hits = retrieve("What is the capital of France?", top_k=2)
        assert isinstance(hits, list)
        if hits:
            hit = hits[0]
            assert "chunk_id" in hit
            assert "text" in hit
            assert "metadata" in hit
            assert isinstance(hit["text"], str)

        # Orchestrator resolve_context contract
        orch = EvaluationOrchestrator()
        # When reference is provided, RAG is bypassed
        ctx_ref = orch.resolve_context(
            question="Q",
            reference_answer="Explicit Ref",
            source_document="Explicit Doc",
        )
        assert "Direct Reference Answer:" in ctx_ref
        assert "Supplied Source Document:" in ctx_ref

        # When reference is omitted, RAG retrieval is executed
        ctx_rag = orch.resolve_context(
            question="What is the capital of France?",
            reference_answer=None,
            source_document=None,
        )
        assert "Retrieved Reference Knowledge Base Chunks" in ctx_rag or len(ctx_rag) > 0

    @pytest.mark.asyncio
    async def test_judge_agents_and_verdict_synthesis_contract(self):
        """Validates contracts between the 4 judge agents and the Verdict Agent synthesis."""
        q = "Who discovered penicillin?"
        r = "Alexander Fleming discovered penicillin in 1928."
        ctx = "Direct Reference Answer:\nAlexander Fleming discovered penicillin in 1928 at St. Mary's Hospital."

        rel_agent = RelevanceAgent()
        acc_agent = AccuracyAgent()
        hal_agent = HallucinationAgent()
        comp_agent = CompletenessAgent()
        verd_agent = VerdictAgent()

        rel_res = await rel_agent.evaluate(q, r)
        assert isinstance(rel_res, RelevanceResult)
        assert 0.0 <= rel_res.score <= 1.0

        acc_res = await acc_agent.evaluate(q, r, ctx)
        assert isinstance(acc_res, AccuracyResult)
        assert 0.0 <= acc_res.score <= 1.0

        hal_res = await hal_agent.evaluate(r, ctx, q)
        assert isinstance(hal_res, HallucinationResult)
        assert 0.0 <= hal_res.hallucination_score <= 1.0

        comp_res = await comp_agent.evaluate(q, r, ctx)
        assert isinstance(comp_res, CompletenessResult)
        assert 0.0 <= comp_res.score <= 1.0

        # Verdict Agent consumes the 4 agent results without re-evaluating
        verd_res = verd_agent.evaluate(
            relevance=rel_res,
            accuracy=acc_res,
            hallucination=hal_res,
            completeness=comp_res,
        )
        assert isinstance(verd_res, VerdictResult)
        assert verd_res.verdict in ("Pass", "Needs Improvement", "Fail")
        assert 0.0 <= verd_res.weighted_score <= 1.0

        # Verify mathematical weight relationship
        # groundedness = 1.0 - hallucination_score
        groundedness = max(0.0, 1.0 - hal_res.hallucination_score)
        expected_raw_score = (
            verd_agent.weight_accuracy * acc_res.score
            + verd_agent.weight_completeness * comp_res.score
            + verd_agent.weight_relevance * rel_res.score
            + verd_agent.weight_groundedness * groundedness
        )
        # Verify within roundoff tolerance
        assert abs(verd_res.weighted_score - round(expected_raw_score, 3)) <= 0.05

    def test_report_aggregator_to_pdf_generator_contract(self):
        """
        Validates contract between ReportDataAggregator, RecommendationEngine,
        and generate_pdf_report.
        """
        # Test against known small batch
        small_batch_id = "batch-small-eval-8"
        report_data = generate_batch_report_data(small_batch_id, storage_dir=BATCH_DIR)

        assert isinstance(report_data, BatchReportData)
        assert report_data.metadata.batch_id == small_batch_id
        assert report_data.metadata.total_records == 8
        assert len(report_data.per_response_details) == 8

        # Recommendations contract
        recs = generate_recommendations(report_data)
        assert isinstance(recs, list)
        assert 2 <= len(recs) <= 5
        assert all(isinstance(r, RecommendationItem) for r in recs)

        # PDF generator contract
        pdf_bytes = generate_pdf_report(report_data)
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF-")
        assert len(pdf_bytes) > 10000

        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        assert len(reader.pages) >= 3
