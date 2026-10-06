# API Endpoints & Backend Services

## Overview

The framework exposes a RESTful HTTP API built with FastAPI, enabling programmatic evaluation, automated CI/CD integration, batch CSV processing, real-time KPI retrieval, and on-demand PDF report streaming.

---

## Endpoint Catalog

| Method | Endpoint | Description | Consumes | Produces |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive HTML5 frontend dashboard | — | `text/html` |
| `POST` | `/evaluate` | Evaluates a single question-response pair | `application/json` | `application/json` |
| `POST` | `/evaluate/batch` | Uploads and processes a multi-row CSV batch | `multipart/form-data` | `application/json` |
| `GET` | `/api/v1/batches` | Lists all historical evaluation batches with summary KPIs | — | `application/json` |
| `GET` | `/api/v1/batch/{batch_id}` | Retrieves full batch details, record items, and statistics | — | `application/json` |
| `DELETE` | `/api/v1/batch/{batch_id}` | Deletes a batch and its persisted JSON record | — | `application/json` |
| `GET` | `/api/v1/batch/{batch_id}/report-data` | Retrieves the aggregated report-data object for reporting | — | `application/json` |
| `GET` | `/api/v1/batch/{batch_id}/export/pdf` | Generates and streams the publication-grade PDF report | — | `application/pdf` |
| `GET` | `/api/v1/batch/{batch_id}/export/csv` | Downloads structured CSV summary of evaluated items | — | `text/csv` |
| `GET` | `/api/v1/evaluation/{eval_id}/export/pdf`| Exports a single submission evaluation to PDF | — | `application/pdf` |
| `GET` | `/api/v1/trends` | Retrieves quality trajectories across historical batches | — | `application/json` |

---

## Detailed Endpoint Reference

### 1. Single Evaluation (`POST /evaluate`)
Evaluates an individual question-response pair across all 4 agents and synthesizes the verdict.

#### Request Body
```json
{
  "question": "What is the capital of France?",
  "ai_response": "The capital of France is Paris.",
  "reference_answer": "Paris is the capital of France.",
  "source_document": null,
  "model_name": "gpt-4o"
}
```

#### Response (`200 OK`)
```json
{
  "submission_id": 42,
  "question": "What is the capital of France?",
  "ai_response": "The capital of France is Paris.",
  "context_used": "Direct Reference Answer:\nParis is the capital of France.",
  "relevance": {
    "score": 1.0,
    "classification": "fully_relevant",
    "reasoning": "Directly and correctly answers the question asked."
  },
  "accuracy": {
    "score": 1.0,
    "classification": "correct",
    "supporting_evidence": ["Paris is the capital of France."],
    "reasoning": "Factual claim matches reference answer."
  },
  "hallucination": {
    "is_hallucinated": false,
    "hallucination_score": 0.0,
    "total_claims": 1,
    "unsupported_claims_count": 0,
    "flagged_claims": [],
    "all_claims": [
      {
        "claim": "The capital of France is Paris.",
        "status": "supported",
        "evidence": "Paris is the capital of France.",
        "explanation": "Supported by reference answer."
      }
    ],
    "reasoning": "All statements verified against reference."
  },
  "completeness": {
    "score": 1.0,
    "classification": "fully_complete",
    "identified_requirements": ["Identify capital of France"],
    "addressed_aspects": ["Identified Paris as capital"],
    "partially_addressed_aspects": [],
    "missing_aspects": [],
    "reasoning": "Fully answers the question prompt."
  },
  "verdict": {
    "weighted_score": 1.0,
    "verdict": "Pass",
    "dimension_scores": {
      "relevance": 1.0,
      "accuracy": 1.0,
      "completeness": 1.0,
      "groundedness": 1.0
    },
    "major_issues": [],
    "strengths": [
      "High Relevance: Directly addresses the user's question and prompt parameters.",
      "High Accuracy: Factual statements align tightly with ground truth evidence.",
      "Grounded Content: Zero hallucinations or ungrounded assertions detected.",
      "Comprehensive Coverage: Answers all sub-questions and key requirements."
    ],
    "consolidated_reasoning": "OVERALL VERDICT: PASS (Weighted Score: 100.0%). The response meets quality standards across all evaluation dimensions."
  },
  "evaluated_at": "2026-09-26T18:00:00.000000+00:00"
}
```

---

### 2. Batch Evaluation (`POST /evaluate/batch`)
Uploads a CSV file for high-throughput batch evaluation.

#### Request Form Data
- `file`: Multipart file upload containing CSV text.

#### Response (`200 OK`)
Returns a `BatchEvaluationSummary` containing the assigned `batch_id`, overall `statistics`, and item array.

```json
{
  "batch_id": "batch-8f92a10b",
  "filename": "qa_eval_set.csv",
  "created_at": "2026-09-26T18:05:00.000000+00:00",
  "statistics": {
    "total_records": 50,
    "successful_records": 50,
    "failed_records": 0,
    "pass_count": 42,
    "needs_improvement_count": 6,
    "fail_count": 2,
    "pass_rate_percent": 84.0,
    "average_weighted_score": 0.884,
    "average_relevance_score": 0.940,
    "average_accuracy_score": 0.895,
    "average_completeness_score": 0.870,
    "average_groundedness_score": 0.920,
    "hallucination_rate_percent": 8.0
  },
  "items": [...]
}
```

---

### 3. Report-Data Aggregation Layer (`GET /api/v1/batch/{batch_id}/report-data`)
Retrieves the structured aggregation payload that feeds the PDF report generator.

#### Query Parameters
- `batch_id`: Unique batch identifier or `"all-batches-combined"` to aggregate across all historical records.

#### Response (`200 OK`)
Returns a `BatchReportData` object containing:
- `metadata`: Batch metadata, execution timestamp, row counts.
- `overall_stats`: Normalized dimensional averages (both 0.0-1.0 and 0-100%).
- `hallucination_frequency`: Total claims extracted, ungrounded claims, and sample flagged assertions.
- `per_response_details`: Array of `PerResponseReportDetail` records.
- `recommendations`: Dynamically generated actionable recommendations with empirical triggers.

---

### 4. PDF Report Export (`GET /api/v1/batch/{batch_id}/export/pdf`)
Renders the executive PDF evaluation report using ReportLab and streams the binary file.

#### Response Headers
- `Content-Type`: `application/pdf`
- `Content-Disposition`: `attachment; filename="evaluation_report_batch-8f92a10b.pdf"`

---

## Backend Services Architecture

The application logic is decoupled into discrete, single-responsibility services:

```
┌────────────────────────────────────────────────────────┐
│                   InputModule (FastAPI)                 │
└──────────────────────────┬─────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
┌────────▼────────┐ ┌──────▼────────┐ ┌──────▼────────┐
│  SQLiteStorage  │ │ BatchEvaluator│ │ ReportService │
│ (submissions.db)│ │(Ingestion/Run)│ │ (Aggregation) │
└─────────────────┘ └──────┬────────┘ └──────┬────────┘
                           │                 │
                           │         ┌───────▼────────┐
                           │         │ Recommendations│
                           │         │  Engine (Scan) │
                           │         └───────┬────────┘
                           │                 │
                           │         ┌───────▼────────┐
                           │         │ PDFReportGen   │
                           │         │  (ReportLab)   │
                           │         └────────────────┘
                           ▼
               ┌───────────────────────┐
               │ EvaluationOrchestrator│
               └───────────────────────┘
```

1. **`SQLiteStorage` ([`src/input_module/storage.py`](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/src/input_module/storage.py))**: Manages relational records for single prompt submissions and retrieval.
2. **`BatchEvaluator` ([`src/agents/batch_evaluator.py`](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/src/agents/batch_evaluator.py))**: Validates CSV schemas, controls concurrency throttles, and calculates batch KPIs.
3. **`ReportService` ([`src/reporting/report_service.py`](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/src/reporting/report_service.py))**: Resolves batch summaries from disk or memory and constructs aggregated `BatchReportData` models.
4. **`RecommendationEngine` ([`src/reporting/recommendations.py`](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/src/reporting/recommendations.py))**: Evaluates aggregate weaknesses against empirical thresholds to construct plain-English engineering actions.
5. **`PDFReportGenerator` ([`src/reporting/pdf_generator.py`](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/src/reporting/pdf_generator.py))**: Compiles ReportLab Flowable elements, tables, and charts into a PDF file with dynamic two-pass pagination.
