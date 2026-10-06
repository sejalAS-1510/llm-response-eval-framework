# LLM Response Evaluation Framework — Technical Documentation

Welcome to the comprehensive technical documentation for the **LLM Response Evaluation Framework**. This multi-agent framework provides rigorous, automated, multi-dimensional evaluation of Large Language Model responses against ground-truth references and retrieved domain knowledge.

---

## Documentation Navigation

This documentation is organized into modular topic guides covering every layer of the architecture, from data ingestion to PDF generation:

| # | Topic | Documentation File | Key Focus Areas |
| :---: | :--- | :--- | :--- |
| **01** | **[System Architecture](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/01-system-architecture.md)** | [`01-system-architecture.md`](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/01-system-architecture.md) | End-to-end component diagram, data flow from UI/CSV to ReportLab, subsystem roles. |
| **02** | **[Agents Responsibility & Workflow](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/02-agents-workflow.md)** | [`02-agents-workflow.md`](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/02-agents-workflow.md) | Relevance, Accuracy, Hallucination, Completeness, Verdict Agent, Orchestrator, Batch Evaluator. |
| **03** | **[Scoring Model & Verdict Methodology](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/03-scoring-and-verdict.md)** | [`03-scoring-and-verdict.md`](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/03-scoring-and-verdict.md) | 0.0–1.0 continuous scales, weighted composite formula, Pass/Needs Improvement/Fail thresholds, critical failure overrides. |
| **04** | **[RAG Pipeline Details](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/04-rag-pipeline.md)** | [`04-rag-pipeline.md`](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/04-rag-pipeline.md) | TruthfulQA & SQuAD ingestion, recursive character chunking (500/80), `all-MiniLM-L6-v2` embeddings, ChromaDB vector store, semantic cosine retrieval, dynamic fallback. |
| **05** | **[Data Models & Schemas](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/05-data-models.md)** | [`05-data-models.md`](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/05-data-models.md) | Pydantic schema hierarchy: single evaluation requests, agent outputs, batch records, summary statistics, report aggregator objects. |
| **06** | **[API Endpoints & Backend Services](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/06-api-and-services.md)** | [`06-api-and-services.md`](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/06-api-and-services.md) | FastAPI REST API specification (`/evaluate`, `/evaluate/batch`, `/api/v1/batch/*`, `/api/v1/batch/*/export/*`), SQLite and JSON stores. |
| **07** | **[CSV Upload Format & Ingestion](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/07-csv-upload-format.md)** | [`07-csv-upload-format.md`](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/07-csv-upload-format.md) | Accepted column names, aliases, delimiter sniffing, BOM handling, row validation, empty-question resilience, error isolation. |
| **08** | **[Dashboard Metric Calculation](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/08-dashboard-metrics.md)** | [`08-dashboard-metrics.md`](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/08-dashboard-metrics.md) | Mathematical formulas for Pass Rate %, Hallucination Rate %, dimensional score averages, groundedness complement, trend trajectory analysis. |
| **09** | **[PDF Report Structure & Generation](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/09-pdf-report-structure.md)** | [`09-pdf-report-structure.md`](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/09-pdf-report-structure.md) | ReportLab layout engine, NumberedCanvas two-pass pagination, Cover/Summary page, dynamic recommendations engine, per-response cards. |
| **10** | **[Deployment & Production Guide](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/10-deployment-guide.md)** | [`10-deployment-guide.md`](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/10-deployment-guide.md) | Local virtual environment, Docker containerization, Docker Compose, cloud hosting (Render, Railway, Hugging Face Spaces), health checks. |
| **★** | **[Final Project Report](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/FINAL_PROJECT_REPORT.md)** | [`FINAL_PROJECT_REPORT.md`](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/docs/FINAL_PROJECT_REPORT.md) | Comprehensive executive narrative covering objectives, methodology, empirical testing results, limitations, and future roadmap. |

---

## Architectural Highlights

```
                                    +----------------------------------+
                                    |    User Browser / Dashboard     |
                                    +-----------------+----------------+
                                                      |
                                                      v
                                        +-------------+-------------+
                                        | FastAPI Input & Routing   |
                                        +-------------+-------------+
                                                      |
                         +----------------------------+----------------------------+
                         |                                                         |
                         v                                                         v
             [Single Submission Mode]                                      [Batch Upload Mode]
                         |                                                         |
                         |                                                         v
                         |                                           +-------------+-------------+
                         |                                           | BatchEvaluator Ingestion  |
                         |                                           +-------------+-------------+
                         |                                                         |
                         +----------------------------+----------------------------+
                                                      |
                                                      v
                                        +-------------+-------------+
                                        |  EvaluationOrchestrator   |
                                        +-------------+-------------+
                                                      |
                          +---------------------------+---------------------------+
                          | Context Resolution: Reference Answer OR RAG ChromaDB  |
                          +---------------------------+---------------------------+
                                                      |
                         +----------------------------+----------------------------+
                         |                            |                            |
                         v                            v                            v
               +---------+--------+         +---------+--------+         +---------+--------+
               | Relevance Agent  |         |  Accuracy Agent  |         | Completeness Agt |
               +---------+--------+         +---------+--------+         +---------+--------+
                         |                            |                            |
                         +----------------------------+----------------------------+
                                                      |
                                                      v
                                        +-------------+-------------+
                                        | Hallucination Agent       |
                                        | (Claim Decomposition)     |
                                        +-------------+-------------+
                                                      |
                                                      v
                                        +-------------+-------------+
                                        | Verdict Agent             |
                                        | (Synthesis & Overrides)   |
                                        +-------------+-------------+
                                                      |
                         +----------------------------+----------------------------+
                         |                                                         |
                         v                                                         v
             +-----------+------------+                              +-------------+-------------+
             | SQLite & JSON Storage  |                              | BatchReportAggregator     |
             +------------------------+                              +-------------+-------------+
                                                                                   |
                                                                                   v
                                                                     +-------------+-------------+
                                                                     | PDFReportGenerator        |
                                                                     | (ReportLab Executive PDF) |
                                                                     +---------------------------+
```

---

## Technology Stack Summary

- **Web Framework**: FastAPI (Python 3.11+), Pydantic v2
- **Vector Retrieval**: ChromaDB (Embedded local persistent vector store)
- **Embedding Model**: `all-MiniLM-L6-v2` via `sentence-transformers` (384-dimensional dense vectors)
- **Text Chunking**: LangChain `RecursiveCharacterTextSplitter` (chunk size: 500, overlap: 80)
- **Reference Datasets**: TruthfulQA (short QA factuality), SQuAD (long-context passage retrieval)
- **Judge LLM**: Google Gemini API (`gemini-1.5-flash` / `gemini-1.5-pro`) with deterministic heuristic fallback
- **Storage**: SQLite (`submissions.db`) for interactive queries, JSON file storage (`data/batches/`) for high-volume batches
- **PDF Generation**: ReportLab Platypus engine with custom `NumberedCanvas` ("Page X of Y"), Flowable tables, and SVG/Pie graphics
- **Frontend Dashboard**: Single-page responsive HTML5/Tailwind-styled interface with interactive KPI grids, Chart.js graphs, and PDF/CSV export controls
