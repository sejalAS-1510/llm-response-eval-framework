# LLM Response Evaluation Platform

<div align="center">

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render%20Cloud-46E3B7.svg?style=for-the-badge&logo=render)](https://llm-response-evaluation-platform.onrender.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=for-the-badge&logo=docker)](https://www.docker.com/)
[![Tests Passing](https://img.shields.io/badge/Tests-86%2F86%20Passed%20(100%25)-brightgreen.svg?style=for-the-badge&logo=pytest)](tests/)

**An automated, enterprise-grade multi-agent LLM evaluation framework designed to audit, score, and benchmark AI responses across four orthogonal dimensions.**

[Live Application](https://llm-response-evaluation-platform.onrender.com) • [API Documentation](https://llm-response-evaluation-platform.onrender.com/docs) • [Technical Docs](docs/) • [Report Guide](docs/FINAL_PROJECT_REPORT.md)

</div>

---

## 🌐 Live Demo

The platform is deployed live on cloud container infrastructure:

- 🚀 **Live Dashboard**: [https://llm-response-evaluation-platform.onrender.com](https://llm-response-evaluation-platform.onrender.com)
- 📑 **Interactive Swagger API Docs**: [https://llm-response-evaluation-platform.onrender.com/docs](https://llm-response-evaluation-platform.onrender.com/docs)
- 🩺 **Health Check API**: [https://llm-response-evaluation-platform.onrender.com/health](https://llm-response-evaluation-platform.onrender.com/health)

*(Note: On the free cloud tier, the container spins down when inactive. If loading for the first time, allow ~30 seconds for the container to wake up).*

---

## 💡 Overview & Key Capabilities

Auditing Large Language Models requires evaluating factual precision, domain grounding, and compliance beyond surface-level fluency. This platform orchestrates specialized AI agents to evaluate responses against reference truth or domain knowledge bases.

- **Multi-Agent Evaluation Core**:
  - **Relevance Agent**: Semantic intent and prompt alignment scoring.
  - **Accuracy Agent**: Factual correctness verification against verified references.
  - **Completeness Agent**: Breadth and depth coverage across required sub-aspects.
  - **Hallucination Agent**: Atomic claim decomposition into standalone assertions audited against source truth.
- **Deterministic Verdict Synthesis**:
  - Transparent weighted composite scoring ($35\%$ Accuracy, $25\%$ Completeness, $20\%$ Relevance, $20\%$ Groundedness).
  - Safety Guardrails: Immediate critical failure overrides for direct contradictions, severe hallucinations ($> 50\%$), and off-topic outputs.
- **Dual Retrieval & Zero-Shot Context**:
  - Direct reference answers or source documents.
  - Automated ChromaDB vector store RAG fallback for open-ended queries.
- **High-Throughput Batch Processing**:
  - Handles $100+$ record CSV uploads with automated column detection, delimiter sniffing, and concurrent async worker pools.
- **Longitudinal Trend Analytics & Drift Detection**:
  - Tracks quality trajectories (`improving`, `stable`, `degrading`) across historical release batches.
- **Executive PDF Reporting**:
  - Generates publication-ready multi-page evaluation audit reports via ReportLab with dynamic improvement recommendations.

---

## 📌 Implementation Milestones

| Milestone | Scope | Status | Component Locations |
|:---:|---|:---:|---|
| **Milestone 1** | Input Module, Schema Validation & ChromaDB Vector Store | **Complete** | [`src/input_module/`](src/input_module), [`src/knowledge_base/`](src/knowledge_base) |
| **Milestone 2** | Multi-Agent Judge Core (Relevance, Accuracy, Claim-Level Hallucination) | **Complete** | [`src/agents/`](src/agents), [`tests/test_milestone2.py`](tests/test_milestone2.py) |
| **Milestone 3** | Completeness Judge, Weighted Verdict Synthesis & 100+ CSV Batch Pipeline | **Complete** | [`src/agents/batch_evaluator.py`](src/agents/batch_evaluator.py), [`src/agents/verdict_agent.py`](src/agents/verdict_agent.py) |
| **Milestone 4** | Interactive Web Dashboard, Quality Trends, Recommendations Engine & PDF Export | **Complete** | [`src/input_module/static/`](src/input_module/static), [`src/reporting/`](src/reporting) |

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│               Interactive SaaS Web Dashboard (HTML5 / Modern CSS)      │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ REST / JSON
┌──────────────────────────────────▼─────────────────────────────────────┐
│                 Input Module & Gateway (FastAPI)                       │
└──────────────────┬──────────────────────────────────┬──────────────────┘
                   │                                  │
    [Single Request Evaluation]                [100+ CSV Batch Ingestion]
                   │                                  │
                   │                        ┌─────────▼────────┐
                   │                        │  BatchEvaluator  │
                   │                        └─────────┬────────┘
                   │                                  │
┌──────────────────▼──────────────────────────────────▼──────────────────┐
│                   Multi-Agent Evaluation Orchestrator                  │
├────────────────────────────────────────────────────────────────────────┤
│ Context Engine: Direct Reference OR ChromaDB RAG Vector Store          │
│                                                                        │
│ Parallel Evaluation (asyncio.gather):                                  │
│ ┌─────────────────┐ ┌─────────────────┐ ┌────────────────────────────┐ │
│ │ Relevance Agent │ │ Accuracy Agent  │ │    Completeness Agent      │ │
│ └─────────────────┘ └─────────────────┘ └────────────────────────────┘ │
│ ┌────────────────────────────────────────────────────────────────────┐ │
│ │          Hallucination Agent (Atomic Claim Decomposition)          │ │
│ └────────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│ Deterministic Synthesis:                                               │
│ ┌────────────────────────────────────────────────────────────────────┐ │
│ │ Verdict Agent (Weighted Scoring Model + Critical Failure Overrides)│ │
│ └────────────────────────────────────────────────────────────────────┘ │
└──────────────────┬──────────────────────────────────┬──────────────────┘
                   │                                  │
┌──────────────────▼──────────────────┐ ┌─────────────▼──────────────────┐
│ SQLite Database & JSON Batch Storage│ │   Report Aggregator Service    │
└─────────────────────────────────────┘ └─────────────┬──────────────────┘
                                                      │
                                        ┌─────────────▼──────────────────┐
                                        │ Dynamic Recommendations Engine │
                                        └─────────────┬──────────────────┘
                                                      │
                                        ┌─────────────▼──────────────────┐
                                        │ ReportLab PDF Report Generator │
                                        └────────────────────────────────┘
```

---

## ⚖️ Scoring Methodology & Safety Guardrails

### Base Composite Scoring Model
The base weighted composite score $S_{\text{weighted}}$ combines all four dimensions:

$$S_{\text{weighted}} = 0.35 \cdot S_{\text{acc}} + 0.25 \cdot S_{\text{comp}} + 0.20 \cdot S_{\text{rel}} + 0.20 \cdot S_{\text{grd}}$$

- **Groundedness Score**: $S_{\text{grd}} = \max(0.0, 1.0 - S_{\text{hallucination}})$
- **Standard Thresholds**:
  - **Pass**: $S_{\text{weighted}} \ge 0.75$
  - **Needs Improvement**: $0.50 \le S_{\text{weighted}} < 0.75$
  - **Fail**: $S_{\text{weighted}} < 0.50$

### Critical Failure Overrides (Hard Safety Guardrails)
To prevent fluent but dangerous or contradictory responses from passing, the `VerdictAgent` enforces deterministic override rules:

```python
# 1. Direct Factual Contradiction
accuracy.classification == "contradictory"              ==>  Verdict = FAIL

# 2. Zero Factual Accuracy
accuracy.score == 0.0                                   ==>  Verdict = FAIL

# 3. Severe Hallucination
is_hallucinated == True and hallucination_score > 0.50   ==>  Verdict = FAIL

# 4. Off-Topic / Unrelated
relevance.score < 0.30 or classification in {"unrelated", "off_topic"}  ==>  Verdict = FAIL
```

---

## 🚀 Quickstart Guide

### Option 1: Docker (Production Container — Recommended)

Run the entire application in a container with zero local dependency management:

```bash
# Clone the repository
git clone https://github.com/sejalAS-1510/llm-response-eval-framework.git
cd llm-response-eval-framework

# Configure your environment
copy .env.example .env

# Build and start container in the background
docker compose up -d
```

Open **`http://localhost:8000`** in your browser.

---

### Option 2: Local Python Environment

```bash
# 1. Create and activate virtual environment
python -m venv .venv

# On Windows:
.\.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Configure environment variables
# Set your GEMINI_API_KEY in .env (an offline engine automatically activates if no key is provided)
cp .env.example .env

# 4. Run application
uvicorn src.input_module.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🧪 Automated Test Suite

The project includes an end-to-end automated test suite covering all agent contracts, scoring algorithms, edge cases, and report pipelines:

```bash
pytest -v
```

```text
============================== 86 passed in ~23s ==============================
```

- **Unit & Property Tests**: Dimension ranges $[0.0, 1.0]$, weight normalization, threshold boundaries.
- **Integration Tests**: Concurrent batch ingestion ($50$, $100$, $200$ records), CSV fault isolation.
- **Safety Tests**: Critical override activation on dangerous contradictions and hallucination thresholds.
- **Reporting Tests**: PDF two-pass page numbering, chart rendering, and recommendation synthesis.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|:---:|---|---|
| `GET` | `/` | Serves the interactive evaluation dashboard |
| `GET` | `/health` | Service health status and milestone readiness check |
| `POST` | `/api/v1/evaluate` | Evaluates a single question and AI response |
| `POST` | `/api/v1/batch/upload-csv` | Ingests and evaluates 100+ CSV dataset with concurrency |
| `GET` | `/api/v1/trends` | Computes historical batch trajectories and quality drift |
| `GET` | `/api/v1/reports/data/{batch_id}` | Retrieves structured JSON audit report data |
| `GET` | `/api/v1/reports/pdf/{batch_id}` | Generates and downloads a publication-grade PDF report |
| `GET` | `/docs` | Interactive Swagger UI API documentation |

---

## 📚 Technical Documentation

Comprehensive architectural design documents and reports are located in [`docs/`](docs/):

- 📄 **[Final Project Report](docs/FINAL_PROJECT_REPORT.md)**: Executive summary, methodology, performance benchmarks, and roadmap.
- 📐 **[01. System Architecture](docs/01-system-architecture.md)**: Subsystem topologies and sequence flows.
- 🤖 **[02. Agents Workflow](docs/02-agents-workflow.md)**: Agent roles and claim decomposition algorithms.
- ⚖️ **[03. Scoring & Verdicts](docs/03-scoring-and-verdict.md)**: Mathematical formulas, weight ceilings, and overrides.
- 🔍 **[04. RAG Pipeline](docs/04-rag-pipeline.md)**: Vector indexing, chunking, and similarity search.
- 📦 **[05. Data Contracts](docs/05-data-models.md)**: Pydantic schemas and serialization models.
- 🌐 **[06. API Specifications](docs/06-api-and-services.md)**: Full REST route parameters and responses.
- 📊 **[07. CSV Ingestion Format](docs/07-csv-upload-format.md)**: Upload specification and validation rules.
- 📈 **[08. Dashboard Metrics](docs/08-dashboard-metrics.md)**: Key performance indicators and calculation formulas.
- 📑 **[09. PDF Report Generation](docs/09-pdf-report-structure.md)**: ReportLab Platypus styles and flowables.
- 🚢 **[10. Cloud Deployment Guide](docs/10-deployment-guide.md)**: Production deployment instructions for Render and Docker.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.
