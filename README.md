# LLM Response Evaluation Platform

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Tests](https://img.shields.io/badge/Tests-86%20Passed%20(100%25)-brightgreen.svg)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker)](https://www.docker.com/)

An automated, multi-agent AI response evaluation framework designed to benchmark, audit, and score Large Language Model (LLM) responses against ground-truth references and retrieved domain knowledge across four orthogonal dimensions: **Relevance**, **Factual Accuracy**, **Hallucination Detection (via atomic claim decomposition)**, and **Completeness**.

---

## 📌 Project Milestones & Implementation Status

| Milestone | Scope | Status | Component Locations |
|:---:|---|:---:|---|
| **Milestone 1** | Input Module & ChromaDB Local RAG Pipeline | **Done** | [`src/input_module/`](src/input_module), [`src/knowledge_base/`](src/knowledge_base) |
| **Milestone 2** | Multi-Agent Judge Core (Relevance, Accuracy, Hallucination) | **Done** | [`src/agents/`](src/agents), [`tests/test_milestone2.py`](tests/test_milestone2.py) |
| **Milestone 3** | Completeness Agent, Verdict Synthesis & 100+ CSV Batch Ingestion | **Done** | [`src/agents/completeness_agent.py`](src/agents/completeness_agent.py), [`src/agents/verdict_agent.py`](src/agents/verdict_agent.py), [`src/agents/batch_evaluator.py`](src/agents/batch_evaluator.py) |
| **Milestone 4** | Interactive SaaS Dashboard, Longitudinal Trends, Dynamic Recommendations & Publication PDF Reports | **Done** | [`src/input_module/static/index.html`](src/input_module/static/index.html), [`src/reporting/`](src/reporting), [`tests/test_milestone4.py`](tests/test_milestone4.py) |

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│               Interactive SaaS Web Dashboard (HTML5/CSS)               │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ HTTP / JSON
┌──────────────────────────────────▼─────────────────────────────────────┐
│                 Input Module & REST Gateway (FastAPI)                  │
└──────────────────┬──────────────────────────────────┬──────────────────┘
                   │                                  │
    [Single Request Routing]               [Batch CSV Ingestion]
                   │                                  │
                   │                        ┌─────────▼────────┐
                   │                        │  BatchEvaluator  │
                   │                        └─────────┬────────┘
                   │                                  │
┌──────────────────▼──────────────────────────────────▼──────────────────┐
│                   Multi-Agent Evaluation Orchestrator                  │
├────────────────────────────────────────────────────────────────────────┤
│ Context Resolution: Direct Reference OR Local ChromaDB RAG Retrieval   │
│                                                                        │
│ Parallel Evaluation: (asyncio.gather)                                  │
│ ┌─────────────────┐ ┌─────────────────┐ ┌────────────────────────────┐ │
│ │ Relevance Agent │ │ Accuracy Agent  │ │    Completeness Agent      │ │
│ └─────────────────┘ └─────────────────┘ └────────────────────────────┘ │
│ ┌────────────────────────────────────────────────────────────────────┐ │
│ │               Hallucination Agent (Claim Decomposition)            │ │
│ └────────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│ Deterministic Synthesis:                                               │
│ ┌────────────────────────────────────────────────────────────────────┐ │
│ │ Verdict Agent (Weighted Scoring + Critical Failure Overrides)      │ │
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

## 🚀 Quickstart Guide

### Option 1: One-Click Runner (Windows / Linux / Mac)

**Windows**:
```cmd
run.bat
```

**Linux / macOS**:
```bash
chmod +x run.sh
./run.sh
```

---

### Option 2: Local Python Virtual Environment

```bash
# 1. Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Configure environment variables (optional; offline heuristic engine runs out of the box)
cp .env.example .env

# 4. Start the FastAPI server and Dashboard
uvicorn src.input_module.main:app --reload --host 0.0.0.0 --port 8000
```

Open **`http://localhost:8000`** in your browser.

---

### Option 3: Docker & Docker Compose (Containerized Production)

```bash
# Build image and run container in background
docker compose up --build -d

# View container logs
docker compose logs -f

# Stop container
docker compose down
```

The container automatically mounts `./data` to persist vector stores and evaluation records.

---

## 🧪 Test Suite & Verification

The framework includes a comprehensive automated test suite with **86 tests** passing with zero failures:

```bash
# Run complete test suite
pytest -v

# Run with concise summary
pytest -q
```

```
============================== 86 passed in ~35s ==============================
```

Test coverage includes:
- Multi-agent scoring and claim decomposition verification
- Mathematical exactness between raw records, dashboard statistics, and PDF exports
- 50, 200, and 500-record batch throughput and memory scaling benchmarks
- Missing reference fallback (automatic RAG retrieval)
- CSV parsing fault tolerance (malformed rows, unescaped quotes, missing columns)
- ReportLab two-pass pagination and dynamic engineering recommendations

---

## 📚 Technical Documentation & Reports

Detailed technical documentation is available in the [`docs/`](docs/) directory:

- 📄 **[Final Project Report](docs/FINAL_PROJECT_REPORT.md)**: Executive narrative covering problem statement, system design, testing results, limitations, and future roadmap.
- 📐 **[01. System Architecture](docs/01-system-architecture.md)**: End-to-end component diagrams and subsystem interactions.
- 🤖 **[02. Agents Workflow](docs/02-agents-workflow.md)**: Agent responsibilities and atomic claim decomposition methodology.
- ⚖️ **[03. Scoring & Verdict Methodology](docs/03-scoring-and-verdict.md)**: Mathematical formulas, weights, and critical failure overrides.
- 🔍 **[04. RAG Pipeline](docs/04-rag-pipeline.md)**: ChromaDB vector store, chunking, and embedding retrieval.
- 📦 **[05. Data Models](docs/05-data-models.md)**: Pydantic schemas and serialization models.
- 🌐 **[06. API & Services](docs/06-api-and-services.md)**: REST endpoints and request/response specifications.
- 📊 **[07. CSV Ingestion Format](docs/07-csv-upload-format.md)**: Format requirements, delimiter detection, and error isolation.
- 📈 **[08. Dashboard Metrics](docs/08-dashboard-metrics.md)**: KPI computation and trend trajectory logic.
- 📑 **[09. PDF Report Generation](docs/09-pdf-report-structure.md)**: ReportLab Platypus structure, styles, and pagination.
- 🚢 **[10. Deployment Guide](docs/10-deployment-guide.md)**: Local, Docker, and cloud hosting (Render, Railway, Hugging Face Spaces).
