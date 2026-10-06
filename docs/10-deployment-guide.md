# 10. Deployment Guide & Production Operations

This document provides complete instructions for deploying the **LLM Response Evaluation Platform** across local, containerized (Docker), and cloud environments.

---

## 1. System Requirements & Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| **Python Version** | Python 3.10+ | Python 3.11+ |
| **Operating System** | Windows 10+, Ubuntu 20.04+, macOS 12+ | Linux (Ubuntu / Debian) or Windows |
| **System Memory (RAM)**| 4 GB | 8 GB+ (for local embeddings & 500+ record batches) |
| **Disk Space** | 2 GB free | 5 GB+ (for ChromaDB vector storage and batch reports) |
| **Optional API Key** | None (heuristic offline mode works out of the box) | Google Gemini API key (`GEMINI_API_KEY` for live LLM completion) |

---

## 2. Deployment Option A: Local Native Deployment

Ideal for local testing, development, and direct presentation to mentors.

### Step 1: Clone Repository & Create Virtual Environment

```bash
# Clone the repository
git clone https://github.com/sejalAS-1510/llm-response-eval-framework.git
cd llm-response-eval-framework

# Create Python virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (CMD):
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

```bash
# Copy example configuration
cp .env.example .env
```

Edit `.env` to configure your keys:
```ini
# Optional: Set your Gemini API key for live LLM evaluations
GEMINI_API_KEY=your_actual_gemini_api_key_here
PORT=8000
HOST=0.0.0.0
```

> [!NOTE]
> If `GEMINI_API_KEY` is not provided, the framework automatically uses its deterministic heuristic offline engine, ensuring 100% of tests, RAG lookups, and evaluation workflows succeed at zero cost.

### Step 4: Run the Application

**Option 1: Using one-click launcher scripts**
- Windows: Double-click `run.bat` or execute `.\run.bat`
- Linux / macOS: Execute `./run.sh`

**Option 2: Direct Uvicorn command**
```bash
uvicorn src.input_module.main:app --reload --host 0.0.0.0 --port 8000
```

Open **`http://localhost:8000`** in your browser.

---

## 3. Deployment Option B: Docker & Docker Compose (Recommended for Production)

Docker eliminates environment inconsistencies, encapsulates dense embedding models, and guarantees repeatable evaluations on any machine.

### Single-Command Launch via Docker Compose

```bash
# Build image and start container
docker compose up --build -d
```

### Checking Container Health & Logs

```bash
# View live container logs
docker compose logs -f

# Verify container health status
docker ps
```

The container includes a built-in health check querying `http://localhost:8000/health`.

### Stopping the Container

```bash
docker compose down
```

> [!IMPORTANT]
> The `./data` directory is mounted into `/app/data` inside the container. All historical batch evaluations, SQLite records, and ChromaDB vector embeddings persist on your host machine even when containers are rebuilt.

---

## 4. Deployment Option C: Cloud Platform Hosting

The framework is packaged for zero-friction cloud deployment.

### 1. Render (Web Service)
1. Fork or push the repository to GitHub.
2. Sign in to [Render.com](https://render.com) and create a **New Web Service**.
3. Connect your repository.
4. Select **Docker** environment or **Python** environment:
   - If Python:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn src.input_module.main:app --host 0.0.0.0 --port $PORT`
5. Under **Environment Variables**, add:
   - `PYTHONUNBUFFERED`: `1`
   - `GEMINI_API_KEY`: *(your key, if available)*
6. Click **Deploy**.

### 2. Railway
1. Go to [Railway.app](https://railway.app) and create a **New Project**.
2. Select **Deploy from GitHub repo**.
3. Railway automatically detects the [`Dockerfile`](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/Dockerfile) and provisions the container on port `8000`.

### 3. Hugging Face Spaces (Docker SDK)
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Choose **Docker** as the Space SDK.
3. Push the repository to the Space git remote.
4. Hugging Face automatically builds the container using [`Dockerfile`](file:///C:/Sejal/Infosys%20Springboard/llm-response-eval-framework/Dockerfile) and serves the evaluation interface on the public Space URL.

---

## 5. Production Health Checks & Verification

After deployment, verify system health using the automated endpoints:

### 1. Service Health Check Endpoint
```bash
curl http://localhost:8000/health
```
**Expected Response**:
```json
{
  "status": "ok",
  "version": "0.3.0",
  "milestones": ["M1", "M2", "M3", "M4"]
}
```

### 2. Swagger / OpenAPI Documentation
Access interactive API documentation at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### 3. Automated Test Suite Execution
Execute the full test suite in the environment:
```bash
pytest -v
```
**Expected Result**: `86 passed in ~35-65s`.
