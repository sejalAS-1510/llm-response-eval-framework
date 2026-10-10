# ==============================================================================
# LLM Response Evaluation Framework - Production Dockerfile
# Optimized for Ultra-Low Memory Cloud Environments (< 200 MB RAM, Render Free tier)
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000

WORKDIR /app

# Install minimal system dependencies required for compilation and health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, datasets, and documentation
COPY src/ /app/src/
COPY data/ /app/data/
COPY docs/ /app/docs/
COPY pytest.ini /app/
COPY tests/ /app/tests/

# Ensure runtime directories exist
RUN mkdir -p /app/data/batches /app/data/chroma /app/data/reference_kb

# Expose port for FastAPI application
EXPOSE 8000

# Container healthcheck querying the /health endpoint dynamically
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default execution command binding to environment PORT (supports Render, Cloud Run, Docker Compose)
CMD ["sh", "-c", "uvicorn src.input_module.main:app --host 0.0.0.0 --port ${PORT}"]
