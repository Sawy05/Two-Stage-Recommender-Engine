FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    torch numpy pydantic fastapi "uvicorn[standard]" pyyaml scikit-learn httpx pytest

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
