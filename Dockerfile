# ─── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install poetry==1.8.3
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.in-project true \
 && poetry install --no-interaction --no-ansi --without dev

# ─── Stage 2: runtime (CPU) ──────────────────────────────────────────────────
FROM python:3.11-slim AS runtime-cpu

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

COPY src/ ./src/
COPY configs/ ./configs/
COPY api/ ./api/
COPY frontend/ ./frontend/

EXPOSE 8000 8501

# ─── Stage 3: runtime (GPU) ──────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime-gpu

RUN apt-get update && apt-get install -y python3.11 python3-pip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

COPY src/ ./src/
COPY configs/ ./configs/
COPY api/ ./api/
COPY frontend/ ./frontend/

EXPOSE 8000 8501
