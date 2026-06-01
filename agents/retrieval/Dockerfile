# PRISM-RAG retrieval service — uv-based build.
#
# Uses uv for ~10-20× faster installs than pip. CPU torch by default;
# override with --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121.
#
# Why `uv pip install --system`: we install into the system Python (the slim
# image's /usr/local) rather than a venv. Inside a container there is no
# "host Python" to protect — a venv would only add an activation layer and a
# longer PATH. `--system` keeps the runtime CMD simple (no venv activation).
FROM python:3.11-slim AS base

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf \
    EMBED_DEVICE=cpu \
    PRISM_DATA_DIR=/data

# Pull the uv binary from the official distroless image — no curl, no shell
# install script, no version drift.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch first from the explicit index (CPU vs CUDA split), then the
# rest from PyPI. uv handles multi-index resolution cleanly.
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        --index-url "${TORCH_INDEX_URL}" \
        --extra-index-url https://pypi.org/simple \
        torch \
    && uv pip install --system -r requirements.txt \
    && uv pip install --system \
        "fastapi>=0.110" "uvicorn[standard]>=0.27" "pydantic>=2.5"

COPY agents/ ./agents/
COPY scripts/ ./scripts/

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8001/healthz || exit 1

# Workers > 1 means N independent encoder copies in memory. Keep at 1 by
# default; bump only after measuring memory headroom.
CMD ["uvicorn", "agents.retrieval.main:app", \
        "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]