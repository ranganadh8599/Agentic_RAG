# syntax=docker/dockerfile:1
# ============================================================================
# Agentic RAG - application image
#
# Default build runs the reranker on CPU (small image). For GPU reranking,
# rebuild with a CUDA torch index, e.g.:
#   docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
# (that override passes TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124)
# ============================================================================
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU torch by default; override with e.g. https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# Runtime system libs needed by the torch/numpy wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install torch from the chosen index first, then everything else
# (the torch requirement in requirements.txt is then already satisfied, so
# pip will not pull the big default CUDA wheel on top of a CPU build).
RUN pip install --no-cache-dir "torch>=2.2" --index-url "$TORCH_INDEX_URL"

# Install the rest of the deps (requirements.txt is copied alone so this
# layer is cached unless requirements.txt changes).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (all .py files + static/ UI).
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
