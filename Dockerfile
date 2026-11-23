# syntax=docker/dockerfile:1
# OpsPilot AI API: FastAPI + CPU-only PyTorch inference.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only wheels from the official PyTorch index: no CUDA libraries in the image.
ARG TORCH_VERSION=2.15.0
RUN pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY opspilot ./opspilot
COPY kb ./kb

# Train at build time: the generator is seeded, so this is deterministic and takes well
# under a minute. The image is self-contained; no model download, no credentials.
RUN python -m opspilot.train --output-dir /app/artifacts

RUN useradd --create-home --uid 10001 opspilot \
    && mkdir -p /app/data \
    && chown opspilot /app/data
USER opspilot

# Non-secret defaults for running inside a container; override any of them with -e or
# compose. Secrets (only OPSPILOT_LLM_API_KEY, optional) are never baked into the image.
ENV OPSPILOT_HOST=0.0.0.0 \
    OPSPILOT_PORT=8000 \
    OPSPILOT_ARTIFACTS_DIR=/app/artifacts \
    OPSPILOT_KB_DIR=/app/kb \
    OPSPILOT_QUEUE_DB_PATH=/app/data/review_queue.db

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import os, sys, urllib.request; url = 'http://127.0.0.1:%s/healthz' % os.environ.get('OPSPILOT_PORT', '8000'); sys.exit(0 if urllib.request.urlopen(url, timeout=2).status == 200 else 1)"

CMD ["python", "-m", "opspilot.api"]
