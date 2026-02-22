# syntax=docker/dockerfile:1
# ToxMol API: CPU inference, classical models only (no torch / jupyter).

FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    TOXMOL_FORCE_CPU=1 \
    DJANGO_SETTINGS_MODULE=config.settings \
    PYTHONPATH=/workspace

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 \
        libxext6 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt \
    && rm -rf /root/.cache/pip /tmp/requirements.txt

COPY src /workspace/src
COPY data/raw /workspace/data/raw
COPY data/processed/.gitkeep /workspace/data/processed/.gitkeep
COPY results/models/.gitkeep /workspace/results/models/.gitkeep
COPY backend /workspace/backend
COPY docker/entrypoint.sh /entrypoint.sh

RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh \
    && find /workspace -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

WORKDIR /workspace/backend

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
