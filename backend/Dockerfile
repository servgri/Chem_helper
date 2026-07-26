# ToxMol Django API (CPU inference)
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
        build-essential curl libxrender1 libxext6 libsm6 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /workspace/backend/requirements.txt
RUN pip install --upgrade pip \
    && pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /workspace/backend/requirements.txt

COPY src /workspace/src
COPY data /workspace/data
COPY results /workspace/results
COPY backend /workspace/backend

WORKDIR /workspace/backend

EXPOSE 8000

CMD ["sh", "-c", "python manage.py migrate --noinput && python manage.py prepare_models && gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 1 --timeout 300"]
