#!/bin/sh
set -e

mkdir -p /workspace/var /workspace/results/models /workspace/data/processed

python manage.py migrate --noinput
python manage.py collectstatic --noinput
python manage.py prepare_models

exec gunicorn config.wsgi:application \
    --bind 0.0.0.0:8000 \
    --workers 1 \
    --threads 2 \
    --timeout 300 \
    --access-logfile - \
    --error-logfile -
