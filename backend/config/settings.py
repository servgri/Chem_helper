"""Django settings for ToxMol API."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# backend/config/settings.py → repo root (parent of backend/)
BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent

# Allow ``from src...`` for research helpers (qsar_utils, viz_rdkit, clearml_tracking).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "toxmol-dev-insecure-change-me-in-production",
)

DEBUG = os.environ.get("DJANGO_DEBUG", "1").strip().lower() not in {"0", "false", "no"}

ALLOWED_HOSTS = [
    h.strip()
    for h in os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,0.0.0.0").split(",")
    if h.strip()
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "rest_framework",
    "molecules",
    "predictions",
    "similarity",
    "retrosynthesis",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.FormParser",
        "rest_framework.parsers.MultiPartParser",
    ],
    "UNAUTHENTICATED_USER": None,
}

# Dev-friendly CORS for Vite frontend (:5173) and optional compose web.
CORS_ALLOW_ALL_ORIGINS = DEBUG
CORS_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
    ).split(",")
    if o.strip()
]
CORS_ALLOW_CREDENTIALS = True

# ToxMol paths (override via env for Docker)
TOXMOL_REPO_ROOT = Path(os.environ.get("TOXMOL_REPO_ROOT", str(REPO_ROOT)))
TOXMOL_MODELS_DIR = Path(
    os.environ.get("TOXMOL_MODELS_DIR", str(TOXMOL_REPO_ROOT / "results" / "models"))
)
TOXMOL_DATA_DIR = Path(os.environ.get("TOXMOL_DATA_DIR", str(TOXMOL_REPO_ROOT / "data")))
TOXMOL_FP_INDEX_PATH = Path(
    os.environ.get(
        "TOXMOL_FP_INDEX_PATH",
        str(TOXMOL_DATA_DIR / "processed" / "tox21_fps.npz"),
    )
)
TOXMOL_N_BITS = int(os.environ.get("TOXMOL_N_BITS", "2048"))
TOXMOL_FP_RADIUS = int(os.environ.get("TOXMOL_FP_RADIUS", "2"))
TOXMOL_SIMILAR_TOP_N = int(os.environ.get("TOXMOL_SIMILAR_TOP_N", "12"))
TOXMOL_RETRO_MAX_DEPTH = int(os.environ.get("TOXMOL_RETRO_MAX_DEPTH", "2"))
TOXMOL_RETRO_BRANCH_LIMIT = int(os.environ.get("TOXMOL_RETRO_BRANCH_LIMIT", "8"))
