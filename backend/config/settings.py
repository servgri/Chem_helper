"""Django settings for ToxMol API."""

from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _env_list(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


DEBUG = _env_flag("DJANGO_DEBUG", "1")

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "").strip()
if not SECRET_KEY:
    if DEBUG:
        SECRET_KEY = "toxmol-dev-insecure-change-me"
    else:
        raise RuntimeError("DJANGO_SECRET_KEY is required when DJANGO_DEBUG=0")

ALLOWED_HOSTS = _env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,0.0.0.0")

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
    "whitenoise.middleware.WhiteNoiseMiddleware",
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
        "NAME": Path(os.environ.get("DJANGO_DB_PATH", str(BASE_DIR / "db.sqlite3"))),
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "ru-ru"
TIME_ZONE = "Europe/Moscow"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "whitenoise.storage.CompressedStaticFilesStorage"},
}

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

CORS_ALLOW_ALL_ORIGINS = DEBUG
CORS_ALLOWED_ORIGINS = _env_list(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:8510,http://127.0.0.1:8510,http://localhost:5173,http://127.0.0.1:5173",
)
CORS_ALLOW_CREDENTIALS = True

CSRF_TRUSTED_ORIGINS = _env_list(
    "CSRF_TRUSTED_ORIGINS",
    ",".join(CORS_ALLOWED_ORIGINS),
)

SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
USE_X_FORWARDED_HOST = True

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "plain": {
            "format": "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "plain",
        }
    },
    "root": {
        "handlers": ["console"],
        "level": os.environ.get("LOG_LEVEL", "INFO"),
    },
}

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
