"""ToxMol API URL configuration."""

from __future__ import annotations

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("molecules.urls")),
    path("api/", include("predictions.urls")),
    path("api/", include("similarity.urls")),
    path("api/", include("retrosynthesis.urls")),
]
