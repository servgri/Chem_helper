from django.urls import path

from molecules.views import FromMolfileView, HealthView, ParseSmilesView

urlpatterns = [
    path("health/", HealthView.as_view(), name="health"),
    path("molecule/parse/", ParseSmilesView.as_view(), name="molecule-parse"),
    path("molecule/from-molfile/", FromMolfileView.as_view(), name="molecule-from-molfile"),
]
