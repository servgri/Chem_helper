from django.urls import path

from retrosynthesis.views import RetrosynthesisView

urlpatterns = [
    path("retrosynthesis/", RetrosynthesisView.as_view(), name="retrosynthesis"),
]
