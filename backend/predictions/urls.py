from django.urls import path

from predictions.views import PredictView

urlpatterns = [
    path("predict/", PredictView.as_view(), name="predict"),
]
