from django.urls import path

from similarity.views import SimilarView

urlpatterns = [
    path("similar/", SimilarView.as_view(), name="similar"),
]
