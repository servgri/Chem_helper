from django.apps import AppConfig


class PredictionsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "predictions"

    def ready(self) -> None:
        from django.contrib import admin

        admin.site.site_header = "ToxMol Admin"
        admin.site.site_title = "ToxMol"
        admin.site.index_title = "Prediction jobs & auth"
