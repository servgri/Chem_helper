from django.contrib import admin

from predictions.models import PredictionJob


@admin.register(PredictionJob)
class PredictionJobAdmin(admin.ModelAdmin):
    list_display = ("id", "smiles_short", "status", "created_at")
    list_filter = ("status", "created_at")
    search_fields = ("smiles",)
    readonly_fields = ("created_at", "updated_at", "payload", "error")

    @admin.display(description="SMILES")
    def smiles_short(self, obj: PredictionJob) -> str:
        s = obj.smiles or ""
        return s if len(s) <= 48 else s[:45] + "..."
