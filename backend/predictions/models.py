from __future__ import annotations

from django.db import models


class PredictionJob(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("done", "Done"),
        ("error", "Error"),
    ]

    smiles = models.CharField(max_length=2048)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="pending")
    payload = models.JSONField(default=dict, blank=True)
    error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"PredictionJob({self.id}, {self.smiles[:40]}, {self.status})"
