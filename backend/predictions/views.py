"""Prediction API."""

from __future__ import annotations

from typing import Any

from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from predictions.models import PredictionJob
from predictions.services import predict_all


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


class PredictView(APIView):
    authentication_classes: list = []
    permission_classes: list = []

    def post(self, request: Request) -> Response:
        smiles = (request.data.get("smiles") or "").strip()
        if not smiles:
            return Response({"detail": "smiles is required"}, status=status.HTTP_400_BAD_REQUEST)

        job = PredictionJob.objects.create(smiles=smiles, status="running")
        try:
            result = _jsonable(predict_all(smiles))
            job.status = "done"
            job.payload = {
                "qsar": result.get("qsar"),
                "admet": result.get("admet"),
                "nr": result.get("nr"),
                "sr": result.get("sr"),
                "errors": result.get("errors"),
            }
            job.save(update_fields=["status", "payload", "updated_at"])
            result["job_id"] = job.id
            return Response(result)
        except ValueError as exc:
            job.status = "error"
            job.error = str(exc)
            job.save(update_fields=["status", "error", "updated_at"])
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
            job.save(update_fields=["status", "error", "updated_at"])
            return Response(
                {"detail": f"Prediction failed: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
