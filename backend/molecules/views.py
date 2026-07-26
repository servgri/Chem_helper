"""Molecule API views."""

from __future__ import annotations

from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from molecules.services import from_molfile, parse_smiles


class HealthView(APIView):
    authentication_classes: list = []
    permission_classes: list = []

    def get(self, request: Request) -> Response:
        from predictions.services import model_status

        return Response({"status": "ok", "models": model_status()})


class ParseSmilesView(APIView):
    authentication_classes: list = []
    permission_classes: list = []

    def post(self, request: Request) -> Response:
        smiles = (request.data.get("smiles") or "").strip()
        if not smiles:
            return Response({"detail": "smiles is required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            return Response(parse_smiles(smiles))
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)


class FromMolfileView(APIView):
    authentication_classes: list = []
    permission_classes: list = []

    def post(self, request: Request) -> Response:
        molfile = request.data.get("molfile") or request.data.get("molblock") or ""
        if not str(molfile).strip():
            return Response({"detail": "molfile is required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            return Response(from_molfile(str(molfile)))
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
