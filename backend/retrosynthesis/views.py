from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from retrosynthesis.services import plan_retrosynthesis


class RetrosynthesisView(APIView):
    authentication_classes: list = []
    permission_classes: list = []

    def post(self, request: Request) -> Response:
        smiles = (request.data.get("smiles") or "").strip()
        if not smiles:
            return Response({"detail": "smiles is required"}, status=status.HTTP_400_BAD_REQUEST)
        max_depth = request.data.get("max_depth", request.data.get("depth"))
        try:
            return Response(plan_retrosynthesis(smiles, max_depth=max_depth))
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
