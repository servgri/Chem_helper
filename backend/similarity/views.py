from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from similarity.services import find_similar


class SimilarView(APIView):
    authentication_classes: list = []
    permission_classes: list = []

    def post(self, request: Request) -> Response:
        smiles = (request.data.get("smiles") or "").strip()
        if not smiles:
            return Response({"detail": "smiles is required"}, status=status.HTTP_400_BAD_REQUEST)
        top_n = request.data.get("top_n")
        try:
            return Response(find_similar(smiles, top_n=top_n))
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
