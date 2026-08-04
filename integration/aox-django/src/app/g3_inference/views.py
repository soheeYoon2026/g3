import gzip
import json

import boto3
import requests
from django.conf import settings
from django.core.cache import cache
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes, permission_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from app.tenants.team_access import require_team_admin, resolve_selected_team


MAX_STL_BYTES = 250 * 1024 * 1024
FORWARDED_FIELDS = (
    "u_x",
    "density",
    "viscosity",
    "temperature",
    "ref_length",
    "ref_area",
    "grid_x",
    "grid_y",
    "grid_z",
)


def _service_config():
    endpoint = getattr(settings, "G3_INFERENCE_URL", "").rstrip("/")
    token = getattr(settings, "G3_INFERENCE_TOKEN", "")
    if endpoint and token:
        return endpoint, token

    cache_key = "g3:inference-service-config"
    cached = cache.get(cache_key)
    if cached:
        return cached

    bucket = getattr(settings, "G3_INFERENCE_CONFIG_BUCKET", "") or getattr(settings, "AWS_S3_BUCKET", "")
    key = getattr(settings, "G3_INFERENCE_CONFIG_KEY", "_private/g3/inference-service.json")
    if not bucket:
        return "", ""
    try:
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        payload = json.loads(body)
        config = str(payload.get("url") or "").rstrip("/"), str(payload.get("token") or "")
    except Exception:
        return "", ""
    if all(config):
        cache.set(cache_key, config, 60)
        return config
    return "", ""


@extend_schema(exclude=True)
@api_view(["POST"])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser])
def create_inference(request):
    selected_team = resolve_selected_team(request, raise_on_ambiguous=True)
    if selected_team is None:
        return Response({"detail": "Team selection is required."}, status=status.HTTP_400_BAD_REQUEST)
    require_team_admin(request.user, selected_team.id)

    endpoint, token = _service_config()
    if not endpoint or not token:
        return Response(
            {"detail": "G3 inference service is not configured."}, status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    stl = request.FILES.get("stl")
    if stl is None or not stl.name.lower().endswith(".stl"):
        return Response({"detail": "An STL file is required."}, status=status.HTTP_400_BAD_REQUEST)
    if stl.size <= 0 or stl.size > MAX_STL_BYTES:
        return Response({"detail": "STL must be between 1 byte and 250 MB."}, status=status.HTTP_400_BAD_REQUEST)

    data = {name: request.data[name] for name in FORWARDED_FIELDS if name in request.data}
    compressed_stl = gzip.compress(stl.read(), compresslevel=6)
    try:
        upstream = requests.post(
            f"{endpoint}/v1/infer",
            headers={"Authorization": f"Bearer {token}"},
            files={"stl": (f"{stl.name}.gz", compressed_stl, "application/gzip")},
            data=data,
            timeout=(5, getattr(settings, "G3_INFERENCE_TIMEOUT_SECONDS", 300)),
        )
    except requests.Timeout:
        return Response({"detail": "G3 inference timed out."}, status=status.HTTP_504_GATEWAY_TIMEOUT)
    except requests.RequestException:
        return Response({"detail": "G3 inference service is unavailable."}, status=status.HTTP_502_BAD_GATEWAY)

    try:
        payload = upstream.json()
    except ValueError:
        payload = {"detail": "G3 inference service returned an invalid response."}
    return Response(payload, status=upstream.status_code)
