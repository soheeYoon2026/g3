"""Authenticated HTTP service for real-time G3 STL inference."""

from __future__ import annotations

import asyncio
import base64
import gzip
import io
import os
import tempfile
import time
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from .canary import ShadowConfig, ShadowDispatcher, request_digest
from .infer_fields import main as infer_fields


MODEL_PATH = Path(os.environ.get("G3_MODEL_PATH", "models/g3_field_g2_v6_final.pt"))
COEFFICIENT_EXPERT = os.environ.get("G3_COEFFICIENT_EXPERT", "g2_su2_clean")
TOKEN_FILE = os.environ.get("G3_API_TOKEN_FILE", "")
API_TOKEN = (
    Path(TOKEN_FILE).read_text().strip()
    if TOKEN_FILE and Path(TOKEN_FILE).is_file()
    else os.environ.get("G3_API_TOKEN", "")
)
MAX_UPLOAD_BYTES = int(os.environ.get("G3_MAX_UPLOAD_BYTES", str(250 * 1024 * 1024)))
INFERENCE_LOCK = asyncio.Lock()
SHADOW = ShadowDispatcher(ShadowConfig.from_env())

app = FastAPI(title="AOX G3 Inference", version="0.1.0")


def _authorize(authorization: str | None) -> None:
    if not API_TOKEN:
        raise HTTPException(status_code=503, detail="G3_API_TOKEN is not configured")
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="invalid inference token")


@app.get("/health")
def health():
    resolved_model = MODEL_PATH.resolve().name if MODEL_PATH.is_file() else MODEL_PATH.name
    return {
        "status": "ok",
        "model_ready": MODEL_PATH.is_file(),
        "model": resolved_model,
        "coefficient_expert": COEFFICIENT_EXPERT,
        "shadow_canary_enabled": SHADOW.config.enabled,
        "shadow_sample_rate": SHADOW.config.sample_rate if SHADOW.config.enabled else 0.0,
    }


@app.post("/v1/infer")
async def infer(
    background_tasks: BackgroundTasks,
    stl: UploadFile = File(...),
    u_x: float = Form(30.0),
    u_y: float = Form(0.0),
    u_z: float = Form(0.0),
    density: float = Form(1.225),
    viscosity: float = Form(1.7894e-5),
    temperature: float = Form(288.15),
    ref_length: float = Form(5.0),
    ref_area: float = Form(1.0),
    grid_x: int = Form(96),
    grid_y: int = Form(64),
    grid_z: int = Form(48),
    include_preview: bool = Form(True),
    authorization: str | None = Header(default=None),
):
    _authorize(authorization)
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="G3 checkpoint is unavailable")
    file_name = (stl.filename or "").lower()
    if not (file_name.endswith(".stl") or file_name.endswith(".stl.gz")):
        raise HTTPException(status_code=400, detail="only STL uploads are accepted")
    if not (8 <= grid_x <= 128 and 8 <= grid_y <= 128 and 8 <= grid_z <= 128):
        raise HTTPException(status_code=400, detail="grid dimensions must be between 8 and 128")

    payload = await stl.read(MAX_UPLOAD_BYTES + 1)
    if file_name.endswith(".gz"):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(payload)) as compressed:
                payload = compressed.read(MAX_UPLOAD_BYTES + 1)
        except (OSError, EOFError) as exc:
            raise HTTPException(status_code=400, detail="invalid compressed STL") from exc
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="STL is empty or exceeds the upload limit")

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="aox-g3-") as temp_dir:
        temp = Path(temp_dir)
        stl_path = temp / "input.stl"
        output_dir = temp / "result"
        stl_path.write_bytes(payload)
        argv = [
            "--stl", str(stl_path),
            "--model", str(MODEL_PATH),
            "--out-dir", str(output_dir),
            "--grid", str(grid_x), str(grid_y), str(grid_z),
            "--u-x", str(u_x),
            "--u-y", str(u_y),
            "--u-z", str(u_z),
            "--density", str(density),
            "--viscosity", str(viscosity),
            "--temperature", str(temperature),
            "--ref-length", str(ref_length),
            "--ref-area", str(ref_area),
            "--coefficient-expert", COEFFICIENT_EXPERT,
        ]
        if include_preview:
            argv.append("--png")
        try:
            async with INFERENCE_LOCK:
                summary = await run_in_threadpool(infer_fields, argv)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"G3 inference failed: {exc}") from exc

        preview = None
        if include_preview:
            preview = base64.b64encode(
                (output_dir / "pressure_streamlines.png").read_bytes()
            ).decode("ascii")

    response = {
        "model": MODEL_PATH.resolve().name,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "grid": summary["grid"],
        "drag_coefficient": summary["drag_coefficient"],
        "lift_coefficient": summary["lift_coefficient"],
        "coefficient_expert": summary.get("coefficient_expert"),
        "deployment_status": summary.get("coefficient_experts", {})
        .get(summary.get("coefficient_expert"), {}).get("deployment_status"),
        "ood_score": summary.get("coefficient_experts", {})
        .get(summary.get("coefficient_expert"), {}).get("ood_score"),
        "in_distribution": summary.get("coefficient_experts", {})
        .get(summary.get("coefficient_expert"), {}).get("in_distribution"),
        "coefficient_warning": summary.get("coefficient_warning"),
        "pressure_range": summary["pressure_range"],
        "speed_range": summary["speed_range"],
        "device": summary["device"],
    }
    if preview is not None:
        response["preview_png"] = f"data:image/png;base64,{preview}"

    if SHADOW.reserve(payload):
        try:
            shadow_path = SHADOW.spool(payload)
            background_tasks.add_task(
                SHADOW.run,
                shadow_path,
                stl.filename or "input.stl",
                {
                    "u_x": u_x,
                    "u_y": u_y,
                    "u_z": u_z,
                    "density": density,
                    "viscosity": viscosity,
                    "temperature": temperature,
                    "ref_length": ref_length,
                    "ref_area": ref_area,
                    "grid_x": grid_x,
                    "grid_y": grid_y,
                    "grid_z": grid_z,
                },
                response,
                request_digest(payload),
            )
        except Exception:
            # Shadow operation is best-effort and must never fail production.
            pass
    return response
