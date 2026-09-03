"""Authenticated HTTP service for real-time G3 STL inference."""

from __future__ import annotations

import asyncio
import base64
import gzip
import io
import math
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
        try:
            from .upload_gate import classify_stl
            upload_gate = classify_stl(str(stl_path))
            upload_gate.pop("path", None)
        except Exception as exc:  # the gate must never block inference
            upload_gate = {"verdict": "unsure", "reasons": [f"gate error: {exc}"], "features": {}}
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
        "upload_gate": upload_gate,
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


# ── /v1/recommend ─────────────────────────────────────────────────────────────
# 플랫폼의 "Recommend lower-Cd shapes". 계산은 aox_g3.recommend 에, 여기는
# /v1/infer 와 같은 인증·업로드·임시 디렉터리·락·오류 규약만 맡는다.
RECOMMEND_MAX_CANDIDATES = int(os.environ.get("G3_RECOMMEND_MAX_CANDIDATES", "12"))
RECOMMEND_MAX_CONTROL_POINTS = 72


def _parse_control_points(raw: str) -> list[list[float]]:
    import json

    try:
        points = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="control_points must be JSON") from exc
    if not isinstance(points, list) or not 1 <= len(points) <= RECOMMEND_MAX_CONTROL_POINTS:
        raise HTTPException(
            status_code=400,
            detail=f"control_points must hold 1 to {RECOMMEND_MAX_CONTROL_POINTS} XYZ points",
        )
    parsed: list[list[float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            raise HTTPException(status_code=400, detail="every control point must be an XYZ triple")
        try:
            values = [float(v) for v in point]
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="control points must be numeric") from exc
        if not all(math.isfinite(v) for v in values):
            raise HTTPException(status_code=400, detail="control points must be finite")
        parsed.append(values)
    return parsed


@app.post("/v1/recommend")
async def recommend(
    stl: UploadFile = File(...),
    control_points: str = Form(...),
    top: int = Form(3),
    symmetric: bool = Form(True),
    u_x: float = Form(30.0),
    density: float = Form(1.225),
    flow_axis: str = Form("+x"),
    ref_length: float = Form(5.0),
    ref_area: float = Form(1.0),
    grid_x: int = Form(96),
    grid_y: int = Form(64),
    grid_z: int = Form(48),
    authorization: str | None = Header(default=None),
):
    import numpy as np
    import trimesh

    from . import recommend as rec

    _authorize(authorization)
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="G3 checkpoint is unavailable")
    file_name = (stl.filename or "").lower()
    if not (file_name.endswith(".stl") or file_name.endswith(".stl.gz")):
        raise HTTPException(status_code=400, detail="only STL uploads are accepted")
    if not (8 <= grid_x <= 128 and 8 <= grid_y <= 128 and 8 <= grid_z <= 128):
        raise HTTPException(status_code=400, detail="grid dimensions must be between 8 and 128")
    points = _parse_control_points(control_points)
    top = max(1, min(int(top), len(points)))

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
    with tempfile.TemporaryDirectory(prefix="aox-g3-rec-") as temp_dir:
        temp = Path(temp_dir)
        base_path = temp / "baseline.stl"
        base_path.write_bytes(payload)
        try:
            mesh = trimesh.load(str(base_path), force="mesh", process=False)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not read STL: {exc}") from exc
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces)
        normals = np.asarray(mesh.vertex_normals, dtype=float)
        if len(vertices) == 0 or len(faces) == 0:
            raise HTTPException(status_code=400, detail="STL has no triangles")

        length, width_centre = rec.vehicle_frame(vertices)
        magnitude = rec.PUSH_FRACTION * length
        radius = rec.RADIUS_FRACTION * length

        def evaluate(stl_path: Path, out_dir: Path) -> dict:
            argv = [
                "--stl", str(stl_path),
                "--model", str(MODEL_PATH),
                "--out-dir", str(out_dir),
                "--grid", str(grid_x), str(grid_y), str(grid_z),
                "--u-x", str(u_x),
                "--density", str(density),
                "--ref-length", str(ref_length),
                "--ref-area", str(ref_area),
                "--coefficient-expert", COEFFICIENT_EXPERT,
            ]
            return infer_fields(argv)

        try:
            async with INFERENCE_LOCK:
                baseline = await run_in_threadpool(evaluate, base_path, temp / "baseline")
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"G3 baseline inference failed: {exc}") from exc
        cd0 = baseline["drag_coefficient"]
        cl0 = baseline["lift_coefficient"]

        chosen = rec.select_candidates(np.asarray(points), RECOMMEND_MAX_CANDIDATES)
        results: list[dict] = []
        for order, control_id in enumerate(chosen):
            pushed, vertex_index, displacement = rec.push_inward(
                vertices, normals, np.asarray(points[control_id]), magnitude, radius,
                symmetric=symmetric, width_centre=width_centre,
            )
            candidate_path = temp / f"candidate-{order:02d}.stl"
            trimesh.Trimesh(pushed, faces, process=False).export(str(candidate_path))
            try:
                # 후보마다 락을 따로 잡는다 — 그 사이 /v1/infer 가 끼어들 수 있게.
                async with INFERENCE_LOCK:
                    summary = await run_in_threadpool(evaluate, candidate_path, temp / f"candidate-{order:02d}")
                cd = summary["drag_coefficient"]
                cl = summary["lift_coefficient"]
            except Exception as exc:  # 한 후보가 죽어도 나머지는 돌려준다
                cd = cl = None
                failure = str(exc)
            else:
                failure = None
            origin = vertices[vertex_index]
            results.append({
                # STL 단위를 모른다(m 도 mm 도 온다). 전장 비율로만 말한다.
                # 프론트가 label 을 SVG id 와 React key 로 쓴다 — 공백·% 없이.
                "label": f"control-{control_id}",
                "control_id": int(control_id),
                "position": [float(v) for v in origin],
                "displacement": [float(v) for v in displacement],
                "influence_radius": float(radius),
                "symmetric": bool(symmetric),
                "preview_points": rec.preview_points(pushed, origin, radius),
                "cd": cd,
                "cl": cl,
                "delta_cd": None if cd is None or cd0 is None else float(cd - cd0),
                "delta_cl": None if cl is None or cl0 is None else float(cl - cl0),
                "error": failure,
            })

    ranked = rec.rank(results, top)
    return {
        "model": MODEL_PATH.resolve().name,
        "device": baseline.get("device"),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "candidate_count": len(results),
        "requested_control_points": len(points),
        "baseline": {"cd": cd0, "cl": cl0},
        "overview_points": rec.overview_points(vertices),
        "overview_bounds": {
            "min": [float(v) for v in vertices.min(axis=0)],
            "max": [float(v) for v in vertices.max(axis=0)],
        },
        "push_fraction": rec.PUSH_FRACTION,
        "radius_fraction": rec.RADIUS_FRACTION,
        "vehicle_length": float(length),
        "recommendations": ranked,
        "limitations": [
            f"Screened {len(results)} of {len(points)} control points, spread along the length axis.",
            f"Each candidate pushes the surface inward by {rec.PUSH_FRACTION:.1%} of the vehicle length "
            f"({magnitude:.4g} in STL units) with an influence radius of {rec.RADIUS_FRACTION:.0%} of length "
            f"({radius:.4g}); outward pushes are not tried.",
            "Ranked by predicted delta Cd only; delta Cl is reported, not ranked.",
            "Finite-difference screening on the surrogate, not a sensitivity solve.",
        ],
    }
