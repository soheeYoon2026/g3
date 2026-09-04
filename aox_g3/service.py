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
import uuid
import threading
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
    from .preview import render_candidate_png

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
            preview_png = None
            try:
                weights = rec.highlight_weights(
                    vertices, origin, radius, symmetric=symmetric, width_centre=width_centre,
                )
                png_path = temp / f"candidate-{order:02d}.png"
                await run_in_threadpool(
                    render_candidate_png, vertices, faces, weights, origin, displacement, png_path,
                )
                preview_png = "data:image/png;base64," + base64.b64encode(png_path.read_bytes()).decode("ascii")
            except Exception:  # 그림이 없어도 추천은 돌려준다 — 렌더 실패가 기능을 막으면 안 된다
                preview_png = None
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
                "preview_png": preview_png,
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


# ── /v1/recommend/optimize — BO+GP 형상 최적화, 비동기 잡 ────────────────────
# 평가 40번이면 1분 안팎이라 요청 안에서 기다리게 하지 않는다. 잡을 만들고 바로
# 돌려주고, 상태는 GET 으로 묻는다. 잡 기록은 프로세스 메모리에만 있다(테스트 박스).
OPTIMIZE_BUDGET = int(os.environ.get("G3_OPTIMIZE_BUDGET", "40"))
OPTIMIZE_MAX_POINTS = int(os.environ.get("G3_OPTIMIZE_MAX_POINTS", "6"))
OPTIMIZE_DELTA_FRACTION = float(os.environ.get("G3_OPTIMIZE_DELTA_FRACTION", "0.02"))
OPTIMIZE_JOBS: dict[str, dict] = {}
OPTIMIZE_JOBS_LOCK = threading.Lock()
OPTIMIZE_JOB_TTL_SECONDS = 6 * 3600


def _optimize_job_get(job_id: str) -> dict | None:
    with OPTIMIZE_JOBS_LOCK:
        job = OPTIMIZE_JOBS.get(job_id)
        return dict(job) if job else None


def _optimize_job_update(job_id: str, **fields) -> None:
    with OPTIMIZE_JOBS_LOCK:
        job = OPTIMIZE_JOBS.get(job_id)
        if job is not None:
            job.update(fields)
            job["updated_at"] = time.time()


def _optimize_jobs_expire() -> None:
    cutoff = time.time() - OPTIMIZE_JOB_TTL_SECONDS
    with OPTIMIZE_JOBS_LOCK:
        for key in [k for k, v in OPTIMIZE_JOBS.items() if v.get("updated_at", 0) < cutoff]:
            OPTIMIZE_JOBS.pop(key, None)


async def _locked_infer(argv: list[str]) -> dict:
    async with INFERENCE_LOCK:
        return await run_in_threadpool(infer_fields, argv)


async def _run_optimize_job(
    job_id: str,
    payload: bytes,
    points: list[list[float]],
    *,
    top: int,
    symmetric: bool,
    budget: int,
    infer_kwargs: dict,
) -> None:
    import numpy as np
    import trimesh

    from . import optimize as opt
    from .preview import render_candidate_png

    loop = asyncio.get_running_loop()
    started = time.perf_counter()
    _optimize_job_update(job_id, status="running")
    try:
        with tempfile.TemporaryDirectory(prefix="aox-g3-opt-") as temp_dir:
            temp = Path(temp_dir)
            base_path = temp / "baseline.stl"
            base_path.write_bytes(payload)
            mesh = trimesh.load(str(base_path), force="mesh", process=False)
            vertices = np.asarray(mesh.vertices, dtype=float)
            faces = np.asarray(mesh.faces)
            normals = np.asarray(mesh.vertex_normals, dtype=float)
            if len(vertices) == 0 or len(faces) == 0:
                raise ValueError("STL has no triangles")

            param = opt.parametrise(
                vertices, normals, np.asarray(points, dtype=float),
                max_points=OPTIMIZE_MAX_POINTS, delta_fraction=OPTIMIZE_DELTA_FRACTION, symmetric=symmetric,
            )
            counter = {"n": 0}

            def argv_for(stl_path: Path, out_dir: Path) -> list[str]:
                return [
                    "--stl", str(stl_path), "--model", str(MODEL_PATH), "--out-dir", str(out_dir),
                    "--grid", str(infer_kwargs["grid_x"]), str(infer_kwargs["grid_y"]), str(infer_kwargs["grid_z"]),
                    "--u-x", str(infer_kwargs["u_x"]), "--density", str(infer_kwargs["density"]),
                    "--ref-length", str(infer_kwargs["ref_length"]), "--ref-area", str(infer_kwargs["ref_area"]),
                    "--coefficient-expert", COEFFICIENT_EXPERT,
                ]

            def evaluate(deformed: np.ndarray) -> dict:
                # optimise() 는 워커 스레드에서 돈다. 추론은 이벤트 루프의 락 아래에서 —
                # /v1/infer 와 GPU 를 번갈아 쓰게 — 코루틴으로 넘기고 결과를 기다린다.
                counter["n"] += 1
                stl_path = temp / f"design-{counter['n']:03d}.stl"
                trimesh.Trimesh(deformed, faces, process=False).export(str(stl_path))
                summary = asyncio.run_coroutine_threadsafe(
                    _locked_infer(argv_for(stl_path, temp / f"design-{counter['n']:03d}")), loop
                ).result()
                expert = summary.get("coefficient_experts", {}).get(summary.get("coefficient_expert"), {})
                # 분포밖(OOD) 판정은 실현불가로 넣는다 — 서로게이트가 모르는 형상의 Cd 로
                # 최적화하면 노이즈를 좇는다. 실현가능성 GP 가 그 지대를 피하게 한다.
                in_distribution = expert.get("in_distribution")
                ok = summary.get("drag_coefficient") is not None and in_distribution is not False
                return {
                    "cd": summary.get("drag_coefficient"),
                    "cl": summary.get("lift_coefficient"),
                    "ok": ok,
                    "error": None if ok else "out of distribution",
                }

            def on_progress(done: int, total: int) -> None:
                _optimize_job_update(job_id, progress={"done": done, "total": total})

            # baseline 은 optimise() 안에서 s=0 으로 한 번 평가된다.
            result = await run_in_threadpool(
                opt.optimise, evaluate, param, vertices, budget=budget, top=top, on_progress=on_progress,
            )
            base = result.baseline if (result.baseline and result.baseline["ok"]) else None
            cd0 = base["cd"] if base else None
            cl0 = base["cl"] if base else None

            recommendations = []
            for item in result.best:
                scales = np.asarray(item["scales"], dtype=float)
                deformed = param.deform(vertices, scales)
                preview_png = None
                try:
                    png_path = temp / f"best-{item['rank']:02d}.png"
                    # 변형된 형상을 그대로 캡처한다 — 사용자 요청: 움직인 형상을 보여 줄 것.
                    biggest = int(np.argmax(np.abs(scales))) if len(scales) else 0
                    await run_in_threadpool(
                        render_candidate_png, deformed, faces, param.weights(vertices, scales),
                        param.origins[biggest], param.inward[biggest] * (1.0 if scales[biggest] >= 0 else -1.0), png_path,
                    )
                    preview_png = "data:image/png;base64," + base64.b64encode(png_path.read_bytes()).decode("ascii")
                except Exception:
                    preview_png = None
                moves = item["moves"]
                lead = max(moves, key=lambda m: abs(m["scale"])) if moves else None
                recommendations.append({
                    "label": f"design-{item['rank']}",
                    "rank": item["rank"],
                    "control_id": lead["control_id"] if lead else None,
                    "position": lead["position"] if lead else None,
                    "displacement": lead["displacement"] if lead else None,
                    "moves": moves,
                    "scales": item["scales"],
                    "influence_radius": float(param.radius),
                    "symmetric": bool(symmetric),
                    "preview_points": [],
                    "preview_png": preview_png,
                    "cd": item["cd"],
                    "cl": item["cl"],
                    "delta_cd": None if cd0 is None else float(item["cd"] - cd0),
                    "delta_cl": None if cl0 is None or item["cl"] is None else float(item["cl"] - cl0),
                    "gp_std": item["gp_std"],
                })

            _optimize_job_update(job_id, status="succeeded", result={
                "model": MODEL_PATH.resolve().name,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "candidate_count": result.evaluated,
                "failed_count": result.failed,
                "budget": budget,
                "delta_fraction": OPTIMIZE_DELTA_FRACTION,
                "max_points": OPTIMIZE_MAX_POINTS,
                "control_ids": param.control_ids,
                "baseline": {"cd": cd0, "cl": cl0},
                "recommendations": recommendations,
                "history": [
                    {"kind": h["kind"], "cd": h["cd"], "cl": h["cl"], "ok": h["ok"], "scales": h["scales"], "error": h["error"]}
                    for h in result.history
                ],
                "limitations": [
                    f"Bayesian optimisation over {param.size} control points, each moving up to "
                    f"{OPTIMIZE_DELTA_FRACTION:.0%} of the vehicle length along its surface normal (in or out).",
                    f"{result.evaluated} surrogate evaluations ({result.failed} failed); ranked by predicted Cd.",
                    "The GP standard deviation is the optimiser's own uncertainty, not the surrogate's validation error.",
                    "Screening on the surrogate, not CFD. Validate the chosen shape with G2 before use.",
                ],
            })
    except Exception as exc:
        _optimize_job_update(job_id, status="failed", error=str(exc))


@app.post("/v1/recommend/optimize", status_code=202)
async def recommend_optimize(
    stl: UploadFile = File(...),
    control_points: str = Form(...),
    top: int = Form(5),
    symmetric: bool = Form(True),
    budget: int = Form(OPTIMIZE_BUDGET),
    u_x: float = Form(30.0),
    density: float = Form(1.225),
    ref_length: float = Form(5.0),
    ref_area: float = Form(1.0),
    grid_x: int = Form(96),
    grid_y: int = Form(64),
    grid_z: int = Form(48),
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
    points = _parse_control_points(control_points)
    top = max(1, min(int(top), 10))
    budget = max(8, min(int(budget), 80))

    payload = await stl.read(MAX_UPLOAD_BYTES + 1)
    if file_name.endswith(".gz"):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(payload)) as compressed:
                payload = compressed.read(MAX_UPLOAD_BYTES + 1)
        except (OSError, EOFError) as exc:
            raise HTTPException(status_code=400, detail="invalid compressed STL") from exc
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="STL is empty or exceeds the upload limit")

    _optimize_jobs_expire()
    job_id = uuid.uuid4().hex
    with OPTIMIZE_JOBS_LOCK:
        OPTIMIZE_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": {"done": 0, "total": budget},
            "created_at": time.time(),
            "updated_at": time.time(),
            "result": None,
            "error": None,
        }
    asyncio.create_task(_run_optimize_job(
        job_id, payload, points, top=top, symmetric=symmetric, budget=budget,
        infer_kwargs={
            "u_x": u_x, "density": density, "ref_length": ref_length, "ref_area": ref_area,
            "grid_x": grid_x, "grid_y": grid_y, "grid_z": grid_z,
        },
    ))
    return {"job_id": job_id, "status": "queued", "progress": {"done": 0, "total": budget}}


@app.get("/v1/recommend/optimize/{job_id}")
def recommend_optimize_status(job_id: str, authorization: str | None = Header(default=None)):
    _authorize(authorization)
    job = _optimize_job_get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown optimisation job")
    return job
