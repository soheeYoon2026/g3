#!/usr/bin/env python3
"""Authenticated HTTP wrapper for the raw-STL AI inference command."""
from __future__ import annotations

import asyncio
import gzip
import hashlib
import hmac
import json
import os
import subprocess
import tempfile
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool


ROOT = Path(os.environ.get("G3_DOMINO_ROOT", "/home/ubuntu/g3-v2"))
PYTHON = Path(os.environ.get("G3_DOMINO_PYTHON", "/home/ubuntu/g3/venv/bin/python"))
SCRIPT = Path(os.environ.get("G3_DOMINO_SCRIPT", ROOT / "scripts/infer_domino_stl.py"))
RECOMMEND_SCRIPT = Path(
    os.environ.get("G3_DOMINO_RECOMMEND_SCRIPT", ROOT / "scripts/recommend_domino_deformations.py")
)
CHECKPOINT = Path(
    os.environ.get(
        "G3_DOMINO_CHECKPOINT",
        ROOT / "var/domino-automotive-runs/decoder-30epoch.pt",
    )
)
# Recommendations only need ΔCd between deformations of one car, and the
# 2026-08-26 gates showed challenger-mixed-v1 wins that axis (5/6 direction,
# ΔCd MAE 0.0051) while losing on absolute Cd. Route the two tasks separately;
# unset the env var to fall back to a single model.
RECOMMEND_CHECKPOINT = Path(
    os.environ.get("G3_DOMINO_RECOMMEND_CHECKPOINT", str(CHECKPOINT))
)
TOKEN_FILE = Path(os.environ.get("G3_API_TOKEN_FILE", "/home/ubuntu/g3/.service-token"))
MAX_UPLOAD_BYTES = int(os.environ.get("G3_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
TIMEOUT_SECONDS = int(os.environ.get("G3_DOMINO_TIMEOUT_SECONDS", "300"))
INFERENCE_LOCK = asyncio.Lock()
RESIDENT_ENABLED = os.environ.get("G3_DOMINO_RESIDENT", "1") != "0"
RESIDENT_ENGINE = None
RECOMMEND_ENGINE = None
TRAINING_SCRIPT = ROOT / "scripts/run_domino_collection_nightly.py"
TRAINING_RUNTIME = ROOT / "var/domino-collector"
TRAINING_SPLIT = ROOT / "data/domino-g2-reaudit-v1/split.automotive-reviewed.json"
MODEL_METRICS = Path(str(CHECKPOINT) + ".metrics.json")

app = FastAPI(title="AOX AI Inference", version="0.1.0")


def _load_resident_engine():
    global RESIDENT_ENGINE
    if RESIDENT_ENGINE is None:
        from scripts.domino_resident_engine import ResidentDominoEngine

        RESIDENT_ENGINE = ResidentDominoEngine(CHECKPOINT, device="cuda:0")
    return RESIDENT_ENGINE


def _load_recommend_engine():
    global RECOMMEND_ENGINE
    if RECOMMEND_CHECKPOINT == CHECKPOINT:
        return _load_resident_engine()
    if RECOMMEND_ENGINE is None:
        from scripts.domino_resident_engine import ResidentDominoEngine

        RECOMMEND_ENGINE = ResidentDominoEngine(RECOMMEND_CHECKPOINT, device="cuda:0")
    return RECOMMEND_ENGINE


@app.on_event("startup")
async def load_model_on_startup() -> None:
    if RESIDENT_ENABLED:
        await run_in_threadpool(_load_resident_engine)


def _model_name() -> str:
    return "AI"


def _token() -> str:
    configured = os.environ.get("G3_API_TOKEN", "").strip()
    return configured or (TOKEN_FILE.read_text().strip() if TOKEN_FILE.is_file() else "")


def _authorize(authorization: str | None) -> None:
    token = _token()
    if not token:
        raise HTTPException(status_code=503, detail="G3 API token is unavailable")
    provided = authorization.removeprefix("Bearer ") if authorization else ""
    configured_hash = os.environ.get("G3_API_TOKEN_SHA256", "").strip().lower()
    direct_match = hmac.compare_digest(provided, token)
    hash_match = bool(configured_hash) and hmac.compare_digest(
        hashlib.sha256(provided.encode()).hexdigest(), configured_hash
    )
    if not direct_match and not hash_match:
        raise HTTPException(status_code=401, detail="invalid inference token")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": _model_name(),
        "model_ready": CHECKPOINT.is_file(),
        "script_ready": SCRIPT.is_file(),
        "python_ready": PYTHON.is_file(),
        "resident_enabled": RESIDENT_ENABLED,
        "resident_ready": RESIDENT_ENGINE is not None,
        "checkpoint": CHECKPOINT.name,
        "recommend_checkpoint": RECOMMEND_CHECKPOINT.name,
        "recommend_expert_split": RECOMMEND_CHECKPOINT != CHECKPOINT,
        "recommend_ready": RECOMMEND_CHECKPOINT.is_file(),
    }


def _run(command: list[str]) -> None:
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )


def _recommend_with_resident_engine(
    stl_path: Path,
    output_dir: Path,
    density: float,
    flow_axis: str,
    top: int,
    symmetric: bool,
    control_points: str,
) -> dict:
    from scripts.recommend_domino_deformations import main as generate_recommendations

    result = generate_recommendations(
        engine=_load_recommend_engine(),
        argv=[
            "--stl", str(stl_path),
            "--checkpoint", str(RECOMMEND_CHECKPOINT),
            "--out", str(output_dir),
            "--density", str(density),
            "--flow-axis", flow_axis,
            "--top", str(top),
            "--symmetric" if symmetric else "--no-symmetric",
        ] + (["--control-points-json", control_points] if control_points else []),
    )
    return _reanchor_to_absolute_model(result, stl_path, density, flow_axis)


def _reanchor_to_absolute_model(result: dict, stl_path: Path, density: float, flow_axis: str) -> dict:
    """Keep the expert's ΔCd but report absolutes from the serving model.

    With two models, the expert's own baseline Cd would contradict the number
    /v1/infer shows for the same car. Anchor every absolute on the serving
    model and add the expert's deltas, so one car has one Cd everywhere.
    """
    if RECOMMEND_CHECKPOINT == CHECKPOINT:
        return result
    baseline = result.get("baseline") or {}
    expert_cd, expert_cl = baseline.get("cd"), baseline.get("cl")
    if expert_cd is None or expert_cl is None:
        return result
    engine = _load_resident_engine()
    anchor = engine.predict(
        stl_path,
        geometry_key=f"anchor:{stl_path}",
        flow_axis=flow_axis,
        density=density,
        reference_area=baseline.get("reference_area_m2"),
    )
    anchor_cd, anchor_cl = anchor["cd"], anchor["cl"]
    baseline.update({"cd": anchor_cd, "cl": anchor_cl,
                     "expert_cd": expert_cd, "expert_cl": expert_cl})
    for row in result.get("recommendations") or []:
        for field, offset in (("cd", anchor_cd - expert_cd), ("cl", anchor_cl - expert_cl)):
            if row.get(field) is not None:
                row[f"expert_{field}"] = row[field]
                row[field] = row[field] + offset
        if row.get("coarse_cd") is not None:
            row["coarse_cd"] = row["coarse_cd"] + (anchor_cd - expert_cd)
    result["absolute_model"] = CHECKPOINT.name
    result["delta_model"] = RECOMMEND_CHECKPOINT.name
    return result


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {} if default is None else default


def _training_running() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "[r]un_domino_collection_nightly.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _start_training_job(*, train: bool) -> None:
    if _training_running():
        raise HTTPException(status_code=409, detail="G3 data pipeline is already running")
    log_dir = TRAINING_RUNTIME / "api-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log = (log_dir / f"{'train' if train else 'refresh'}-{stamp}.log").open("a", encoding="utf-8")
    command = [
        str(PYTHON), str(TRAINING_SCRIPT), "--root", str(ROOT),
        "--out", "data/domino-g2-clean-v1", "--minimum-free-gb", "10",
    ]
    if train:
        command += ["--train", "--min-new-cases", "0", "--epochs", "5", "--device", "cuda:0"]
    subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    log.close()


@app.get("/v1/training/status")
def training_status(authorization: str | None = Header(default=None)):
    _authorize(authorization)
    latest = _read_json(TRAINING_RUNTIME / "latest.json")
    collection = latest.get("collection") or {}
    split = _read_json(TRAINING_SPLIT)
    metrics = _read_json(MODEL_METRICS)
    validation = metrics.get("finetuned_validation") or {}
    history = metrics.get("validation_history") or []
    running = _training_running()
    failed = latest.get("status") in {"failed", "training-failed"}
    split_counts = {
        "train": len(split.get("train_cases") or []),
        "validation": len(split.get("validation_cases") or []),
        "test": len(split.get("test_cases") or []),
        "excluded": len(split.get("excluded_cases") or []),
        "unassigned": 0,
    }
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "collector": {
            "status": "running" if running else latest.get("status", "unknown"),
            "lastRunAt": latest.get("finished_at") or latest.get("started_at"),
            "nextRunAt": None,
            "accepted": int(collection.get("accepted") or 0),
            "acceptedThisRun": int(collection.get("accepted_this_run") or latest.get("new_accepted_cases") or 0),
            "attemptedSources": int(collection.get("attempted_sources") or 0),
            "rejectedSources": int(collection.get("rejected_sources") or 0),
            "uniqueGroups": int(collection.get("unique_groups") or 0),
        },
        "split": split_counts,
        "model": {
            "name": _model_name(),
            "checkpoint": str(CHECKPOINT),
            "epochs": len(history) or None,
            "trainedAt": datetime.fromtimestamp(CHECKPOINT.stat().st_mtime, timezone.utc).isoformat() if CHECKPOINT.is_file() else None,
            "cdMae": validation.get("cd_mae"),
            "clMae": validation.get("cl_mae"),
            "promoted": True,
        },
        "pipeline": [
            {"key": "collect", "label": "데이터 수집", "status": "running" if running else ("failed" if failed else "complete"), "detail": f"유효 데이터 {int(collection.get('accepted') or 0)}건"},
            {"key": "quality", "label": "품질 검사", "status": "waiting" if running else ("warning" if failed else "complete"), "detail": f"제외 {int(collection.get('rejected_sources') or 0)}건"},
            {"key": "classify", "label": "형상 분류", "status": "complete", "detail": f"자동차 외 형상 {split_counts['excluded']}건 제외"},
            {"key": "split", "label": "데이터 분할", "status": "complete", "detail": f"학습 {split_counts['train']} · 검증 {split_counts['validation']} · 테스트 {split_counts['test']}"},
            {"key": "train", "label": "모델 학습", "status": "running" if running and latest.get("training_requested") else ("warning" if failed else "complete"), "detail": f"AI · {len(history) or 30} epochs"},
            {"key": "evaluate", "label": "모델 평가", "status": "waiting" if running else "complete", "detail": "운영 모델 자동 교체 없음"},
        ],
    }


@app.post("/v1/training/refresh")
def training_refresh(authorization: str | None = Header(default=None)):
    _authorize(authorization)
    _start_training_job(train=False)
    return {"status": "started"}


@app.post("/v1/training/train")
def training_train(authorization: str | None = Header(default=None)):
    _authorize(authorization)
    _start_training_job(train=True)
    return {"status": "started"}


async def _save_upload(stl: UploadFile, directory: Path) -> Path:
    name = (stl.filename or "").lower()
    if not (name.endswith(".stl") or name.endswith(".stl.gz")):
        raise HTTPException(status_code=400, detail="only STL uploads are accepted")
    upload_path = directory / ("input.stl.gz" if name.endswith(".gz") else "input.stl")
    size = 0
    with upload_path.open("wb") as output:
        while chunk := await stl.read(8 * 1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=400, detail="STL exceeds the 2048 MB upload limit")
            output.write(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="STL is empty")
    if not name.endswith(".gz"):
        return upload_path
    stl_path = directory / "input.stl"
    try:
        with gzip.open(upload_path, "rb") as source, stl_path.open("wb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    except (OSError, EOFError) as exc:
        raise HTTPException(status_code=400, detail="invalid compressed STL") from exc
    return stl_path


@app.post("/v1/recommend")
async def recommend(
    stl: UploadFile = File(...),
    density: float = Form(1.225),
    flow_axis: str = Form("+x"),
    top: int = Form(3),
    symmetric: bool = Form(True),
    control_points: str = Form(""),
    authorization: str | None = Header(default=None),
):
    _authorize(authorization)
    if flow_axis not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
        raise HTTPException(status_code=400, detail="invalid flow_axis")
    if density <= 0 or not 1 <= top <= 5:
        raise HTTPException(status_code=400, detail="invalid density or top")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="aox-domino-recommend-") as temp_dir:
        temp = Path(temp_dir)
        try:
            stl_path = await _save_upload(stl, temp)
            output_dir = temp / "recommendations"
            async with INFERENCE_LOCK:
                result = await run_in_threadpool(
                    _recommend_with_resident_engine,
                    stl_path,
                    output_dir,
                    density,
                    flow_axis,
                    top,
                    symmetric,
                    control_points,
                )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail="recommendation timed out") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip().splitlines()[-1]
            raise HTTPException(status_code=422, detail=f"recommendation failed: {detail}") from exc
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail=f"recommendation failed: {exc}") from exc
    return {
        "model": _model_name(),
        "device": "cuda:0",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "baseline": result["baseline"],
        "overview_points": result.get("overview_points", []),
        "overview_bounds": result.get("overview_bounds"),
        "recommendations": result["recommendations"],
        "candidate_count": len(result["candidates"]),
        "absolute_model": result.get("absolute_model", CHECKPOINT.name),
        "delta_model": result.get("delta_model", CHECKPOINT.name),
        "limitations": ["Surface-control sensitivity screening; validate the applied shape with G2 CFD."],
    }


@app.post("/v1/infer")
async def infer(
    stl: UploadFile = File(...),
    speed: float = Form(30.0),
    u_x: float | None = Form(None),
    density: float = Form(1.225),
    ref_area: float | None = Form(None),
    flow_axis: str = Form("+x"),
    authorization: str | None = Header(default=None),
):
    _authorize(authorization)
    if not all((CHECKPOINT.is_file(), SCRIPT.is_file(), PYTHON.is_file())):
        raise HTTPException(status_code=503, detail="AI service files are unavailable")
    if flow_axis not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
        raise HTTPException(status_code=400, detail="invalid flow_axis")
    if speed <= 0 or density <= 0:
        raise HTTPException(status_code=400, detail="speed and density must be positive")
    if ref_area is not None and ref_area <= 0:
        raise HTTPException(status_code=400, detail="reference area must be positive")

    name = (stl.filename or "").lower()
    if not (name.endswith(".stl") or name.endswith(".stl.gz")):
        raise HTTPException(status_code=400, detail="only STL uploads are accepted")
    started = time.perf_counter()
    timings = {}
    with tempfile.TemporaryDirectory(prefix="aox-domino-") as temp_dir:
        temp = Path(temp_dir)
        upload_path = temp / ("input.stl.gz" if name.endswith(".gz") else "input.stl")
        size = 0
        digest = hashlib.sha256()
        upload_started = time.perf_counter()
        with upload_path.open("wb") as output:
            while chunk := await stl.read(8 * 1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=400, detail="STL exceeds the 2048 MB upload limit")
                output.write(chunk)
                digest.update(chunk)
        timings["upload_receive_seconds"] = time.perf_counter() - upload_started
        if size == 0:
            raise HTTPException(status_code=400, detail="STL is empty")

        stl_path = temp / "input.stl"
        if name.endswith(".gz"):
            decompress_started = time.perf_counter()
            try:
                with gzip.open(upload_path, "rb") as source, stl_path.open("wb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            except (OSError, EOFError) as exc:
                raise HTTPException(status_code=400, detail="invalid compressed STL") from exc
            timings["decompress_seconds"] = time.perf_counter() - decompress_started
        # Warn when the upload is not a complete car: the model's accuracy
        # claims only hold for road cars (2026-08-26 gate work). Advisory only.
        try:
            from aox_g3.upload_gate import classify_stl

            upload_gate = await run_in_threadpool(classify_stl, str(stl_path))
            upload_gate.pop("path", None)
        except Exception as exc:
            upload_gate = {"verdict": "unsure", "reasons": [f"gate error: {exc}"], "features": {}}

        try:
            worker_started = time.perf_counter()
            async with INFERENCE_LOCK:
                if RESIDENT_ENABLED:
                    engine = await run_in_threadpool(_load_resident_engine)
                    result = await run_in_threadpool(
                        engine.predict,
                        stl_path,
                        digest.hexdigest(),
                        flow_axis,
                        density,
                        ref_area,
                    )
                else:
                    result_path = temp / "result.json"
                    command = [
                        str(PYTHON), str(SCRIPT), "--stl", str(stl_path),
                        "--checkpoint", str(CHECKPOINT), "--out", str(result_path),
                        "--speed", str(speed if u_x is None else u_x),
                        "--density", str(density), "--flow-axis", flow_axis,
                        "--device", "cuda:0",
                    ]
                    if ref_area is not None:
                        command.extend(["--ref-area", str(ref_area)])
                    await run_in_threadpool(_run, command)
                    result = json.loads(result_path.read_text())
            timings["worker_process_seconds"] = time.perf_counter() - worker_started
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail="AI prediction timed out") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip().splitlines()[-1]
            raise HTTPException(status_code=422, detail=f"AI prediction failed: {detail}") from exc
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail=f"AI result is invalid: {exc}") from exc

    timings.update(result.get("timings") or {})
    if RESIDENT_ENABLED:
        timings.update({
            "domino_prepare_seconds": result.get("preparation_seconds", 0.0),
            "gpu_inference_seconds": result.get("inference_seconds", 0.0),
            "coefficient_integration_seconds": result.get("coefficient_integration_seconds", 0.0),
        })
    timings["service_total_seconds"] = time.perf_counter() - started
    return {
        "model": _model_name(),
        "device": "cuda:0",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "upload_gate": upload_gate,
        "inference_seconds": result.get("inference_seconds"),
        "cache_hit": result.get("cache_hit", False),
        "timings": {key: round(value, 6) for key, value in timings.items()},
        "drag_coefficient": result.get("cd"),
        "lift_coefficient": result.get("cl"),
        "reference_area": result.get("reference_area_m2"),
        "reference_area_source": result.get("reference_area_source"),
        "flow_axis": result.get("flow_axis") or flow_axis,
        "speed": result.get("speed_mps") or (speed if u_x is None else u_x),
        "density": result.get("density_kg_m3") or density,
        "limitations": result.get("limitations") or [
            "checkpoint is trained near its dataset conditions",
            "STL units must be metres and the selected flow axis must be correct",
        ],
    }
