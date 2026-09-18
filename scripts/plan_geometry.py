"""Rule-based controller: diagnose -> route -> run -> verify -> retry, and stop
where only the customer can decide.

    plan_geometry.py --in X.stp|X.stl --out var/runs/x [--answers answers.json] [--assume-defaults]

The controller never invents a tool. It calls the same scripts a person would
(prepare_geometry, flat_floor_wrap, close_with_wrap, resurface_noclose, ...),
reads their reports, and applies the rules in ROUTES below. Every decision is
written to plan.json with the measurement that triggered it. Decisions that
are the customer's (units, floor height, closed rims, which openings to keep)
become entries in questions.json with a proposal, a reason and evidence files;
the run pauses there (status "waiting_for_answers") unless --assume-defaults,
in which case the proposals are used and marked as assumptions.

Answers are a JSON object {question_id: value}; a value of null keeps the proposal.
"""

import argparse
import atexit
import os
import gc
import json
import re
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from aox_g3 import ledger
from aox_g3.run_state import RunLock, write_json
from aox_g3.measurement_cache import cached_thickness
from aox_g3.quality_display import healing_result
from aox_g3.controller_state import mesh_checkpoint, outcome
from aox_g3.mesh_cleanup import cleanup, validate

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in", dest="src", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--answers", type=Path, help="answers.json from the UI or the customer")
ap.add_argument("--assume-defaults", action="store_true", help="use every proposal instead of pausing")
ap.add_argument("--max-retries", type=int, default=2)
ap.add_argument("--time-budget", type=int, default=1500, help="seconds per script run; a run over budget is stopped and the plan falls back to a coarser route")
ap.add_argument("--no-render", action="store_true")
ap.add_argument("--force", action="store_true", help="ignore the ledger and run every step again")
args = ap.parse_args()

import trimesh  # noqa: E402

args.out.mkdir(parents=True, exist_ok=True)
try:
    RUN_LOCK = RunLock(args.out)
except RuntimeError as exc:
    print(str(exc), flush=True)
    sys.exit(2)
atexit.register(RUN_LOCK.close)
PLAN = args.out / "plan.json"
QUESTIONS = args.out / "questions.json"
LOG = open(args.out / "plan_log.txt", "a", encoding="utf-8")
python = sys.executable
STATE_IO_LOCK = threading.RLock()

plan = json.loads(PLAN.read_text(encoding="utf-8")) if PLAN.exists() else {
    "input": str(args.src), "out": str(args.out), "status": "running", "diagnosis": {}, "route": None,
    "decisions": [], "runs": [], "checks": [], "questions": [], "answers": {}, "assumptions": [],
    "deliverables": [], "warnings": []}
answers = dict(plan.get("answers", {}))
if args.answers and args.answers.exists():
    answers.update({k: v for k, v in json.loads(args.answers.read_text()).items() if v is not None})
plan["answers"] = answers
# Current checks/questions must describe this attempt, not an earlier failed route.
plan.setdefault("attempt_history", []).append({
    "status": plan.get("status"), "checks": plan.get("checks", []),
    "questions": plan.get("questions", []), "deliverables": plan.get("deliverables", [])})
for key in ("checks", "questions", "deliverables"):
    plan[key] = []
plan["retries"] = 0
plan["status"] = "running"
plan.pop("result", None)
plan["runtime"] = {"protocol": 1, "pid": os.getpid(), "started_at": time.time()}


def finalize_runtime():
    if plan.get("status") == "running":
        plan["status"] = "failed"
        plan["runtime"].setdefault("error", "완료 상태 없이 실행이 종료되었습니다.")
    plan["runtime"]["ended_at"] = time.time()
    write_json(PLAN, plan)
    LOG.close()


def runtime_exception(kind, value, traceback):
    plan["status"] = "failed"
    plan["runtime"]["error"] = f"{kind.__name__}: {value}"
    sys.__excepthook__(kind, value, traceback)


sys.excepthook = runtime_exception
atexit.register(finalize_runtime)
write_json(PLAN, plan)


def log(msg):
    # Keep the public vocabulary bounded; stage and state belong in the message.
    tags = {
        "실행": ("진행", ""), "입력": ("진행", "입력: "),
        "초기 검사": ("진단", "초기 검사: "), "분류": ("진단", "분류: "),
        "미리보기": ("진행", "미리보기: "), "두께 측정": ("진행", "두께 측정: "),
        "바닥 검사": ("진단", "바닥 검사: "), "재사용": ("진행", "재사용: "),
        "건너뜀": ("진행", "완료 단계 재사용: "), "경로 선택": ("수리", "방법 선택: "),
        "결정": ("수리", "방법 선택: "), "가정": ("질문", "제안값 적용: "),
        "답변 대기": ("질문", "답변 대기 — "), "경고": ("진행", "주의 — "),
        "끝": ("진행", "종료: "), "형상 보고": ("진단", "힐링 결과: "),
    }
    match = re.match(r"^\[([^]]+)\]\s*(.*)$", msg)
    if match and match.group(1) in tags:
        tag, description = tags[match.group(1)]
        msg = f"[{tag}] {description}{match.group(2)}"
    with STATE_IO_LOCK:
        if msg.startswith("["):
            plan["runtime"]["current_stage"] = msg
            plan["runtime"]["updated_at"] = time.time()
            write_json(PLAN, plan)
        print(msg, flush=True)
        LOG.write(msg + "\n")
        LOG.flush()


def ray_hits_batched(intersector, origins, directions, batch_size=25, progress=log):
    """First hits with global ray indices, without one huge candidate array."""
    locations, ray_indices = [], []
    start, total = 0, len(origins)
    while start < total:
        stop = min(start + batch_size, total)
        try:
            loc, ray, _ = intersector.intersects_location(
                origins[start:stop], directions[start:stop], multiple_hits=False)
        except MemoryError:
            gc.collect()
            if batch_size == 1:
                raise RuntimeError(
                    "두께 측정 메모리 부족: 광선 1개도 계산할 수 없습니다. "
                    "측정값을 추정하지 않고 중단합니다.") from None
            batch_size = max(1, batch_size // 2)
            progress(f"[두께 측정] 메모리 부족 — 광선 묶음을 {batch_size}개로 줄여 재시도")
            continue
        if len(ray):
            locations.append(loc)
            ray_indices.append(np.asarray(ray, dtype=np.int64) + start)
        start = stop
        progress(f"[두께 측정] 광선 {start:,}/{total:,} (묶음 {batch_size}개)")
    return (np.concatenate(locations) if locations else np.empty((0, 3)),
            np.concatenate(ray_indices) if ray_indices else np.empty(0, dtype=np.int64))


def save():
    with STATE_IO_LOCK:
        write_json(PLAN, plan)


def decide(what, because, evidence=None, tag=None):
    if tag is None:
        tag = "진단" if what == "바닥이 없는 껍질" else "경로 선택"
    plan["decisions"].append({"what": what, "because": because, "evidence": evidence or {}})
    log(f"[{tag}] {what}  ← {because}")
    save()


def check(name, ok, detail, on_fail=None):
    plan["checks"].append({"name": name, "ok": bool(ok), "detail": detail, "on_fail": on_fail})
    log(f"[검증] {'통과' if ok else '실패'} {name}: {detail}" + (f"  → {on_fail}" if (not ok and on_fail) else ""))
    save()
    return bool(ok)


OUTPUT_FLAGS = ("--out", "--output", "--report", "--caps-stl", "--field", "--o")


def path_args(arguments):
    """Split the arguments into input files and output files.

    A file that the step is about to write must not be hashed as an input: if it happens to
    exist from an earlier run the identity changes and the same work looks new (2026-09-14).
    """
    inputs, outputs = [], []
    previous = ""
    for a in arguments:
        text = str(a)
        if any(text.endswith(ext) for ext in (".stl", ".stp", ".step", ".obj", ".json", ".txt", ".png")):
            (outputs if previous in OUTPUT_FLAGS else inputs).append(Path(text))
        previous = text
    return inputs, outputs


def run(script, arguments, capture_name):
    cmd = [python, str(HERE / script)] + [str(a) for a in arguments]
    t0 = time.time()
    in_paths, out_paths = path_args(arguments)
    paths = in_paths + out_paths
    inputs = [p for p in in_paths if p.exists()]
    # a step usually writes into a directory it was given, so watch those too
    dirs = [Path(str(a)) for a in arguments if str(a).startswith(str(args.out)) and not any(path_args([a]))]
    def snapshot():
        seen = {}
        for p in paths:
            if p.exists():
                seen[p] = (p.stat().st_mtime_ns, p.stat().st_size)
        for d in dirs:
            if d.is_dir():
                for f in d.rglob("*"):
                    if f.is_file():
                        seen[f] = (f.stat().st_mtime_ns, f.stat().st_size)
        return seen
    before = snapshot()
    ident = ledger.identity(script, arguments, inputs, args.out)
    hit = ledger.find(args.out, ident)
    if hit and not args.force:
        log(f"[건너뜀] {script} — 같은 입력·인자로 {hit['at']} 에 끝난 단계 (산출물 {len(hit['outputs'])}개, {hit['seconds']} s)")
        plan["runs"].append({"script": script, "args": [str(a) for a in arguments], "skipped": True,
                             "from": hit["at"], "log": f"{capture_name}.txt"})
        save()
        prior = args.out / f"{capture_name}.txt"
        return True, prior.read_text(encoding="utf-8", errors="replace") if prior.exists() else ""
    log(f"[실행] {script} {' '.join(str(a) for a in arguments)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.time_budget)
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        text = out.decode() if isinstance(out, bytes) else out
        (args.out / f"{capture_name}.txt").write_text(text, encoding="utf-8")
        plan["runs"].append({"script": script, "args": [str(a) for a in arguments], "seconds": round(time.time() - t0, 1),
                             "returncode": "timeout", "log": f"{capture_name}.txt"})
        plan["warnings"].append(f"{script} 시간 예산 {args.time_budget} s 초과로 중단")
        log(f"   시간 예산 {args.time_budget} s 초과 — 중단")
        save()
        return False, "timeout"
    text = "\n".join(line for line in (proc.stdout + proc.stderr).splitlines() if "swig/python detected" not in line)
    (args.out / f"{capture_name}.txt").write_text(text, encoding="utf-8")
    plan["runs"].append({"script": script, "args": [str(a) for a in arguments], "seconds": round(time.time() - t0, 1),
                         "returncode": proc.returncode, "log": f"{capture_name}.txt"})
    save()
    after = snapshot()
    produced = [p for p, v in after.items() if before.get(p) != v]
    ledger.record(args.out, script, arguments, inputs, produced + [args.out / f"{capture_name}.txt"], ident,
                  time.time() - t0, status="done" if proc.returncode == 0 else "failed")
    if proc.returncode == 4 and script == "prepare_geometry.py":
        log("[경고] 계산은 끝났지만 랩 품질 미통과 — 후속 검증·대체 경로 진행")
        return True, text
    if proc.returncode != 0:
        log(f"   실패 (코드 {proc.returncode}): {text.strip().splitlines()[-1] if text.strip() else ''}")
        if script != "flat_floor_wrap.py" or "--no-wrap" not in [str(a) for a in arguments]:
            plan["status"] = "failed"
            plan["warnings"].append(f"{script} 실패: {text.strip().splitlines()[-1] if text.strip() else ''}")
            save()
            sys.exit(2)
    return proc.returncode == 0, text


def ask(qid, question, proposal, reason, kind="number", unit="", evidence=None, choices=None, where=None):
    """Register a customer question. Returns the answer if known, else the proposal (assumed) or None (pause).

    where: what the viewer should show for this question - {"points": [[x,y,z,r,label], ...]}
    and/or {"plane_z": z}; the page flies to it and, for plane_z, moves a plane with the answer.
    """
    entry = {"id": qid, "question": question, "type": kind, "unit": unit, "proposal": proposal, "reason": reason,
             "evidence": evidence or [], "choices": choices, "where": where or {}}
    plan["questions"] = [q for q in plan["questions"] if q["id"] != qid] + [entry]
    if qid in answers:
        if kind == "choice" and choices and answers[qid] not in choices:
            raise ValueError(f"{qid}: 지원하지 않는 답변 {answers[qid]!r}; 선택지 {choices}")
        entry["answer"] = answers[qid]
        save()
        return answers[qid]
    if args.assume_defaults:
        plan["assumptions"].append({"id": qid, "value": proposal, "reason": reason})
        entry["assumed"] = True
        save()
        log(f"[가정] {qid} = {proposal}  ({reason})")
        return proposal
    save()
    return None


def viewer_mesh(path, target=150_000, loaded_mesh=None, face_count=None):
    """A light copy of the mesh for the page's 3D viewer (viewer.stl)."""
    out = args.out / "viewer.stl"
    try:
        identity = {"input_hash": ledger.file_hash(Path(path)), "target": target, "max_error": 5.0, "version": 1}
        if not args.force and out.exists() and plan.get("viewer_identity") == identity:
            log("[미리보기] 저장된 프록시 재사용 — 같은 입력·설정")
            return out
        log("[미리보기] 원본 프록시 준비 중 — 수리 계산은 하지 않습니다")
        m = loaded_mesh
        if face_count is None:
            m = trimesh.load(path, force="mesh") if m is None else m
            face_count = len(m.faces)
        if face_count > target:
            from meshlib import mrmeshpy as MR, mrmeshnumpy as MN
            ml = MR.loadMesh(str(path))
            ds = MR.DecimateSettings()
            ds.maxDeletedFaces = int(face_count - target)
            ds.maxError = 5.0
            ds.packMesh = True
            MR.decimateMesh(ml, ds)
            m = trimesh.Trimesh(np.asarray(MN.getNumpyVerts(ml)), np.asarray(MN.getNumpyFaces(ml.topology)), process=False)
        m.export(out)
        with STATE_IO_LOCK:
            plan["viewer_mesh"] = "viewer.stl"
            plan["viewer_identity"] = identity
            plan["viewer_bbox"] = [m.bounds[0].round(1).tolist(), m.bounds[1].round(1).tolist()]
            save()
        log("[미리보기] 원본 프록시 생성 완료")
    except Exception as exc:
        plan["warnings"].append(f"viewer mesh 실패: {type(exc).__name__}: {exc}")
        log(f"[경고] 프록시 미리보기 생성 실패: {type(exc).__name__}")
        save()
    return out


def focus_render(mesh_path, point, span, name, title, mirror=False):
    """Close-up render at a point, as evidence for a question (skipped with --no-render)."""
    if args.no_render:
        return None
    out = args.out / f"focus_{name}.png"
    arguments = ["--step", str(mesh_path), "--out", str(out), f"--focus-point={point[0]:.0f},{point[1]:.0f},{point[2]:.0f}",
                 "--focus-span", str(span), "--no-holes", "--title", title] + (["--mirror"] if mirror else [])
    subprocess.run([python, str(HERE / "render_geometry.py")] + arguments, capture_output=True, text=True)
    return str(out) if out.exists() else None


def pause():
    pending = [q for q in plan["questions"] if "answer" not in q and not q.get("assumed")]
    write_json(QUESTIONS, pending)
    plan["status"] = "waiting_for_answers"
    save()
    log(f"[질문] 답변이 필요한 항목 {len(pending)}개")
    log("[답변 대기] 브라우저에서 답을 선택하고 ‘답 저장 후 이어서 실행’을 누르세요.")
    sys.exit(3)


def finish(status="done"):
    status, code = outcome(plan, status)
    if status == "waiting_for_answers":
        pause()
    plan["status"] = status
    plan["result"] = {"status": status, "exit_code": code,
                      "quality_passed": status == "done"}
    write_json(QUESTIONS, [])
    save()
    log(f"[끝] {status}  산출물: " + ", ".join(plan["deliverables"]))
    sys.exit(code)


# ----------------------------------------------------------------- diagnosis
src = args.src
is_step = src.suffix.lower() in (".stp", ".step")
diag = plan["diagnosis"]
diag["format"] = "STEP" if is_step else "mesh"

if not is_step:
    # Show native coordinates before asking which units/axis they represent.
    # This only creates a display proxy; repair remains behind customer answers.
    initial_identity = {"hash": ledger.file_hash(src), "version": 1,
                        "trimesh": trimesh.__version__, "numpy": np.__version__}
    cached_initial = plan.get("initial_mesh_diagnosis", {})
    reuse_initial = (not args.force and cached_initial.get("identity") == initial_identity
                     and cached_initial.get("status") == "done")
    log("[입력] 파일 읽는 중 — 이어갈 단계 준비")
    def load_native():
        native = trimesh.load(src, force="mesh")
        native.merge_vertices()
        return native
    mesh, native_reused = mesh_checkpoint(args.out, "native_mesh", initial_identity,
                                          load_native, force=args.force)
    if native_reused:
        log("[재사용] 저장된 메시 배열 — 원본 파싱·정점 병합 생략")
    face_count = len(mesh.faces)
    if reuse_initial:
        diag.update(cached_initial["diagnosis"])
        ext = np.asarray(cached_initial["extents_exact"], dtype=float)
        open_share = cached_initial["boundary_edge_share_exact"]
        log("[초기 검사] 저장된 결과 재사용 — 같은 입력·검사 설정")
        viewer_mesh(src, loaded_mesh=mesh.copy() if face_count <= 150_000 else None, face_count=face_count)
        # Vertex normalization is needed by downstream geometry operations, not a re-diagnosis.
    else:
        proxy_source = mesh.copy() if face_count <= 150_000 else None
        log("[초기 검사] 검사 중 — 미리보기 생성과 병렬 진행")
        with ThreadPoolExecutor(max_workers=1) as preview_pool:
            preview_future = preview_pool.submit(viewer_mesh, src, 150_000, proxy_source, face_count)
            ext = mesh.extents
            _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
            open_share = float((counts == 1).sum() / max(1, len(counts)))
            parts = mesh.split(only_watertight=False)
            measured = {"triangles": int(len(mesh.faces)), "extents": ext.round(2).tolist(),
                        "bbox_min": mesh.bounds[0].round(2).tolist(), "bodies": int(len(parts)),
                        "boundary_edge_share": round(open_share, 5), "watertight": bool(mesh.is_watertight),
                        "winding_consistent": bool(mesh.is_winding_consistent),
                        "finite_coordinates": bool(np.isfinite(mesh.vertices).all())}
            log("[초기 검사] 완료 — 미리보기 생성 종료 확인 중")
            preview_future.result()
        diag.update(measured)
        plan["initial_mesh_diagnosis"] = {"identity": initial_identity, "status": "done",
                                          "diagnosis": measured, "extents_exact": ext.tolist(),
                                          "boundary_edge_share_exact": open_share}
        save()
        log(f"[진단] 메쉬 삼각형 {len(mesh.faces):,}  치수 {ext.round(1).tolist()}  몸체 {diag['bodies']}  경계 모서리 비율 {open_share*100:.3f} %")

    # Upgrade legacy diagnostics once, without repeating split/boundary analysis.
    if "winding_consistent" not in diag or "finite_coordinates" not in diag:
        diag["winding_consistent"] = bool(mesh.is_winding_consistent)
        diag["finite_coordinates"] = bool(np.isfinite(mesh.vertices).all())
        plan["initial_mesh_diagnosis"]["diagnosis"].update({
            "winding_consistent": diag["winding_consistent"],
            "finite_coordinates": diag["finite_coordinates"]})
        save()
    if not diag["finite_coordinates"]:
        raise ValueError("입력 좌표에 NaN/무한대가 있습니다. 단위 추정·수리를 중단합니다.")

    # units: a car is 3-6 m long
    longest = float(ext.max())
    if longest < 20:
        unit_guess, scale = "m", 1000.0
    elif longest < 400:
        unit_guess, scale = "inch", 25.4
    else:
        unit_guess, scale = "mm", 1.0
    diag["unit_guess"] = unit_guess
    unit = ask("units", f"이 STL의 단위가 {unit_guess}로 보입니다(가장 긴 치수 {longest:.1f}). 맞습니까? (mm / m / inch)", unit_guess,
               f"가장 긴 치수 {longest:.1f}: 20 미만이면 m, 400 미만이면 inch, 그 이상이면 mm 로 봅니다", kind="choice",
               choices=["mm", "m", "inch"])
    length_axis = int(np.argmax(ext))
    axis_ans = ask("length_axis_now", f"지금 파일에서 차의 길이 방향은 어느 축입니까? ({'xyz'[length_axis]}축으로 보입니다. x 면 그대로 두고, y/z 면 x 로 돌립니다)",
                   "xyz"[length_axis], "파이프라인은 x 를 길이·유동 방향으로 가정합니다; 답은 현재 파일의 길이축", kind="choice", choices=["x", "y", "z"])
    if unit is None or axis_ans is None:
        pause()
    scale = {"mm": 1.0, "m": 1000.0, "inch": 25.4}[unit]
    work = args.out / "input_mm.stl"
    if src.resolve() == work.resolve():
        # Pointed at our own converted copy (a page restarted with --in <out>/input_mm.stl).
        # Converting again would apply the units and the axis swap twice: on 2026-09-14 that
        # turned a 4.6 m GT-R into a 117 m one and put the question markers 56 m wide.
        decide("이미 변환된 입력이라 단위·축 변환 생략", f"입력이 출력 폴더의 input_mm.stl 과 같은 파일: {src}")
        scale, axis_ans = 1.0, "x"
    normalized_identity = {"source": initial_identity, "scale": scale,
                           "axis": axis_ans, "version": 1}
    def normalize_mesh():
        v = mesh.vertices * scale
        if axis_ans == "y":
            v = np.column_stack([v[:, 1], -v[:, 0], v[:, 2]])
        elif axis_ans == "z":
            v = np.column_stack([v[:, 2], v[:, 1], -v[:, 0]])
        return trimesh.Trimesh(v, mesh.faces, process=False)
    mesh, normalized_reused = mesh_checkpoint(args.out, "normalized_mesh", normalized_identity,
                                              normalize_mesh, force=args.force, stl=True)
    if normalized_reused:
        log("[재사용] mm·x길이축 메시 — 변환·STL 저장 생략")
    if scale != 1.0 or axis_ans != "x":
        decide(f"입력을 mm·x길이축으로 변환 ({unit}, 길이축 {axis_ans})", "단위와 축은 자동 감지가 안 되므로 질문 뒤 변환", {"file": str(work)})
    ext = mesh.extents
    diag["extents_mm"] = ext.round(1).tolist()
    if scale != 1.0 or axis_ans != "x":
        viewer_mesh(work)

    diag["surface_area_m2"] = round(float(mesh.area) / 1e6, 2)

    # A routing threshold is not evidence that geometry needs repair.
    if open_share < 1e-3:
        validation_identity = {"source": normalized_identity, "version": 2}
        validation = plan.get("basic_mesh_validation", {})
        if args.force or validation.get("identity") != validation_identity:
            log("[초기 검사] mm 기준 퇴화 면 확인 — 수밀·면 방향·좌표는 초기 결과 재사용")
            validation = {"identity": validation_identity,
                          "watertight": diag["watertight"],
                          "winding_consistent": diag["winding_consistent"],
                          "degenerate_faces": int((~mesh.nondegenerate_faces()).sum()),
                          "finite_coordinates": diag["finite_coordinates"],
                          "self_intersections": "not_checked",
                          "component_intersections": "not_checked"}
            plan["basic_mesh_validation"] = validation
            save()
        check("원본 메시 수밀", validation["watertight"], str(validation["watertight"]))
        check("원본 면 방향 일관성", validation["winding_consistent"], str(validation["winding_consistent"]))
        check("원본 퇴화 면 없음", validation["degenerate_faces"] == 0, f"{validation['degenerate_faces']}개")
        check("원본 좌표 유효", validation["finite_coordinates"], str(validation["finite_coordinates"]))
        passed = all(c["ok"] for c in plan["checks"])
        if passed:
            plan["route"] = "preserve-basic-valid"
            decide("기본 검사 통과 → 형상 유지", "두께 측정·재표면화를 자동 실행할 근거 없음")
            plan["deliverables"].append(str(src if scale == 1.0 and axis_ans == "x" else work))
            plan["validation_scope"] = "basic_only; self/component intersections not checked"
            finish()
        if not validation["winding_consistent"] or validation["degenerate_faces"]:
            decide("면 방향·퇴화 면 자동 정리", "정점을 이동하지 않고 발견된 문제만 정리")
            mesh, reused_cleanup = mesh_checkpoint(args.out, "cleaned_mesh",
                {"source": normalized_identity, "cleanup_version": 1},
                lambda: cleanup(mesh), force=args.force)
            cleaned = args.out / "cleaned.stl"
            if not reused_cleanup or not cleaned.exists() or plan.get("cleaned_hash") != ledger.file_hash(cleaned):
                mesh.export(cleaned)
                plan["cleaned_hash"] = ledger.file_hash(cleaned)
            work = cleaned
            repaired_validation = validate(mesh)
            plan["cleanup_result"] = repaired_validation
            for c in plan["checks"]:
                if not c["ok"]:
                    c["resolved"] = "자동 정리 후 아래 결과로 재검사"
            check("정리 결과 수밀", repaired_validation["watertight"], str(repaired_validation["watertight"]))
            check("정리 결과 면 방향 일관성", repaired_validation["winding_consistent"], str(repaired_validation["winding_consistent"]))
            check("정리 결과 퇴화 면 없음", repaired_validation["degenerate_faces"] == 0, str(repaired_validation["degenerate_faces"]))
            check("정리 결과 좌표 유효", repaired_validation["finite_coordinates"], str(repaired_validation["finite_coordinates"]))
            if all(c["ok"] or c.get("resolved") for c in plan["checks"]):
                plan["route"] = "automatic-local-cleanup"
                plan["deliverables"].append(str(cleaned))
                plan["validation_scope"] = "basic_only; self/component intersections not checked"
                finish()
        if not mesh.is_winding_consistent or len(mesh.faces) == 0:
            plan["deliverables"].append(str(work))
            finish()
        # Do not mark input failures resolved until output checks actually pass.
        input_failed_checks = [c for c in plan["checks"] if not c["ok"] and not c.get("resolved")]

    if open_share < 1e-3:
        # ------------------------------------------------ closed mesh: resurface
        log(f"[분류] 열림 경계가 적은 메쉬 — 경계 비율 {open_share*100:.3f} % < 0.1 % (수밀 보증 아님)")
        decide("경로 D: 닫지 않는 재표면화", "열림 경계 비율이 재표면화 경로 기준 미만")
        # Only this route consumes wall thickness to propose its voxel size.
        def measure_thickness():
            big = max(mesh.split(only_watertight=False), key=lambda p: len(p.faces))
            pts, fid = trimesh.sample.sample_surface(big, 3000, seed=0)
            n = big.face_normals[fid]
            log("[두께 측정] 측정 중 — 광선 3,000개를 작은 묶음으로 계산")
            loc, ray = ray_hits_batched(big.ray, pts - n * 0.05, -n)
            thick = np.linalg.norm(loc - (pts - n * 0.05)[ray], axis=1) if len(ray) else np.array([50.0])
            return float(np.percentile(thick, 5))

        settings = {"version": 1, "samples": 3000, "seed": 0, "offset_mm": 0.05,
                    "percentile": 5, "batch_size": 25, "no_hit_mm": 50.0,
                    "component": "largest_face_count", "numpy": np.__version__,
                    "trimesh": trimesh.__version__}
        t5, reused = cached_thickness(args.out / "thickness_measurement.json", work,
                                      settings, measure_thickness, force=args.force)
        diag["thickness_p5_mm"] = round(t5, 2)
        diag["thickness_measurement"] = {"status": "reused" if reused else "measured",
                                         "file": str(args.out / "thickness_measurement.json")}
        if reused:
            log(f"[두께 측정] 저장값 재사용 — {t5:.2f} mm, 같은 입력·설정")
        else:
            log(f"[두께 측정] 완료 — 두께 5% 분위 {t5:.2f} mm")
        save()
        voxel_prop = float(np.clip(round(t5 / 2.5, 1), 0.6, 3.0))
        est = mesh.area / (voxel_prop ** 2) * 0.7
        if est > 1.5e8:
            voxel_prop = float(np.ceil(np.sqrt(mesh.area * 0.7 / 1.5e8) * 10) / 10)
            plan["warnings"].append(f"삼각형 추정 {est/1e6:.0f} M → 복셀을 {voxel_prop} mm 로 올림")
        voxel = ask("voxel_mm", f"재표면화 복셀 크기 (제안 {voxel_prop} mm). 이 크기로 진행할까요?", voxel_prop,
                    f"두께 5% 분위 {t5:.1f} mm·계산량 기준; 복셀이 커지면 작은 틈·얇은 형상이 달라질 수 있습니다", unit="mm")
        if voxel is None:
            pause()
        if isinstance(voxel, bool) or not isinstance(voxel, (int, float)) or not np.isfinite(voxel) or voxel <= 0:
            raise ValueError("복셀 크기는 유한한 양수여야 합니다.")
        plan["route"] = "D-resurface"
        ok, text = run("prepare_geometry.py", ["--in", work, "--out", args.out / "run", "--no-mirror", "--resurface", voxel]
                       + (["--no-render"] if args.no_render else []), "run_resurface")
        res = args.out / "run" / "resurfaced.stl"
        if ok and res.exists():
            r = trimesh.load(res, force="mesh"); r.merge_vertices()
            _, c2 = np.unique(r.edges_sorted, axis=0, return_counts=True)
            wt = bool((c2 == 1).sum() == 0 and (c2 > 2).sum() == 0)
            dv = abs(r.volume) / max(1e-9, abs(mesh.volume)) - 1
            check("수밀", wt, f"경계 {(c2==1).sum()} 비다양체 {(c2>2).sum()}")
            check("체적 변화 < 3 %", abs(dv) < 0.03, f"{dv*100:+.1f} %", "복셀을 줄여 재실행")
            check("경계상자 유지", np.allclose(r.extents, mesh.extents, atol=2 * voxel + 1), f"{r.extents.round(0).tolist()} vs {mesh.extents.round(0).tolist()}")
            if abs(dv) >= 0.03 and voxel > 0.6 and plan.get("retries", 0) < args.max_retries:
                plan["retries"] = plan.get("retries", 0) + 1
                nv = max(0.6, round(voxel / 2, 1))
                decide(f"복셀 {voxel} → {nv} mm 로 재실행", f"체적 변화 {dv*100:+.1f} % 가 3 % 를 넘음")
                ok, text = run("prepare_geometry.py", ["--in", work, "--out", args.out / "run", "--no-mirror", "--resurface", nv, "--no-render"], "run_resurface2")
                if ok and res.exists():
                    plan["checks"] = [c for c in plan["checks"] if c["name"] not in
                                      ("수밀", "체적 변화 < 3 %", "경계상자 유지")]
                    r = trimesh.load(res, force="mesh"); r.merge_vertices()
                    _, c2 = np.unique(r.edges_sorted, axis=0, return_counts=True)
                    dv = abs(r.volume) / max(1e-9, abs(mesh.volume)) - 1
                    check("수밀", (c2 == 1).sum() == 0 and (c2 > 2).sum() == 0,
                          f"재시도 경계 {(c2 == 1).sum()} 비다양체 {(c2 > 2).sum()}")
                    check("체적 변화 < 3 %", abs(dv) < 0.03, f"재시도 {dv * 100:+.1f} %")
                    check("경계상자 유지", np.allclose(r.extents, mesh.extents, atol=2 * nv + 1),
                          f"재시도 {r.extents.round(0).tolist()}")
            if str(res) not in plan["deliverables"]:
                plan["deliverables"].append(str(res))
            output_checks = [c for c in plan["checks"] if c["name"] in ("수밀", "체적 변화 < 3 %", "경계상자 유지")]
            if ok and len(output_checks) == 3 and all(c["ok"] for c in output_checks):
                for c in input_failed_checks:
                    c["resolved"] = "재표면화 결과 검사 통과"
            finish()
        else:
            finish("failed")
    else:
        # ------------------------------------------------ open mesh
        diag.pop("thickness_p5_mm", None)
        diag["thickness_measurement"] = {"status": "not_required", "reason": "열린 메쉬 랩 경로는 두께를 사용하지 않음"}
        decide("열린 메쉬", f"경계 모서리 비율 {open_share*100:.3f} % ≥ 0.1 %", tag="분류")
        log("[두께 측정] 생략 — 열린 메쉬 경로에서는 사용하지 않음")
        floor_summary = args.out / "floorscan" / "summary.json"
        floor_identity = {"mesh_hash": ledger.file_hash(work), "version": 1,
                          "scan_script": ledger.file_hash(HERE / "flat_floor_wrap.py"),
                          "trimesh": trimesh.__version__, "numpy": np.__version__,
                          "grid": [60, 24], "low_fraction": 0.3, "section_step_mm": 20}
        floor_cache = plan.get("floor_diagnosis_cache", {})
        reuse_floor = (not args.force and floor_cache.get("identity") == floor_identity
                       and floor_cache.get("status") == "done")
        if reuse_floor:
            coverage = floor_cache["coverage"]
            floor_prop = floor_cache["floor_prop"]
            diag.update(floor_cache["diagnosis"])
            log("[재사용] 저장된 바닥 진단 — 광선·단면 검사 생략; 바닥 높이 답변은 수리에만 적용")
        else:
            log("[바닥 검사] 검사 중 — 아래쪽 광선으로 바닥 덮임 확인")
            # Is the underside there? Area of downward-facing surface in the lower third,
            # against the footprint. GT-R (no floor at all) has almost none; a car with
            # open panel seams still has its floor. The section scan alone cannot tell
            # the two apart: open seams keep any outline from closing.
            # Seen from below: cast rays up through a grid over the footprint; with a floor
            # the first hit is low (under 30 % of the height), with the floor missing the
            # rays go on to the cabin ceiling and engine parts. Face-area counts do not
            # work here: inner skins and underbody parts gave GT-R 173 % "coverage".
            lo, hi = mesh.bounds
            H = hi[2] - lo[2]
            gx = np.linspace(lo[0] + 0.05 * (hi[0] - lo[0]), hi[0] - 0.05 * (hi[0] - lo[0]), 60)
            gy = np.linspace(lo[1] + 0.1 * (hi[1] - lo[1]), hi[1] - 0.1 * (hi[1] - lo[1]), 24)
            origins = np.array([[x, y, lo[2] - 10.0] for x in gx for y in gy])
            dirs = np.tile([0, 0, 1.0], (len(origins), 1))
            hit, ray, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
            first = np.full(len(origins), np.nan)
            if len(ray):
                first[ray] = hit[:, 2]
            seen = np.isfinite(first)
            low = seen & (first < lo[2] + 0.3 * H)
            coverage = float(low.sum() / max(1, seen.sum()))
            diag["underside_coverage"] = round(coverage, 3)
            log(f"[바닥 검사] 광선 검사 완료 — 바닥 덮임률 {coverage*100:.1f} %")
            diag["underside_first_hit_median_frac"] = round(float(np.nanmedian((first - lo[2]) / H)), 3) if seen.any() else None
            ok, text = run("flat_floor_wrap.py", ["--in", work, "--out", args.out / "floorscan", "--no-wrap"], "floorscan")
            floor_summary = args.out / "floorscan" / "summary.json"
            floor_prop = json.loads(floor_summary.read_text()).get("floor_z") if floor_summary.exists() else None
            if floor_prop is None and coverage < 0.3:
                # no floor: propose the first height where sections cover 30 % of the box, else 15 % of the height
                best = None
                for z in np.arange(lo[2] + 20.0, lo[2] + 0.45 * H, 20.0):
                    sec = mesh.section(plane_origin=[0, 0, float(z)], plane_normal=[0, 0, 1])
                    if sec is None:
                        continue
                    try:
                        planar, _ = sec.to_2D()
                        area = sum(pg.area for pg in planar.polygons_full)
                    except Exception:
                        area = 0.0
                    if area >= 0.3 * (hi[0] - lo[0]) * (hi[1] - lo[1]):
                        best = float(z) + 20.0
                        break
                floor_prop = round(best if best is not None else float(lo[2] + 0.15 * H), 1)
            plan["floor_diagnosis_cache"] = {"identity": floor_identity, "status": "done",
                "coverage": coverage, "floor_prop": floor_prop,
                "diagnosis": {k: diag[k] for k in ("underside_coverage", "underside_first_hit_median_frac")}}
            save()
        if floor_prop is not None and coverage < 0.3:
            decide("바닥이 없는 껍질", f"바닥 덮임률 {coverage*100:.0f} % (30 % 미만); 평바닥 위치 확인 필요")
        if floor_prop is None:
            decide("경로 3: 구멍을 지키는 랩", f"언더사이드 덮임률 {coverage*100:.0f} % (바닥 있음) 인데 단면이 안 닫힘 → 부품 사이 틈이 문제")
            from aox_g3 import fair
            loops, _ = fair.boundary_loops(mesh)
            sizes = sorted((float(np.ptp(mesh.vertices[lp], axis=0).max()) for lp in loops), reverse=True)
            diag["open_loops"] = len(loops)
            diag["open_loop_sizes_mm"] = [round(s) for s in sizes[:12]]
            keep_prop = 13.0
            loop_pts, loop_lines = [], []
            for lp in sorted(loops, key=lambda l: -np.ptp(mesh.vertices[l], axis=0).max())[:12]:
                P = mesh.vertices[lp]
                size = float(np.ptp(P, axis=0).max())
                # cap the marker: a scan's seams form loops metres across, and a sphere that big
                # swallows the car in the viewer (2026-09-14)
                loop_pts.append([*P.mean(0).round(1).tolist(), float(min(size / 2, 200.0)), f"열린 고리 {size:.0f} mm"])
                # the loop itself, thinned, so the page can draw the opening instead of a blob
                idx = np.round(np.linspace(0, len(P) - 1, min(len(P), 160))).astype(int)
                loop_lines.append(P[idx].round(1).tolist())
            keep = ask("keep_openings_mm", f"유동이 지나야 하는 가장 작은 구멍 크기 (제안 {keep_prop} mm). 이보다 좁은 틈은 랩이 닫습니다.", keep_prop,
                       "윙 슬롯·덕트·그릴 중 가장 작은 것; 랩 알파는 이 값의 절반", unit="mm",
                       evidence=[str(args.out / "run" / "render_mesh.png")], where={"points": loop_pts, "lines": loop_lines})
            if keep is None:
                pause()
            est = mesh.area / ((keep / 2) ** 2) * 0.7
            area_m2 = mesh.area / 1e6
            diag_mm = float(np.linalg.norm(mesh.extents))
            # An open mesh cannot take the local re-wrap (the cut piece needs a closed
            # reference), and a fine alpha over a big open car does not finish: GT-R
            # (77 m², 55k open edges) at 6.5 mm ran past an hour. Coarse alpha then,
            # and the closed-openings table goes to the customer instead of a promise.
            coarse = area_m2 >= 30.0 or len(mesh.faces) >= 1_000_000
            plan["route"] = "3-wrap-keep-openings" + ("-coarse" if coarse else "")
            if coarse:
                decide("거친 알파 15 mm 로 감싸고 닫힌 자리는 보고", f"열린 메쉬 표면적 {area_m2:.0f} m² / 삼각형 {len(mesh.faces):,}: 가는 알파 전역 랩은 시간 안에 안 끝남")
                arguments = ["--in", work, "--out", args.out / "run", "--no-mirror", "--wrap", "--wrap-alpha-div", round(diag_mm / 15.0, 1),
                             "--force-wrap", "--smooth-seams"] + (["--no-render"] if args.no_render else [])
            else:
                arguments = ["--in", work, "--out", args.out / "run", "--no-mirror", "--wrap", "--keep-openings-above", keep,
                             "--force-wrap", "--smooth-seams"] + (["--no-render"] if args.no_render else [])
            ok, text = run("prepare_geometry.py", arguments, "run_wrap")
            wrap_status = None
            try:
                wrap_status = json.loads((args.out / "run" / "summary.json").read_text())["stages"]["wrap"]["status"]
            except Exception:
                pass
            if text == "timeout" and not coarse:
                decide("거친 알파 15 mm 로 재실행", "가는 알파 랩이 시간 예산을 넘음")
                coarse = True
                plan["route"] = "3-wrap-keep-openings-coarse"
                arguments = ["--in", work, "--out", args.out / "run", "--no-mirror", "--wrap", "--wrap-alpha-div", round(diag_mm / 15.0, 1),
                             "--force-wrap", "--smooth-seams"] + (["--no-render"] if args.no_render else [])
                ok, text = run("prepare_geometry.py", arguments, "run_wrap2")
            out_stl = args.out / "run" / ("wrap_smooth.stl" if (args.out / "run" / "wrap_smooth.stl").exists() else "wrap.stl")
            if out_stl.exists() or wrap_status == "hollow":
                if out_stl.exists():
                    w = trimesh.load(out_stl, force="mesh"); w.merge_vertices()
                    fill = abs(w.volume) / float(np.prod(w.extents))
                    detail = f"{fill:.2f}, 삼각형 {len(w.faces):,}"
                else:
                    fill = 0.0
                    detail = "랩 층이 속 빈 결과로 거부 (summary: hollow)"
                hollow = not check("랩이 속 찬 차 (채움률 0.3~0.6)", 0.3 <= fill <= 0.6, detail,
                                   "이음새로 랩이 안으로 샘 → 평바닥 가정으로 전환")
                if hollow and plan.get("retries", 0) < args.max_retries:
                    plan["retries"] = plan.get("retries", 0) + 1
                    fz = ask("floor_z_mm", f"랩이 이음새로 새어 속이 비었습니다. 평바닥을 z = {round(float(lo[2] + 0.15 * H), 1)} mm (높이의 15 %)에 두고 다시 감쌀까요?",
                             round(float(lo[2] + 0.15 * H), 1), "단면 훑기가 닫힘을 못 찾아 높이의 15 % 를 제안", unit="mm")
                    if fz is None:
                        pause()
                    decide("경로 C 로 전환: 평바닥 가정 + 랩", f"채움률 {fill:.2f} < 0.3")
                    plan["route"] = "C-flat-floor-wrap (after leak)"
                    ok, text = run("flat_floor_wrap.py", ["--in", work, "--out", args.out / "assumed", "--floor-z", fz, "--alpha-div", 360], "flatfloor")
                    out_stl = args.out / "assumed" / "wrapped.stl"
                    fill = 0.0
                    if out_stl.exists():
                        w = trimesh.load(out_stl, force="mesh"); w.merge_vertices()
                        fill = abs(w.volume) / float(np.prod(w.extents))
                    if not check("평바닥 랩 채움률 0.3~0.6", 0.3 <= fill <= 0.6, f"{fill:.2f}", "알파를 29 mm(대각선/180)로 올려 이음새를 덮고 재시도"):
                        # the seams are wider than the alpha (GT-R: panel gaps force a 29 mm floor, measured before)
                        decide("알파 29 mm 로 재시도", f"평바닥을 넣어도 채움률 {fill:.2f}: 이음새가 랩 알파보다 넓음")
                        ok, text = run("flat_floor_wrap.py", ["--in", work, "--out", args.out / "assumed29", "--floor-z", fz, "--alpha-div", 180], "flatfloor29")
                        out29 = args.out / "assumed29" / "wrapped.stl"
                        fill29 = 0.0
                        if out29.exists():
                            w = trimesh.load(out29, force="mesh"); w.merge_vertices()
                            fill29 = abs(w.volume) / float(np.prod(w.extents))
                        if check("평바닥 랩 29 mm 채움률 0.3~0.6", 0.3 <= fill29 <= 0.6, f"{fill29:.2f}"):
                            for c in plan["checks"]:
                                if "채움률" in c["name"] and not c["ok"]:
                                    c["resolved"] = "29 mm 재시도로 해결"
                            out_stl = out29
                            acc = ask("accept_coarse_29mm", "이음새가 넓어 29 mm 알파로만 속이 찹니다. 그 결과를 원본에 다시 붙여 날카롭게 만든 판(재메쉬→투영→6.5 mm 재랩)을 쓸까요, 정리된 모델을 주시겠습니까?",
                                      "accept", "29 mm 미만 틈은 닫힌 채, 표면은 원본으로 되돌림 (8월 GT-R 처방)", kind="choice", choices=["accept", "provide_clean_model"])
                            if acc == "accept":
                                # the August recipe: the coarse wrap is closed, so remesh it, pull it back onto
                                # the original where that is within reach, and wrap finely - no seam to leak
                                decide("거친 랩을 원본에 투영 후 6.5 mm 재랩", "닫힌 거친 랩은 가는 알파로 다시 감쌀 수 있고 투영이 형상을 되돌림")
                                ok, text = run("wrap_project_rewrap.py", ["--wrap", out29, "--reference", work, "--out", args.out / "sharp.stl",
                                                                          "--report", args.out / "sharp.json", "--edge", 10, "--max-move", 13, "--fine-alpha", 6.5], "sharpen")
                                sharp = args.out / "sharp.stl"
                                if ok and sharp.exists():
                                    rep_s = json.loads((args.out / "sharp.json").read_text())
                                    if check("투영·재랩 결과 속 찬 차 (채움률 0.3~0.6)", 0.3 <= rep_s.get("fill", 0) <= 0.6,
                                             f"채움률 {rep_s.get('fill')}, 원본과 p50 {rep_s.get('dev_p50_mm')} mm, 삼각형 {rep_s.get('fine_faces'):,}"):
                                        out_stl = sharp
                        else:
                            ask("seams_too_wide", "평바닥과 29 mm 알파로도 속이 찹니다. 이음새가 그보다 넓습니다. 정리된 모델(이음새 봉합)을 주시거나, 닫을 자리를 지정해 주세요.",
                                "provide_clean_model", f"채움률 15 mm {fill:.2f}, 29 mm {fill29:.2f}", kind="choice", choices=["provide_clean_model", "specify_closures"])
                            plan["status"] = "needs_customer"
                            save()
                            if not args.assume_defaults:
                                pause()
            closed_txt = args.out / "run" / "wrap_closed.txt"
            if closed_txt.exists():
                rows_c = re.findall(r"([\d.]+) cm²\s+\(\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\).*?틈 ≈\s*([\d.]+) mm", closed_txt.read_text())
                closed = [{"area_cm2": float(a), "centre": [float(x), float(y), float(z)], "gap_mm": float(g)} for a, x, y, z, g in rows_c]
                plan["closed_openings"] = closed
                gaps = [c["gap_mm"] for c in closed]
                bad = [c for c in closed if c["gap_mm"] >= keep]
                pts = [[*c["centre"], max(30.0, c["gap_mm"] * 3), f"{c['gap_mm']:.0f} mm"] for c in closed[:20]]
                if coarse:
                    accepted = ask("accept_coarse_closures", f"거친 알파(15 mm)라 {keep:.0f} mm 이상 틈 {len(bad)}곳이 닫혔습니다 (wrap_closed.txt). 이대로 쓸까요, 아니면 정리된 모델을 주시겠습니까?",
                        "accept", "열린 패널 이음새가 많은 메쉬는 가는 알파로 감쌀 수 없음", kind="choice", choices=["accept", "provide_clean_model"],
                        evidence=[str(closed_txt)], where={"points": pts})
                    if accepted is None:
                        pause()
                    if accepted == "provide_clean_model":
                        plan["status"] = "needs_customer"
                        check("정리된 입력 모델 필요", False, "사용자가 거친 랩 결과를 거부함")
                else:
                    check(f"{keep:.0f} mm 이상 구멍은 열려 있음", not bad, f"닫힌 자리 중 {keep:.0f} mm 이상: {len(bad)}개")
            if out_stl.exists():
                plan["deliverables"].append(str(out_stl))
                finish("done" if ok else "failed")
            else:
                finish("failed")
        else:
            decide("경로 C: 평바닥 가정 + 랩", f"바닥 높이 제안 z={floor_prop} (단면 닫힘 또는 30 % 덮임 높이; 언더사이드 덮임률 {coverage*100:.0f} %)")
            fz = ask("floor_z_mm", f"평바닥을 z = {floor_prop} mm 에 둘까요? (단면 훑기가 찾은 첫 닫힘 높이)", floor_prop,
                     "지면 간극·언더바디 형상은 설계 의도라 고객 확인 필요; 절대 Cd 가 달라짐", unit="mm",
                     # Do not render the full-resolution mesh before registering
                     # the question: this can block for tens of minutes. The
                     # viewer's plane_z marker supplies the visual floor guide.
                     evidence=[str(floor_summary)] if floor_summary.exists() else [],
                     where={"plane_z": float(floor_prop)})
            if fz is None:
                pause()
            plan["route"] = "C-flat-floor-wrap"
            ok, text = run("flat_floor_wrap.py", ["--in", work, "--out", args.out / "assumed", "--floor-z", fz, "--alpha-div", 360], "flatfloor")
            wrap = args.out / "assumed" / "wrapped.stl"
            if wrap.exists():
                w = trimesh.load(wrap, force="mesh"); w.merge_vertices()
                fill = abs(w.volume) / float(np.prod(w.extents))
                check("랩 수밀", w.is_watertight, f"몸체 {w.body_count}")
                check("채움률 0.3~0.6 (속 찬 차)", 0.3 <= fill <= 0.6, f"{fill:.2f}", "바닥 높이를 다시 물을 것")
                plan["deliverables"].append(str(wrap))
            finish("done" if ok else "failed")

else:
    # ---------------------------------------------------------------- STEP
    from aox_g3 import cad
    step_identity = {"hash": ledger.file_hash(src), "version": 1,
                     "diagnostics": ledger.file_hash(HERE.parent / "aox_g3" / "cad.py")}
    step_validation = plan.get("initial_step_validation", {})
    if args.force or step_validation.get("identity") != step_identity:
        log("[초기 검사] STEP 원본 자유 모서리·열린 셸·면 유효성 검사")
        shape, report = cad.read_step(src)
        if shape is None:
            raise RuntimeError("STEP 읽기 실패: " + str(report.warnings))
        cad.diagnose(shape, report)
        step_validation = {"identity": step_identity, "report": report.as_dict()}
        plan["initial_step_validation"] = step_validation
        del shape
        save()
    step_report = step_validation["report"]
    if (step_report.get("faces", 0) > 0 and step_report.get("solids", 0) > 0
            and step_report.get("free_edges") == 0 and step_report.get("open_shells") == 0
            and step_report.get("invalid_faces") == 0):
        plan["route"] = "preserve-step-basic-valid"
        plan["validation_scope"] = "basic CAD checks only; solid validity/intersections not fully checked"
        plan["deliverables"].append(str(src))
        check("STEP 기본 검사 통과", True, "자유 모서리·열린 셸·무효 면 없음")
        decide("STEP 형상 유지", "기본 검사에서 수리 근거가 발견되지 않음; 원본 구조 보존")
        finish()
    ok, text = run("propose_parameters.py", ["--in", src, "--out", args.out / "params.json"], "propose")
    params = json.loads((args.out / "params.json").read_text()) if (args.out / "params.json").exists() else {}
    diag["params"] = {k: v for k, v in params.items() if k != "questions"}
    half = params.get("half_model")
    mirror_ans = ask("half_model", f"반쪽 모델로 보입니다({half}). 미러해서 전체 차로 볼까요?", bool(half) if half is not None else True,
                     "대칭 확인은 고객 몫; 미러하면 전면 면적·랩이 전체 차 기준", kind="bool")
    seal_prop = params.get("seal_below")
    seal = ask("seal_below_mm", f"이 크기보다 작은 구멍은 묻지 않고 닫습니다 (제안 {seal_prop} mm). 유동이 지나야 하는 가장 작은 구멍보다 작아야 합니다.",
               seal_prop, "구멍 크기 분포에서 가장 넓은 로그 간격", unit="mm")
    rims = params.get("close_near") or []          # [[x, y, z, r], ...] from autotune.detect_wheels
    rim_pts = [[float(r[0]), float(r[1]), float(r[2]), float((list(r) + [450])[3]), f"림 {k+1}"] for k, r in enumerate(rims)]
    rims_ans = ask("close_rims", f"바퀴 림처럼 보이는 고리 {len(rims)}곳을 닫힌 원판으로 볼까요? (스포크 구멍은 Cd 를 14 % 바꿉니다)", True,
                   "림 위치·크기는 자동 감지; 원판 처리 여부는 해석 의도", kind="bool", where={"points": rim_pts})
    if mirror_ans is None or seal is None or rims_ans is None:
        pause()
    arguments = ["--in", src, "--out", args.out / "run", "--seal-below", seal] + ([] if mirror_ans else ["--no-mirror"]) \
        + (["--no-render"] if args.no_render else [])
    if rims_ans and rims:
        for r in rims:
            x, y, z, rad = (list(r) + [450])[:4]
            arguments += [f"--close-near={x:.0f},{y:.0f},{z:.0f},{rad:.0f}"]
    plan["route"] = "A-heal"
    ok, text = run("prepare_geometry.py", arguments, "run_heal")
    summary = json.loads((args.out / "run" / "summary.json").read_text()) if (args.out / "run" / "summary.json").exists() else {}
    heal = json.loads((args.out / "run" / "heal.json").read_text()).get("heal", {}) if (args.out / "run" / "heal.json").exists() else {}
    nums = summary.get("numbers", {})
    plan["healing_result"] = healing_result(summary)
    check("힐링 단계 실행", plan["healing_result"]["execution_status"] == "ok", "heal: " + plan["healing_result"]["execution_status"] + " (형상 품질 검사와 별개)")
    check("힐링 STEP 형상 유효", nums.get("valid") is True, f"valid={nums.get('valid')}")
    check("힐링 부유 캡 없음", nums.get("floating_caps") == 0, f"floating_caps={nums.get('floating_caps')}")
    log(f"[형상 보고] run/healed.stp: closed={nums.get('closed')}, valid={nums.get('valid')}, floating_caps={nums.get('floating_caps')}, 잔여 자유경계={nums.get('free_boundaries_measured')}")
    plan["deliverables"].append(str(args.out / "run" / "healed.stp"))
    full_mesh0 = args.out / "run" / ("mesh_full.stl" if mirror_ans else "mesh.stl")
    if full_mesh0.exists():
        viewer_mesh(full_mesh0)
        for k, r in enumerate(rims):
            ev = focus_render(full_mesh0, r[:3], 900, f"rim{k+1}", f"림 {k+1}")
            for q in plan["questions"]:
                if q["id"] == "close_rims" and ev:
                    q["evidence"].append(ev)
    left = heal.get("left_open", [])
    big_left = [b for b in left if float(b.get("size", 0)) > 1000]
    if not big_left:
        decide("큰 구멍 없음 → 힐링 STEP 이 인도물", f"남은 구멍 {len(left)}개 전부 1 m 이하")
        finish("done" if ok else "failed")
    decide("경로 C+E: 평바닥 가정 랩 → 랩 형상 캡으로 STEP 닫기", f"남은 구멍 중 1 m 이상 {len(big_left)}개 (최대 {max(float(b['size']) for b in big_left):.0f} mm): 언더바디·캐빈은 봉합 크기로 못 닫음")
    plan["route"] = "A-heal + C-flat-floor-wrap + E-wrapcaps"
    full_mesh = args.out / "run" / ("mesh_full.stl" if mirror_ans else "mesh.stl")
    ok, text = run("flat_floor_wrap.py", ["--in", full_mesh, "--out", args.out / "floorscan", "--no-wrap"], "floorscan")
    fsum = args.out / "floorscan" / "summary.json"
    floor_prop = json.loads(fsum.read_text()).get("floor_z") if fsum.exists() else None
    open_pts = []
    for b in big_left:
        c = json.loads(b["centre"]) if isinstance(b["centre"], str) else b["centre"]
        open_pts.append([float(c[0]), float(c[1]), float(c[2]), float(b["size"]) / 2, f"열린 구멍 {float(b['size']):.0f} mm"])
    fz = ask("floor_z_mm", f"언더바디가 열려 있어 평바닥을 가정합니다. 높이 z = {floor_prop} mm (단면 훑기의 첫 닫힘)로 둘까요?", floor_prop,
             "지면 간극·언더바디는 설계 의도; 이 높이가 절대 Cd 를 바꿈", unit="mm", evidence=[str(fsum), str(args.out / "run" / "render_mesh.png")],
             where={"plane_z": float(floor_prop) if floor_prop is not None else None, "points": open_pts})
    if fz is None:
        pause()
    ok, text = run("flat_floor_wrap.py", ["--in", full_mesh, "--out", args.out / "assumed", "--floor-z", fz, "--alpha-div", 360], "flatfloor")
    wrap = args.out / "assumed" / "wrapped.stl"
    if wrap.exists():
        w = trimesh.load(wrap, force="mesh"); w.merge_vertices()
        fill = abs(w.volume) / float(np.prod(w.extents))
        check("랩 수밀", w.is_watertight, f"몸체 {w.body_count}")
        check("채움률 0.3~0.6", 0.3 <= fill <= 0.6, f"{fill:.2f}")
        plan["deliverables"].append(str(wrap))
    ok, text = run("close_with_wrap.py", ["--step", args.out / "run" / "healed.stp", "--wrap", wrap, "--out", args.out / "healed_wrapcaps.stp",
                                          "--report", args.out / "wrapcaps.json", "--floor-z", fz], "wrapcaps")
    rep = json.loads((args.out / "wrapcaps.json").read_text()) if (args.out / "wrapcaps.json").exists() else {}
    skipped = [c for c in rep.get("caps", []) if c.get("skipped") or c.get("failed")]
    check("STEP 자유경계 = 대칭면 + 뜻 필요한 곳뿐", rep.get("big_loops_after", 99) <= 1 + len(skipped),
          f"남은 고리 {rep.get('big_loops_after')} (건너뛴 캡 {len(skipped)})")
    if skipped:
        wheel_pts = [[*c.get("centre", [0, 0, 0]), max(60.0, c.get("perimeter_mm", 300) / 6), f"고리 {c.get('perimeter_mm')} mm"] for c in skipped if c.get("centre")]
        ev = []
        for k, c in enumerate(skipped[:2]):
            if c.get("centre"):
                r = focus_render(full_mesh0, c["centre"], 600, f"wheel{k+1}", f"못 닫은 바퀴 고리 {k+1}")
                if r:
                    ev.append(r)
        treatment = ask("wheel_treatment", f"바퀴 주변 고리 {len(skipped)}곳을 닫지 않은 검토용 결과로 남길까요? 원판·타이어 접합 자동 처리는 아직 지원하지 않습니다.", "leave",
            "남은 고리는 별도 형상 수정이 필요합니다", kind="choice", choices=["leave"],
            evidence=[str(args.out / "wrapcaps.json")] + ev, where={"points": wheel_pts})
        if treatment is None:
            pause()
        if treatment != "leave":
            raise ValueError("지원하지 않는 wheel_treatment: " + str(treatment))
        decide("바퀴 고리를 열린 채 검토용으로 유지", "사용자가 leave를 선택함")
        check("바퀴 고리 후속 수정 필요", False, f"열린 고리 {len(skipped)}곳; 자동 수정 미지원")
    plan["deliverables"].append(str(args.out / "healed_wrapcaps.stp"))
    finish("done" if ok else "failed")
