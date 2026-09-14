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
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in", dest="src", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--answers", type=Path, help="answers.json from the UI or the customer")
ap.add_argument("--assume-defaults", action="store_true", help="use every proposal instead of pausing")
ap.add_argument("--max-retries", type=int, default=2)
ap.add_argument("--time-budget", type=int, default=1500, help="seconds per script run; a run over budget is stopped and the plan falls back to a coarser route")
ap.add_argument("--no-render", action="store_true")
args = ap.parse_args()

import trimesh  # noqa: E402

args.out.mkdir(parents=True, exist_ok=True)
PLAN = args.out / "plan.json"
QUESTIONS = args.out / "questions.json"
LOG = open(args.out / "plan_log.txt", "a", encoding="utf-8")
python = sys.executable

plan = json.loads(PLAN.read_text()) if PLAN.exists() else {
    "input": str(args.src), "out": str(args.out), "status": "running", "diagnosis": {}, "route": None,
    "decisions": [], "runs": [], "checks": [], "questions": [], "answers": {}, "assumptions": [],
    "deliverables": [], "warnings": []}
answers = dict(plan.get("answers", {}))
if args.answers and args.answers.exists():
    answers.update({k: v for k, v in json.loads(args.answers.read_text()).items() if v is not None})
plan["answers"] = answers
plan["status"] = "running"


def log(msg):
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


def save():
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=1))


def decide(what, because, evidence=None):
    plan["decisions"].append({"what": what, "because": because, "evidence": evidence or {}})
    log(f"[결정] {what}  ← {because}")
    save()


def check(name, ok, detail, on_fail=None):
    plan["checks"].append({"name": name, "ok": bool(ok), "detail": detail, "on_fail": on_fail})
    log(f"[검증] {'통과' if ok else '실패'} {name}: {detail}" + (f"  → {on_fail}" if (not ok and on_fail) else ""))
    save()
    return bool(ok)


def run(script, arguments, capture_name):
    cmd = [python, str(HERE / script)] + [str(a) for a in arguments]
    t0 = time.time()
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


def viewer_mesh(path, target=150_000):
    """A light copy of the mesh for the page's 3D viewer (viewer.stl)."""
    out = args.out / "viewer.stl"
    try:
        m = trimesh.load(path, force="mesh")
        if len(m.faces) > target:
            from meshlib import mrmeshpy as MR, mrmeshnumpy as MN
            ml = MR.loadMesh(str(path))
            ds = MR.DecimateSettings()
            ds.maxDeletedFaces = int(len(m.faces) - target)
            ds.maxError = 5.0
            ds.packMesh = True
            MR.decimateMesh(ml, ds)
            m = trimesh.Trimesh(np.asarray(MN.getNumpyVerts(ml)), np.asarray(MN.getNumpyFaces(ml.topology)), process=False)
        m.export(out)
        plan["viewer_mesh"] = "viewer.stl"
        plan["viewer_bbox"] = [m.bounds[0].round(1).tolist(), m.bounds[1].round(1).tolist()]
        save()
    except Exception as exc:
        plan["warnings"].append(f"viewer mesh 실패: {type(exc).__name__}: {exc}")
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
    QUESTIONS.write_text(json.dumps(pending, ensure_ascii=False, indent=1))
    plan["status"] = "waiting_for_answers"
    save()
    log(f"[정지] 고객 답이 필요한 질문 {len(pending)}개 → {QUESTIONS}")
    log("      답을 answers.json 에 {id: value} 로 적고 --answers 로 다시 실행하거나, plan_ui.py 로 답하세요.")
    sys.exit(3)


def finish(status="done"):
    if status == "done" and any(not c["ok"] for c in plan["checks"][-3:]):
        status = "done_with_failed_checks"
    if plan.get("status") == "needs_customer":
        status = "needs_customer"
    plan["status"] = status
    QUESTIONS.write_text("[]")
    save()
    log(f"[끝] {status}  산출물: " + ", ".join(plan["deliverables"]))


# ----------------------------------------------------------------- diagnosis
src = args.src
is_step = src.suffix.lower() in (".stp", ".step")
diag = plan["diagnosis"]
diag["format"] = "STEP" if is_step else "mesh"

if not is_step:
    mesh = trimesh.load(src, force="mesh")
    mesh.merge_vertices()
    ext = mesh.extents
    _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    open_share = float((counts == 1).sum() / max(1, len(counts)))
    parts = mesh.split(only_watertight=False)
    diag.update({"triangles": int(len(mesh.faces)), "extents": ext.round(2).tolist(), "bbox_min": mesh.bounds[0].round(2).tolist(),
                 "bodies": int(len(parts)), "boundary_edge_share": round(open_share, 5), "watertight": bool(mesh.is_watertight)})
    log(f"[진단] 메쉬 삼각형 {len(mesh.faces):,}  치수 {ext.round(1).tolist()}  몸체 {len(parts)}  경계 모서리 비율 {open_share*100:.3f} %")

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
    v = mesh.vertices * scale
    if axis_ans == "y":
        v = np.column_stack([v[:, 1], -v[:, 0], v[:, 2]])
    elif axis_ans == "z":
        v = np.column_stack([v[:, 2], v[:, 1], -v[:, 0]])
    mesh = trimesh.Trimesh(v, mesh.faces, process=False)
    mesh.export(work)
    if scale != 1.0 or axis_ans != "x":
        decide(f"입력을 mm·x길이축으로 변환 ({unit}, 길이축 {axis_ans})", "단위와 축은 자동 감지가 안 되므로 질문 뒤 변환", {"file": str(work)})
    ext = mesh.extents
    diag["extents_mm"] = ext.round(1).tolist()
    viewer_mesh(work)

    # thin walls: inward first hit
    big = max(mesh.split(only_watertight=False), key=lambda p: len(p.faces))
    pts, fid = trimesh.sample.sample_surface(big, 3000, seed=0)
    n = big.face_normals[fid]
    loc, ray, _ = big.ray.intersects_location(pts - n * 0.05, -n, multiple_hits=False)
    thick = np.linalg.norm(loc - (pts - n * 0.05)[ray], axis=1) if len(ray) else np.array([50.0])
    t5 = float(np.percentile(thick, 5))
    diag["thickness_p5_mm"] = round(t5, 2)
    diag["surface_area_m2"] = round(float(mesh.area) / 1e6, 2)

    if open_share < 1e-3:
        # ------------------------------------------------ closed mesh: resurface
        decide("경로 D: 닫지 않는 재표면화", f"경계 모서리 비율 {open_share*100:.3f} % < 0.1 % (닫힌 입력)")
        voxel_prop = float(np.clip(round(t5 / 2.5, 1), 0.6, 3.0))
        est = mesh.area / (voxel_prop ** 2) * 0.7
        if est > 1.5e8:
            voxel_prop = float(np.ceil(np.sqrt(mesh.area * 0.7 / 1.5e8) * 10) / 10)
            plan["warnings"].append(f"삼각형 추정 {est/1e6:.0f} M → 복셀을 {voxel_prop} mm 로 올림")
        voxel = ask("voxel_mm", f"재표면화 복셀 크기 (제안 {voxel_prop} mm). 판 두께 5 % 분위 {t5:.1f} mm 의 절반 이하가 규칙입니다.", voxel_prop,
                    f"가장 얇은 부위(5 % 분위) {t5:.1f} mm; 복셀은 그 절반 이하, 0.6~3 mm", unit="mm")
        if voxel is None:
            pause()
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
            plan["deliverables"].append(str(res))
            finish()
        else:
            finish("failed")
    else:
        # ------------------------------------------------ open mesh
        decide("열린 메쉬", f"경계 모서리 비율 {open_share*100:.2f} % ≥ 0.1 %")
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
            decide("바닥이 없는 껍질", f"아래에서 쏜 광선의 첫 충돌이 높이 30 % 아래인 비율 {coverage*100:.0f} % (30 % 미만); 단면 닫힘 없음")
        if floor_prop is None:
            decide("경로 3: 구멍을 지키는 랩", f"언더사이드 덮임률 {coverage*100:.0f} % (바닥 있음) 인데 단면이 안 닫힘 → 부품 사이 틈이 문제")
            from aox_g3 import fair
            loops, _ = fair.boundary_loops(mesh)
            sizes = sorted((float(np.ptp(mesh.vertices[lp], axis=0).max()) for lp in loops), reverse=True)
            diag["open_loops"] = len(loops)
            diag["open_loop_sizes_mm"] = [round(s) for s in sizes[:12]]
            keep_prop = 13.0
            loop_pts = []
            for lp in sorted(loops, key=lambda l: -np.ptp(mesh.vertices[l], axis=0).max())[:12]:
                P = mesh.vertices[lp]
                loop_pts.append([*P.mean(0).round(1).tolist(), float(np.ptp(P, axis=0).max() / 2), f"열린 고리 {np.ptp(P, axis=0).max():.0f} mm"])
            keep = ask("keep_openings_mm", f"유동이 지나야 하는 가장 작은 구멍 크기 (제안 {keep_prop} mm). 이보다 좁은 틈은 랩이 닫습니다.", keep_prop,
                       "윙 슬롯·덕트·그릴 중 가장 작은 것; 랩 알파는 이 값의 절반", unit="mm",
                       evidence=[str(args.out / "run" / "render_mesh.png")], where={"points": loop_pts})
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
                            out_stl = out29
                            ask("accept_coarse_29mm", "이음새가 넓어 29 mm 알파로만 속이 찹니다. 29 mm 결과(작은 틈·덕트는 닫힘)를 쓸까요, 정리된 모델을 주시겠습니까?",
                                "accept", "29 mm 미만 틈은 전부 닫힌 상태", kind="choice", choices=["accept", "provide_clean_model"])
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
                    ask("accept_coarse_closures", f"거친 알파(15 mm)라 {keep:.0f} mm 이상 틈 {len(bad)}곳이 닫혔습니다 (wrap_closed.txt). 이대로 쓸까요, 아니면 정리된 모델을 주시겠습니까?",
                        "accept", "열린 패널 이음새가 많은 메쉬는 가는 알파로 감쌀 수 없음", kind="choice", choices=["accept", "provide_clean_model"],
                        evidence=[str(closed_txt)], where={"points": pts})
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
                     evidence=[str(floor_summary)] + ([e] if (e := focus_render(work, [float(lo[0] + 0.5 * (hi[0] - lo[0])), float((lo[1] + hi[1]) / 2), float(floor_prop)], 1500, "floor", "평바닥 높이 후보 (아래에서)")) else []),
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
        + (["--auto"] if not rims else []) + (["--no-render"] if args.no_render else [])
    if rims_ans and rims:
        for r in rims:
            x, y, z, rad = (list(r) + [450])[:4]
            arguments += [f"--close-near={x:.0f},{y:.0f},{z:.0f},{rad:.0f}"]
    plan["route"] = "A-heal"
    ok, text = run("prepare_geometry.py", arguments, "run_heal")
    summary = json.loads((args.out / "run" / "summary.json").read_text()) if (args.out / "run" / "summary.json").exists() else {}
    heal = json.loads((args.out / "run" / "heal.json").read_text()).get("heal", {}) if (args.out / "run" / "heal.json").exists() else {}
    nums = summary.get("numbers", {})
    check("STEP 되읽기·경계상자", summary.get("stages", {}).get("heal", {}).get("status") == "ok", f"구멍 {nums.get('holes_found')} 중 {nums.get('holes_filled')} 닫힘, 실측 잔여 {nums.get('free_boundaries_measured')}")
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
        if not (args.out / "run" / "wrap.stl").exists():
            ok, text = run("prepare_geometry.py", arguments + ["--wrap", "--force-wrap", "--smooth-seams"], "run_wrap")
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
        ask("wheel_treatment", f"바퀴 주변 고리 {len(skipped)}곳은 접힘 때문에 자동으로 못 닫았습니다. 어떻게 할까요?", "leave",
            "림·타이어 접합은 설계·해석 의도(회전 휠, 원판, 실제 타이어 형상)", kind="choice", choices=["leave", "disc", "tyre_contact"],
            evidence=[str(args.out / "wrapcaps.json")] + ev, where={"points": wheel_pts})
    plan["deliverables"].append(str(args.out / "healed_wrapcaps.stp"))
    finish("done" if ok else "failed")
