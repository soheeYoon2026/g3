"""One command from a supplier's STEP to everything the pipeline can produce.

    prepare_geometry.py --in CAS-A.stp --out var/runs/cas-a \\
        --seal-below 900 --close-near=-41,-910,76,450 --close-near=2689,-905,62,450

Stages, each writing its own report and each surviving the failure of the others:

    cad      read, diagnose, sew                       -> cad.json
    heal     close the small holes as B-rep, write STEP -> healed.stp, heal.json
    intent   what was left open and why, for the engineer -> intent.md
    mesh     tessellate, stitch, patch the rest, STL     -> mesh.stl, mesh_full.stl, mesh.json
    area     projected frontal area of the full car       -> frontal_area.txt
    render   pictures of the STEP and the mesh            -> render_step.png, render_mesh.png
    wrap     (with --wrap) CGAL alpha wrap of the mesh    -> wrap.stl, wrap.json

A mesh input (STL/OBJ/PLY) skips cad, heal and intent. summary.json at the end
says which stages ran, how long, and the numbers that matter.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in", dest="src", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True, help="output directory")
ap.add_argument("--seal-below", type=float,
                help="close holes up to this size (default: diagonal/6)")
ap.add_argument("--close-near", action="append", default=[], metavar="X,Y,Z,R",
                help="deliberate simplification such as a closed rim; repeatable")
ap.add_argument("--no-mirror", action="store_true",
                help="the model is already a full car; do not mirror for area/full mesh")
ap.add_argument("--auto", action="store_true",
                help="measure the model and propose sealing size, sewing ladder, "
                     "closed-rim points and mirroring; explicit flags still win")
ap.add_argument("--params", type=Path,
                help="a proposal JSON from propose_parameters.py to use instead of --auto")
ap.add_argument("--wrap", action="store_true", help="also run the CGAL wrap tier")
ap.add_argument("--no-render", action="store_true")
ap.add_argument("--curvature-angle", type=float, default=15.0)
args = ap.parse_args()

args.out.mkdir(parents=True, exist_ok=True)
log_path = args.out / "log.txt"
log_file = open(log_path, "w", encoding="utf-8")
summary = {"input": str(args.src), "stages": {}, "numbers": {}}
python = sys.executable


def log(line=""):
    print(line, flush=True)
    log_file.write(line + "\n")
    log_file.flush()


def stage(name):
    """Context manager-ish bookkeeping: time, status, exception isolation."""
    class _Stage:
        def __enter__(self):
            self.t0 = time.time()
            summary["stages"].setdefault(name, {})
            log(f"\n== {name} ==")
            return self

        def __exit__(self, exc_type, exc, tb):
            seconds = time.time() - self.t0
            entry = summary["stages"].setdefault(name, {})
            entry["seconds"] = round(seconds, 1)
            if exc is None:
                entry.setdefault("status", "ok")
                log(f"   {name}: 완료 {seconds:.1f}s")
            else:
                entry["status"] = "failed"
                entry["error"] = f"{exc_type.__name__}: {str(exc)[:300]}"
                log(f"   {name}: 실패 — {entry['error']}")
            return True  # swallow: the next stage still runs
    return _Stage()


def run_script(script, arguments, capture_to=None):
    """Run one of the standalone scripts, keeping its stdout in the run folder."""
    cmd = [python, "-u", str(HERE / script), *arguments]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    text = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout + proc.stderr)
    text = "\n".join(ln for ln in text.splitlines()
                     if ln.strip() and not any(k in ln for k in (
                         "Transferring Shape", "WorkSession", "Step File Name",
                         "****", "RuntimeWarning", "d1[is_ab]")))
    if capture_to is not None:
        Path(capture_to).write_text(text, encoding="utf-8")
    if proc.returncode not in (0, 1):   # 1 is "not closed", which is informative
        raise RuntimeError(f"{script} exit {proc.returncode}: {text[-400:]}")
    return text


is_mesh_input = args.src.suffix.lower() in (".stl", ".obj", ".ply")
healed_step = args.out / "healed.stp"
heal_json = args.out / "heal.json"
mesh_stl = args.out / "mesh.stl"
mesh_full_stl = args.out / "mesh_full.stl"
mirror = not args.no_mirror

# ------------------------------------------------------------------ cad + heal
if not is_mesh_input:
    from aox_g3 import brep, cad

    proposal = None
    with stage("cad"):
        shape, cad_report = cad.read_step(args.src)
        if shape is None:
            raise RuntimeError(f"STEP reader refused: {cad_report.warnings}")
        cad.diagnose(shape, cad_report)
        stages_to_use = None
        if args.params and args.params.exists():
            proposal = json.loads(args.params.read_text())
            log(f"   파라미터 제안 읽음: {args.params}")
        elif args.auto:
            from aox_g3 import autotune
            t_auto = time.time()
            prop, _ = autotune.propose(shape, cad_report, do_sweep=True)
            proposal = prop.as_dict()
            (args.out / "params.json").write_text(
                json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")
            log(f"   자동 제안 {time.time() - t_auto:.0f}s → params.json")
            for line in prop.rationale:
                log(f"   · {line}")
        if proposal:
            stages_to_use = proposal.get("sew_stages") or None
            if args.seal_below is None and proposal.get("seal_below"):
                args.seal_below = float(proposal["seal_below"])
            if not args.close_near and proposal.get("close_near"):
                args.close_near = [",".join(str(v) for v in p) for p in proposal["close_near"]]
            if not args.no_mirror and proposal.get("half_model") is False:
                args.no_mirror = True
                mirror = False
            summary["numbers"]["params_source"] = "params" if args.params else "auto"
        shape, cad_report = cad.sew_progressive(shape, cad_report, tolerances=stages_to_use)
        after = cad.CadReport()
        cad.diagnose(shape, after)
        (args.out / "cad.json").write_text(json.dumps(
            {"before": cad_report.as_dict(), "after_sew": after.as_dict()},
            ensure_ascii=False, indent=2))
        diag = float(np.linalg.norm(np.array(cad_report.bbox[3:]) - np.array(cad_report.bbox[:3])))
        summary["numbers"].update({
            "faces": cad_report.faces, "shells_before": cad_report.shells,
            "shells_after_sew": after.shells,
            "free_edges_before": cad_report.free_edges,
            "free_edges_after_sew": after.free_edges,
            "diagonal": round(diag, 1), "units_hint": cad_report.units_hint})
        for w in cad_report.warnings:
            log(f"   경고: {w}")
        log(f"   면 {cad_report.faces:,}  셸 {cad_report.shells:,}→{after.shells:,}  "
            f"자유모서리 {cad_report.free_edges:,}→{after.free_edges:,}")

    with stage("heal"):
        seal = args.seal_below if args.seal_below is not None else diag / 6.0
        close_near = []
        for spec in args.close_near:
            parts = [float(v) for v in spec.replace(" ", "").split(",")]
            if len(parts) != 4:
                raise ValueError(f"--close-near must be X,Y,Z,R: {spec!r}")
            close_near.append(tuple(parts))
        healed, heal_report = brep.heal(shape, sealing_size=seal, close_near=close_near)
        brep.write_step(healed, healed_step, heal_report,
                        units="MM" if cad_report.units_hint == "mm" else "M")
        heal_json.write_text(json.dumps(
            {"cad": cad_report.as_dict(), "heal": heal_report.as_dict()},
            ensure_ascii=False, indent=2))
        summary["numbers"].update({
            "sealing_size": round(seal, 1),
            "holes_found": heal_report.boundaries_found,
            "holes_filled": heal_report.boundaries_filled,
            "holes_left": heal_report.boundaries_left,
            "free_boundaries_measured": heal_report.free_boundaries_after,
            "floating_caps": heal_report.floating_caps,
            "closed": heal_report.closed, "valid": heal_report.valid,
            "area_before_m2": round(heal_report.area_before / 1e6, 3),
            "area_after_m2": round(heal_report.area / 1e6, 3),
            "step_mb": round(healed_step.stat().st_size / 1e6, 1)})
        log(f"   봉합크기 {seal:.0f}  구멍 {heal_report.boundaries_found}개 중 "
            f"{heal_report.boundaries_filled} 메움, 실측 잔여 자유경계 "
            f"{heal_report.free_boundaries_after}  닫힘 {heal_report.closed}")
        for w in heal_report.warnings:
            log(f"   경고: {w}")

    with stage("intent"):
        lines = [f"# {args.src.name} — 엔지니어 결정 항목", ""]
        held = sorted(heal_report.left_open, key=lambda b: -b.size)
        if not held:
            lines.append("남은 열린 경계가 없습니다.")
        for b in held:
            c = b.centre
            note = b.note
            if "larger than the sealing size" in note:
                kind = "봉합 크기 초과 — 닫을지, 어떻게 닫을지 결정 필요"
            elif "overlapping panel" in note or "surface under it" in note:
                kind = "겹친 판 — 트림된 서피스가 필요 (자동 해소 불가)"
            elif "simplification" in note:
                kind = "요청에 따라 단순화됨"
            else:
                kind = f"도구 한계: {note[:80]}"
            lines.append(f"- 크기 **{b.size:.0f}** 둘레 {b.length:.0f}  "
                         f"중심 ({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f}) — {kind}")
        simplified = [b for b in heal_report.filled if "simplification" in b.note]
        if simplified:
            lines += ["", "## 요청에 따라 닫은 것 (단순화)", ""]
            for b in simplified:
                c = b.centre
                lines.append(f"- 크기 {b.size:.0f} 중심 ({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f})")
        (args.out / "intent.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary["numbers"]["intent_items"] = len(held)
        log(f"   결정 항목 {len(held)}개 → intent.md")

# ------------------------------------------------------------------------ mesh
with stage("mesh"):
    arguments = ["--in", str(args.src if is_mesh_input else healed_step),
                 "--out", str(mesh_stl), "--report", str(args.out / "mesh.json"),
                 "--curvature-angle", str(args.curvature_angle)]
    if args.seal_below is not None:
        arguments += ["--seal-below", str(args.seal_below)]
    if not is_mesh_input and heal_json.exists():
        arguments += ["--heal-report", str(heal_json)]
    text = run_script("fair_mesh.py", arguments, args.out / "mesh.txt")
    m = re.search(r"결과 삼각형 ([\d,]+)", text)
    summary["numbers"]["mesh_triangles"] = int(m.group(1).replace(",", "")) if m else None
    m = re.search(r"메움 (\d+)", text)
    summary["numbers"]["mesh_loops_filled"] = int(m.group(1)) if m else None
    if mirror:
        import trimesh
        mesh = trimesh.load(mesh_stl, force="mesh")
        flipped = mesh.copy()
        flipped.vertices[:, 1] *= -1.0
        flipped.faces = flipped.faces[:, ::-1]
        (mesh + flipped).export(mesh_full_stl)
    log("   " + "\n   ".join(text.splitlines()[-4:]))

# ------------------------------------------------------------------------ area
with stage("area"):
    arguments = ["--in", str(mesh_full_stl if mirror else mesh_stl)]
    text = run_script("frontal_area.py", arguments, args.out / "frontal_area.txt")
    m = re.search(r"전면 면적 \(투영 합집합\) ([\d.]+) m²", text)
    summary["numbers"]["frontal_area_m2"] = float(m.group(1)) if m else None
    m = re.search(r"경계상자 W×H ([\d.]+) m²", text)
    summary["numbers"]["bbox_wh_m2"] = float(m.group(1)) if m else None
    log(f"   전면 면적 {summary['numbers']['frontal_area_m2']} m²  "
        f"(경계상자 {summary['numbers']['bbox_wh_m2']} m²)")

# ---------------------------------------------------------------------- render
if not args.no_render:
    with stage("render"):
        if not is_mesh_input:
            run_script("render_geometry.py", [
                "--step", str(healed_step), "--out", str(args.out / "render_step.png"),
                "--label-top", "4", "--title", f"{args.src.name} · B-rep 봉합"]
                + (["--mirror"] if mirror else []))
        run_script("render_geometry.py", [
            "--step", str(mesh_stl), "--out", str(args.out / "render_mesh.png"),
            "--label-top", "4", "--title", f"{args.src.name} · 메쉬 봉합"]
            + (["--mirror"] if mirror else []))

# ------------------------------------------------------------------------ wrap
if args.wrap:
    with stage("wrap"):
        text = run_script("seal_geometry.py", [
            "--in", str(mesh_full_stl if mirror else mesh_stl),
            "--out", str(args.out / "wrap.stl"),
            "--report", str(args.out / "wrap.json")], args.out / "wrap.txt")
        # The wrap "fails" honestly while the large openings are open: it reports
        # a hollow result rather than a solid. Record that as its own status.
        summary["stages"]["wrap"]["status"] = "hollow" if "0/1 성공" in text else "ok"
        log("   " + "\n   ".join(text.splitlines()[-3:]))

# --------------------------------------------------------------------- summary
(args.out / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
log("\n== 요약 ==")
for name, entry in summary["stages"].items():
    log(f"   {name:<7} {entry.get('status', '?'):<7} {entry.get('seconds', 0):6.1f}s")
for key, value in summary["numbers"].items():
    log(f"   {key}: {value}")
log(f"\n출력 폴더: {args.out}")
log_file.close()
