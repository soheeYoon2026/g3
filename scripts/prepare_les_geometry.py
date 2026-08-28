"""Turn selected G2 surfaces into watertight LES inputs.

The v8/v9 label campaigns ran LES on open-shell upload STLs and got four times the
run-to-run scatter. G2's solved surface is nearly closed already, so this reports
what is open, fills it, and refuses to export anything still leaking -- an
unnoticed open shell is exactly how those campaigns wasted GPU days.
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pyvista as pv
import trimesh


def to_trimesh(mesh) -> trimesh.Trimesh:
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    return trimesh.Trimesh(vertices=np.asarray(mesh.points, dtype=np.float64),
                           faces=faces, process=False)


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--runs", type=Path, required=True, help="one run id per line")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--hole-size", type=float, default=0.5)
args = ap.parse_args()

args.out.mkdir(parents=True, exist_ok=True)
runs = [line.strip() for line in args.runs.read_text().splitlines() if line.strip()]

report = []
for run in runs:
    run_dir = args.root / f"run_{run}"
    surface = pv.read(run_dir / f"boundary_{run}.vtp").extract_surface().triangulate().clean()
    before = to_trimesh(surface)

    sealed = surface.fill_holes(args.hole_size).triangulate().clean()
    tri = to_trimesh(sealed)
    tri.fill_holes()
    tri.update_faces(tri.nondegenerate_faces())
    tri.remove_unreferenced_vertices()
    tri.fix_normals()

    entry = {
        "run": run,
        "watertight_before": bool(before.is_watertight),
        "watertight_after": bool(tri.is_watertight),
        "faces": int(len(tri.faces)),
        "added_faces": int(len(tri.faces) - len(before.faces)),
        "volume": float(tri.volume),
        "area": float(tri.area),
        "extents": np.round(tri.extents, 4).tolist(),
        "degenerate": int((tri.area_faces <= 0).sum()),
    }
    if tri.is_watertight and entry["degenerate"] == 0:
        # LES reads millimetres for these uploads; keep metres and let the runner scale
        tri.export(args.out / f"run_{run}.stl")
        entry["exported"] = True
    else:
        entry["exported"] = False
    report.append(entry)
    print(f"run_{run}: 수밀 {entry['watertight_before']} -> {entry['watertight_after']}  "
          f"추가면 {entry['added_faces']}  체적 {entry['volume']:.4f}  "
          f"치수 {entry['extents']}  {'내보냄' if entry['exported'] else '보류'}")

(args.out / "seal_report.json").write_text(json.dumps(report, indent=1) + "\n")
ok = sum(1 for r in report if r["exported"])
print(f"\n{ok}/{len(report)}개 LES 입력 준비 완료 -> {args.out}")
if ok != len(report):
    print("일부가 수밀이 아니다 — 그대로 LES에 넣으면 v8/v9의 4배 노이즈를 반복한다")
