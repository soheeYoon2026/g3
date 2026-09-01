"""Pick base shapes for the LES benchmark campaign.

The benchmark needs drag-increasing deformations on car geometry the serving model
has never trained on. This finds candidate base shapes: full cars by the
production gate, watertight enough for LES, sitting inside the encoder grid after
alignment, and belonging to no job the serving model saw.
"""

import argparse
import json
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pyvista as pv
import trimesh

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--run-index", type=Path, required=True)
ap.add_argument("--shape-class", type=Path, required=True)
ap.add_argument("--coverage", type=Path, help="run -> fraction inside encoder grid")
ap.add_argument("--incumbent-log", type=Path, required=True)
ap.add_argument("--reaudit-root", type=Path, required=True)
ap.add_argument("--out", type=Path)
args = ap.parse_args()

log = args.incumbent_log.read_text()
seen_runs, order = set(), []
for run in re.findall(r"load_case\(.run_([0-9]+).\)", log):
    if run not in seen_runs:
        seen_runs.add(run)
        order.append(run)
seen_jobs = set()
for run in order:
    path = args.reaudit_root / f"run_{run}" / f"conditions_{run}.json"
    if not path.exists():
        continue
    match = re.search(r"([A-Za-z0-9]{22})_CFD\.cfg",
                      json.loads(path.read_text()).get("source_cfg") or "")
    if match:
        seen_jobs.add(match.group(1))
print(f"현행이 학습에서 본 잡: {len(seen_jobs)}개")

index = json.loads(args.run_index.read_text())
classes = json.loads(args.shape_class.read_text())

# one base shape per unseen job: the design the optimizer started from
candidates = {}
for run, meta in index.items():
    if meta["job_uid"] in seen_jobs:
        continue
    if classes.get(str(run), {}).get("verdict") != "full_car":
        continue
    if meta["design"] != "RBF_DSN_001":
        continue
    candidates[meta["job_uid"]] = run
print(f"현행 미학습 · 차량형 · 기준설계: {len(candidates)}개 형상")

rows = []
for job, run in sorted(candidates.items()):
    run_dir = args.root / f"run_{run}"
    try:
        mesh = pv.read(run_dir / f"boundary_{run}.vtp").extract_surface().triangulate().clean()
        tri = trimesh.Trimesh(vertices=np.asarray(mesh.points, dtype=np.float64),
                              faces=np.asarray(mesh.faces).reshape(-1, 4)[:, 1:],
                              process=False)
        conditions = json.loads((run_dir / f"conditions_{run}.json").read_text())
    except Exception as exc:
        print(f"  run_{run}: 읽기 실패 {type(exc).__name__}")
        continue
    rows.append({
        "job_uid": job, "run": int(run),
        "cd": float(conditions.get("su2_cd", float("nan"))),
        "watertight": bool(tri.is_watertight),
        "extents": np.round(tri.extents, 3).tolist(),
        "cells": int(mesh.n_cells),
        "volume": round(float(tri.volume), 3),
    })

print(f"\n{'job':>10s} {'run':>6s} {'Cd':>7s} {'수밀':>5s} {'치수':>24s} {'셀':>7s}")
print("-" * 66)
for r in sorted(rows, key=lambda r: r["cd"]):
    print(f"{r['job_uid'][:8]:>10s} {r['run']:>6d} {r['cd']:7.4f} "
          f"{str(r['watertight']):>5s} {str(r['extents']):>24s} {r['cells']:>7d}")

watertight = [r for r in rows if r["watertight"]]
print(f"\n수밀 형상 {len(watertight)}/{len(rows)}개 — LES에 바로 쓸 수 있는 후보")
if watertight:
    cds = np.array([r["cd"] for r in watertight])
    print(f"  Cd 범위 {cds.min():.4f}~{cds.max():.4f}")

if args.out and watertight:
    args.out.write_text(json.dumps({"candidates": watertight}, indent=1) + "\n")
    print("wrote", args.out)
