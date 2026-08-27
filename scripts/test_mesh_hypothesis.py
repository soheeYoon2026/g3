"""Where does the model's own +0.0045 ΔCd offset come from?

The fitted bias splits into two parts: the benchmark's skew toward drag reduction,
which varies by fold, and a stable positive offset in the predictions themselves.
G2 re-meshes every variant, so a candidate explanation is that the predicted
difference carries a mesh-change component alongside the shape-change one. This
correlates each pair's prediction residual against how much the mesh changed, and
against how much the shape changed, to see which one it tracks.
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pyvista as pv


def load(path: Path):
    rows = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("model") == "fine_tuned" and "case" in row:
            rows[row["case"]] = row
    return rows


def mesh_stats(run_dir: Path, tag: str):
    mesh = pv.read(run_dir / f"boundary_{tag}.vtp").extract_surface().triangulate()
    sized = mesh.compute_cell_sizes(length=False, volume=False)
    areas = np.asarray(sized.cell_data["Area"], dtype=np.float64)
    conditions = json.loads((run_dir / f"conditions_{tag}.json").read_text())
    return {
        "cells": float(mesh.n_cells),
        "wetted": float(areas.sum()),
        "mean_cell": float(areas.mean()),
        "ref_area": float(conditions["ref_area"]),
    }


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--cv-dir", type=Path, required=True)
ap.add_argument("--root", type=Path, required=True)
args = ap.parse_args()

records = []
for fold in sorted(args.cv_dir.glob("fold*")):
    rows = load(fold / "eval-test.jsonl")
    for pair in json.loads((fold / "pairs-test.json").read_text())["pairs"]:
        base, var = pair["baseline"], pair["variant"]
        if f"run_{base}" not in rows or f"run_{var}" not in rows:
            continue
        sb = mesh_stats(args.root / f"run_{base}", str(base))
        sv = mesh_stats(args.root / f"run_{var}", str(var))
        pred = rows[f"run_{var}"]["pred_cd"] - rows[f"run_{base}"]["pred_cd"]
        records.append({
            "residual": pred - pair["true_delta_cd"],
            "pred": pred,
            "true": pair["true_delta_cd"],
            "d_cells": (sv["cells"] - sb["cells"]) / sb["cells"],
            "d_wetted": (sv["wetted"] - sb["wetted"]) / sb["wetted"],
            "d_mean_cell": (sv["mean_cell"] - sb["mean_cell"]) / sb["mean_cell"],
            "d_ref_area": (sv["ref_area"] - sb["ref_area"]) / sb["ref_area"],
        })

print(f"쌍 {len(records)}개\n")
residual = np.array([r["residual"] for r in records])
print(f"잔차(예측−정답) 평균 {residual.mean():+.5f}  표준편차 {residual.std():.5f}")

print(f"\n{'설명변수':>14s} {'평균':>10s} {'잔차와 상관':>11s} {'예측과 상관':>11s} {'정답과 상관':>11s}")
print("-" * 62)
for key, name in (("d_cells", "셀 수 변화"), ("d_wetted", "젖은면적 변화"),
                  ("d_mean_cell", "평균셀크기 변화"), ("d_ref_area", "기준면적 변화")):
    x = np.array([r[key] for r in records])
    if x.std() == 0:
        print(f"{name:>14s} {x.mean():+10.5f}  (변화 없음)")
        continue
    c_res = float(np.corrcoef(x, residual)[0, 1])
    c_pred = float(np.corrcoef(x, [r["pred"] for r in records])[0, 1])
    c_true = float(np.corrcoef(x, [r["true"] for r in records])[0, 1])
    print(f"{name:>14s} {x.mean():+10.5f} {c_res:+11.2f} {c_pred:+11.2f} {c_true:+11.2f}")

print("\n메시가 원인이라면 잔차가 셀 수/평균셀크기 변화를 따라가야 하고,")
print("형상이 원인이라면 잔차가 젖은면적·기준면적 변화를 따라가야 한다.")
