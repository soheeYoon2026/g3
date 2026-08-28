"""Are the model's ΔCd disagreements concentrated on non-car geometry?

Before spending GPU days adjudicating with LES, check the cheaper explanation:
the disagreements may not mean the labels are wrong, they may mean the model is
being asked about shapes unlike anything it was trained on. If disagreement rate
tracks how car-like a geometry is, the ceiling is coverage, not labels.
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


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--cv-dir", type=Path, required=True)
ap.add_argument("--root", type=Path, required=True)
args = ap.parse_args()

cache = {}


def shape_of(run):
    if run in cache:
        return cache[run]
    mesh = pv.read(args.root / f"run_{run}" / f"boundary_{run}.vtp").extract_surface()
    span = np.sort(np.diff(np.asarray(mesh.bounds).reshape(3, 2), axis=1).ravel())[::-1]
    sized = mesh.triangulate().compute_cell_sizes(length=False, volume=False)
    cache[run] = {
        "length": float(span[0]), "width": float(span[1]), "height": float(span[2]),
        "area": float(np.asarray(sized.cell_data["Area"]).sum()),
    }
    return cache[run]


records = []
for fold in sorted(args.cv_dir.glob("fold*")):
    rows = load(fold / "eval-test.jsonl")
    for pair in json.loads((fold / "pairs-test.json").read_text())["pairs"]:
        base, var = pair["baseline"], pair["variant"]
        if f"run_{base}" not in rows or f"run_{var}" not in rows:
            continue
        pred = rows[f"run_{var}"]["pred_cd"] - rows[f"run_{base}"]["pred_cd"]
        shape = shape_of(base)
        # a road car is about 4.5-5.5 m long, 1.8-2.1 wide, 1.3-1.6 tall
        car_like = (1.2 <= shape["height"] <= 1.8 and 1.7 <= shape["width"] <= 2.3)
        records.append({
            "agree": bool(np.sign(pred) == np.sign(pair["true_delta_cd"])),
            "car_like": car_like,
            **shape,
        })

n = len(records)
car = [r for r in records if r["car_like"]]
odd = [r for r in records if not r["car_like"]]
print(f"채점 쌍 {n}개 — 차량형 {len(car)}개 / 비차량형 {len(odd)}개\n")

for name, group in (("차량형", car), ("비차량형", odd)):
    if not group:
        continue
    hits = sum(1 for r in group if r["agree"])
    print(f"{name}: 방향 일치 {hits}/{len(group)} = {100*hits/len(group):.0f}%")
    dims = np.array([[r["length"], r["width"], r["height"]] for r in group])
    print(f"   치수 중앙값 {np.round(np.median(dims, axis=0), 2).tolist()} m  "
          f"면적 중앙값 {np.median([r['area'] for r in group]):.1f} m^2")

if car and odd:
    p_car = sum(r["agree"] for r in car) / len(car)
    p_odd = sum(r["agree"] for r in odd) / len(odd)
    print(f"\n차이 {100*(p_car-p_odd):+.0f}%p")
    print("→ 차량형에서 뚜렷이 높으면, 69% 천장은 라벨이 아니라 형상 분포 문제다")
