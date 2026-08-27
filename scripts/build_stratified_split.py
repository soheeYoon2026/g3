"""Build a Cd-stratified train/val split over the car-like labelled pool.

The split in use put 40 of 54 training cases inside one narrow drag bin and gave
validation a Cd spread of 0.006, so validation loss could not distinguish models
and the trained model over-amplified whenever asked to extrapolate. This draws
from every drag stratum instead, and gives validation a case from each so it
actually spans the range the gate asks about.
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import trimesh

ap = argparse.ArgumentParser()
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--gate", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--val-per-stratum", type=int, default=2)
ap.add_argument("--max-speed", type=float, default=100.0)
ap.add_argument("--min-ref-area", type=float, default=1.0)
ap.add_argument("--min-stl-points", type=int, default=0,
                help="drop cases the training sampler cannot fill (0 disables)")
ap.add_argument("--edges", type=float, nargs="+",
                default=[0.0, 0.23, 0.27, 0.31, 0.40, 10.0])
args = ap.parse_args()

gate = {str(r["run"]) for r in json.loads(args.gate.read_text())["test_cases"]}

pool, undersized = [], []
for run_dir in sorted(args.root.glob("run_*")):
    run = run_dir.name.split("_", 1)[1]
    cfg = run_dir / f"conditions_{run}.json"
    if run in gate or not cfg.exists():
        continue
    c = json.loads(cfg.read_text())
    cd = c.get("su2_cd")
    if cd is None or not np.isfinite(float(cd)):
        continue
    if float(c.get("speed") or 0) > args.max_speed:
        continue
    if float(c.get("ref_area") or 0) < args.min_ref_area:
        continue
    if args.min_stl_points:
        stl = run_dir / f"drivaer_{run}.stl"
        if not stl.exists():
            continue
        mesh = trimesh.load(stl, force="mesh", process=False)
        if len(mesh.vertices) < args.min_stl_points:
            undersized.append((run, len(mesh.vertices)))
            continue
    pool.append({"run": run, "cd": float(cd)})

pool.sort(key=lambda r: r["cd"])
print(f"게이트 제외 차량형 라벨 케이스: {len(pool)}")
if undersized:
    print(f"  표본 미달로 제외: {len(undersized)}개 " +
          ", ".join(f"run_{r}({n:,}점)" for r, n in undersized))

train, val = [], []
for i in range(len(args.edges) - 1):
    lo, hi = args.edges[i], args.edges[i + 1]
    stratum = [r for r in pool if lo <= r["cd"] < hi]
    if not stratum:
        print(f"  [{lo:.2f}-{hi:.2f}) 비어 있음")
        continue
    # spread the validation picks across the stratum rather than taking one end
    take = min(args.val_per_stratum, max(1, len(stratum) // 4))
    idx = np.linspace(0, len(stratum) - 1, take).round().astype(int)
    picked = {stratum[j]["run"] for j in idx}
    val += [r for r in stratum if r["run"] in picked]
    train += [r for r in stratum if r["run"] not in picked]
    print(f"  [{lo:.2f}-{hi:.2f}) {len(stratum):3d}개 -> train {len(stratum)-len(picked)}, val {len(picked)}")

for name, rows in (("train", train), ("val", val)):
    cds = np.array([r["cd"] for r in rows])
    print(f"{name}: {len(rows)}개  Cd {cds.min():.4f}~{cds.max():.4f} std {cds.std():.4f}")

args.out.write_text(json.dumps({
    "train_cases": [{"run": int(r["run"]), "group_id": None, "su2_cd": r["cd"]} for r in train],
    "val_cases": [{"run": int(r["run"]), "group_id": None, "su2_cd": r["cd"]} for r in val],
}, indent=2))
print("wrote", args.out)
