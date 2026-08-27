"""Report the drag distribution of the whole labelled pool, minus the frozen gate,
to see whether a Cd-stratified training split is even possible."""

import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--root", type=Path, required=True, help="v3 dataset root holding conditions_*.json")
ap.add_argument("--gate", type=Path, required=True, help="standing gate split json")
ap.add_argument("--npy-root", type=Path, required=True, help="exported train/ and val/ .npy dirs")
args = ap.parse_args()

gate = {str(r["run"]) for r in json.loads(args.gate.read_text())["test_cases"]}
used = set()
for split in ("train", "val"):
    used |= {p.stem.split("_", 1)[1] for p in (args.npy_root / split).glob("*.npy")}

rows = []
for run_dir in sorted(args.root.glob("run_*")):
    run = run_dir.name.split("_", 1)[1]
    cfg = run_dir / f"conditions_{run}.json"
    if not cfg.exists():
        continue
    conditions = json.loads(cfg.read_text())
    cd = conditions.get("su2_cd")
    if cd is None or not np.isfinite(float(cd)):  # some reaudit cases carry NaN labels
        continue
    rows.append({"run": run, "cd": float(cd),
                 "speed": float(conditions.get("speed") or 0),
                 "mach": conditions.get("mach"),
                 "ref_area": float(conditions.get("ref_area") or 0)})

print(f"라벨 있는 전체 케이스: {len(rows)}")
cds = np.array([r["cd"] for r in rows])
print(f"  Cd {cds.min():.4f}~{cds.max():.4f} 평균 {cds.mean():.4f} std {cds.std():.4f}")

# transonic wing cases pollute the pool; car-like cases are the ones we serve
cars = [r for r in rows if r["speed"] <= 100 and r["ref_area"] >= 1.0]
print(f"\n차량형(속도<=100 m/s, 기준면적>=1.0 m^2): {len(cars)}")
cc = np.array([r["cd"] for r in cars])
if len(cc):
    print(f"  Cd {cc.min():.4f}~{cc.max():.4f} 평균 {cc.mean():.4f} std {cc.std():.4f}")
    hist, edges = np.histogram(cc, bins=8)
    for i in range(len(hist)):
        print(f"   [{edges[i]:.3f}-{edges[i+1]:.3f}] {'#'*hist[i]} {hist[i]}")

avail = [r for r in cars if r["run"] not in gate and r["run"] not in used]
print(f"\n게이트/현행 split 어디에도 안 쓰인 차량형 여유분: {len(avail)}")
if avail:
    av = np.array([r["cd"] for r in avail])
    print(f"  Cd {av.min():.4f}~{av.max():.4f} std {av.std():.4f}")
    hist, edges = np.histogram(av, bins=8)
    for i in range(len(hist)):
        print(f"   [{edges[i]:.3f}-{edges[i+1]:.3f}] {'#'*hist[i]} {hist[i]}")
