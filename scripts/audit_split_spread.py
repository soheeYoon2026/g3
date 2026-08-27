"""Report the label spread of each split, since a model can only interpolate over
the range it was trained on."""

import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--root", type=Path, required=True, help="v3 dataset root holding conditions_*.json")
ap.add_argument("--npy-root", type=Path, required=True, help="exported train/ and val/ .npy dirs")
ap.add_argument("--gate", type=Path, required=True, help="standing gate split json")
args = ap.parse_args()


def cd_of(run: str):
    path = args.root / f"run_{run}" / f"conditions_{run}.json"
    if not path.exists():
        return None
    cd = json.loads(path.read_text()).get("su2_cd")
    return cd if cd is not None and np.isfinite(float(cd)) else None


groups = {}
for split in ("train", "val"):
    runs = [p.stem.split("_", 1)[1] for p in sorted((args.npy_root / split).glob("*.npy"))]
    groups[split] = runs
groups["gate"] = [str(r["run"]) for r in json.loads(args.gate.read_text())["test_cases"]]

for name, runs in groups.items():
    values = [cd_of(r) for r in runs]
    found = np.array([v for v in values if v is not None], dtype=float)
    missing = len(values) - len(found)
    if not len(found):
        print(f"{name:>6s}: {len(runs)}개, 라벨 확인 불가")
        continue
    quart = np.percentile(found, [25, 50, 75])
    print(f"{name:>6s}: {len(runs):3d}개 (라벨없음 {missing:2d}) "
          f"Cd {found.min():.4f}~{found.max():.4f} "
          f"평균 {found.mean():.4f} std {found.std():.4f} "
          f"사분위 {quart[0]:.3f}/{quart[1]:.3f}/{quart[2]:.3f}")
    hist, edges = np.histogram(found, bins=6)
    print("        분포 " + "  ".join(f"[{edges[i]:.2f}-{edges[i+1]:.2f}]:{hist[i]}"
                                     for i in range(len(hist))))
