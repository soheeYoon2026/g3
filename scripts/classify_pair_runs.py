"""Label every run in the pair dataset with the production upload-gate verdict.

Scoring already has to be split by shape class; training should be able to use the
same labels, so this writes them once instead of re-classifying in each script.
"""

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pyvista as pv
import trimesh

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--gate-module", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

sys.path.insert(0, str(args.gate_module))
from aox_g3.upload_gate import classify_mesh

verdicts = {}
run_dirs = sorted((p for p in args.root.glob("run_*") if p.is_dir()),
                  key=lambda p: int(p.name.split("_")[1]))
for run_dir in run_dirs:
    tag = run_dir.name.split("_", 1)[1]
    surface = pv.read(run_dir / f"boundary_{tag}.vtp").extract_surface().triangulate()
    mesh = trimesh.Trimesh(vertices=np.asarray(surface.points, dtype=np.float64),
                           faces=np.asarray(surface.faces).reshape(-1, 4)[:, 1:],
                           process=False)
    try:
        result = classify_mesh(mesh)
        verdicts[tag] = {"verdict": result.get("verdict", "unknown"),
                         "features": result.get("features", {})}
    except Exception as exc:
        verdicts[tag] = {"verdict": "error", "error": f"{type(exc).__name__}: {exc}"}

counts = Counter(v["verdict"] for v in verdicts.values())
print(f"런 {len(verdicts)}개 분류")
for verdict, count in counts.most_common():
    print(f"  {verdict}: {count}")

pairs = json.loads((args.root / "pairs.json").read_text())["pairs"]
pair_counts = Counter(verdicts[str(p["baseline"])]["verdict"] for p in pairs
                      if str(p["baseline"]) in verdicts)
print(f"\n쌍 {len(pairs)}개 (기준 형상의 판정 기준)")
for verdict, count in pair_counts.most_common():
    print(f"  {verdict}: {count}")

args.out.write_text(json.dumps(verdicts, indent=1) + "\n")
print("wrote", args.out)
