"""Sanity-check the benchmark shapes before spending GPU days on them.

Two things to establish: the production gate still calls them cars after
deformation, and the deformations are large enough for 4-level LES to resolve
(measured scatter cd_std 0.0003-0.0007, so |ΔCd| above about 0.005 is safe).
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import trimesh

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--dir", type=Path, required=True)
ap.add_argument("--gate-module", type=Path, required=True)
args = ap.parse_args()

sys.path.insert(0, str(args.gate_module))
from aox_g3.upload_gate import classify_mesh

for stl in sorted(args.dir.glob("*.stl")):
    # STL stores every triangle's corners separately, so watertightness only shows
    # after welding — and that is exactly how les_service_run.py loads them
    mesh = trimesh.load(stl, force="mesh")
    try:
        result = classify_mesh(mesh)
    except Exception as exc:
        print(f"{stl.stem:>20s}  분류 실패 {type(exc).__name__}")
        continue
    features = result.get("features", {})
    print(f"{stl.stem:>20s}  {result.get('verdict', '?'):>14s}  "
          f"L{features.get('length_m', 0):5.2f} W{features.get('width_m', 0):5.2f} "
          f"H{features.get('height_m', 0):5.2f}  "
          f"정면적 {features.get('frontal_area_m2', 0):5.2f}  "
          f"체적 {mesh.volume:7.3f}  수밀 {mesh.is_watertight}")
    reasons = result.get("reasons") or []
    if reasons:
        print(f"{'':>20s}  사유: {'; '.join(reasons)[:100]}")
