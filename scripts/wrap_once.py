"""Wrap once at a given alpha and write the result, so it can be inspected."""

import argparse
from pathlib import Path

import numpy as np
import trimesh

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--mesh", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--alpha-div", type=float, default=90.0)
ap.add_argument("--offset", type=float,
                help="wrap offset in model units (default alpha/30). A larger offset rounds "
                     "plate edges and junctions at the cost of inflating every surface by it")
args = ap.parse_args()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aox_g3.seal import alpha_wrap  # noqa: E402

mesh = trimesh.load(args.mesh, force="mesh")
diag = float(np.linalg.norm(np.asarray(mesh.extents, dtype=float)))
alpha = diag / args.alpha_div
offset = args.offset if args.offset is not None else alpha / 30.0
wrapped = alpha_wrap(mesh, alpha, offset)
wrapped.export(args.out)
print(f"alpha {alpha:.1f} (대각선/{args.alpha_div:.0f})  offset {offset:.2f}")
print(f"삼각형 {len(wrapped.faces):,}  -> {args.out}")
