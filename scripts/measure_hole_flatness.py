"""How many holes could be capped on a fitted plane instead of a solved surface?

Free-form filling keeps producing patches that leave the hole, and raising the
continuity does not help: G1 tangency to the neighbouring face is accepted on all
407 boundary edges and the patches still run away. That is a sign the approach is
wrong rather than mistuned - a single surface that is tangent to a dozen unrelated
panels along a convoluted 3D loop may simply not exist, and asking for one gets
whatever the solver settles on.

Projection avoids the whole question. Fit a plane to the boundary, build the patch
on that plane, and the patch cannot wander because the surface was chosen rather
than solved. The cost is the error where the boundary is not actually flat, so
measure it: for each hole, the out-of-plane spread of its boundary against the
hole's own size decides whether a plane is a fair cap or a lie.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aox_g3 import brep, cad  # noqa: E402

from OCP.BRep import BRep_Tool  # noqa: E402
from OCP.BRepAdaptor import BRepAdaptor_Curve  # noqa: E402
from OCP.TopAbs import TopAbs_EDGE  # noqa: E402
from OCP.TopoDS import TopoDS  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--step", type=Path, required=True)
ap.add_argument("--max-size", type=float, default=900.0)
args = ap.parse_args()


def sample(wire, per_edge=16):
    pts = []
    for e in brep._explore(wire, TopAbs_EDGE):
        edge = TopoDS.Edge_s(e)
        curve = BRepAdaptor_Curve(edge)
        first, last = BRep_Tool.Range_s(edge)
        for t in np.linspace(first, last, per_edge):
            p = curve.Value(float(t))
            pts.append((p.X(), p.Y(), p.Z()))
    return np.asarray(pts)


shape, report = cad.read_step(args.step)
cad.diagnose(shape, report)
shape, report = cad.sew_progressive(shape, report)
holes, _ = brep.free_boundaries(shape)

print(f"{'구멍크기':>9} {'평면밖 최대':>11} {'RMS':>9} {'최대/크기':>10} {'모서리':>7}")
rows = []
for boundary, wire in holes:
    if boundary.size > args.max_size:
        continue
    pts = sample(wire)
    centre = pts.mean(axis=0)
    centred = pts - centre
    # The plane through the centroid whose normal is the least-variance direction
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[2]
    offset = centred @ normal
    worst = float(np.abs(offset).max())
    rms = float(np.sqrt((offset ** 2).mean()))
    ratio = worst / boundary.size if boundary.size else float("inf")
    rows.append(ratio)
    print(f"{boundary.size:9.1f} {worst:11.2f} {rms:9.2f} {ratio:10.4f} "
          f"{boundary.edges:7d}")

if rows:
    ordered = sorted(rows)
    print(f"\n{len(ordered)}개   중앙 {ordered[len(ordered) // 2]:.4f}   "
          f"최대 {ordered[-1]:.4f}")
    print("  분포:", " ".join(f"{r:.3f}" for r in ordered))
    for cut in (0.01, 0.02, 0.05, 0.10, 0.20):
        n = sum(1 for r in ordered if r <= cut)
        print(f"  평면밖/크기 {cut:>5.2f} 이하: {n:3d}개 / {len(ordered)}개")
