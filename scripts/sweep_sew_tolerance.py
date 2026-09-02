"""How coarse can the sewing tolerance be before it damages the CAD?

The mesh path picked diagonal x 0.002 - about 10.5 mm on a car - because it only
had to bridge the gaps before tessellation, and a tessellated triangle does not
care what tolerance produced it. Handing STEP back changes that. OCC stores the
tolerance on the edges it creates, a downstream CAD system has to honour it, and
10 mm of slop on a body panel edge is not a repaired model. The round trip shows
the cost: an input with zero invalid faces comes back with 196.

So sweep it. Reading the STEP is the slow part and does not depend on the
tolerance, so read once and sew repeatedly. What is worth watching is not just
whether shells merge but whether the faces survive: shells and free edges say how
much got stitched, invalid faces say what it cost.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from aox_g3 import brep, cad  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--step", type=Path, required=True)
ap.add_argument("--tolerances", type=float, nargs="+",
                default=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.51])
args = ap.parse_args()

from OCP.BRepGProp import BRepGProp  # noqa: E402
from OCP.GProp import GProp_GProps  # noqa: E402

t0 = time.time()
shape, base = cad.read_step(args.step)
if shape is None:
    raise SystemExit(f"읽기 실패: {base.warnings}")
cad.diagnose(shape, base)
print(f"읽기 {time.time() - t0:.1f}s   면 {base.faces:,}  셸 {base.shells:,}  "
      f"자유모서리 {base.free_edges:,}  무효면 {base.invalid_faces:,}")

props = GProp_GProps()
BRepGProp.SurfaceProperties_s(shape, props)
area0 = props.Mass()
print(f"원본 표면적 {area0 / 1e6:.3f} m2\n")

print(f"{'허용오차':>9} {'셸':>7} {'자유모서리':>10} {'무효면':>7} "
      f"{'구멍':>6} {'표면적 m2':>10} {'면적변화':>9} {'초':>6}")
for tol in args.tolerances:
    t0 = time.time()
    report = cad.CadReport(bbox=base.bbox)
    try:
        sewn, report = cad.sew(shape, report, tol)
    except Exception as exc:
        print(f"{tol:9.2f}   실패 {type(exc).__name__}")
        continue
    after = cad.CadReport()
    cad.diagnose(sewn, after)
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(sewn, props)
    area = props.Mass()
    holes, _ = brep.free_boundaries(sewn)
    print(f"{tol:9.2f} {after.shells:7,} {after.free_edges:10,} "
          f"{after.invalid_faces:7,} {len(holes):6,} {area / 1e6:10.3f} "
          f"{100 * (area - area0) / area0:+8.2f}% {time.time() - t0:6.1f}")
