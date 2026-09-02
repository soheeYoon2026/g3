"""Read the written STEP back and check it survived.

Writing a STEP file that no CAD system opens is worse than writing nothing, so the
output has to be re-read and compared against what went in: same faces, same area,
same free boundaries. The file size is checked too, because OCC writes a pcurve for
every edge by default and that alone can triple the file for no benefit to a
downstream CAD system, which recomputes them anyway.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aox_g3 import brep, cad  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--step", type=Path, nargs="+", required=True)
args = ap.parse_args()

from OCP.BRepGProp import BRepGProp  # noqa: E402
from OCP.GProp import GProp_GProps  # noqa: E402

for path in args.step:
    if not path.exists():
        print(f"{path.name}: 없음")
        continue
    t0 = time.time()
    shape, report = cad.read_step(path)
    if shape is None:
        print(f"{path.name}: 되읽기 실패 — {report.warnings}")
        continue
    cad.diagnose(shape, report)
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, props)
    holes, dangling = brep.free_boundaries(shape)

    size = path.stat().st_size / 1e6
    print(f"\n{path.name}  ({size:.1f} MB)")
    print(f"  되읽기 {time.time() - t0:.1f}s")
    print(f"  면 {report.faces:,}  셸 {report.shells:,}  고체 {report.solids:,}  "
          f"무효면 {report.invalid_faces:,}")
    print(f"  표면적 {props.Mass() / 1e6:.2f} m2   "
          f"구멍 {len(holes)}개   고리 못 이룬 경계 {len(dangling)}개")
    print(f"  부품명 {len(report.part_names)}개  색 {report.colours}개  "
          f"단위추정 {report.units_hint}")
