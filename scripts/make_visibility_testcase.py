"""Build a STEP whose answer is known, so step 8 can be checked rather than trusted.

CAS-A cannot validate the hidden-geometry classification: it leaks, so the flood
fill reaches the cabin and the honest response is to refuse. Refusing correctly is
worth something, but it says nothing about whether the classification is right when
it does run.

So make a case with an answer known in advance - the same discipline as the
140-hole mesh used to check the sealing cascade:

    an outer box          6 faces, every one of them skin
    an inner box inside   6 faces, every one hidden, reachable by nothing
    a plate outside it    2 large faces with flow on both sides - baffles,
                          plus 4 thin edge faces

Anything that gets these wrong will get a car wrong.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from OCP.BRep import BRep_Builder  # noqa: E402
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox  # noqa: E402
from OCP.gp import gp_Pnt  # noqa: E402
from OCP.TopoDS import TopoDS_Compound  # noqa: E402

from aox_g3 import brep  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

builder = BRep_Builder()
compound = TopoDS_Compound()
builder.MakeCompound(compound)

# The body: closed, so the flood fill has an outside and an inside to tell apart
outer = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1000.0, 500.0, 400.0).Shape()
builder.Add(compound, outer)

# Buried: no path to it at any passage size
inner = BRepPrimAPI_MakeBox(gp_Pnt(400, 150, 150), 200.0, 200.0, 100.0).Shape()
builder.Add(compound, inner)

# A baffle: a thin plate standing clear of the body, air on both of its faces.
# 4 mm is under the 17 mm voxel pitch a car would use, which is the case that
# matters — a baffle thinner than the pitch is exactly what "zero thickness" means
# to a mesher.
plate = BRepPrimAPI_MakeBox(gp_Pnt(1200, 100, 50), 4.0, 300.0, 300.0).Shape()
builder.Add(compound, plate)

ok = brep.write_step(compound, args.out, units="MM")
print(f"{'저장' if ok else '실패'}: {args.out}")
print("기대값:  외피 6면 (바깥 상자)   숨음 6면 (안쪽 상자)   "
      "baffle 2면 + 얇은 옆면 4개 (판)")
