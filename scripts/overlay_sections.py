"""Overlay sections of a reference mesh and a candidate (wrap) to show what was bridged.

Black: reference. Red: candidate. Each panel is one cutting plane given as
axis=value[,ymin,ymax,zmin,zmax] (window in the plane's own 2-D coordinates).
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--reference", type=Path, required=True)
ap.add_argument("--candidate", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--cut", action="append", required=True, metavar="AXIS=V[,a0,a1,b0,b1]",
                help="e.g. x=-565,-700,700,40,220   window axes are the two remaining coordinates in xyz order")
ap.add_argument("--title", default="")
args = ap.parse_args()

fp = FontProperties(fname="/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc")
ref = trimesh.load(args.reference, force="mesh")
cand = trimesh.load(args.candidate, force="mesh")
AX = {"x": 0, "y": 1, "z": 2}
fig, axes = plt.subplots(1, len(args.cut), figsize=(6.2 * len(args.cut), 5.4))
axes = np.atleast_1d(axes)
for ax, spec in zip(axes, args.cut):
    axis, rest = spec.split("=")
    vals = [float(v) for v in rest.split(",")]
    k = AX[axis]; v = vals[0]
    others = [i for i in range(3) if i != k]
    o = [0, 0, 0]; n = [0, 0, 0]; o[k] = v; n[k] = 1
    for mesh, colour, lw, label in ((ref, "k", 1.0, "원본"), (cand, "r", 1.0, "랩")):
        sec = mesh.section(plane_origin=o, plane_normal=n)
        if sec is None:
            continue
        first = True
        for ent in sec.entities:
            p = sec.vertices[ent.points]
            ax.plot(p[:, others[0]], p[:, others[1]], colour, lw=lw, label=label if first else None)
            first = False
    if len(vals) == 5:
        ax.set_xlim(vals[1], vals[2]); ax.set_ylim(vals[3], vals[4])
    ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel("xyz"[others[0]] + " (mm)"); ax.set_ylabel("xyz"[others[1]] + " (mm)")
    ax.set_title(f"{axis} = {v:.0f} mm", fontproperties=fp)
    ax.legend(loc="upper right", prop=fp)
if args.title:
    fig.suptitle(args.title, fontproperties=fp, fontsize=13)
fig.tight_layout()
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=110)
print("saved", args.out)
