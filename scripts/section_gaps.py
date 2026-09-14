"""Sections that say which part of the closed surface is stuck to the original.

    section_gaps.py --reference original.stl --candidate closed.stl --out s.png \
        --cut "x=430,-1100,1100,100,600" --cut "y=0,-2400,-1700,100,800"

The overlay of two outlines does not answer "where does it not stick": both lines
are drawn the same and the eye has to guess which red piece has a black line under
it. Here every point of the candidate's section is measured against the original
mesh: within --web-distance it is drawn grey (stuck), beyond it red and thick (the
bridge the closing step added). The original is drawn as a thin black line under
both. Axes are equal, so a 60 mm gap looks like 60 mm.
"""
import argparse
from pathlib import Path

import igl
import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--reference", type=Path, required=True)
ap.add_argument("--candidate", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--cut", action="append", required=True, metavar="AXIS=V[,a0,a1,b0,b1][;label]",
                help="plane and window in the two remaining coordinates, in xyz order; text after ';' labels the panel")
ap.add_argument("--web-distance", type=float, default=1.5)
ap.add_argument("--cols", type=int, default=3)
ap.add_argument("--title", default="")
args = ap.parse_args()

fp = FontProperties(fname="/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc")
AX = {"x": 0, "y": 1, "z": 2}
ref = trimesh.load(args.reference, force="mesh")
cand = trimesh.load(args.candidate, force="mesh")
RV = np.ascontiguousarray(ref.vertices, np.float64)
RF = np.ascontiguousarray(ref.faces, np.int64)


def segments(mesh, axis, value):
    """Section as an (n, 2, 3) array of line segments."""
    normal = np.zeros(3); normal[axis] = 1.0
    origin = np.zeros(3); origin[axis] = value
    path = mesh.section(plane_origin=origin, plane_normal=normal)
    if path is None:
        return np.zeros((0, 2, 3))
    V = np.asarray(path.vertices, float)
    ent = []
    for e in path.entities:
        pts = np.asarray(e.points, int)
        ent.append(np.stack([V[pts[:-1]], V[pts[1:]]], axis=1))
    return np.concatenate(ent) if ent else np.zeros((0, 2, 3))


cuts = []
for spec in args.cut:
    body, _, label = spec.partition(";")
    head, _, rest = body.partition("=")
    axis = AX[head.strip().lower()]
    nums = [float(v) for v in rest.split(",")]
    value, window = nums[0], (nums[1:] if len(nums) >= 5 else None)
    cuts.append((axis, value, window, label.strip()))

rows = int(np.ceil(len(cuts) / args.cols))
cols = min(args.cols, len(cuts))
fig, axes = plt.subplots(rows, cols, figsize=(5.6 * cols, 4.9 * rows), squeeze=False)
for k, (axis, value, window, label) in enumerate(cuts):
    ax = axes[k // cols][k % cols]
    keep = [i for i in range(3) if i != axis]
    ref_seg = segments(ref, axis, value)
    cnd_seg = segments(cand, axis, value)
    mid = cnd_seg.mean(axis=1)
    d = np.sqrt(igl.point_mesh_squared_distance(np.ascontiguousarray(mid, np.float64), RV, RF)[0]) if len(mid) else np.zeros(0)
    added = d > args.web_distance
    if len(ref_seg):
        ax.add_collection(LineCollection(ref_seg[:, :, keep], colors="black", linewidths=1.5, zorder=2))
    if len(cnd_seg):
        ax.add_collection(LineCollection(cnd_seg[~added][:, :, keep], colors="#8a99a8", linewidths=2.4, zorder=1))
        ax.add_collection(LineCollection(cnd_seg[added][:, :, keep], colors="#e03e30", linewidths=3.4, zorder=3))
    if window:
        ax.set_xlim(window[0], window[1]); ax.set_ylim(window[2], window[3])
    else:
        pts = np.concatenate([ref_seg.reshape(-1, 3), cnd_seg.reshape(-1, 3)])[:, keep]
        lo, hi = pts.min(0), pts.max(0); pad = 0.06 * (hi - lo).max()
        ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    names = "xyz"
    ax.set_xlabel(f"{names[keep[0]]} (mm)"); ax.set_ylabel(f"{names[keep[1]]} (mm)")
    head = f"{names[axis]} = {value:.0f} mm"
    share = float(np.linalg.norm(np.diff(cnd_seg[added], axis=1), axis=2).sum()) if added.any() else 0.0
    total = float(np.linalg.norm(np.diff(cnd_seg, axis=1), axis=2).sum()) if len(cnd_seg) else 1.0
    ax.set_title(f"{head}   {label}\n덧댄 길이 {share/10:.0f} cm / 단면 둘레 {total/10:.0f} cm" if label
                 else f"{head}\n덧댄 길이 {share/10:.0f} cm / 단면 둘레 {total/10:.0f} cm",
                 fontproperties=fp, fontsize=12)
    if k == 0:
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color="black", lw=1.5, label="원본 패널"),
                           Line2D([], [], color="#8a99a8", lw=3.0, label="원본에 붙은 면"),
                           Line2D([], [], color="#e03e30", lw=3.4, label="틈을 건너뛴 면")],
                  prop=fp, fontsize=9, loc="upper right", framealpha=0.9)
    print(f"{head} {label}: 덧댄 {share/10:.0f} cm / {total/10:.0f} cm")
for k in range(len(cuts), rows * cols):
    axes[k // cols][k % cols].axis("off")
fig.suptitle(args.title + "    검정 = 원본 패널 · 회색 = 원본에 붙은 면 · 빨강 = 틈을 건너뛴 면",
             fontproperties=fp, fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=115)
print(f"저장 {args.out}")
