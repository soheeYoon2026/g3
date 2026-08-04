"""Demonstrate the STL -> point cloud + SDF pipeline on a real car body.

Runs the actual sampler on an Ahmed-body STL and reports what the surrogate
would see. Optionally writes a scatter PNG (matplotlib) for a visual check.

    python scripts/sample_stl_demo.py --stl ../g4-docker-image/ahmed_1.stl --png out.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aox_g3.config import SampleConfig
from aox_g3.geometry.stl_sampler import load_mesh, stl_to_pointcloud, signed_distance


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", default="../g4-docker-image/ahmed_1.stl")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--png", default=None, help="optional scatter output path")
    args = ap.parse_args(argv)

    mesh = load_mesh(args.stl)
    print(f"mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
          f"watertight={mesh.is_watertight}")
    print(f"bbox extent: {mesh.extents}")

    cfg = SampleConfig(n_surface_points=args.n)
    pc = stl_to_pointcloud(args.stl, cfg)
    print(f"sampled {len(pc.points)} surface points (normalised to unit box)")
    print(f"  point range: {pc.points.min(0)} .. {pc.points.max(0)}")
    print(f"  normalisation: center={pc.center}, scale={pc.scale:.3f}")
    print(f"  feature matrix (xyz+normal): {pc.features.shape}")

    # SDF at a few probe points along the normalised x-axis (should be
    # negative outside the body, more negative further away).
    probes = np.array([[-2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=float)
    probes_raw = probes * pc.scale + pc.center
    sdf = signed_distance(mesh, probes_raw)
    print(f"  SDF at 3 external probes: {np.round(sdf, 3)} (negative = outside)")

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(9, 3))
        for k, (a, b, lbl) in enumerate([(0, 2, "x-z (side)"),
                                         (0, 1, "x-y (top)"),
                                         (1, 2, "y-z (rear)")]):
            ax = fig.add_subplot(1, 3, k + 1)
            ax.scatter(pc.points[:, a], pc.points[:, b], s=1, c=pc.points[:, 2],
                       cmap="viridis")
            ax.set_title(lbl); ax.set_aspect("equal"); ax.axis("off")
        fig.suptitle(f"G3 surface sampling — {Path(args.stl).name}")
        fig.tight_layout()
        fig.savefig(args.png, dpi=120)
        print(f"  wrote {args.png}")


if __name__ == "__main__":
    main()
