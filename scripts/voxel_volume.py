"""Measure enclosed volume by voxel occupancy instead of the divergence theorem.

Every judgement so far - the seal cascade's hollow test, the alpha sweep's fill
fraction - rests on trimesh's `.volume`, which integrates over the faces and is
meaningless if the winding is inconsistent. A wrap loaded from an OFF file is
exactly the case where that can happen, so before concluding the wrap is hollow,
count what it actually encloses: voxelise, flood the outside, and whatever is
neither wall nor outside is enclosed.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--mesh", type=Path, nargs="+", required=True)
ap.add_argument("--pitch-div", type=float, default=250.0)
args = ap.parse_args()

for path in args.mesh:
    mesh = trimesh.load(path, force="mesh")
    lo, hi = np.asarray(mesh.bounds[0]), np.asarray(mesh.bounds[1])
    diag = float(np.linalg.norm(hi - lo))
    pitch = diag / args.pitch_div
    pad = 3 * pitch
    origin = lo - pad
    shape = tuple(int(np.ceil((hi[i] + pad - origin[i]) / pitch)) + 1 for i in range(3))

    wall = np.zeros(shape, dtype=bool)
    tris = np.asarray(mesh.triangles, dtype=float)
    a = np.clip(np.floor((tris.min(axis=1) - origin) / pitch).astype(np.int64),
                0, np.array(shape) - 1)
    b = np.clip(np.floor((tris.max(axis=1) - origin) / pitch).astype(np.int64),
                0, np.array(shape) - 1)
    for (i0, j0, k0), (i1, j1, k1) in zip(a, b):
        wall[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1] = True

    labels, _ = ndimage.label(~wall)
    outside = labels == labels[0, 0, 0]
    enclosed = ~wall & ~outside

    voxel = pitch ** 3
    box = float(np.prod(hi - lo))
    enclosed_volume = enclosed.sum() * voxel
    wall_volume = wall.sum() * voxel
    print(f"\n{path.name}")
    print(f"  격자 {shape}  피치 {pitch:.1f}")
    print(f"  divergence 체적 {abs(mesh.volume) / 1e9:8.3f} m3  "
          f"({100 * abs(mesh.volume) / box:.1f}% of box)")
    print(f"  복셀 내부 체적  {enclosed_volume / 1e9:8.3f} m3  "
          f"({100 * enclosed_volume / box:.1f}% of box)")
    print(f"  벽 복셀 체적    {wall_volume / 1e9:8.3f} m3")
    ratio = (enclosed_volume / abs(mesh.volume)) if mesh.volume else float("inf")
    print(f"  복셀/divergence = {ratio:.1f}x "
          f"{'← divergence 가 크게 과소' if ratio > 3 else ''}")
