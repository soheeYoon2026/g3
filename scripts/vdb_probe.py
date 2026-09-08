"""OpenVDB level-set resurfacing of a mesh (runs in the conda env python with pyopenvdb).

Reads verts/faces .npy, builds a level set at --voxel, writes the iso-0 surface and a
morphological closing (dilate = mesh at iso +r, level set again, erode = mesh at iso -r;
OpenVDB level sets are negative inside, so +r is the dilation) as .npy pairs.
"""
import argparse, time
import numpy as np
import openvdb

ap = argparse.ArgumentParser()
ap.add_argument("--verts", required=True); ap.add_argument("--faces", required=True)
ap.add_argument("--out-prefix", required=True)
ap.add_argument("--voxel", type=float, default=2.0)
ap.add_argument("--close-radius", type=float, default=8.0)
ap.add_argument("--adaptivity", type=float, default=0.5)
args = ap.parse_args()

V = np.load(args.verts).astype(np.float64); F = np.load(args.faces).astype(np.int32)
xf = openvdb.createLinearTransform(voxelSize=args.voxel)

def level_set(points, tris, half_width):
    return openvdb.FloatGrid.createLevelSetFromPolygons(points, triangles=tris, transform=xf, halfWidth=half_width)

def to_tris(points, tris, quads):
    tris = np.asarray(tris, np.int64).reshape(-1, 3) if len(tris) else np.zeros((0, 3), np.int64)
    if len(quads):
        q = np.asarray(quads, np.int64).reshape(-1, 4)
        tris = np.vstack([tris, q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    return np.asarray(points, np.float64), tris

t0 = time.time()
hw = max(3.0, args.close_radius / args.voxel + 2)
g = level_set(V, F, hw)
print(f"level set: voxel {args.voxel} half-width {hw:.0f} active voxels {g.activeVoxelCount():,}  {time.time()-t0:.1f}s")
p, t, q = g.convertToPolygons(isovalue=0.0, adaptivity=args.adaptivity)
P, T = to_tris(p, t, q); np.save(args.out_prefix + "_iso0_v.npy", P); np.save(args.out_prefix + "_iso0_f.npy", T)
print(f"iso 0: {len(T):,} tris  {time.time()-t0:.1f}s")
r = args.close_radius
p, t, q = g.convertToPolygons(isovalue=+r, adaptivity=0.0)   # dilate
P1, T1 = to_tris(p, t, q)
g2 = level_set(P1, T1, hw)
p, t, q = g2.convertToPolygons(isovalue=-r, adaptivity=args.adaptivity)   # erode back
P2, T2 = to_tris(p, t, q); np.save(args.out_prefix + f"_close{r:.0f}_v.npy", P2); np.save(args.out_prefix + f"_close{r:.0f}_f.npy", T2)
print(f"closing ±{r}: {len(T2):,} tris  {time.time()-t0:.1f}s")
