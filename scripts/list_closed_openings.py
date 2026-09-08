"""Which openings did a wrap close? Lists the patches of added material.

A wrap face whose centroid is farther than --web-distance from the reference is
material the wrap added. Connected patches of such faces are listed largest first
with their centre, extent and the gap they bridged (about twice the largest
centroid distance in the patch). Also reports whether the hollow roll-hoop-style
tubes came out filled, by counting nested loops in x-sections.
"""

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import trimesh
from shapely.geometry import Polygon

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--reference", type=Path, required=True)
ap.add_argument("--candidates", type=Path, nargs="+", required=True)
ap.add_argument("--web-distance", type=float, default=1.5)
ap.add_argument("--min-area", type=float, default=2000.0, help="mm² — smaller patches are not listed")
ap.add_argument("--top", type=int, default=14)
ap.add_argument("--tube-x", type=str, default="", help="x range 'a,b' to scan for hollow tubes (nested loops)")
args = ap.parse_args()

ref = trimesh.load(args.reference, force="mesh")
ref.merge_vertices()


def hollow_tubes(mesh, x0, x1):
    n = 0
    for x in np.arange(x0, x1, 50.0):
        sec = mesh.section(plane_origin=[x, 0, 0], plane_normal=[1, 0, 0])
        if sec is None:
            continue
        p2, _ = sec.to_2D()
        polys = [Polygon(np.asarray(e.discrete(p2.vertices))) for e in p2.entities if len(e.points) > 3]
        polys = [p for p in polys if p.is_valid and 1 < p.area < 5000]
        n += sum(1 for i, a in enumerate(polys) for j, b in enumerate(polys) if i != j and a.contains(b))
    return n


for path in args.candidates:
    m = trimesh.load(path, force="mesh")
    m.merge_vertices()
    c = m.triangles_center
    _, d, _ = trimesh.proximity.closest_point(ref, c)
    web = d > args.web_distance
    adj = m.face_adjacency
    g = nx.Graph()
    g.add_nodes_from(np.flatnonzero(web))
    g.add_edges_from(adj[web[adj].all(axis=1)])
    comps = sorted(nx.connected_components(g), key=lambda s: -m.area_faces[list(s)].sum())
    total = m.area_faces[web].sum()
    print(f"\n{path.name}: 삼각형 {len(m.faces):,}  수밀 {m.is_watertight}  몸체 {m.body_count}  "
          f"체적 {abs(m.volume)/1e9:.4f} m³  덧댄 면적 {total/100:.0f} cm² ({total/m.area*100:.1f}%)")
    if args.tube_x:
        a, b = (float(v) for v in args.tube_x.split(","))
        print(f"  속 빈 관 단면(중첩 고리) x {a:.0f}~{b:.0f}: {hollow_tubes(m, a, b)}  (원본 {hollow_tubes(ref, a, b)})")
    print("  덧댄 자리 — 면적, 중심, 크기, 건너뛴 틈 ≈ 2×최대거리")
    shown = 0
    for comp in comps:
        idx = np.fromiter(comp, int)
        a = m.area_faces[idx].sum()
        if a < args.min_area or shown >= args.top:
            break
        cc = c[idx]
        ext = cc.max(0) - cc.min(0)
        print(f"    {a/100:6.0f} cm²  ({cc.mean(0)[0]:6.0f},{cc.mean(0)[1]:5.0f},{cc.mean(0)[2]:4.0f})  "
              f"{ext[0]:4.0f}×{ext[1]:4.0f}×{ext[2]:3.0f} mm  틈 ≈ {2*d[idx].max():4.1f} mm")
        shown += 1
