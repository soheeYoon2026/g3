"""What a wrap closed: patches of material the wrap added across gaps.

A wrap face whose centroid lies farther than `web_distance` from the reference
is material the wrap added (the reference is what was wrapped). Connected
patches of such faces are the closed openings; the gap each one bridged is
about twice the largest centroid distance inside the patch (the wrap spans the
gap at its mouth, so its middle sits half a gap from either side).
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
import trimesh


@dataclass
class ClosedPatch:
    faces: np.ndarray
    area: float          # mm²
    center: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    gap: float           # mm, ≈ 2 × max centroid distance

    @property
    def extent(self):
        return self.bbox_max - self.bbox_min

    def as_dict(self):
        return {"area_mm2": round(float(self.area), 1), "center": self.center.round(1).tolist(),
                "bbox_min": self.bbox_min.round(1).tolist(), "bbox_max": self.bbox_max.round(1).tolist(),
                "gap_mm": round(float(self.gap), 1), "faces": int(len(self.faces))}


def closed_patches(reference, candidate, web_distance=1.5, min_area=2000.0):
    """List the patches of added material on `candidate`, largest first."""
    c = candidate.triangles_center
    _, d, _ = trimesh.proximity.closest_point(reference, c)
    web = d > web_distance
    adj = candidate.face_adjacency
    g = nx.Graph()
    g.add_nodes_from(np.flatnonzero(web))
    g.add_edges_from(adj[web[adj].all(axis=1)])
    out = []
    for comp in nx.connected_components(g):
        idx = np.fromiter(comp, int)
        area = float(candidate.area_faces[idx].sum())
        if area < min_area:
            continue
        cc = c[idx]
        out.append(ClosedPatch(idx, area, cc.mean(0), cc.min(0), cc.max(0), 2.0 * float(d[idx].max())))
    out.sort(key=lambda p: -p.area)
    return out, web
