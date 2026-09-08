"""How rough is a wrapped/smoothed surface where it departs from the original?

Splits the candidate's faces into a "webbed" zone (face centroid farther than
--web-distance from the reference, i.e. material the wrap added across gaps, slots
and plate edges) and a "faithful" zone, and reports the dihedral-angle
distribution of each, the two-way sampled deviation, and volume. The webbed
zone is where a viewer sees jagged seams; the faithful zone must not move.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--reference", type=Path, required=True, help="original mesh")
ap.add_argument("--candidates", type=Path, nargs="+", required=True)
ap.add_argument("--web-distance", type=float, default=1.0)
ap.add_argument("--samples", type=int, default=8000)
ap.add_argument("--json", type=Path)
args = ap.parse_args()

ref = trimesh.load(args.reference, force="mesh")
ref.merge_vertices()
rows = []
print(f"{'candidate':30s} {'faces':>8s} {'web%':>6s} | webbed dihedral p50/p90/p99  >60° | faithful p50/p90 | dev c→r p90/max | r→c p90/max | vol m³")
for path in args.candidates:
    m = trimesh.load(path, force="mesh")
    m.merge_vertices()
    c = m.triangles_center
    _, d, _ = trimesh.proximity.closest_point(ref, c)
    web = d > args.web_distance
    adj = m.face_adjacency
    ang = np.degrees(m.face_adjacency_angles)
    pw = web[adj].any(axis=1)
    aw, af = ang[pw], ang[~pw]
    pts, _ = trimesh.sample.sample_surface(m, args.samples, seed=1)
    _, d1, _ = trimesh.proximity.closest_point(ref, pts)
    pts, _ = trimesh.sample.sample_surface(ref, args.samples, seed=1)
    _, d2, _ = trimesh.proximity.closest_point(m, pts)
    row = {
        "file": path.name, "faces": int(len(m.faces)), "watertight": bool(m.is_watertight),
        "bodies": int(m.body_count), "volume_m3": float(abs(m.volume)) / 1e9,
        "web_area_share": float(m.area_faces[web].sum() / m.area),
        "web_dihedral_p50": float(np.percentile(aw, 50)) if len(aw) else 0.0,
        "web_dihedral_p90": float(np.percentile(aw, 90)) if len(aw) else 0.0,
        "web_dihedral_p99": float(np.percentile(aw, 99)) if len(aw) else 0.0,
        "web_over_60": float(np.mean(aw > 60)) if len(aw) else 0.0,
        "faithful_dihedral_p50": float(np.percentile(af, 50)),
        "faithful_dihedral_p90": float(np.percentile(af, 90)),
        "cand_to_ref_p90": float(np.percentile(d1, 90)), "cand_to_ref_max": float(d1.max()),
        "ref_to_cand_p90": float(np.percentile(d2, 90)), "ref_to_cand_max": float(d2.max()),
    }
    rows.append(row)
    print(f"{row['file']:30s} {row['faces']:8,d} {row['web_area_share']*100:5.1f}% | "
          f"{row['web_dihedral_p50']:5.1f}/{row['web_dihedral_p90']:5.1f}/{row['web_dihedral_p99']:5.1f}  {row['web_over_60']*100:4.1f}% | "
          f"{row['faithful_dihedral_p50']:4.1f}/{row['faithful_dihedral_p90']:4.1f} | "
          f"{row['cand_to_ref_p90']:4.2f}/{row['cand_to_ref_max']:4.1f} | {row['ref_to_cand_p90']:4.2f}/{row['ref_to_cand_max']:4.1f} | {row['volume_m3']:.4f}"
          + ("" if row["watertight"] and row["bodies"] == 1 else f"   ! watertight={row['watertight']} bodies={row['bodies']}"))
if args.json:
    args.json.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
