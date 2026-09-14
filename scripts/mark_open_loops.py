"""Put the real outline of each opening into the question file, so the viewer can draw it.

    mark_open_loops.py --dir var/runs/plan-gtr6 --mesh var/runs/plan-gtr6/input_mm.stl [--top 12]

A sphere at the centroid says "around here". The boundary loop itself says exactly
which hole. This walks the mesh's open edges, keeps the largest loops, thins each
polyline to --max-points, and writes them into questions.json and plan.json under
where.lines, next to the points that were already there.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aox_g3 import fair

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dir", type=Path, required=True)
ap.add_argument("--mesh", type=Path, required=True)
ap.add_argument("--question", default="keep_openings_mm", help="which question the loops belong to")
ap.add_argument("--top", type=int, default=12)
ap.add_argument("--max-points", type=int, default=160)
ap.add_argument("--near", type=float, default=0.0, help="mm; keep only loops whose centre is within this of an existing marker (0 = take the largest)")
args = ap.parse_args()

mesh = trimesh.load(args.mesh, force="mesh")
mesh.merge_vertices()
loops, _ = fair.boundary_loops(mesh)
print(f"열린 고리 {len(loops)}개")

def thin(P, n):
    if len(P) <= n:
        return P
    idx = np.round(np.linspace(0, len(P) - 1, n)).astype(int)
    return P[idx]

ranked = sorted(loops, key=lambda l: -np.ptp(mesh.vertices[l], axis=0).max())

def entry_for(lp):
    P = np.asarray(mesh.vertices[lp], float)
    size = float(np.ptp(P, axis=0).max())
    return {"size_mm": round(size, 1), "centre": P.mean(0).round(1).tolist(),
            "line": thin(P, args.max_points).round(1).tolist()}

qpath = args.dir / "questions.json"
plan_path = args.dir / "plan.json"
questions = json.loads(qpath.read_text()) if qpath.exists() else []
plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
targets = [q for q in questions + plan.get("questions", []) if q["id"] == args.question]
if not targets:
    raise SystemExit(f"질문 {args.question} 이 없습니다")

existing = (targets[0].get("where") or {}).get("points") or []
chosen = []
if args.near > 0 and existing:
    centres = np.array([[p[0], p[1], p[2]] for p in existing])
    for lp in ranked:
        c = mesh.vertices[lp].mean(0)
        if np.linalg.norm(centres - c, axis=1).min() <= args.near:
            chosen.append(lp)
        if len(chosen) >= args.top:
            break
else:
    chosen = ranked[: args.top]

lines = [entry_for(lp) for lp in chosen]
if args.near > 0 and existing:
    # keep every marker the question already had; attach an outline only where one matched,
    # so running this never removes locations the pipeline found (2026-09-14)
    used = [False] * len(lines)
    points, outlines = [], []
    for pt in existing:
        c = np.array(pt[:3], float)
        best, bi = None, -1
        for i, e in enumerate(lines):
            if used[i]:
                continue
            dist = float(np.linalg.norm(np.array(e["centre"]) - c))
            if dist <= args.near and (best is None or dist < best):
                best, bi = dist, i
        points.append(pt)
        if bi >= 0:
            used[bi] = True
            outlines.append(lines[bi]["line"])
        else:
            outlines.append(None)
    print(f"  기존 표식 {len(existing)}개 중 테두리를 찾은 것 {sum(1 for o in outlines if o)}개")
else:
    points = [[*e["centre"], 0.0, f"열린 고리 {e['size_mm']:.0f} mm"] for e in lines]
    outlines = [e["line"] for e in lines]
for q in targets:
    w = q.get("where") or {}
    w["lines"] = outlines
    w["points"] = points
    q["where"] = w
qpath.write_text(json.dumps(questions, ensure_ascii=False, indent=1))
plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=1))
print(f"{args.question} 에 테두리 {len(lines)}개 기록: " + ", ".join(f"{e['size_mm']:.0f} mm({len(e['line'])}점)" for e in lines[:6]) + (" …" if len(lines) > 6 else ""))
