#!/usr/bin/env python3
"""Build paired G2 deformation samples from accepted DoMINO surface cases."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pyvista as pv


def case_by_source(manifest: dict, job_uid: str) -> list[dict]:
    prefix = f"{job_uid}:RBF_DSN_"
    rows = [
        row for row in manifest.get("cases", [])
        if row.get("accepted") and str(row.get("source", {}).get("source_id", "")).startswith(prefix)
    ]
    return sorted(rows, key=lambda row: row["source"]["source_id"])


def available_jobs(manifest: dict) -> list[str]:
    counts: dict[str, int] = {}
    pattern = re.compile(r"^([^:]+):RBF_DSN_\d+$")
    for row in manifest.get("cases", []):
        if not row.get("accepted"):
            continue
        match = pattern.match(str(row.get("source", {}).get("source_id", "")))
        if match:
            counts[match.group(1)] = counts.get(match.group(1), 0) + 1
    return sorted(job for job, count in counts.items() if count >= 2)


def build_pair(root: Path, baseline: dict, target: dict, output: Path) -> dict:
    base_run, target_run = int(baseline["run"]), int(target["run"])
    base_mesh = pv.read(root / f"run_{base_run}" / f"boundary_{base_run}.vtp").triangulate()
    target_mesh = pv.read(root / f"run_{target_run}" / f"boundary_{target_run}.vtp").triangulate()
    if base_mesh.n_points != target_mesh.n_points:
        raise ValueError(
            f"topology mismatch: {base_mesh.n_points} != {target_mesh.n_points}"
        )
    base_points = np.asarray(base_mesh.points, dtype=np.float32)
    target_points = np.asarray(target_mesh.points, dtype=np.float32)
    displacement = target_points - base_points
    magnitude = np.linalg.norm(displacement, axis=1)
    scale = max(float(np.linalg.norm(np.ptp(base_points, axis=0))), 1e-12)
    moved = magnitude > scale * 1e-7
    base_cd = float(baseline["integrated"]["su2_cd"])
    target_cd = float(target["integrated"]["su2_cd"])
    base_cl = float(baseline["integrated"]["su2_cl"])
    target_cl = float(target["integrated"]["su2_cl"])

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        base_points=base_points,
        target_points=target_points,
        displacement=displacement,
        moved_mask=moved,
        faces=np.asarray(base_mesh.faces, dtype=np.int64),
        base_cd=np.float32(base_cd),
        target_cd=np.float32(target_cd),
        delta_cd=np.float32(target_cd - base_cd),
        base_cl=np.float32(base_cl),
        target_cl=np.float32(target_cl),
        delta_cl=np.float32(target_cl - base_cl),
    )
    return {
        "baseline_run": base_run,
        "target_run": target_run,
        "file": output.name,
        "points": int(base_mesh.n_points),
        "moved_points": int(moved.sum()),
        "max_displacement": float(magnitude.max(initial=0)),
        "mean_moved_displacement": float(magnitude[moved].mean()) if moved.any() else 0.0,
        "point_correspondence": "stable_vtk_point_index",
        "base_cd": base_cd,
        "target_cd": target_cd,
        "delta_cd": target_cd - base_cd,
        "base_cl": base_cl,
        "target_cl": target_cl,
        "delta_cl": target_cl - base_cl,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job-uid")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = available_jobs(manifest) if args.all else [args.job_uid]
    all_pairs = []
    job_summaries = []
    for job_uid in jobs:
        rows = case_by_source(manifest, job_uid)
        if len(rows) < 2:
            continue
        job_out = args.out / job_uid if args.all else args.out
        job_out.mkdir(parents=True, exist_ok=True)
        baseline = rows[0]
        pairs = [
            build_pair(args.dataset, baseline, row, job_out / f"{job_uid}_{row['run']}.npz")
            for row in rows[1:]
        ]
        for pair in pairs:
            pair["job_uid"] = job_uid
            pair["file"] = str((job_out / pair["file"]).relative_to(args.out))
        all_pairs.extend(pairs)
        job_summaries.append({"job_uid": job_uid, "baseline_run": baseline["run"], "pairs": len(pairs)})
    payload = {
        "schema_version": 1,
        "format": "g2-deformation-pairs-v1",
        "jobs": job_summaries,
        "summary": {"jobs": len(job_summaries), "pairs": len(all_pairs)},
        "pairs": all_pairs,
    }
    (args.out / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
