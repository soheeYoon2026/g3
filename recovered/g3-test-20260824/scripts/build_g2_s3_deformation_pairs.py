#!/usr/bin/env python3
"""Recover G2 original-to-RBF deformation pairs directly from S3 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pyvista as pv

from build_domino_s3_v4 import parse_rbf_objectives
from training_event_inventory import load_event_rows


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def list_keys(client, bucket: str, prefix: str) -> set[str]:
    keys: set[str] = set()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        keys.update(item["Key"] for item in page.get("Contents", []))
    return keys


def load_surface(client, bucket: str, key: str) -> pv.PolyData:
    with tempfile.NamedTemporaryFile(suffix=".vtu") as handle:
        client.download_file(bucket, key, handle.name)
        return pv.read(handle.name).extract_surface().triangulate()


def recover_job(client, row: dict, out: Path) -> tuple[list[dict], str | None]:
    bucket = str(row["s3_bucket"])
    prefix = str(row["s3_output_prefix"]).rstrip("/")
    job_uid = str(row["job_uid"])
    keys = list_keys(client, bucket, prefix)
    history_key = prefix + "/rbf_optimization_history.csv"
    if history_key not in keys:
        return [], "missing rbf_optimization_history.csv"
    history = client.get_object(Bucket=bucket, Key=history_key)["Body"].read()
    objectives = parse_rbf_objectives(history.decode("utf-8-sig", "replace"))
    designs = sorted(
        design for design in objectives
        if prefix + f"/{design}/surface_flow.vtu" in keys
    )
    if len(designs) < 2:
        return [], f"fewer than two usable RBF surfaces ({len(designs)})"

    base_design = designs[0]
    base_mesh = load_surface(client, bucket, prefix + f"/{base_design}/surface_flow.vtu")
    base_points = np.asarray(base_mesh.points, dtype=np.float32)
    diagonal = max(float(np.linalg.norm(np.ptp(base_points, axis=0))), 1e-12)
    base_cd = float(objectives[base_design])
    job_out = out / job_uid
    job_out.mkdir(parents=True, exist_ok=True)
    pairs = []
    for design in designs[1:]:
        target_mesh = load_surface(client, bucket, prefix + f"/{design}/surface_flow.vtu")
        if target_mesh.n_points != base_mesh.n_points:
            continue
        target_points = np.asarray(target_mesh.points, dtype=np.float32)
        displacement = target_points - base_points
        magnitude = np.linalg.norm(displacement, axis=1)
        if not np.isfinite(displacement).all() or float(magnitude.max(initial=0)) > diagonal * 0.25:
            continue
        moved = magnitude > diagonal * 1e-7
        if not moved.any():
            continue
        target_cd = float(objectives[design])
        if not math.isfinite(target_cd):
            continue
        file = job_out / f"{design}.npz"
        np.savez_compressed(
            file,
            base_points=base_points,
            target_points=target_points,
            displacement=displacement,
            moved_mask=moved,
            base_cd=np.float32(base_cd),
            target_cd=np.float32(target_cd),
            delta_cd=np.float32(target_cd - base_cd),
        )
        pairs.append({
            "job_uid": job_uid,
            "base_design": base_design,
            "target_design": design,
            "file": str(file.relative_to(out)),
            "points": int(base_mesh.n_points),
            "moved_points": int(moved.sum()),
            "max_displacement": float(magnitude.max()),
            "base_cd": base_cd,
            "target_cd": target_cd,
            "delta_cd": target_cd - base_cd,
            "bucket": bucket,
            "output_prefix": prefix,
            "conditions": row.get("conditions") or {},
        })
    return pairs, None if pairs else "no deformation passed geometry checks"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-inventory", action="append", type=Path)
    parser.add_argument("--csv", action="append", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.event_inventory and not args.csv:
        parser.error("provide --event-inventory and/or --csv")
    rows = load_event_rows(args.event_inventory or [])
    for path in args.csv or []:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows.extend(csv.DictReader(handle))
    jobs = {}
    for row in rows:
        if str(row.get("solver", "")).upper() == "G2" and row.get("job_uid"):
            jobs[str(row["job_uid"])] = row
    args.out.mkdir(parents=True, exist_ok=True)
    clients, pairs, rejected = {}, [], []
    for index, (job_uid, row) in enumerate(sorted(jobs.items()), 1):
        bucket = str(row["s3_bucket"])
        client = clients.setdefault(bucket, boto3.client("s3"))
        try:
            recovered, reason = recover_job(client, row, args.out)
            pairs.extend(recovered)
            if reason:
                rejected.append({"job_uid": job_uid, "reason": reason})
        except Exception as exc:
            rejected.append({"job_uid": job_uid, "reason": f"{type(exc).__name__}: {exc}"})
        print(f"[{index}/{len(jobs)}] {job_uid}: +{len(pairs)} total", flush=True)
    payload = {
        "schema_version": 1,
        "format": "g2-s3-deformation-pairs-v1",
        "summary": {"g2_jobs": len(jobs), "paired_jobs": len({p['job_uid'] for p in pairs}), "pairs": len(pairs), "unpaired_jobs": len(rejected)},
        "pairs": pairs,
        "unpaired": rejected,
    }
    atomic_json(args.out / "manifest.json", payload)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
