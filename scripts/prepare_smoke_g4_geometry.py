#!/usr/bin/env python3
"""Build G4 geometry/Cd expert samples from successful job outputs.

Both the historical smoke-report schema (``job_status`` /
``s3_output_prefix`` / ``test_case``) and the all-buckets inventory schema
(``status`` / ``output_s3_key`` / ``project_uid``) are accepted.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import boto3
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aox_g3.geometry.stl_sampler import load_mesh, sample_surface
from aox_g3.geometry.surface_sampling import GEOMETRY_PREPROCESSING_VERSION


def list_objects(client, bucket, prefix):
    objects = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        objects.extend(item for item in page.get("Contents", []))
    return objects


def normalize_inventory_row(row):
    normalized = dict(row)
    normalized["job_status"] = row.get("job_status") or row.get("status", "")
    normalized["s3_output_prefix"] = (
        row.get("s3_output_prefix") or row.get("output_s3_key", "")
    )
    normalized["group_key"] = (
        row.get("test_case") or row.get("project_uid") or row.get("job_uid", "")
    )
    normalized["source_kind"] = "smoke" if row.get("test_case") else "inventory"
    return normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=100_000)
    args = parser.parse_args()

    rows = [
        normalize_inventory_row(row)
        for row in csv.DictReader(args.csv.open(newline="", encoding="utf-8-sig"))
    ]
    rows = [
        row for row in rows
        if row.get("solver") == "G4"
        and row["job_status"] == "succeeded"
        and row.get("s3_bucket")
        and row["s3_output_prefix"]
    ]
    deduplicated = {row["group_key"]: row for row in rows}
    cases, skipped, clients = [], [], {}
    cases_dir = args.out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    for number, row in enumerate(deduplicated.values(), 1):
        case_id = f"{row['source_kind']}_{row['group_key']}_{row['job_uid']}"
        try:
            bucket, prefix = row["s3_bucket"], row["s3_output_prefix"]
            client = clients.setdefault(bucket, boto3.client("s3"))
            objects = list_objects(client, bucket, prefix)
            stl = next(item for item in objects if item["Key"].lower().endswith("_optimized.stl"))
            result = next(item for item in objects if item["Key"].lower().endswith("_results.json"))
            result_data = json.loads(client.get_object(Bucket=bucket, Key=result["Key"])["Body"].read())
            cd = float(result_data["cd_best"])
            with tempfile.TemporaryDirectory(prefix=f"g4_{case_id}_") as temp_name:
                stl_path = Path(temp_name) / "optimized.stl"
                client.download_file(bucket, stl["Key"], str(stl_path))
                mesh = load_mesh(stl_path)
                points, normals = sample_surface(mesh, args.max_points, seed=number)
                lo, hi = np.asarray(mesh.bounds, dtype=np.float64)
            center, scale = (lo + hi) / 2.0, float(np.max(hi - lo))
            points = ((points - center) / max(scale, 1e-12)).astype(np.float32)
            target = cases_dir / f"{case_id}.npz"
            np.savez_compressed(
                target,
                geometry_points=points,
                geometry_normals=normals.astype(np.float32),
                cd_final=np.float32(cd),
                center=center.astype(np.float32),
                scale=np.float32(scale),
                geometry_preprocessing_version=np.asarray(
                    GEOMETRY_PREPROCESSING_VERSION
                ),
            )
            try:
                cfg = json.loads(row.get("run_config") or "{}")
            except json.JSONDecodeError:
                cfg = {}
            velocity = float(cfg.get("U", 30.0))
            cases.append({
                "case_id": case_id,
                "group_id": f"{row['source_kind']}_{row['group_key']}",
                "npz": str(target.relative_to(args.out_dir)),
                "cd_final": cd,
                "coefficient_expert": "g4_lbm",
                "coefficient_quality": "successful_g4_results_json",
                "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
                "geometry_sampling": "area_weighted_triangle_face_normal",
                "surface_points": len(points),
                "conditions": {
                    "u_x": velocity, "u_y": 0.0, "u_z": 0.0,
                    "density": 1.225, "viscosity": 1.7894e-5,
                    "temperature": 288.15, "ref_length": scale, "ref_area": 1.0,
                },
                "smoke_source": {
                    "test_case": row.get("test_case"),
                    "project_uid": row.get("project_uid"),
                    "job_id": row.get("job_id"),
                    "job_uid": row["job_uid"], "bucket": bucket,
                    "stl_key": stl["Key"], "result_key": result["Key"],
                },
            })
            print(f"OK {case_id}: points={len(points)} Cd={cd:.6f}")
        except Exception as exc:
            skipped.append({"case_id": case_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"SKIP {case_id}: {type(exc).__name__}: {exc}")

    payload = {
        "format": "geometry-coefficient-v1", "coefficient_expert": "g4_lbm",
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "source_csv": str(args.csv), "cases": cases, "skipped": skipped,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"cases": len(cases), "skipped": len(skipped)}, indent=2))


if __name__ == "__main__":
    main()
