#!/usr/bin/env python3
"""Materialize clean G2 field cases from a solver job inventory.

Only successful G2 runs are considered.  Root and ``RBF_DSN_*`` flow cases
are converted, low-resolution surfaces are rejected, and duplicate geometries
from repeated DRAG/LIFT runs are removed before writing the manifest.

Both the historical smoke-report schema (``job_status`` /
``s3_output_prefix`` / ``test_case``) and the all-buckets inventory schema
(``status`` / ``output_s3_key`` / ``project_uid``) are accepted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

import boto3
from prepare_g2_fields import prepare_case


REQUIRED_NAMES = {"flow.vtu", "surface_flow.vtu", "history.csv"}


def normalize_inventory_row(row: dict[str, str]) -> dict[str, str]:
    """Map supported CSV schemas to the fields used by the materializer."""
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


def list_keys(client, bucket: str, prefix: str) -> list[str]:
    keys = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        keys.extend(item["Key"] for item in page.get("Contents", []))
    return keys


def case_files(prefix: str, keys: list[str]) -> dict[str, dict[str, str]]:
    base = prefix.rstrip("/") + "/"
    cases: dict[str, dict[str, str]] = {}
    for key in keys:
        relative = key[len(base):] if key.startswith(base) else key
        parts = relative.split("/")
        if len(parts) == 1:
            case_name, name = "FINAL", parts[0]
        elif len(parts) == 2 and parts[0].startswith("RBF_DSN_"):
            case_name, name = parts
        else:
            continue
        if name in REQUIRED_NAMES or name.endswith("_CFD.cfg"):
            cases.setdefault(case_name, {})[name] = key
    return {
        name: files for name, files in cases.items()
        if "flow.vtu" in files and "surface_flow.vtu" in files
        and any(key.endswith("_CFD.cfg") for key in files)
    }


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--n-volume", type=int, default=200_000)
    parser.add_argument("--n-surface", type=int, default=20_000)
    parser.add_argument("--min-surface-points", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = [
        normalize_inventory_row(row)
        for row in csv.DictReader(args.csv.open(newline="", encoding="utf-8-sig"))
    ]
    rows = [
        row for row in rows
        if row.get("solver") == "G2"
        and row["job_status"] == "succeeded"
        and row.get("s3_bucket")
        and row["s3_output_prefix"]
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    clients, cases, skipped, digests = {}, [], [], {}
    number = 0

    for row in rows:
        bucket, prefix = row["s3_bucket"], row["s3_output_prefix"]
        client = clients.setdefault(bucket, boto3.client("s3"))
        available = case_files(prefix, list_keys(client, bucket, prefix))
        for design, files in sorted(available.items()):
            number += 1
            case_id = (
                f"{row['source_kind']}_{row['group_key']}_{row['job_uid']}_{design}"
            )
            target = args.out_dir / "cases" / f"{case_id}.npz"
            try:
                with tempfile.TemporaryDirectory(prefix=f"{case_id}_") as temp_name:
                    temp = Path(temp_name)
                    for name, key in files.items():
                        client.download_file(bucket, key, str(temp / name))
                    digest = file_digest(temp / "surface_flow.vtu")
                    if digest in digests:
                        skipped.append({
                            "case_id": case_id, "reason": "duplicate_geometry_and_fields",
                            "duplicate_of": digests[digest],
                        })
                        print(f"SKIP duplicate {case_id} -> {digests[digest]}")
                        continue
                    prepared = prepare_case(
                        temp, target, args.n_volume, args.n_surface, args.seed + number
                    )
                surface_count = int(prepared["counts"]["surface"])
                if surface_count < args.min_surface_points:
                    target.unlink(missing_ok=True)
                    raise ValueError(
                        f"surface points {surface_count} < {args.min_surface_points} quality gate"
                    )
                digests[digest] = case_id
                prepared.update({
                    "case_id": case_id,
                    "group_id": f"{row['source_kind']}_{row['group_key']}",
                    "npz": str(target.resolve().relative_to(args.manifest.parent.resolve()))
                    if target.resolve().is_relative_to(args.manifest.parent.resolve())
                    else str(target.resolve()),
                    "coefficient_expert": "g2_su2_clean",
                    "coefficient_quality": "surface_points_gte_5000",
                    "smoke_source": {
                        "test_case": row.get("test_case"),
                        "project_uid": row.get("project_uid"),
                        "job_id": row.get("job_id"),
                        "job_uid": row["job_uid"], "design": design,
                        "bucket": bucket, "output_prefix": prefix,
                    },
                })
                cases.append(prepared)
                coeff = prepared["coefficients"]
                print(
                    f"OK {case_id}: surface={surface_count} "
                    f"Cd={coeff.get('cd'):.6f} Cl={coeff.get('cl'):.6f}"
                )
            except Exception as exc:
                target.unlink(missing_ok=True)
                skipped.append({"case_id": case_id, "reason": f"{type(exc).__name__}: {exc}"})
                print(f"SKIP {case_id}: {type(exc).__name__}: {exc}")

    payload = {
        "schema_version": 2,
        "format": "g2-smoke-fields-v1",
        "source_csv": str(args.csv),
        "quality_gate": {"min_surface_points": args.min_surface_points},
        "cases": cases,
        "skipped": skipped,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"cases": len(cases), "skipped": len(skipped)}, indent=2))


if __name__ == "__main__":
    main()
