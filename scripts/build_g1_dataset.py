#!/usr/bin/env python3
"""Inspect and materialize G1 job outputs for G3 training.

The input CSV is the AOX export containing one S3 output prefix per G1 job.
This first-stage utility intentionally keeps inventory and download separate so
large OpenFOAM fields are never copied blindly.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import re
from collections import Counter
from pathlib import Path

import boto3


REGION_RE = re.compile(r"-g1-([a-z0-9-]+)-node-")


def region_from_bucket(bucket: str) -> str:
    match = REGION_RE.search(bucket)
    if not match:
        raise ValueError(f"Cannot infer AWS region from bucket: {bucket}")
    return match.group(1)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"tenant_uid", "team_uid", "project_uid", "job_uid", "s3_bucket", "output_s3_key"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    return rows


def inventory(
    rows: list[dict[str, str]],
    limit: int | None = None,
    embed_json: bool = False,
    job_root: bool = False,
) -> dict:
    clients: dict[str, object] = {}
    jobs = []
    suffixes: Counter[str] = Counter()
    total_bytes = 0

    for row in rows[:limit]:
        bucket = row["s3_bucket"]
        prefix = row["output_s3_key"]
        if job_root and prefix.endswith("output/"):
            prefix = prefix[: -len("output/")]
        region = region_from_bucket(bucket)
        client = clients.setdefault(region, boto3.client("s3", region_name=region))
        objects = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                relative = item["Key"][len(prefix):]
                size = int(item["Size"])
                suffix = Path(relative).suffix.lower() or "<none>"
                suffixes[suffix] += 1
                total_bytes += size
                record = {"key": item["Key"], "relative": relative, "size": size}
                if embed_json and suffix == ".json" and size <= 1_000_000:
                    body = client.get_object(Bucket=bucket, Key=item["Key"])["Body"].read()
                    try:
                        record["json"] = json.loads(body)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        record["text"] = body.decode(errors="replace")
                objects.append(record)
        jobs.append({
            "tenant_uid": row["tenant_uid"],
            "team_uid": row["team_uid"],
            "project_uid": row["project_uid"],
            "job_uid": row["job_uid"],
            "bucket": bucket,
            "region": region,
            "prefix": prefix,
            "objects": objects,
        })

    return {
        "job_count": len(jobs),
        "object_count": sum(len(job["objects"]) for job in jobs),
        "total_bytes": total_bytes,
        "suffix_counts": dict(suffixes.most_common()),
        "jobs": jobs,
    }


def download_matches(
    rows: list[dict[str, str]],
    destination: Path,
    patterns: list[str],
    limit: int | None = None,
    job_root: bool = False,
) -> dict:
    clients: dict[str, object] = {}
    downloaded = 0
    downloaded_bytes = 0
    for row in rows[:limit]:
        bucket = row["s3_bucket"]
        prefix = row["output_s3_key"]
        if job_root and prefix.endswith("output/"):
            prefix = prefix[: -len("output/")]
        region = region_from_bucket(bucket)
        client = clients.setdefault(region, boto3.client("s3", region_name=region))
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                relative = item["Key"][len(prefix):]
                if not any(fnmatch.fnmatch(relative, pattern) for pattern in patterns):
                    continue
                target = destination / row["job_uid"] / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or target.stat().st_size != int(item["Size"]):
                    client.download_file(bucket, item["Key"], str(target))
                downloaded += 1
                downloaded_bytes += int(item["Size"])
    return {"downloaded_files": downloaded, "downloaded_bytes": downloaded_bytes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--embed-json", action="store_true")
    parser.add_argument("--job-root", action="store_true",
                        help="Inventory the whole job prefix instead of output/ only")
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--download-pattern", action="append", default=[])
    args = parser.parse_args()

    rows = load_rows(args.csv)
    if args.inventory:
        result = inventory(rows, args.limit, args.embed_json, args.job_root)
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        args.inventory.write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result[k] for k in ("job_count", "object_count", "total_bytes", "suffix_counts")}, indent=2))
    if args.download_dir:
        if not args.download_pattern:
            parser.error("--download-dir requires at least one --download-pattern")
        print(json.dumps(download_matches(
            rows, args.download_dir, args.download_pattern, args.limit, args.job_root
        ), indent=2))
    if not args.inventory and not args.download_dir:
        parser.error("provide --inventory and/or --download-dir")


if __name__ == "__main__":
    main()
