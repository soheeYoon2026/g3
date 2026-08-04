#!/usr/bin/env python3
"""Read-only audit of S3 bucket access and representative key layouts."""

from __future__ import annotations

import argparse
import json

import boto3
from botocore.config import Config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bucket", nargs="+")
    parser.add_argument("--max-keys", type=int, default=25)
    args = parser.parse_args()

    base = boto3.client("s3", config=Config(retries={"max_attempts": 3}))
    results = []
    for bucket in args.bucket:
        try:
            location = base.get_bucket_location(Bucket=bucket).get("LocationConstraint")
            region = location or "us-east-1"
            client = boto3.client("s3", region_name=region)
            page = client.list_objects_v2(Bucket=bucket, MaxKeys=args.max_keys)
            results.append({
                "bucket": bucket,
                "region": region,
                "accessible": True,
                "sample_count": page.get("KeyCount", 0),
                "is_truncated": page.get("IsTruncated", False),
                "sample_keys": [item["Key"] for item in page.get("Contents", [])],
            })
        except Exception as exc:
            results.append({
                "bucket": bucket,
                "accessible": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
