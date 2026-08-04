#!/usr/bin/env python3
"""Merge manifests, deduplicate case IDs, and exclude frozen holdout cases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return payload["cases"] if isinstance(payload, dict) else payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    excluded = {
        str(row["case_id"])
        for path in args.exclude_manifest
        for row in rows(path)
    }
    output_parent = args.out.resolve().parent
    merged, seen = [], set()
    for source in args.manifest:
        source = source.resolve()
        for original in rows(source):
            case_id = str(original["case_id"])
            if case_id in excluded or case_id in seen:
                continue
            row = dict(original)
            npz = Path(row["npz"])
            if not npz.is_absolute():
                npz = (source.parent / npz).resolve()
            if not npz.is_file():
                raise FileNotFoundError(f"missing NPZ for {case_id}: {npz}")
            row["npz"] = os.path.relpath(npz, output_parent)
            merged.append(row)
            seen.add(case_id)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema_version": 1,
        "format": "g3-filtered-training-manifest-v1",
        "sources": [str(path.resolve()) for path in args.manifest],
        "excluded_sources": [str(path.resolve()) for path in args.exclude_manifest],
        "excluded_case_ids": sorted(excluded),
        "cases": merged,
    }, indent=2) + "\n")
    print(json.dumps({
        "cases": len(merged),
        "excluded": len(excluded),
        "output": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()

