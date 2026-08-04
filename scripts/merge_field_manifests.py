#!/usr/bin/env python3
"""Merge field manifests while preserving resolvable NPZ paths and provenance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    output_parent = args.out.resolve().parent
    rows, skipped = [], []
    case_ids: dict[str, Path] = {}
    sources: set[str] = set()
    for manifest in args.manifest:
        manifest = manifest.resolve()
        payload = json.loads(manifest.read_text())
        for item in payload.get("skipped", []):
            skipped.append({"manifest": str(manifest), **item})
        for original in payload["cases"]:
            row = dict(original)
            npz = Path(row["npz"])
            if not npz.is_absolute():
                npz = (manifest.parent / npz).resolve()
            if not npz.exists():
                raise FileNotFoundError(f"missing NPZ for {row['case_id']}: {npz}")
            source = str(Path(row.get("source", {}).get("case_dir", npz)).resolve())
            if source in sources:
                continue
            case_id = str(row["case_id"])
            if case_id in case_ids:
                raise ValueError(
                    f"duplicate case_id {case_id}: {case_ids[case_id]} and {npz}"
                )
            row["npz"] = os.path.relpath(npz, output_parent)
            row["source_manifest"] = str(manifest)
            rows.append(row)
            sources.add(source)
            case_ids[case_id] = npz

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema_version": 1,
        "source_manifests": [str(path.resolve()) for path in args.manifest],
        "cases": rows,
        "skipped": skipped,
    }, indent=2))
    print(f"Wrote {len(rows)} unique cases ({len(skipped)} skipped records) -> {args.out}")


if __name__ == "__main__":
    main()
