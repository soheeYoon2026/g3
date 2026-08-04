#!/usr/bin/env python3
"""Merge preprocessing manifests and copy their NPZ cases into one dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(f"output already exists: {args.out_dir}")

    cases_dir = args.out_dir / "cases"
    cases_dir.mkdir(parents=True)
    merged: list[dict] = []
    seen_ids: set[str] = set()
    sources: list[str] = []

    for manifest_path in args.manifest:
        manifest_path = manifest_path.resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(payload.get("geometry_preprocessing_version", 0)) != 2:
            raise ValueError(f"manifest is not preprocessing v2: {manifest_path}")
        sources.append(str(manifest_path))
        for row in payload.get("cases", []):
            case_id = str(row["case_id"])
            if case_id in seen_ids:
                raise ValueError(f"duplicate case_id: {case_id}")
            seen_ids.add(case_id)
            source_npz = Path(row["npz"])
            if not source_npz.is_absolute():
                source_npz = manifest_path.parent / source_npz
            if not source_npz.is_file():
                raise FileNotFoundError(source_npz)
            target_npz = cases_dir / f"{case_id}.npz"
            if target_npz.exists():
                raise FileExistsError(target_npz)
            shutil.copy2(source_npz, target_npz)
            merged_row = dict(row)
            merged_row["npz"] = f"cases/{target_npz.name}"
            merged.append(merged_row)

    output = {
        "format": "g3-preprocessed-merged-v2",
        "geometry_preprocessing_version": 2,
        "source_manifests": sources,
        "cases": merged,
        "skipped": [],
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cases": len(merged), "output": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
