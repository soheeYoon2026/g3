#!/usr/bin/env python
"""Build and physically validate a complete DoMINO v3 dataset from an audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_domino_su2_v3 import convert_case
from validate_domino_su2_v3 import integrate_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-cd-relative-error", type=float, default=0.03)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.out}")

    audit = json.loads(args.audit.read_text())
    sources = [row for row in audit["cases"] if row["ready"]]
    manifest = []
    for index, source in enumerate(sources, 1):
        conditions = convert_case(source["vtu"], source["cfg"], args.out, str(index))
        run_dir = args.out / f"run_{index}"
        integrated = integrate_case(run_dir)
        cd_scale = max(abs(integrated["su2_cd"]), 1e-8)
        cd_relative_error = abs(integrated["cd"] - integrated["su2_cd"]) / cd_scale
        row = {
            "run": index,
            "legacy_run": source["legacy_run"],
            "group_id": f"mesh_{source['geometry_digest'][:16]}",
            "geometry_digest": source["geometry_digest"],
            "conditions": conditions,
            "integrated": integrated,
            "cd_relative_error": cd_relative_error,
            "accepted": cd_relative_error <= args.max_cd_relative_error,
        }
        manifest.append(row)
        print(
            f"[{index:02d}/{len(sources)}] AOA={conditions['aoa']:7.2f} "
            f"Cd={integrated['su2_cd']:+.5f} integrated={integrated['cd']:+.5f} "
            f"error={cd_relative_error * 100:.2f}%"
        )

    accepted = sum(row["accepted"] for row in manifest)
    output = {
        "schema_version": 1,
        "format": "domino-su2-flow-aligned-v3",
        "source_audit": str(args.audit.resolve()),
        "cases": manifest,
        "summary": {
            "generated": len(manifest),
            "accepted": accepted,
            "rejected": len(manifest) - accepted,
            "unique_groups": len({row["group_id"] for row in manifest if row["accepted"]}),
            "max_cd_relative_error": args.max_cd_relative_error,
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["summary"], indent=2))
    raise SystemExit(0 if accepted == len(manifest) else 2)


if __name__ == "__main__":
    main()
