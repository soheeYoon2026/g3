#!/usr/bin/env python3
"""Merge field manifests while quality-gating Cd/Cl supervision.

Every readable case remains available to the shared Cp/velocity backbone, but
cases below the surface-resolution gate have global coefficient labels removed
so mesh artifacts cannot teach the Cd/Cl expert.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-surface-points", type=int, default=5_000)
    parser.add_argument("--expert", default="g2_su2_clean")
    args = parser.parse_args()

    cases, seen, sources = [], set(), []
    supervised = field_only = duplicates = 0
    for manifest_path in args.manifest:
        payload = json.loads(manifest_path.read_text())
        rows = payload["cases"] if isinstance(payload, dict) else payload
        sources.append(str(manifest_path))
        for original in rows:
            row = copy.deepcopy(original)
            identity = (str(row.get("group_id", "")), str(row["case_id"]))
            if identity in seen:
                duplicates += 1
                continue
            seen.add(identity)
            npz = Path(row["npz"])
            if not npz.is_absolute():
                row["npz"] = str((manifest_path.parent / npz).resolve())
            count = int(row.get("counts", {}).get("surface", 0))
            coefficients = row.setdefault("coefficients", {})
            valid = count >= args.min_surface_points and coefficients.get("cd") is not None
            if valid:
                row["coefficient_expert"] = args.expert
                row["coefficient_quality"] = "accepted"
                supervised += 1
            else:
                row["raw_coefficients_rejected"] = copy.deepcopy(coefficients)
                row["coefficients"] = {"cd": None, "cl": None}
                row["coefficient_quality"] = (
                    f"field_only_surface_points_{count}_below_{args.min_surface_points}"
                )
                field_only += 1
            cases.append(row)

    output = {
        "schema_version": 2,
        "format": "g2-quality-gated-expert-v1",
        "sources": sources,
        "quality_gate": {
            "rule": "surface_points >= threshold",
            "min_surface_points": args.min_surface_points,
            "expert": args.expert,
        },
        "summary": {
            "cases": len(cases), "coefficient_supervised": supervised,
            "field_only": field_only, "duplicates_removed": duplicates,
        },
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
