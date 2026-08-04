#!/usr/bin/env python3
"""Split quality-gated G2 coefficient rows into normal/high-drag experts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normal-out", type=Path, required=True)
    parser.add_argument("--high-out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    normal, high = [], []
    for source in payload["cases"]:
        row = copy.deepcopy(source)
        cd = row.get("coefficients", {}).get("cd")
        if cd is None:
            continue
        row["npz"] = str(
            (args.manifest.parent / row["npz"]).resolve()
            if not Path(row["npz"]).is_absolute() else Path(row["npz"])
        )
        if float(cd) > args.threshold:
            row["coefficient_expert"] = "g2_su2_high_drag"
            high.append(row)
        else:
            row["coefficient_expert"] = "g2_su2_clean"
            normal.append(row)

    common = {
        "format": "g2-coefficient-domain-v1", "source": str(args.manifest),
        "threshold": args.threshold,
    }
    for path, name, rows in (
        (args.normal_out, "g2_su2_clean", normal),
        (args.high_out, "g2_su2_high_drag", high),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**common, "coefficient_expert": name, "cases": rows}, indent=2))
    print(json.dumps({"normal": len(normal), "high_drag": len(high)}, indent=2))


if __name__ == "__main__":
    main()
