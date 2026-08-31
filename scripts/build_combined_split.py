"""Build the split for combined absolute-Cd + pair training.

The pair runs are added to training only. Validation and test keep the
incumbent's own cases, otherwise the comparison against it stops being an A/B.
"""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--base-split", type=Path, required=True)
ap.add_argument("--pairs", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

base = json.loads(args.base_split.read_text())
pairs = json.loads(args.pairs.read_text())["pairs"]
pair_runs = sorted({r for p in pairs for r in (p["baseline"], p["variant"])})

held = {r["run"] for r in base.get("validation_cases", []) + base["test_cases"]}
overlap = held & set(pair_runs)
if overlap:
    raise SystemExit(f"쌍 런이 검증/테스트와 겹친다: {sorted(overlap)[:5]}")

combined = {
    "train_cases": base["train_cases"] + [{"run": r, "group_id": None} for r in pair_runs],
    "validation_cases": base.get("validation_cases", []),
    "test_cases": base["test_cases"],
}
args.out.write_text(json.dumps(combined, indent=1) + "\n")
print(f"train {len(combined['train_cases'])} "
      f"(절대 {len(base['train_cases'])} + 쌍 {len(pair_runs)}) / "
      f"val {len(combined['validation_cases'])} / test {len(combined['test_cases'])}")
print("wrote", args.out)
