"""Report the three standing gates side by side for a promotion decision."""

import argparse
import json
from pathlib import Path


def summary(path: Path):
    if not path.exists():
        return None
    for line in reversed(path.read_text().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if row.get("model") == "fine_tuned" and "cases" in row:
            return row
    return None


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--prefix", default="/tmp/promo")
args = ap.parse_args()

header = (f"{'모델':>12s} {'gate15 MAE/순위':>18s} {'Windsor':>9s} "
          f"{'gtr Cd':>8s} {'ΔCd 방향':>9s} {'ΔCd MAE':>9s}")
print(header)
print("-" * len(header))
for tag in args.tags:
    gate = summary(Path(f"{args.prefix}-gate15-{tag}.jsonl"))
    windsor = summary(Path(f"{args.prefix}-windsor-{tag}.jsonl"))
    gtr = summary(Path(f"{args.prefix}-gtr-{tag}.jsonl"))
    if not (gate and windsor and gtr):
        print(f"{tag:>12s}  결과 없음")
        continue
    delta = gtr.get("delta_cd") or {}
    pairs = delta.get("direction_pairs", 0)
    hits = round(delta.get("direction_accuracy", 0) * pairs)
    print(f"{tag:>12s} {gate['cd_mae']:9.4f}/{gate['cd_spearman']:+.2f} "
          f"{windsor['cd_mae']:9.4f} {gtr['cd_mae']:8.4f} "
          f"{f'{hits}/{pairs}':>9s} {delta.get('mae', float('nan')):9.4f}")
