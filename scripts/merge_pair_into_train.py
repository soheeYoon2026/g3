"""Merge the deformation-pair runs into the aligned training root.

The absolute-Cd fine-tune reads one dataset and the ΔCp pair loss another, so
combining the two losses needs both in a single root with non-colliding run ids.
The pair runs keep their own ids offset by a fixed base, and the pair manifest is
rewritten to match.

Only pairs whose geometry is already in the absolute training set are taken, so
this adds a new *signal* without adding new shapes - the standing gates stay as
uncontaminated as they were.
"""

import argparse
import json
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--train-root", type=Path, required=True, help="aligned absolute-Cd dataset")
ap.add_argument("--pair-root", type=Path, required=True, help="aligned pair dataset")
ap.add_argument("--pairs", type=Path, required=True, help="pairs to merge")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--offset", type=int, default=10000)
args = ap.parse_args()

pairs = json.loads(args.pairs.read_text())["pairs"]
runs = sorted({r for p in pairs for r in (p["baseline"], p["variant"])})

if args.out.exists():
    shutil.rmtree(args.out)
shutil.copytree(args.train_root, args.out)
base_runs = sorted(int(p.name.split("_")[1]) for p in args.out.glob("run_*") if p.is_dir())
print(f"절대 Cd 학습셋 {len(base_runs)}런 복사")

copied = []
for run in runs:
    src = args.pair_root / f"run_{run}"
    new_id = run + args.offset
    dst = args.out / f"run_{new_id}"
    if dst.exists():
        raise SystemExit(f"run id 충돌: {new_id}")
    shutil.copytree(src, dst)
    # the v3 loader keys every file on the run id, so rename in place
    for old in list(dst.iterdir()):
        if f"_{run}." in old.name:
            old.rename(dst / old.name.replace(f"_{run}.", f"_{new_id}."))
    copied.append(new_id)

print(f"쌍 런 {len(copied)}개 추가 (id +{args.offset})")

remapped = [{**p,
             "baseline": p["baseline"] + args.offset,
             "variant": p["variant"] + args.offset}
            for p in pairs]
(args.out / "pairs.json").write_text(json.dumps({"pairs": remapped}, indent=1) + "\n")
print(f"쌍 매니페스트 {len(remapped)}개 -> {args.out / 'pairs.json'}")
print(f"합계 {len(base_runs) + len(copied)}런")
