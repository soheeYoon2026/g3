"""Materialise the G2 deformation-pair benchmark as a v3 dataset.

The pair labels have been sitting in g2-s3-deformation-pairs-v1 as point clouds
and coefficients; the surface meshes they refer to are still in S3, one per RBF
design step at about 700 kB. This fetches those, runs each through the existing
v3 preparation, and emits a pairs manifest keyed by the run ids it assigned -- so
the same evaluators that handle the six-pair benchmark can score this one.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import boto3

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--benchmark", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--prepare", type=Path, required=True, help="prepare_domino_su2_v3.py")
ap.add_argument("--python", default=sys.executable)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

payload = json.loads(args.benchmark.read_text())
meshes = payload["meshes"]
if args.limit:
    wanted = set()
    for pair in payload["pairs"][:args.limit]:
        wanted.add((pair["job_uid"], pair["base_design"]))
        wanted.add((pair["job_uid"], pair["target_design"]))
    meshes = [m for m in meshes if (m["job_uid"], m["design"]) in wanted]

s3 = boto3.client("s3")
args.out.mkdir(parents=True, exist_ok=True)

# The per-design history.csv is the adjoint history and carries no CD column, so
# take the coefficients from the pair manifest, which already resolved them.
cd_of = {}
for pair in payload["pairs"]:
    cd_of[(pair["job_uid"], pair["base_design"])] = pair["base_cd"]
    cd_of[(pair["job_uid"], pair["target_design"])] = pair["target_cd"]

run_of, failures = {}, []
for index, mesh in enumerate(meshes, start=1):
    key = (mesh["job_uid"], mesh["design"])
    prefix = f"{mesh['output_prefix']}/{mesh['design']}"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        try:
            objects = s3.list_objects_v2(Bucket="aoxlabs-stage-static", Prefix=prefix)
            names = {Path(o["Key"]).name: o["Key"] for o in objects.get("Contents", [])}
            if "surface_flow.vtu" not in names:
                raise FileNotFoundError("surface_flow.vtu 없음")
            cfg = next((n for n in names if n.endswith("_CFD.cfg")), None)
            if cfg is None:
                raise FileNotFoundError("CFD.cfg 없음")
            for name in ("surface_flow.vtu", "history.csv", cfg):
                if name in names:
                    s3.download_file("aoxlabs-stage-static", names[name], str(tmp / name))

            run_id = str(index)
            proc = subprocess.run(
                [args.python, str(args.prepare), "--vtu", str(tmp / "surface_flow.vtu"),
                 "--cfg", str(tmp / cfg), "--out", str(args.out), "--run", run_id],
                capture_output=True, text=True, timeout=900)
            if proc.returncode != 0:
                raise RuntimeError((proc.stderr or proc.stdout).strip()[-200:])

            conditions_path = args.out / f"run_{run_id}" / f"conditions_{run_id}.json"
            conditions = json.loads(conditions_path.read_text())
            conditions["su2_cd"] = cd_of[key]
            conditions["su2_cl"] = None  # the pair manifest carries drag only
            conditions["job_uid"], conditions["design"] = key
            conditions_path.write_text(json.dumps(conditions, indent=2) + "\n")
            run_of[key] = run_id
        except Exception as exc:
            failures.append((key, f"{type(exc).__name__}: {exc}"))
            continue
    if index % 20 == 0:
        print(f"  {index}/{len(meshes)} 처리", flush=True)

print(f"\n메시 {len(run_of)}/{len(meshes)} 변환 성공")
for key, why in failures[:8]:
    print(f"  실패 {key[0][:8]}/{key[1]}: {why}")

pairs, dropped = [], 0
for pair in payload["pairs"]:
    base = run_of.get((pair["job_uid"], pair["base_design"]))
    target = run_of.get((pair["job_uid"], pair["target_design"]))
    if base is None or target is None:
        dropped += 1
        continue
    pairs.append({"baseline": int(base), "variant": int(target),
                  "true_delta_cd": pair["delta_cd"],
                  "note": f"{pair['job_uid'][:8]}:{pair['target_design']}"})

manifest = args.out / "pairs.json"
manifest.write_text(json.dumps({"pairs": pairs}, indent=1) + "\n")
index = args.out / "run_index.json"
index.write_text(json.dumps(
    {run: {"job_uid": job, "design": design, "su2_cd": cd_of[(job, design)]}
     for (job, design), run in sorted(run_of.items(), key=lambda kv: int(kv[1]))},
    indent=1) + "\n")
print(f"쌍 {len(pairs)}개 기록 (메시 누락으로 {dropped}개 제외) -> {manifest}")
print(f"런 대응표 -> {index}")
