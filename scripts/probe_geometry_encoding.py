"""Does the geometry encoder even see these deformations?

The paired trainer caches geo_rep_surface with detach(), so whatever the encoder
fails to distinguish, no amount of decoder training can recover. This measures how
far apart the encodings of a pair's two shapes are, relative to how far apart
different cars are, and whether that distance tracks the measured ΔCd.

If pair encodings are near-identical against the between-car scale, the encoder is
the bottleneck and the decoder is being asked to read a signal that never arrives.
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import torch

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--pairs", type=Path, required=True)
ap.add_argument("--checkpoint", type=Path)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--limit", type=int, default=40)
args = ap.parse_args()

from huggingface_hub import snapshot_download
from physicsnemo.cfd.evaluation.datasets.adapters.drivaerml import DrivAerMLAdapter
from physicsnemo.cfd.evaluation.models.wrappers.domino.wrapper import DominoWrapper

snap = Path(snapshot_download("nvidia/domino_drivaerml")) / "domino_drivaerml_surface_checkpoint"
wrapper = DominoWrapper().load(checkpoint_path=str(snap / "DoMINO.0.501.mdlus"),
                               stats_path=str(snap / "scaling_factors.pkl"),
                               device=args.device, domino_config=str(snap / "config.yaml"))
model = wrapper._model
if args.checkpoint:
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
model.eval()
adapter = DrivAerMLAdapter(root=str(args.root))

cache = {}


def encode(run: int):
    if run in cache:
        return cache[run]
    data = wrapper.prepare_inputs(adapter.load_case(f"run_{run}"))["data_dict"]
    lo, hi = data["surface_min_max"][:, 0], data["surface_min_max"][:, 1]
    geometry = 2.0 * (data["geometry_coordinates"] - lo) / (hi - lo) - 1
    with torch.no_grad():
        encoding = model.geo_rep_surface(geometry, data["surf_grid"], data["sdf_surf_grid"])
    cache[run] = encoding.detach().float().reshape(-1).cpu().numpy()
    return cache[run]


pairs = json.loads(args.pairs.read_text())["pairs"][:args.limit]
within, true_delta = [], []
for pair in pairs:
    try:
        a, b = encode(pair["baseline"]), encode(pair["variant"])
    except Exception as exc:
        print(f"  건너뜀 {pair['baseline']}->{pair['variant']}: {type(exc).__name__}")
        continue
    within.append(float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12)))
    true_delta.append(pair["true_delta_cd"])

runs = sorted(cache)
between = []
for i in range(len(runs)):
    for j in range(i + 1, len(runs)):
        a, b = cache[runs[i]], cache[runs[j]]
        between.append(float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12)))

within = np.array(within)
between = np.array(between)
print(f"\n쌍 {len(within)}개, 서로 다른 형상 조합 {len(between)}개")
print(f"쌍 내부 인코딩 거리(상대): 중앙값 {np.median(within):.5f}  "
      f"범위 {within.min():.5f}~{within.max():.5f}")
print(f"형상 간 인코딩 거리(상대): 중앙값 {np.median(between):.5f}  "
      f"범위 {between.min():.5f}~{between.max():.5f}")
print(f"비율(쌍 내부 / 형상 간): {np.median(within)/np.median(between):.3f}")

if len(within) > 2 and within.std() > 0:
    corr = float(np.corrcoef(within, np.abs(true_delta))[0, 1])
    print(f"\n인코딩 거리 vs |정답 ΔCd| 상관: {corr:+.2f}")
    print("→ 상관이 0에 가까우면 인코더가 변형의 세기를 구분하지 못한다는 뜻이다")
