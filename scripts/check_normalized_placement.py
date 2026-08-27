"""Where do our shapes land inside the encoder's normalized domain?

geo_rep_surface is fed 2*(coords - min)/(max - min) - 1 against a fixed
DrivAerML box, and its SDF grid covers [-1, 1]. Anything outside that range is
not merely coarsely sampled, it is off the grid entirely -- which would explain an
encoder that cannot tell a deformation from no deformation at all.
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
ap.add_argument("--runs", type=int, default=8)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

from huggingface_hub import snapshot_download
from physicsnemo.cfd.evaluation.datasets.adapters.drivaerml import DrivAerMLAdapter
from physicsnemo.cfd.evaluation.models.wrappers.domino.wrapper import DominoWrapper

snap = Path(snapshot_download("nvidia/domino_drivaerml")) / "domino_drivaerml_surface_checkpoint"
wrapper = DominoWrapper().load(checkpoint_path=str(snap / "DoMINO.0.501.mdlus"),
                               stats_path=str(snap / "scaling_factors.pkl"),
                               device=args.device, domino_config=str(snap / "config.yaml"))
adapter = DrivAerMLAdapter(root=str(args.root))

run_dirs = sorted((p for p in args.root.glob("run_*") if p.is_dir()),
                  key=lambda p: int(p.name.split("_")[1]))[:args.runs]

print(f"{'run':>8s} {'정규화 x':>18s} {'정규화 y':>18s} {'정규화 z':>18s} {'상자 안 비율':>10s}")
print("-" * 78)
inside_fractions = []
for run_dir in run_dirs:
    run = run_dir.name.split("_", 1)[1]
    data = wrapper.prepare_inputs(adapter.load_case(f"run_{run}"))["data_dict"]
    lo, hi = data["surface_min_max"][:, 0], data["surface_min_max"][:, 1]
    geometry = (2.0 * (data["geometry_coordinates"] - lo) / (hi - lo) - 1)
    g = geometry.detach().cpu().numpy().reshape(-1, 3)
    inside = float(np.all((g >= -1) & (g <= 1), axis=1).mean())
    inside_fractions.append(inside)
    ranges = [f"[{g[:,k].min():+.2f},{g[:,k].max():+.2f}]" for k in range(3)]
    print(f"{run_dir.name:>8s} {ranges[0]:>18s} {ranges[1]:>18s} {ranges[2]:>18s} "
          f"{100*inside:9.1f}%")

print(f"\n정규화 상자 [-1, 1] 안에 든 정점 비율: 평균 {100*np.mean(inside_fractions):.1f}%")
print(f"사용된 상자: min {np.round(lo.detach().cpu().numpy().ravel(),3).tolist()} "
      f"max {np.round(hi.detach().cpu().numpy().ravel(),3).tolist()}")
print("\n비율이 0에 가까우면 형상이 인코더 격자 밖에 있다는 뜻이고,")
print("그렇다면 해상도 문제가 아니라 좌표계 문제가 먼저다.")
