"""Can the pretrained checkpoint still load at a finer geometry grid?

Raising interp_res only helps if the pretrained weights survive it. If the
geometry representation is convolutional the kernels are resolution-independent
and it loads; if anything is a fixed-size linear layer over the flattened grid,
it does not, and the fix would cost the pretrained model.
"""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import torch

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--resolutions", nargs="+", default=["128,64,64", "256,128,128"])
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from physicsnemo.models.domino.model import DoMINO

snap = Path(snapshot_download("nvidia/domino_drivaerml")) / "domino_drivaerml_surface_checkpoint"
config = OmegaConf.load(snap / "config.yaml")
reference = torch.load(snap.parent / "domino_drivaerml_surface_checkpoint" /
                       "DoMINO.0.501.mdlus", map_location="cpu", weights_only=False) \
    if False else None

from physicsnemo import Module

pretrained = Module.from_checkpoint(str(snap / "DoMINO.0.501.mdlus"))
weights = pretrained.state_dict()
print(f"사전학습 텐서 {len(weights)}개")

surface_vars = len(config.variables.surface.solution)
for spec in args.resolutions:
    res = [int(v) for v in spec.split(",")]
    model_cfg = OmegaConf.merge(config.model, OmegaConf.create({"interp_res": res}))
    try:
        model = DoMINO(input_features=3, output_features_vol=None,
                       output_features_surf=surface_vars,
                       global_features=config.model.global_features
                       if "global_features" in config.model else 2,
                       model_parameters=model_cfg)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        shape_bad = [k for k, v in model.state_dict().items()
                     if k in weights and v.shape != weights[k].shape]
        print(f"interp_res={res}: 누락 {len(missing)}  잉여 {len(unexpected)}  "
              f"모양불일치 {len(shape_bad)}"
              + ("  -> 사전학습 유지 가능" if not (missing or shape_bad) else "  -> 불가"))
        for name in (list(missing)[:3] + shape_bad[:3]):
            print(f"    {name}")
    except Exception as exc:
        print(f"interp_res={res}: 생성 실패 {type(exc).__name__}: {str(exc)[:160]}")
