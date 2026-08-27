"""Convert an official-training .mdlus checkpoint into the plain state_dict our
evaluators load, and sanity-check that it matches the pretrained architecture."""

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import torch
from huggingface_hub import snapshot_download
from physicsnemo.cfd.evaluation.models.wrappers.domino.wrapper import DominoWrapper

ap = argparse.ArgumentParser()
ap.add_argument("--mdlus", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

snapshot = Path(snapshot_download("nvidia/domino_drivaerml"))
surf = snapshot / "domino_drivaerml_surface_checkpoint"
wrapper = DominoWrapper().load(checkpoint_path=str(surf / "DoMINO.0.501.mdlus"),
                               stats_path=str(surf / "scaling_factors.pkl"),
                               device=args.device, domino_config=str(surf / "config.yaml"))
reference = wrapper._model.state_dict()

from physicsnemo import Module

model = Module.from_checkpoint(str(args.mdlus)).to(args.device)
trained = model.state_dict()

missing = [k for k in reference if k not in trained]
extra = [k for k in trained if k not in reference]
mismatch = [k for k in reference if k in trained and reference[k].shape != trained[k].shape]
print(f"reference {len(reference)} tensors | trained {len(trained)}")
print(f"missing {len(missing)} | extra {len(extra)} | shape mismatch {len(mismatch)}")
for name in (missing[:3] + extra[:3] + mismatch[:3]):
    print("   ", name)
if missing or extra or mismatch:
    raise SystemExit("architecture mismatch — cannot reuse the existing evaluators")

changed = sum(1 for k in reference if not torch.equal(reference[k].cpu(), trained[k].cpu()))
print(f"pretrained 대비 변경된 텐서: {changed}/{len(reference)}")
torch.save(trained, args.out)
print("wrote", args.out)
