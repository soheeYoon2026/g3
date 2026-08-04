"""Harness for the production surrogate: NVIDIA PhysicsNeMo DoMINO.

DoMINO is the recommended G3 engine: a point-based neural operator that maps an
STL to surface pressure / wall-shear (and optionally volume fields), which then
integrate to Cd/Cl. This script is the seam that will replace the Phase 0
baseline once the GPU stack is in place.

It does NOT ship weights or physicsnemo (multi-GB, GPU, license terms — see
domino/README.md). Run it as-is and it detects what is missing and tells you
exactly how to get it, so the seam is testable before the heavy install.

    python domino/run_domino_inference.py --stl ../g4-docker-image/ahmed_1.stl
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import textwrap


FETCH_HELP = textwrap.dedent(
    """
    To enable DoMINO inference:

    1. Install PhysicsNeMo (Apache-2.0 code) into a torch+CUDA env:
         pip install nvidia-physicsnemo
       (or clone github.com/NVIDIA/physicsnemo and its physicsnemo-cfd helper)

    2. Get weights. Two routes, MIND THE LICENCE:
       a) Pretrained checkpoint from Hugging Face (NVIDIA Open Model License,
          which permits commercial use) — confirm the exact repo id on the
          model card, e.g. the DoMINO / DrivAerML external-aero checkpoints
          under the `nvidia/` org. Use only as a *seed*; mint your own weights
          for production.
       b) TRAIN YOUR OWN on DrivAerML (CC-BY-SA, commercial-clean) plus labels
          from G1/G2/G4 — see aox_g3.data.label_interface. This is the
          recommended production path (no redistribution constraints).

       Do NOT train production weights on DrivAerNet/DrivAerNet++ — that data is
       CC-BY-NC (non-commercial).

    3. Set DOMINO_CHECKPOINT=/path/to/checkpoint and re-run.

    Reference example:
       physicsnemo/examples/cfd/external_aerodynamics/domino
    """
)


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def preflight() -> list[str]:
    missing = []
    if not _have("torch"):
        missing.append("torch (+CUDA)")
    if not _have("physicsnemo"):
        missing.append("physicsnemo")
    if not os.environ.get("DOMINO_CHECKPOINT"):
        missing.append("DOMINO_CHECKPOINT env var")
    return missing


def run(stl_path: str, checkpoint: str) -> dict:
    """Load DoMINO, predict the surface field, integrate to Cd/Cl.

    Left as the integration point: wire to the physicsnemo DoMINO inference API
    for your checkpoint. Sample the STL with aox_g3.geometry.stl_to_pointcloud
    (+ SDF) so the input matches how the model was trained, run the forward
    pass to get surface pressure & wall-shear, then integrate over area for the
    force coefficients.
    """
    raise NotImplementedError(
        "Wire this to the physicsnemo DoMINO inference API for your checkpoint. "
        "Input: stl_to_pointcloud(stl) + SDF. Output: surface p/tau -> Cd/Cl."
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="DoMINO surrogate inference (STL -> Cd/Cl).")
    ap.add_argument("--stl", required=True)
    ap.add_argument("--checkpoint", default=os.environ.get("DOMINO_CHECKPOINT"))
    args = ap.parse_args(argv)

    missing = preflight()
    if missing:
        print("DoMINO inference is not wired up yet. Missing:")
        for m in missing:
            print(f"  - {m}")
        print(FETCH_HELP)
        return 2

    result = run(args.stl, args.checkpoint)
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
