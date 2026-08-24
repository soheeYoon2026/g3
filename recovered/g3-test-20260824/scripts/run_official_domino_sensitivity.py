#!/usr/bin/env python3
"""Call NVIDIA's official sensitivity workflow without its optional CLI package."""

import argparse
import sys
import types
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stl", required=True)
    ap.add_argument("--speed", type=float, default=30.0)
    ap.add_argument("--density", type=float, default=1.225)
    ap.add_argument("--stencil-size", type=int, default=7)
    args = ap.parse_args()

    workflow = Path(args.workflow).resolve()
    sys.path.insert(0, str(workflow))
    # tyro is used only by the optional `if __name__ == '__main__'` CLI.
    sys.modules.setdefault("tyro", types.SimpleNamespace())
    import main as official

    official.main(
        model_checkpoint_path=Path(args.checkpoint),
        input_file=Path(args.stl),
        stream_velocity=args.speed,
        air_density=args.density,
        stencil_size=args.stencil_size,
        verbose=False,
    )


if __name__ == "__main__":
    main()
