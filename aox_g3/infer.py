"""Inference: STL in -> Cd/Cl out, in the AOX product contract.

    python -m aox_g3.infer --stl body.stl --model g3_model.pkl

This is the seam the platform calls. It samples the STL with the *same*
``stl_to_pointcloud`` used in training and returns the objective vector. The
production version additionally returns the surface pressure/shear field the
force integral is derived from.
"""

from __future__ import annotations

import argparse
import json

from .config import TARGETS, DEFAULT_SAMPLE
from .geometry.stl_sampler import stl_to_pointcloud
from .models.baseline_sklearn import PooledMLPRegressor


def predict_stl(stl_path: str, model_path: str, cfg=DEFAULT_SAMPLE) -> dict:
    cloud = stl_to_pointcloud(stl_path, cfg)
    model = PooledMLPRegressor.load(model_path)
    pred = model.predict([cloud])[0]
    return {name: float(pred[j]) for j, name in enumerate(TARGETS)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Predict Cd/Cl from an STL.")
    ap.add_argument("--stl", required=True)
    ap.add_argument("--model", default="g3_model.pkl")
    args = ap.parse_args(argv)

    result = predict_stl(args.stl, args.model)
    print(json.dumps(result, indent=2))
    print(f"\nCd = {result['cd']:.4f}  ({result['cd'] * 1e4:.0f} drag counts)"
          f"   Cl = {result['cl']:.4f}")
    return result


if __name__ == "__main__":
    main()
