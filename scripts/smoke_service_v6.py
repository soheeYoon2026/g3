#!/usr/bin/env python3
"""Run an authenticated G3 service smoke test without printing credentials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8003")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--expected-model", default="g3_field_g2_v6_final.pt")
    args = parser.parse_args()

    token = args.token_file.read_text().strip()
    with args.stl.open("rb") as handle:
        response = requests.post(
            args.url.rstrip("/") + "/v1/infer",
            headers={"Authorization": f"Bearer {token}"},
            files={"stl": (args.stl.name, handle, "model/stl")},
            data={"grid_x": 8, "grid_y": 8, "grid_z": 8},
            timeout=300,
        )
    response.raise_for_status()
    payload = response.json()
    if payload.get("model") != args.expected_model:
        raise RuntimeError(f"unexpected service model: {payload.get('model')}")
    summary_keys = (
        "model", "coefficient_expert", "deployment_status", "ood_score",
        "in_distribution", "coefficient_warning", "drag_coefficient",
        "lift_coefficient", "pressure_range", "speed_range", "grid", "device",
        "elapsed_seconds",
    )
    print(json.dumps({key: payload.get(key) for key in summary_keys}, indent=2))


if __name__ == "__main__":
    main()
