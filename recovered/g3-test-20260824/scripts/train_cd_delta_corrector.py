#!/usr/bin/env python3
"""Train a small, separate Cd-delta corrector without changing DoMINO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def features(row: dict) -> list[float]:
    p = np.asarray(row["position"], dtype=float)
    d = np.asarray(row["displacement"], dtype=float)
    magnitude = float(np.linalg.norm(d))
    # Compact edit descriptor. Quadratic position terms let a linear ridge
    # model represent different behavior around nose/cabin/tail without a
    # high-capacity neural network that would memorize 27 samples.
    return [
        *p.tolist(),
        *d.tolist(),
        magnitude,
        float(row.get("influence_radius", 0.0)),
        float(row.get("amplitude_scale", 1.0)),
        float(row.get("predicted_delta_cd") or 0.0),
        p[0] * p[0], p[1] * p[1], p[2] * p[2],
        p[0] * p[1], p[0] * p[2], p[1] * p[2],
    ]


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    signs = np.sign(y) == np.sign(pred)
    return {
        "count": int(len(y)),
        "mae": float(np.mean(np.abs(y - pred))),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "direction_accuracy": float(np.mean(signs)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.dataset).read_text())
    rows = data["samples"]
    splits = {name: [r for r in rows if r["split"] == name]
              for name in ("train", "validation", "test")}

    x_train = np.asarray([features(r) for r in splits["train"]])
    y_train = np.asarray([r["g2_delta_cd"] for r in splits["train"]])
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-9] = 1.0

    def design(items: list[dict]) -> np.ndarray:
        x = (np.asarray([features(r) for r in items]) - mean) / scale
        return np.column_stack([np.ones(len(x)), x])

    xt = design(splits["train"])
    xv = design(splits["validation"])
    yv = np.asarray([r["g2_delta_cd"] for r in splits["validation"]])
    candidates = []
    for alpha in np.logspace(-5, 3, 81):
        penalty = np.eye(xt.shape[1]) * alpha
        penalty[0, 0] = 0.0
        weights = np.linalg.solve(xt.T @ xt + penalty, xt.T @ y_train)
        pred = xv @ weights
        score = np.mean(np.abs(yv - pred))
        candidates.append((score, float(alpha), weights))
    _, alpha, weights = min(candidates, key=lambda item: item[0])

    report = {}
    predictions = []
    for name, items in splits.items():
        y = np.asarray([r["g2_delta_cd"] for r in items])
        pred = design(items) @ weights
        report[name] = metrics(y, pred)
        for row, value in zip(items, pred):
            predictions.append({
                "split": name,
                "label": row["label"],
                "actual_delta_cd": row["g2_delta_cd"],
                "base_ai_delta_cd": row.get("predicted_delta_cd"),
                "corrected_delta_cd": float(value),
            })

    test_rows = splits["test"]
    base_test = np.asarray([r.get("predicted_delta_cd", 0.0) for r in test_rows])
    actual_test = np.asarray([r["g2_delta_cd"] for r in test_rows])
    report["base_ai_test"] = metrics(actual_test, base_test)
    payload = {
        "schema_version": 1,
        "model_type": "ridge_cd_delta_corrector",
        "scope": "gtr_r35_nismo_single_control_edit",
        "feature_order": [
            "px", "py", "pz", "dx", "dy", "dz", "displacement_magnitude",
            "influence_radius", "amplitude_scale", "base_ai_delta_cd",
            "px2", "py2", "pz2", "px_py", "px_pz", "py_pz",
        ],
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "intercept": float(weights[0]),
        "weights": weights[1:].tolist(),
        "ridge_alpha": alpha,
        "metrics": report,
        "predictions": predictions,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"output": str(out), "alpha": alpha, "metrics": report}, indent=2))


if __name__ == "__main__":
    main()
