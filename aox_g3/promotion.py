"""Offline and shadow-canary gates used before a G3 model promotion."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


def nested_value(payload: dict, dotted_path: str) -> float:
    value: Any = payload
    for part in dotted_path.split("."):
        value = value[part]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric at {dotted_path}")
    return result


def evaluate_offline_gates(
    production: dict,
    challenger: dict,
    gates: dict[str, dict[str, float]],
) -> dict:
    checks = []
    for metric, policy in gates.items():
        current = nested_value(production, metric)
        candidate = nested_value(challenger, metric)
        allowed = current * float(policy.get("max_regression_ratio", 1.0))
        absolute_max = policy.get("absolute_max")
        passed = candidate <= allowed
        if absolute_max is not None:
            passed = passed and candidate <= float(absolute_max)
        checks.append({
            "metric": metric,
            "production": current,
            "challenger": candidate,
            "allowed": allowed,
            "absolute_max": absolute_max,
            "passed": passed,
        })
    return {"passed": bool(checks) and all(row["passed"] for row in checks), "checks": checks}


def load_jsonl(path: str | Path) -> Iterable[dict]:
    source = Path(path)
    if not source.exists():
        return []
    rows = []
    for line in source.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, TypeError):
            continue
    return rows


def evaluate_shadow_gates(
    rows: Iterable[dict],
    *,
    challenger_model: str | None = None,
    min_samples: int = 20,
    max_error_rate: float = 0.01,
    max_p95_abs_cd_delta: float = 0.05,
    max_p95_abs_cl_delta: float = 0.05,
    max_p95_latency_ratio: float = 1.5,
) -> dict:
    selected = [
        row for row in rows
        if challenger_model is None or row.get("challenger_model") == challenger_model
    ]
    successful = [row for row in selected if row.get("status") == "ok"]

    def percentile(field: str, *, absolute: bool = False) -> float | None:
        values = []
        for row in successful:
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(abs(value) if absolute else value)
        if not values:
            return None
        values.sort()
        index = min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)
        return values[index]

    error_rate = (
        (len(selected) - len(successful)) / len(selected) if selected else 1.0
    )
    cd_p95 = percentile("delta_cd", absolute=True)
    cl_p95 = percentile("delta_cl", absolute=True)
    latency_p95 = percentile("latency_ratio")
    checks = {
        "sample_count": len(selected) >= min_samples,
        "error_rate": error_rate <= max_error_rate,
        "cd_drift": cd_p95 is not None and cd_p95 <= max_p95_abs_cd_delta,
        "cl_drift": cl_p95 is not None and cl_p95 <= max_p95_abs_cl_delta,
        "latency": latency_p95 is not None and latency_p95 <= max_p95_latency_ratio,
    }
    return {
        "passed": all(checks.values()),
        "samples": len(selected),
        "successful": len(successful),
        "error_rate": error_rate,
        "p95_abs_cd_delta": cd_p95,
        "p95_abs_cl_delta": cl_p95,
        "p95_latency_ratio": latency_p95,
        "checks": checks,
    }

