"""Shared coefficient metrics for G3 checkpoint evaluation.

Implements the unified evaluation contract: absolute Cd/Cl error, rank
correlation, and pairwise ``ΔCd = Cd(variant) - Cd(baseline)`` error with
direction accuracy. Pairs come either from an explicit baseline/variant
manifest or from every same-geometry-group combination in the evaluated rows,
so the DoMINO and PointNet evaluators report identical metric definitions.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np


def spearman(actual, predicted):
    """Return Spearman rank correlation, including average ranks for ties."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        result = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            result[order[start:end]] = 0.5 * (start + end - 1) + 1.0
            start = end
        return result

    if len(actual) < 2:
        return None
    actual_rank, predicted_rank = ranks(actual), ranks(predicted)
    if np.std(actual_rank) == 0 or np.std(predicted_rank) == 0:
        return None
    return float(np.corrcoef(actual_rank, predicted_rank)[0, 1])


def load_pairs(path: Path):
    """Read an explicit pair manifest: a list (or {"pairs": list}) of
    {"baseline": <case key>, "variant": <case key>} objects."""
    payload = json.loads(Path(path).read_text())
    pairs = payload["pairs"] if isinstance(payload, dict) else payload
    if not isinstance(pairs, list):
        raise ValueError("pair manifest must be a list or contain a 'pairs' list")
    for pair in pairs:
        if "baseline" not in pair or "variant" not in pair:
            raise ValueError(f"pair entry needs 'baseline' and 'variant': {pair}")
    return pairs


def derive_group_pairs(rows, group_key="group_id", case_key="case"):
    """Every unordered same-group combination, oriented and ordered by case key."""
    grouped = {}
    for index, row in enumerate(rows):
        group = row.get(group_key)
        if group is not None:
            grouped.setdefault(group, []).append(index)
    pairs = []
    for group in sorted(grouped, key=str):
        members = sorted(grouped[group], key=lambda index: str(rows[index][case_key]))
        pairs.extend(itertools.combinations(members, 2))
    return pairs


def resolve_explicit_pairs(rows, pairs, case_key="case"):
    """Map baseline/variant case keys onto row indices, failing on unknown keys."""
    by_case = {str(row[case_key]): index for index, row in enumerate(rows)}
    resolved = []
    for pair in pairs:
        baseline, variant = str(pair["baseline"]), str(pair["variant"])
        if baseline not in by_case or variant not in by_case:
            missing = baseline if baseline not in by_case else variant
            raise ValueError(f"pair references case '{missing}' absent from evaluated rows")
        resolved.append((by_case[baseline], by_case[variant]))
    return resolved


def _signed_direction(value, tolerance):
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def delta_metrics(rows, index_pairs, true_key, pred_key, direction_tolerance=0.0):
    """MAE, direction accuracy, and rank correlation of pairwise deltas.

    Direction accuracy only counts pairs whose true delta magnitude exceeds
    ``direction_tolerance``; a zero predicted delta never matches a nonzero
    true delta.
    """
    if not index_pairs:
        return None
    true_delta, pred_delta = [], []
    for baseline, variant in index_pairs:
        true_delta.append(rows[variant][true_key] - rows[baseline][true_key])
        pred_delta.append(rows[variant][pred_key] - rows[baseline][pred_key])
    true_delta = np.asarray(true_delta, dtype=float)
    pred_delta = np.asarray(pred_delta, dtype=float)

    directional = np.abs(true_delta) > direction_tolerance
    matched = [
        _signed_direction(t, direction_tolerance) == _signed_direction(p, 0.0)
        for t, p in zip(true_delta[directional], pred_delta[directional])
    ]
    return {
        "pairs": int(len(index_pairs)),
        "mae": float(np.mean(np.abs(true_delta - pred_delta))),
        "direction_pairs": int(np.sum(directional)),
        "direction_accuracy": float(np.mean(matched)) if matched else None,
        "spearman": spearman(true_delta, pred_delta),
    }


def coefficient_summary(rows, pairs=None, case_key="case", direction_tolerance=0.0):
    """Unified summary: absolute Cd/Cl MAE + Spearman, and ΔCd/ΔCl pair metrics.

    ``rows`` need ``true_cd``/``pred_cd``/``true_cl``/``pred_cl``; group-derived
    pairing additionally needs ``group_id``. ``pairs`` (from ``load_pairs``)
    overrides group derivation.
    """
    summary = {
        "cases": len(rows),
        "cd_mae": float(np.mean([abs(r["true_cd"] - r["pred_cd"]) for r in rows])),
        "cl_mae": float(np.mean([abs(r["true_cl"] - r["pred_cl"]) for r in rows])),
        "cd_spearman": spearman([r["true_cd"] for r in rows], [r["pred_cd"] for r in rows]),
        "cl_spearman": spearman([r["true_cl"] for r in rows], [r["pred_cl"] for r in rows]),
    }
    if pairs is not None:
        index_pairs = resolve_explicit_pairs(rows, pairs, case_key=case_key)
        summary["pair_source"] = "explicit"
    else:
        index_pairs = derive_group_pairs(rows, case_key=case_key)
        summary["pair_source"] = "group" if index_pairs else None
    for name, true_key, pred_key in (("delta_cd", "true_cd", "pred_cd"), ("delta_cl", "true_cl", "pred_cl")):
        metrics = delta_metrics(rows, index_pairs, true_key, pred_key, direction_tolerance)
        if metrics is None:
            summary[name] = None
        else:
            summary[name] = metrics
    return summary
