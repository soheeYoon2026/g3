"""Convert synced training-event JSON inventories to materializer rows."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _find_number(value: Any, keys: tuple[str, ...]) -> float | None:
    if isinstance(value, dict):
        normalized = {str(key).lower(): child for key, child in value.items()}
        for key in keys:
            if key in normalized:
                direct = _number(normalized[key])
                if direct is not None:
                    return direct
                nested = _find_number(normalized[key], ("final", "best", "value", "result"))
                if nested is not None:
                    return nested
        for child in value.values():
            nested = _find_number(child, keys)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for child in value:
            nested = _find_number(child, keys)
            if nested is not None:
                return nested
    return None


def load_event_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("cases", []):
            event = record["event"]
            event_id = str(event["event_id"])
            if event_id in seen:
                continue
            seen.add(event_id)
            summary = event.get("result_summary") or {}
            objective = str(event.get("objective") or "drag").lower()
            cd = _find_number(summary, ("cd_final", "cd_best", "best_cd", "cd"))
            cl = _find_number(summary, ("cl_final", "cl_best", "best_cl", "cl"))
            target = cl if objective == "lift" else cd
            prefix = str(event.get("s3_output_prefix") or "").rstrip("/")
            rows.append({
                "solver": str(event.get("solver") or "").upper(),
                "job_status": "succeeded",
                "status": "succeeded",
                "test_case": event_id.replace(":", "_"),
                "job_id": str(event.get("job_id") or ""),
                "job_uid": str(event.get("job_uid") or event_id),
                "project_uid": str(event.get("project_uid") or event_id),
                "s3_bucket": str(event.get("s3_output_bucket") or ""),
                "s3_output_prefix": prefix,
                "output_s3_key": prefix + "/",
                "objective_function": objective,
                "cd_initial": "",
                "cd_final": "" if target is None else str(target),
                "cl_final": "" if cl is None else str(cl),
                "run_config": json.dumps({
                    **(event.get("conditions") or {}), "objective": objective
                }),
            })
    return rows
