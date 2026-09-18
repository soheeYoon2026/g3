"""Reuse only completed measurements with identical input bytes and settings."""
import hashlib
import json
import math
import time
from pathlib import Path

from .run_state import write_json


def cached_thickness(path, input_path, settings, measure, force=False):
    digest = hashlib.sha256()
    with open(input_path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    identity = {"input_sha256": digest.hexdigest(), "settings": settings}
    path = Path(path)
    if not force and path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            value = cached["thickness_p5_mm"]
            if (cached.get("status") == "done" and cached.get("identity") == identity
                    and type(value) in (int, float) and math.isfinite(value) and value >= 0):
                return float(value), True
        except (OSError, ValueError, KeyError, TypeError):
            pass
    started = time.time()
    value = float(measure())
    if not math.isfinite(value) or value < 0:
        raise ValueError("두께 측정 결과가 유효하지 않습니다.")
    write_json(path, {"status": "done", "identity": identity,
                      "thickness_p5_mm": value, "seconds": time.time() - started,
                      "completed_at": time.time()})
    return value, False
