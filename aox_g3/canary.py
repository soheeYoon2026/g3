"""Shadow-canary dispatch and audit logging for the G3 inference service.

The production response is never changed by this module.  A sampled request is
replayed to a localhost-only challenger service after the primary response has
been produced, and a compact comparison is appended to a JSONL audit file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def should_sample(payload: bytes, rate: float) -> bool:
    """Deterministically sample identical STL bytes at the same rate."""
    rate = min(max(float(rate), 0.0), 1.0)
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(2**64) < rate


def compare_results(primary: dict, challenger: dict) -> dict:
    """Return runtime and prediction drift without copying large preview data."""
    result: dict[str, Any] = {
        "primary_model": primary.get("model"),
        "challenger_model": challenger.get("model"),
        "primary_expert": primary.get("coefficient_expert"),
        "challenger_expert": challenger.get("coefficient_expert"),
    }
    for key, output_key in (
        ("drag_coefficient", "cd"),
        ("lift_coefficient", "cl"),
        ("elapsed_seconds", "latency_seconds"),
        ("ood_score", "ood_score"),
    ):
        left = _float(primary.get(key))
        right = _float(challenger.get(key))
        result[f"primary_{output_key}"] = left
        result[f"challenger_{output_key}"] = right
        result[f"delta_{output_key}"] = (
            right - left if left is not None and right is not None else None
        )
    primary_latency = result["primary_latency_seconds"]
    challenger_latency = result["challenger_latency_seconds"]
    result["latency_ratio"] = (
        challenger_latency / primary_latency
        if primary_latency and challenger_latency is not None
        else None
    )
    return result


@dataclass(frozen=True)
class ShadowConfig:
    url: str = ""
    token_file: str = ""
    sample_rate: float = 0.0
    timeout_seconds: int = 300
    audit_path: str = "var/canary/shadow.jsonl"
    spool_dir: str = "var/canary/spool"
    model_pointer: str = ""
    max_pending: int = 1

    @classmethod
    def from_env(cls) -> "ShadowConfig":
        return cls(
            url=os.environ.get("G3_SHADOW_URL", "").rstrip("/"),
            token_file=os.environ.get("G3_SHADOW_TOKEN_FILE", ""),
            sample_rate=float(os.environ.get("G3_SHADOW_SAMPLE_RATE", "0")),
            timeout_seconds=int(os.environ.get("G3_SHADOW_TIMEOUT_SECONDS", "300")),
            audit_path=os.environ.get("G3_SHADOW_AUDIT_PATH", "var/canary/shadow.jsonl"),
            spool_dir=os.environ.get("G3_SHADOW_SPOOL_DIR", "var/canary/spool"),
            model_pointer=os.environ.get("G3_SHADOW_MODEL_POINTER", ""),
            max_pending=max(1, int(os.environ.get("G3_SHADOW_MAX_PENDING", "1"))),
        )

    @property
    def enabled(self) -> bool:
        configured = bool(self.url and self.token_file and self.sample_rate > 0.0)
        return configured and (
            not self.model_pointer or Path(self.model_pointer).is_file()
        )


class ShadowDispatcher:
    """Bounded, best-effort replay client used by FastAPI BackgroundTasks."""

    def __init__(self, config: ShadowConfig):
        self.config = config
        self._slots = threading.BoundedSemaphore(config.max_pending)
        self._audit_lock = threading.Lock()

    def reserve(self, payload: bytes) -> bool:
        return (
            self.config.enabled
            and should_sample(payload, self.config.sample_rate)
            and self._slots.acquire(blocking=False)
        )

    def spool(self, payload: bytes) -> Path:
        directory = Path(self.config.spool_dir)
        directory.mkdir(parents=True, exist_ok=True)
        descriptor, raw_path = tempfile.mkstemp(prefix="shadow-", suffix=".stl", dir=directory)
        path = Path(raw_path)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
            os.chmod(path, 0o600)
        except Exception:
            path.unlink(missing_ok=True)
            self._slots.release()
            raise
        return path

    def release(self) -> None:
        self._slots.release()

    def _write_audit(self, row: dict) -> None:
        path = Path(self.config.audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, separators=(",", ":"), allow_nan=False)
        with self._audit_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def run(
        self,
        stl_path: Path,
        filename: str,
        form_data: dict[str, Any],
        primary: dict,
        request_hash: str,
    ) -> None:
        """Replay one request.  All failures are recorded and never propagated."""
        started = datetime.now(timezone.utc)
        row: dict[str, Any] = {
            "timestamp": started.isoformat(),
            "request_hash": request_hash,
            "status": "failed",
        }
        try:
            import requests

            token = Path(self.config.token_file).read_text().strip()
            data = {key: str(value) for key, value in form_data.items()}
            data["include_preview"] = "false"
            with stl_path.open("rb") as handle:
                response = requests.post(
                    f"{self.config.url}/v1/infer",
                    headers={"Authorization": f"Bearer {token}"},
                    files={"stl": (filename, handle, "model/stl")},
                    data=data,
                    timeout=(5, self.config.timeout_seconds),
                )
            row["http_status"] = response.status_code
            if response.ok:
                row.update(compare_results(primary, response.json()))
                row["status"] = "ok"
            else:
                row["error"] = f"challenger-http-{response.status_code}"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
        finally:
            row["completed_at"] = datetime.now(timezone.utc).isoformat()
            try:
                self._write_audit(row)
            finally:
                stl_path.unlink(missing_ok=True)
                self._slots.release()


def request_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
