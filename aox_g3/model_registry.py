"""Atomic production/challenger model pointers with rollback history."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.state_path = self.root / "registry.json"

    def _read(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema_version": 1, "history": []}
        return json.loads(self.state_path.read_text())

    def _write(self, state: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / f".registry-{os.getpid()}.json"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        os.replace(temporary, self.state_path)

    def _target(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {target}")
        return target

    def _set_pointer(self, name: str, target: Path) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        pointer = self.root / f"{name}.pt"
        temporary = self.root / f".{name}-{os.getpid()}.pt"
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(os.path.relpath(target, self.root))
        os.replace(temporary, pointer)
        return pointer

    def initialize(self, production: str | Path) -> dict[str, Any]:
        target = self._target(production)
        self._set_pointer("production", target)
        state = self._read()
        state.update({
            "production": str(target),
            "previous_production": state.get("previous_production"),
            "updated_at": _utc_now(),
        })
        self._write(state)
        return state

    def stage(self, challenger: str | Path, report: str | Path | None = None) -> dict[str, Any]:
        target = self._target(challenger)
        self._set_pointer("challenger", target)
        state = self._read()
        state.update({
            "challenger": str(target),
            "challenger_report": str(Path(report).resolve()) if report else None,
            "challenger_staged_at": _utc_now(),
            "updated_at": _utc_now(),
        })
        self._write(state)
        return state

    def promote(self, reason: str) -> dict[str, Any]:
        state = self._read()
        challenger = state.get("challenger")
        production = state.get("production")
        if not challenger:
            raise RuntimeError("no challenger is staged")
        target = self._target(challenger)
        if production:
            previous = self._target(production)
            self._set_pointer("previous", previous)
        self._set_pointer("production", target)
        (self.root / "challenger.pt").unlink(missing_ok=True)
        event = {
            "action": "promote",
            "at": _utc_now(),
            "from": production,
            "to": str(target),
            "reason": reason,
        }
        state.setdefault("history", []).append(event)
        state.update({
            "previous_production": production,
            "production": str(target),
            "challenger": None,
            "promoted_at": event["at"],
            "updated_at": event["at"],
        })
        self._write(state)
        return state

    def rollback(self, reason: str) -> dict[str, Any]:
        state = self._read()
        previous = state.get("previous_production")
        production = state.get("production")
        if not previous:
            raise RuntimeError("no previous production checkpoint is available")
        target = self._target(previous)
        self._set_pointer("production", target)
        event = {
            "action": "rollback",
            "at": _utc_now(),
            "from": production,
            "to": str(target),
            "reason": reason,
        }
        state.setdefault("history", []).append(event)
        state.update({
            "production": str(target),
            "previous_production": production,
            "rolled_back_at": event["at"],
            "updated_at": event["at"],
        })
        self._write(state)
        return state
