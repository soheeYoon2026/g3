#!/usr/bin/env python3
"""Run the midnight G3 train, offline gate, shadow stage, and promotion cycle."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aox_g3.model_registry import ModelRegistry
from aox_g3.promotion import (
    evaluate_offline_gates,
    evaluate_shadow_gates,
    load_jsonl,
)


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def case_ids(manifests: list[str]) -> set[str]:
    result = set()
    for raw_path in manifests:
        path = Path(raw_path)
        payload = read_json(path)
        rows = payload.get("cases", payload)
        for row in rows:
            result.add(str(row["case_id"]))
    return result


def render_command(parts: list[str], variables: dict[str, str]) -> list[str]:
    return [part.format_map(variables) for part in parts]


def run_commands(commands: list[list[str]], variables: dict[str, str], cwd: Path) -> None:
    for command in commands:
        subprocess.run(render_command(command, variables), cwd=cwd, check=True)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def maybe_promote(config: dict, registry: ModelRegistry) -> dict:
    state = registry._read()
    challenger = state.get("challenger")
    offline_passed = state.get("challenger_offline_gate_passed", False)
    if not challenger or not offline_passed:
        return {"status": "no-eligible-challenger"}
    policy = config["shadow_gates"]
    report = evaluate_shadow_gates(
        load_jsonl(config["shadow_audit"]),
        challenger_model=Path(challenger).name,
        **policy,
    )
    if not report["passed"]:
        if report["samples"] >= int(policy.get("min_samples", 20)):
            state["challenger"] = None
            state["challenger_offline_gate_passed"] = False
            state["challenger_rejected_at"] = datetime.now(timezone.utc).isoformat()
            state["challenger_shadow_gate"] = report
            registry._write(state)
            (registry.root / "challenger.pt").unlink(missing_ok=True)
            return {"status": "shadow-gate-failed", "report": report}
        return {"status": "shadow-gate-pending", "report": report}
    registry.promote("offline and shadow canary gates passed")
    return {"status": "promoted", "report": report, "checkpoint": challenger}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--record-baseline", action="store_true")
    args = parser.parse_args()

    config = read_json(args.config)
    root = Path(config.get("working_directory", ".")).resolve()
    state_path = root / config.get("state_path", "var/nightly/state.json")
    lock_path = root / config.get("lock_path", "var/nightly/lifecycle.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry = ModelRegistry(root / config.get("registry", "models/registry"))
        promotion = maybe_promote(config, registry)

        sync_variables = {"root": str(root)}
        run_commands(config.get("sync_commands", []), sync_variables, root)
        manifests = [str(root / path) for path in config["dataset_manifests"]]
        current_cases = case_ids(manifests)
        state = read_json(state_path) if state_path.exists() else {}
        trained_cases = set(state.get("trained_case_ids", []))
        new_cases = sorted(current_cases - trained_cases)
        now = datetime.now(timezone.utc)
        result = {
            "started_at": now.isoformat(),
            "promotion": promotion,
            "dataset_cases": len(current_cases),
            "new_cases": len(new_cases),
        }

        if promotion["status"] == "shadow-gate-pending":
            state.update({
                "last_checked_at": now.isoformat(),
                "last_result": "waiting-for-shadow-gate",
                "pending_new_cases": len(new_cases),
            })
            write_json_atomic(state_path, state)
            result["status"] = "waiting-for-shadow-gate"
            print(json.dumps(result, indent=2))
            return

        if not state or args.record_baseline:
            state.update({
                "trained_case_ids": sorted(current_cases),
                "baseline_recorded_at": now.isoformat(),
                "last_result": "baseline-recorded",
            })
            write_json_atomic(state_path, state)
            result["status"] = "baseline-recorded"
            print(json.dumps(result, indent=2))
            return

        minimum = int(config.get("minimum_new_cases", 10))
        if len(new_cases) < minimum:
            state.update({
                "last_checked_at": now.isoformat(),
                "last_result": "insufficient-new-cases",
                "pending_new_cases": len(new_cases),
            })
            write_json_atomic(state_path, state)
            result["status"] = "insufficient-new-cases"
            print(json.dumps(result, indent=2))
            return

        run_id = now.strftime("%Y%m%dT%H%M%SZ")
        run_dir = root / config.get("runs_directory", "var/nightly/runs") / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        variables = {
            "root": str(root),
            "run_id": run_id,
            "run_dir": str(run_dir),
            "challenger": str(run_dir / f"g3-{run_id}.pt"),
            "challenger_report": str(run_dir / "challenger.evaluation.json"),
            "production": str(registry.root / "production.pt"),
            "production_report": str(run_dir / "production.evaluation.json"),
        }
        run_commands(config.get("prepare_commands", []), variables, root)
        run_commands(config["train_commands"], variables, root)
        run_commands(config["evaluate_production_commands"], variables, root)
        run_commands(config["evaluate_challenger_commands"], variables, root)

        production_report = read_json(variables["production_report"])
        challenger_report = read_json(variables["challenger_report"])
        gate = evaluate_offline_gates(
            production_report, challenger_report, config["offline_gates"]
        )
        write_json_atomic(run_dir / "offline-gate.json", gate)
        if gate["passed"]:
            registry_state = registry.stage(
                variables["challenger"], variables["challenger_report"]
            )
            registry_state["challenger_offline_gate_passed"] = True
            registry_state["challenger_offline_gate"] = gate
            registry._write(registry_state)
            status = "challenger-staged"
        else:
            status = "offline-gate-failed"

        # Cases are consumed after a completed training attempt.  A failed gate
        # remains auditable and is not retrained identically every night.
        state.update({
            "trained_case_ids": sorted(current_cases),
            "last_trained_at": now.isoformat(),
            "last_run_id": run_id,
            "last_result": status,
            "offline_gate": gate,
        })
        write_json_atomic(state_path, state)
        result.update({"status": status, "run_id": run_id, "offline_gate": gate})
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
