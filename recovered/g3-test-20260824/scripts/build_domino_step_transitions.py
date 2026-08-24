#!/usr/bin/env python3
"""Build ordered G2 optimisation-step transitions from a DoMINO manifest.

RBF_DSN_001 is the submitted/initial design, subsequent RBF_DSN entries are
ordered optimisation designs, and FINAL is the terminal result.  Transitions
are split by parent G2 job so one trajectory cannot leak across partitions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


RBF_PATTERN = re.compile(r"^RBF_DSN_(\d+)$")


def source_parts(case: dict) -> tuple[str, str]:
    source_id = str(case.get("source", {}).get("source_id", ""))
    if ":" not in source_id:
        raise ValueError(f"case run_{case.get('run')} has no source design")
    return tuple(source_id.split(":", 1))


def design_order(design: str) -> tuple[int, int]:
    match = RBF_PATTERN.match(design)
    if match:
        return 0, int(match.group(1))
    if design == "FINAL":
        return 1, 0
    raise ValueError(f"unsupported G2 design label: {design}")


def coefficient(case: dict, name: str) -> float:
    value = case.get("conditions", {}).get(f"su2_{name}")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"run_{case.get('run')} has invalid {name}: {value}")
    return float(value)


def geometry_files(root: Path, case: dict) -> dict:
    run = int(case["run"])
    directory = root / f"run_{run}"
    return {
        "run": run,
        "stl": str(directory / f"drivaer_{run}.stl"),
        "surface": str(directory / f"boundary_{run}.vtp"),
        "conditions": str(directory / f"conditions_{run}.json"),
        "geometry_digest": case.get("geometry_digest"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation-jobs", type=int, default=4)
    parser.add_argument("--test-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for case in manifest["cases"]:
        job, design = source_parts(case)
        grouped[job].append((design, case))

    transitions_by_job: dict[str, list[dict]] = {}
    rejected = []
    for job, values in grouped.items():
        try:
            ordered = sorted(values, key=lambda item: design_order(item[0]))
            rbf_numbers = [design_order(design)[1] for design, _ in ordered if design != "FINAL"]
            if not rbf_numbers or rbf_numbers[0] != 1:
                raise ValueError("trajectory does not start at RBF_DSN_001")
            if rbf_numbers != list(range(1, max(rbf_numbers) + 1)):
                raise ValueError(f"trajectory has missing RBF steps: {rbf_numbers}")
            if ordered[-1][0] != "FINAL":
                raise ValueError("trajectory has no FINAL design")
            rows = []
            for index, ((from_design, before), (to_design, after)) in enumerate(zip(ordered, ordered[1:]), start=1):
                from_cd, to_cd = coefficient(before, "cd"), coefficient(after, "cd")
                from_cl, to_cl = coefficient(before, "cl"), coefficient(after, "cl")
                rows.append({
                    "transition_id": f"{job}:{from_design}->{to_design}",
                    "source_job_uid": job,
                    "source_kind": "g2_rbf_step",
                    "trajectory_step": index,
                    "from_design": from_design,
                    "to_design": to_design,
                    "from_geometry": geometry_files(root, before),
                    "to_geometry": geometry_files(root, after),
                    "from_cd": from_cd,
                    "to_cd": to_cd,
                    "delta_cd": to_cd - from_cd,
                    "from_cl": from_cl,
                    "to_cl": to_cl,
                    "delta_cl": to_cl - from_cl,
                    "conditions": {
                        key: before.get("conditions", {}).get(key)
                        for key in ("speed", "density", "ref_area", "aoa", "sideslip")
                    },
                })
            transitions_by_job[job] = rows
        except ValueError as error:
            rejected.append({"source_job_uid": job, "reason": str(error)})

    jobs = sorted(transitions_by_job)
    random.Random(args.seed).shuffle(jobs)
    if args.validation_jobs + args.test_jobs >= len(jobs):
        raise SystemExit("validation and test job counts leave no training jobs")
    validation = set(jobs[: args.validation_jobs])
    test = set(jobs[args.validation_jobs : args.validation_jobs + args.test_jobs])
    train = set(jobs) - validation - test

    def rows(selected: set[str]) -> list[dict]:
        return [row for job in jobs if job in selected for row in transitions_by_job[job]]

    case_by_run = {int(case["run"]): case for case in manifest["cases"]}

    def cases(selected: set[str]) -> list[dict]:
        run_ids = {
            int(geometry["run"])
            for transition in rows(selected)
            for geometry in (transition["from_geometry"], transition["to_geometry"])
        }
        return [case_by_run[run] for run in sorted(run_ids)]

    result = {
        "schema_version": 1,
        "format": "g3-domino-step-transitions-v1",
        "source_manifest": str(manifest_path),
        "ordering": "RBF_DSN_001 -> ... -> RBF_DSN_N -> FINAL",
        "seed": args.seed,
        "jobs": len(transitions_by_job),
        "transitions": sum(len(value) for value in transitions_by_job.values()),
        "train_jobs": len(train),
        "validation_jobs": len(validation),
        "test_jobs": len(test),
        "train_transitions": rows(train),
        "validation_transitions": rows(validation),
        "test_transitions": rows(test),
        "train_cases": cases(train),
        "validation_cases": cases(validation),
        "test_cases": cases(test),
        "rejected_jobs": rejected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "jobs", "transitions", "train_jobs", "validation_jobs", "test_jobs")}, indent=2))
    print(json.dumps({
        "train_transitions": len(result["train_transitions"]),
        "validation_transitions": len(result["validation_transitions"]),
        "test_transitions": len(result["test_transitions"]),
        "rejected_jobs": len(rejected),
    }, indent=2))


if __name__ == "__main__":
    main()
