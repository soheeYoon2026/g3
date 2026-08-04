#!/usr/bin/env python3
"""Freeze each deployed checkpoint's held-out expert cases for future gates."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        nargs="+",
        action="append",
        metavar="VALUE",
        required=True,
        help="LABEL EXPERT OUTPUT SOURCE [SOURCE ...]",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    for specification in args.dataset:
        if len(specification) < 4:
            parser.error("--dataset requires LABEL EXPERT OUTPUT and at least one SOURCE")
        label, expert_name, output_raw, *sources_raw = specification
        expert = checkpoint.get("coefficient_experts", {}).get(expert_name)
        if expert is None:
            raise KeyError(f"checkpoint has no expert {expert_name!r}")
        validation_ids = [str(value) for value in expert.get("val_cases", [])]
        available = {}
        for raw_source in sources_raw:
            source = Path(raw_source).resolve()
            payload = json.loads(source.read_text())
            source_rows = payload["cases"] if isinstance(payload, dict) else payload
            for original in source_rows:
                row = copy.deepcopy(original)
                npz = Path(row["npz"])
                if not npz.is_absolute():
                    npz = (source.parent / npz).resolve()
                row["npz"] = str(npz)
                available[str(row["case_id"])] = row
        missing = [case_id for case_id in validation_ids if case_id not in available]
        if missing:
            raise ValueError(f"{label}: validation cases missing from sources: {missing}")
        output = Path(output_raw)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "schema_version": 1,
            "format": "g3-frozen-validation-v1",
            "label": label,
            "expert": expert_name,
            "checkpoint": str(args.checkpoint.resolve()),
            "sources": [str(Path(path).resolve()) for path in sources_raw],
            "cases": [available[case_id] for case_id in validation_ids],
        }, indent=2) + "\n")
        print(f"{label}: froze {len(validation_ids)} cases -> {output}")


if __name__ == "__main__":
    main()
