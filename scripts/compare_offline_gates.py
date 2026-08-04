#!/usr/bin/env python3
"""Compare production and challenger reports using configured offline gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aox_g3.promotion import evaluate_offline_gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    production = json.loads(args.production.read_text())
    challenger = json.loads(args.challenger.read_text())
    report = evaluate_offline_gates(
        production, challenger, config["offline_gates"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
