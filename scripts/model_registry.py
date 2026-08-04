#!/usr/bin/env python3
"""Manage atomic G3 production/challenger checkpoint pointers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aox_g3.model_registry import ModelRegistry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="models/registry")
    subparsers = parser.add_subparsers(dest="action", required=True)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("checkpoint")
    stage = subparsers.add_parser("stage")
    stage.add_argument("checkpoint")
    stage.add_argument("--report")
    promote = subparsers.add_parser("promote")
    promote.add_argument("--reason", required=True)
    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("--reason", required=True)
    subparsers.add_parser("status")
    args = parser.parse_args()

    registry = ModelRegistry(args.registry)
    if args.action == "initialize":
        state = registry.initialize(args.checkpoint)
    elif args.action == "stage":
        state = registry.stage(args.checkpoint, args.report)
    elif args.action == "promote":
        state = registry.promote(args.reason)
    elif args.action == "rollback":
        state = registry.rollback(args.reason)
    else:
        state = registry._read()
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
