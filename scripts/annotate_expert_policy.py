#!/usr/bin/env python3
"""Add deployment policy metadata to a multi-expert checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-validated-cases", type=int, default=20)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    for name, expert in checkpoint.get("coefficient_experts", {}).items():
        count = len(expert.get("train_cases", [])) + len(expert.get("val_cases", []))
        status = "validated" if count >= args.min_validated_cases else "experimental"
        expert["deployment_status"] = status
        expert["deployment_policy"] = {
            "training_cases": count,
            "minimum_validated_cases": args.min_validated_cases,
            "requires_solver_verification": status == "experimental",
        }
        print(f"{name}: {count} cases -> {status}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
