#!/usr/bin/env python3
"""Add integrated Cd/Cy/Cl labels from raw G2 surface results to a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prepare_g2_fields import _read_vtu, surface_force_coefficients, volume_reference_pressure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    rows = payload["cases"] if isinstance(payload, dict) else payload
    for number, row in enumerate(rows, 1):
        conditions = row["conditions"]
        velocity = np.asarray([
            conditions["u_x"], conditions["u_y"], conditions["u_z"]
        ], dtype=float)
        speed = float(np.linalg.norm(velocity))
        if speed <= 0.0:
            raise ValueError(f"{row['case_id']}: zero freestream velocity")
        surface = _read_vtu(Path(row["source"]["surface"]))
        volume = _read_vtu(Path(row["source"]["volume"]))
        q_ref = 0.5 * float(conditions["density"]) * speed**2
        row["coefficients"] = surface_force_coefficients(
            surface, float(conditions["ref_area"]), velocity / speed,
            q_ref=q_ref, p_ref=volume_reference_pressure(volume),
        )
        print(f"[{number}/{len(rows)}] {row['case_id']}: "
              f"Cd={row['coefficients']['cd']:.6f} Cl={row['coefficients']['cl']:.6f}")
    args.manifest.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
