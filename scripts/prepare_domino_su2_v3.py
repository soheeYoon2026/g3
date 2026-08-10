#!/usr/bin/env python
"""Convert SU2 surface results to a flow-aligned DoMINO dataset.

Unlike the legacy converter, this module preserves Mach/AoA/sideslip and rotates
the geometry and vector fields into a canonical frame where the inlet velocity
always points along +X.  The original dataset is never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


def _setting(text: str, key: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*(.+)$", text, re.I | re.M)
    return match.group(1).split("%", 1)[0].strip() if match else None


def _number(text: str | None, default: float | None = None) -> float | None:
    if text is None:
        return default
    values = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return float(values[0]) if values else default


def flow_frame(aoa_deg: float, sideslip_deg: float) -> np.ndarray:
    """Return world-to-flow rotation with rows (drag, side, lift)."""
    alpha = math.radians(aoa_deg)
    beta = math.radians(sideslip_deg)
    ca, sa, cb, sb = math.cos(alpha), math.sin(alpha), math.cos(beta), math.sin(beta)
    drag = np.array([ca * cb, sb, sa * cb], dtype=np.float64)
    side = np.array([-ca * sb, cb, -sa * sb], dtype=np.float64)
    lift = np.array([-sa, 0.0, ca], dtype=np.float64)
    rotation = np.stack((drag, side, lift))
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12):
        raise ValueError("invalid aerodynamic rotation")
    return rotation


def flow_frame_from_velocity(velocity: np.ndarray) -> np.ndarray:
    """Return a stable world-to-flow frame using an explicit velocity vector."""
    velocity = np.asarray(velocity, dtype=np.float64)
    speed = float(np.linalg.norm(velocity))
    if speed <= 0:
        raise ValueError("zero inlet velocity")
    drag = velocity / speed
    world_side = np.array([0.0, 1.0, 0.0])
    side = world_side - np.dot(world_side, drag) * drag
    if np.linalg.norm(side) < 1e-10:
        fallback = np.array([0.0, 0.0, 1.0])
        side = fallback - np.dot(fallback, drag) * drag
    side /= np.linalg.norm(side)
    lift = np.cross(drag, side)
    return np.stack((drag, side, lift))


def read_flow_conditions(cfg_path: str | Path) -> dict[str, object]:
    text = Path(cfg_path).read_text(errors="ignore")
    mach = _number(_setting(text, "MACH_NUMBER"))
    configured_aoa = _number(_setting(text, "AOA"))
    configured_sideslip = _number(_setting(text, "SIDESLIP_ANGLE"))
    density = float(_number(_setting(text, "INC_DENSITY_INIT"), 1.225))
    ref_area = _number(_setting(text, "REF_AREA"))

    raw_velocity = _setting(text, "INC_VELOCITY_INIT")
    if raw_velocity:
        components = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw_velocity)]
        if len(components) != 3:
            raise ValueError(f"expected 3 INC_VELOCITY_INIT components in {cfg_path}")
        velocity = np.asarray(components, dtype=np.float64)
        speed = float(np.linalg.norm(velocity))
        rotation = flow_frame_from_velocity(velocity)
        aoa = math.degrees(math.atan2(velocity[2], velocity[0]))
        sideslip = math.degrees(math.asin(np.clip(velocity[1] / speed, -1.0, 1.0)))
    elif mach is not None:
        speed = float(mach * 340.3)
        aoa = float(configured_aoa or 0.0)
        sideslip = float(configured_sideslip or 0.0)
        rotation = flow_frame(aoa, sideslip)
        velocity = rotation[0] * speed
    else:
        raise ValueError(f"no velocity or Mach number in {cfg_path}")
    return {
        "speed": speed,
        "velocity": velocity,
        "mach": mach,
        "aoa": aoa,
        "sideslip": sideslip,
        "configured_aoa": configured_aoa,
        "configured_sideslip": configured_sideslip,
        "density": density,
        "ref_area": ref_area,
        "rotation": rotation,
    }


def read_su2_coefficients(case_dir: str | Path) -> tuple[float, float]:
    for path in sorted(Path(case_dir).glob("history*.csv")):
        with path.open(newline="") as stream:
            rows = list(csv.reader(stream))
        if len(rows) < 2:
            continue
        header = [value.strip().strip('"') for value in rows[0]]
        if "CD" in header:
            cd_index = header.index("CD")
            cl_index = header.index("CL") if "CL" in header else None
            return float(rows[-1][cd_index]), (
                float(rows[-1][cl_index]) if cl_index is not None else float("nan")
            )
    return float("nan"), float("nan")


def convert_case(vtu_path: str | Path, cfg_path: str | Path, out_root: str | Path, run_id: str) -> dict[str, object]:
    import pyvista as pv

    vtu_path, cfg_path = Path(vtu_path), Path(cfg_path)
    flow = read_flow_conditions(cfg_path)
    mesh = pv.read(vtu_path)
    if "Pressure_Coefficient" not in mesh.point_data or "Skin_Friction_Coefficient" not in mesh.point_data:
        raise ValueError(f"missing Cp/Cf in {vtu_path}")

    cp = np.asarray(mesh.point_data["Pressure_Coefficient"], np.float64).reshape(-1)
    cf_world = np.asarray(mesh.point_data["Skin_Friction_Coefficient"], np.float64)
    if not np.isfinite(cp).all() or not np.isfinite(cf_world).all():
        raise ValueError(f"non-finite Cp/Cf in {vtu_path}")
    if float(np.mean(np.abs(cp) > 5.0)) > 0.01:
        raise ValueError(f"diverged Cp in {vtu_path}")

    rotation = np.asarray(flow["rotation"])
    mesh.points = np.asarray(mesh.points) @ rotation.T
    cf_flow = cf_world @ rotation.T
    q_kinematic = 0.5 * float(flow["speed"]) ** 2
    mesh.point_data["pMeanTrim"] = (np.clip(cp, -8.0, 8.0) * q_kinematic).astype(np.float32)
    mesh.point_data["wallShearStressMeanTrim"] = (cf_flow * q_kinematic).astype(np.float32)

    surface = mesh.extract_surface().triangulate()
    cell = surface.point_data_to_cell_data()
    output = pv.PolyData(surface.points, surface.faces)
    output.cell_data["pMeanTrim"] = np.asarray(cell.cell_data["pMeanTrim"], np.float32).reshape(-1)
    output.cell_data["wallShearStressMeanTrim"] = np.asarray(
        cell.cell_data["wallShearStressMeanTrim"], np.float32
    )

    run_dir = Path(out_root) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output.save(run_dir / f"boundary_{run_id}.vtp")
    surface.save(run_dir / f"drivaer_{run_id}.stl")
    cd, cl = read_su2_coefficients(vtu_path.parent)
    metadata = {
        "speed": flow["speed"],
        "velocity_world": np.asarray(flow["velocity"]).tolist(),
        "mach": flow["mach"],
        "aoa": flow["aoa"],
        "sideslip": flow["sideslip"],
        "density": flow["density"],
        "ref_area": flow["ref_area"],
        "su2_cd": cd,
        "su2_cl": cl,
        "flow_aligned": True,
        "source_vtu": str(vtu_path),
        "source_cfg": str(cfg_path),
    }
    (run_dir / f"conditions_{run_id}.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vtu", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    print(json.dumps(convert_case(args.vtu, args.cfg, args.out, args.run), indent=2))


if __name__ == "__main__":
    main()
