#!/usr/bin/env python3
"""Convert G2/SU2 VTU results into compact G3 field-training NPZ cases."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

from aox_g3.geometry.surface_sampling import (
    GEOMETRY_PREPROCESSING_VERSION,
    interpolate_vertex_values,
    sample_triangle_surface,
)

try:
    import vtk
    from vtk.util import numpy_support as nps
except ImportError as exc:  # pragma: no cover
    raise SystemExit("VTK is required: pip install vtk") from exc


def _read_vtu(path: Path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    if data is None or data.GetNumberOfPoints() == 0:
        raise ValueError(f"empty or unreadable VTU: {path}")
    return data


def _array(data, *names: str) -> np.ndarray | None:
    arrays = data.GetPointData()
    lookup = {str(arrays.GetArrayName(i)).lower(): arrays.GetArray(i)
              for i in range(arrays.GetNumberOfArrays())}
    for name in names:
        arr = lookup.get(name.lower())
        if arr is not None:
            return np.asarray(nps.vtk_to_numpy(arr))
    return None


def _points(data) -> np.ndarray:
    return np.asarray(nps.vtk_to_numpy(data.GetPoints().GetData()), dtype=np.float64)


def _triangles(poly) -> np.ndarray:
    """Return triangle indices, fan-triangulating polygonal VTK cells."""
    result = []
    for cell_index in range(poly.GetNumberOfCells()):
        cell = poly.GetCell(cell_index)
        ids = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]
        result.extend(
            [ids[0], ids[i], ids[i + 1]] for i in range(1, len(ids) - 1)
        )
    if not result:
        raise ValueError("surface mesh contains no triangles")
    return np.asarray(result, dtype=np.int64)


def _surface_with_normals(data):
    surface = vtk.vtkDataSetSurfaceFilter()
    surface.SetInputData(data)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(surface.GetOutputPort())
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOff()
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.Update()
    result = normals.GetOutput()
    n = _array(result, "Normals")
    if n is None:
        raise ValueError("failed to compute surface normals")
    return result, np.asarray(n, dtype=np.float64)


def volume_reference_pressure(volume) -> float:
    """Match the far-field pressure reference used for pressure-only outputs."""
    pressure = _array(volume, "Pressure")
    if pressure is None:
        return 0.0
    points = _points(volume)
    center = (points.min(0) + points.max(0)) / 2.0
    radius = 0.35 * np.max(points.max(0) - points.min(0))
    outer = np.linalg.norm(points - center, axis=1) > radius
    return float(np.median(pressure[outer])) if outer.any() else float(np.median(pressure))


def surface_force_coefficients(
    surface,
    ref_area: float,
    flow_direction: np.ndarray,
    q_ref: float = 1.0,
    p_ref: float = 0.0,
) -> dict[str, float]:
    """Integrate SU2 surface Cp and skin friction into Cd, Cy, and Cl."""
    if ref_area <= 0.0:
        raise ValueError("REF_AREA must be positive for force coefficients")
    points = _points(surface)
    cp = _array(surface, "Pressure_Coefficient", "Cp")
    if cp is None:
        pressure = _array(surface, "Pressure")
        if pressure is None:
            raise ValueError("surface pressure data is required for force coefficients")
        cp = (pressure - p_ref) / max(q_ref, 1e-12)
    cf = _array(surface, "Skin_Friction_Coefficient")
    pressure_integral = np.zeros(3, dtype=np.float64)
    friction_integral = np.zeros(3, dtype=np.float64)
    for cell_index in range(surface.GetNumberOfCells()):
        cell = surface.GetCell(cell_index)
        ids = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]
        if len(ids) < 3:
            continue
        triangles = ([ids] if len(ids) == 3
                     else [[ids[0], ids[i], ids[i + 1]] for i in range(1, len(ids) - 1)])
        for triangle in triangles:
            vertices = points[triangle]
            area_vector = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]) / 2.0
            area = float(np.linalg.norm(area_vector))
            if area <= 0.0:
                continue
            pressure_integral += float(np.mean(cp[triangle])) * area_vector
            if cf is not None:
                friction_integral += np.mean(cf[triangle], axis=0) * area
    pressure_force = -pressure_integral / ref_area
    friction_force = friction_integral / ref_area
    force = pressure_force + friction_force
    # Repair a globally reversed VTK surface winding.  The SU2 convention is
    # positive drag along the freestream direction.
    if float(np.dot(force, flow_direction)) < 0.0:
        pressure_force *= -1.0
        force = pressure_force + friction_force
    return {
        "cd": float(np.dot(force, flow_direction)),
        "cy": float(force[1]),
        "cl": float(force[2]),
        "pressure_cd": float(np.dot(pressure_force, flow_direction)),
        "friction_cd": float(np.dot(friction_force, flow_direction)),
    }


def _cfg_value(text: str, name: str, default: float) -> float:
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*([^%\n]+)", text, re.MULTILINE)
    return float(match.group(1).strip()) if match else float(default)


def _cfg_vector(text: str, name: str, default=(1.0, 0.0, 0.0)) -> np.ndarray:
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*\(([^)]+)\)", text, re.MULTILINE)
    if not match:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v.strip()) for v in match.group(1).split(",")], dtype=np.float64)


def read_conditions(cfg: Path) -> dict[str, float]:
    text = cfg.read_text(errors="replace")
    velocity = _cfg_vector(text, "INC_VELOCITY_INIT")
    return {
        "u_x": float(velocity[0]), "u_y": float(velocity[1]), "u_z": float(velocity[2]),
        "density": _cfg_value(text, "INC_DENSITY_INIT", 1.225),
        "viscosity": _cfg_value(text, "MU_CONSTANT", 1.7894e-5),
        "temperature": _cfg_value(text, "INC_TEMPERATURE_INIT", 288.15),
        "ref_length": _cfg_value(text, "REF_LENGTH", 1.0),
        "ref_area": _cfg_value(text, "REF_AREA", 1.0),
    }


def _sample_indices(
    points: np.ndarray,
    body_lo: np.ndarray,
    body_hi: np.ndarray,
    flow_direction: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Near-body/wake-biased subsampling plus global coverage."""
    n = len(points)
    if n <= count:
        return np.arange(n)
    span = float(np.max(body_hi - body_lo)) or 1.0
    near_lo, near_hi = body_lo - 0.35 * span, body_hi + 0.35 * span
    near = np.flatnonzero(np.all((points >= near_lo) & (points <= near_hi), axis=1))

    axis = int(np.argmax(np.abs(flow_direction)))
    sign = 1.0 if flow_direction[axis] >= 0 else -1.0
    wake_lo, wake_hi = body_lo - 0.75 * span, body_hi + 0.75 * span
    if sign > 0:
        wake_lo[axis] = body_hi[axis]
        wake_hi[axis] = body_hi[axis] + 4.0 * span
    else:
        wake_hi[axis] = body_lo[axis]
        wake_lo[axis] = body_lo[axis] - 4.0 * span
    wake = np.flatnonzero(np.all((points >= wake_lo) & (points <= wake_hi), axis=1))

    def choose(pool, k):
        if len(pool) == 0 or k <= 0:
            return np.empty(0, dtype=np.int64)
        return rng.choice(pool, size=min(k, len(pool)), replace=False)

    selected = np.unique(np.concatenate([
        choose(near, count // 2),
        choose(wake, count // 4),
        choose(np.arange(n), count),
    ]))
    if len(selected) > count:
        selected = rng.choice(selected, size=count, replace=False)
    elif len(selected) < count:
        remaining = np.setdiff1d(np.arange(n), selected, assume_unique=False)
        selected = np.concatenate([selected, choose(remaining, count - len(selected))])
    return selected


def prepare_case(
    case_dir: Path,
    output: Path,
    n_volume: int,
    n_surface: int,
    seed: int,
) -> dict:
    volume_path = case_dir / "flow.vtu"
    surface_path = case_dir / "surface_flow.vtu"
    cfg_candidates = sorted(case_dir.glob("*CFD.cfg")) or sorted(case_dir.glob("*.cfg"))
    if not volume_path.exists() or not surface_path.exists() or not cfg_candidates:
        raise FileNotFoundError(f"case needs flow.vtu, surface_flow.vtu and a CFG: {case_dir}")

    conditions = read_conditions(cfg_candidates[0])
    u = np.asarray([conditions["u_x"], conditions["u_y"], conditions["u_z"]])
    u_ref = float(np.linalg.norm(u)) or 1.0
    flow_dir = u / u_ref
    rho = conditions["density"]
    q_ref = 0.5 * rho * u_ref * u_ref

    surface_raw = _read_vtu(surface_path)
    surface, _ = _surface_with_normals(surface_raw)
    surface_points_raw = _points(surface)
    surface_faces = _triangles(surface)
    body_lo, body_hi = surface_points_raw.min(0), surface_points_raw.max(0)
    center = (body_lo + body_hi) / 2.0
    scale = float(np.max(body_hi - body_lo)) or 1.0

    volume = _read_vtu(volume_path)
    volume_points_raw = _points(volume)
    volume_velocity = _array(volume, "Velocity")
    if volume_velocity is None:
        raise ValueError(f"Velocity field missing: {volume_path}")
    volume_cp = _array(volume, "Pressure_Coefficient", "Cp")
    volume_pressure = _array(volume, "Pressure")
    if volume_cp is None:
        if volume_pressure is None:
            raise ValueError(f"pressure field missing: {volume_path}")
        p_ref = volume_reference_pressure(volume)
        volume_cp = (volume_pressure - p_ref) / q_ref
    else:
        p_ref = 0.0

    surface_cp = _array(surface, "Pressure_Coefficient", "Cp")
    if surface_cp is None:
        surface_pressure = _array(surface, "Pressure")
        if surface_pressure is None:
            raise ValueError(f"surface pressure field missing: {surface_path}")
        surface_cp = (surface_pressure - p_ref) / q_ref
    coefficients = surface_force_coefficients(
        surface, conditions["ref_area"], flow_dir, q_ref=q_ref, p_ref=p_ref
    )

    finite = np.isfinite(volume_points_raw).all(1) & np.isfinite(volume_velocity).all(1) & np.isfinite(volume_cp)
    volume_points_raw = volume_points_raw[finite]
    volume_velocity = np.asarray(volume_velocity[finite], dtype=np.float64)
    volume_cp = np.asarray(volume_cp[finite], dtype=np.float64).reshape(-1)

    rng = np.random.default_rng(seed)
    vi = _sample_indices(volume_points_raw, body_lo, body_hi, flow_dir, n_volume, rng)
    surface_samples = sample_triangle_surface(
        surface_points_raw, surface_faces, n_surface, seed=seed + 1_000_003
    )
    sampled_surface_cp = interpolate_vertex_values(
        np.asarray(surface_cp).reshape(-1), surface_faces, surface_samples
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        geometry_points=((surface_samples.points - center) / scale).astype(np.float32),
        geometry_normals=surface_samples.normals.astype(np.float32),
        volume_points=((volume_points_raw[vi] - center) / scale).astype(np.float32),
        volume_velocity=(volume_velocity[vi] / u_ref).astype(np.float32),
        volume_cp=volume_cp[vi].astype(np.float32),
        surface_points=((surface_samples.points - center) / scale).astype(np.float32),
        surface_cp=np.asarray(sampled_surface_cp, dtype=np.float32).reshape(-1),
        center=center.astype(np.float64), scale=np.asarray(scale),
        geometry_preprocessing_version=np.asarray(GEOMETRY_PREPROCESSING_VERSION),
    )
    reynolds = rho * u_ref * conditions["ref_length"] / conditions["viscosity"]
    return {
        "case_id": case_dir.name,
        "npz": output.name,
        "conditions": conditions,
        "normalization": {"center": center.tolist(), "scale": scale, "u_ref": u_ref,
                          "q_ref": q_ref, "p_ref": p_ref},
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "geometry_sampling": "area_weighted_triangle_face_normal",
        "reynolds": reynolds,
        "coefficients": coefficients,
        "source": {"case_dir": str(case_dir), "volume": str(volume_path),
                   "surface": str(surface_path), "cfg": str(cfg_candidates[0])},
        "counts": {"volume": int(len(vi)), "surface": int(len(surface_samples.points))},
    }


def case_identity(case_dir: Path) -> tuple[str, str]:
    if case_dir.name.upper() == "DIRECT" and case_dir.parent.name.upper().startswith("DSN_"):
        designs_dir = case_dir.parent.parent
        if designs_dir.name == "DESIGNS":
            family = designs_dir.parent.name
        elif designs_dir.name.startswith("DESIGNS_"):
            # Some older runs put DSN folders directly below a named
            # ``DESIGNS_rs6``/``DESIGNS_sidemirror`` root.  Treat that root as
            # one family so a group split cannot leak adjacent optimization
            # designs into both train and validation sets.
            family = designs_dir.name
        else:
            family = designs_dir.parent.name
        return f"{family}__{case_dir.parent.name}", family
    return f"{case_dir.parent.name}__{case_dir.name}", case_dir.parent.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--case-glob", action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--group-id", default=None)
    parser.add_argument("--n-volume", type=int, default=200_000)
    parser.add_argument("--n-surface", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-invalid", action="store_true",
                        help="record unreadable/incomplete cases and continue")
    args = parser.parse_args()

    paths = [Path(p).resolve() for p in args.case]
    for pattern in args.case_glob:
        paths.extend(Path(p).resolve() for p in sorted(glob.glob(pattern)))
    paths = list(dict.fromkeys(paths))
    if not paths:
        parser.error("provide --case and/or --case-glob")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for index, case_dir in enumerate(paths):
        case_id, inferred_group = case_identity(case_dir)
        output = args.out_dir / f"{case_id}.npz"
        print(f"[{index + 1}/{len(paths)}] {case_dir}")
        try:
            row = prepare_case(case_dir, output, args.n_volume, args.n_surface, args.seed + index)
        except Exception as exc:
            if not args.skip_invalid:
                raise
            skipped.append({"case_dir": str(case_dir), "error": f"{type(exc).__name__}: {exc}"})
            print(f"  SKIP: {type(exc).__name__}: {exc}")
            continue
        row["case_id"] = case_id
        row["group_id"] = args.group_id or inferred_group
        row["npz"] = str(output.resolve().relative_to(args.manifest.parent.resolve())) \
            if output.resolve().is_relative_to(args.manifest.parent.resolve()) else str(output.resolve())
        rows.append(row)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({
        "schema_version": 1,
        "geometry_preprocessing_version": GEOMETRY_PREPROCESSING_VERSION,
        "cases": rows,
        "skipped": skipped,
    }, indent=2))
    print(f"Wrote {len(rows)} cases ({len(skipped)} skipped) -> {args.manifest}")


if __name__ == "__main__":
    main()
