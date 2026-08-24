#!/usr/bin/env python3
"""Rank smooth surface-control moves with the resident DoMINO model."""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np
import pyvista as pv

from scripts.domino_resident_engine import ResidentDominoEngine


def _lateral_axis(flow_axis: str) -> int:
    return 1 if flow_axis[-1] == "x" else 0


def _sample_controls(points: np.ndarray, count: int, lateral_axis: int) -> np.ndarray:
    center = (points.min(0) + points.max(0)) * 0.5
    pool = np.flatnonzero(points[:, lateral_axis] >= center[lateral_axis])
    if pool.size == 0:
        pool = np.arange(len(points))
    # Seed at the most forward point, then use farthest-point sampling so the
    # controls cover the actual surface instead of named vehicle sections.
    selected = [int(pool[np.argmin(points[pool, 0])])]
    minimum_distance = np.full(pool.size, np.inf)
    for _ in range(1, min(count, pool.size)):
        delta = points[pool] - points[selected[-1]]
        minimum_distance = np.minimum(minimum_distance, np.einsum("ij,ij->i", delta, delta))
        selected.append(int(pool[np.argmax(minimum_distance)]))
    return np.asarray(selected, dtype=np.int64)


def _deform(
    mesh: pv.PolyData,
    origin: np.ndarray,
    displacement: np.ndarray,
    radius: float,
    lateral_axis: int,
    symmetric: bool,
) -> pv.PolyData:
    result = mesh.copy(deep=True)
    points = np.asarray(result.points, dtype=np.float64)
    center = (points.min(0) + points.max(0)) * 0.5
    origins = [origin]
    moves = [displacement]
    mirrored = origin.copy()
    mirrored[lateral_axis] = 2 * center[lateral_axis] - origin[lateral_axis]
    if symmetric and abs(mirrored[lateral_axis] - origin[lateral_axis]) > radius * 0.05:
        mirrored_move = displacement.copy()
        mirrored_move[lateral_axis] *= -1
        origins.append(mirrored)
        moves.append(mirrored_move)
    total = np.zeros_like(points)
    for source, move in zip(origins, moves, strict=True):
        distance = np.linalg.norm(points - source, axis=1)
        weight = np.square(np.clip(1.0 - distance / radius, 0.0, 1.0))
        total += weight[:, None] * move
    result.points = points + total
    return result


def _preview_points(points: np.ndarray, origin: np.ndarray, radius: float) -> list[list[float]]:
    local = points[np.linalg.norm(points - origin, axis=1) <= radius * 1.25]
    if len(local) > 420:
        local = local[np.linspace(0, len(local) - 1, 420, dtype=np.int64)]
    return np.round((local - origin) / radius, 4).tolist()


def _overview_points(points: np.ndarray) -> list[list[float]]:
    sampled = points
    if len(sampled) > 900:
        sampled = sampled[np.linspace(0, len(sampled) - 1, 900, dtype=np.int64)]
    minimum, maximum = points.min(0), points.max(0)
    center = (minimum + maximum) * 0.5
    scale = max(float(np.max(maximum - minimum)) * 0.5, 1e-9)
    return np.round((sampled - center) / scale, 4).tolist()


def _render_recommendation(mesh: pv.PolyData, row: dict, path: Path) -> str | None:
    """Render a real-mesh overview and close-up; never fail recommendation."""
    try:
        origin = np.asarray(row["position"], dtype=float)
        displacement = np.asarray(row["displacement"], dtype=float)
        surface_normal = np.asarray(row["surface_normal"], dtype=float)
        surface_normal /= max(float(np.linalg.norm(surface_normal)), 1e-9)
        radius = float(row["influence_radius"])
        center = np.asarray(mesh.center, dtype=float)
        length = max(float(mesh.length), radius)
        move_end = origin + displacement
        surface_points = np.asarray(mesh.points)
        display_origins = [origin]
        display_moves = [displacement]
        if row.get("symmetric", True):
            lateral_axis = int(row.get("lateral_axis", 1))
            mirrored_origin = origin.copy()
            mirrored_origin[lateral_axis] = 2 * center[lateral_axis] - origin[lateral_axis]
            if abs(mirrored_origin[lateral_axis] - origin[lateral_axis]) > radius * 0.05:
                mirrored_move = displacement.copy()
                mirrored_move[lateral_axis] *= -1
                display_origins.append(mirrored_origin)
                display_moves.append(mirrored_move)
        distance = np.min(
            np.stack([np.linalg.norm(surface_points - source, axis=1) for source in display_origins]),
            axis=0,
        )
        influence = np.square(np.clip(1.0 - distance / radius, 0.0, 1.0))
        colored_mesh = mesh.copy(deep=True)
        base_color = np.array([135.0, 147.0, 160.0])
        influence_color = np.array([249.0, 115.0, 22.0])
        colors = base_color[None, :] * (1.0 - influence[:, None]) + influence_color[None, :] * influence[:, None]
        colored_mesh.point_data["recommendation_rgb"] = colors.astype(np.uint8)
        plotter = pv.Plotter(off_screen=True, window_size=(900, 480))
        plotter.set_background("#111820")
        plotter.add_mesh(colored_mesh, scalars="recommendation_rgb", rgb=True, smooth_shading=True)
        current_points = np.stack(display_origins)
        target_points = np.stack([source + move for source, move in zip(display_origins, display_moves, strict=True)])
        plotter.add_points(current_points, color="#1687ff", point_size=20, render_points_as_spheres=True)
        plotter.add_points(target_points, color="#ff4d45", point_size=16, render_points_as_spheres=True)
        # Keep a stable vehicle-scale three-quarter view. Looking straight
        # along an arbitrary surface normal can turn a complete car into a
        # nearly edge-on line. Only mirror the view to the side containing
        # the selected control point.
        lateral_side = 1.0 if origin[1] >= center[1] else -1.0
        view_vector = np.array([-1.15, lateral_side * 1.35, 0.72])
        view_vector /= np.linalg.norm(view_vector)
        overview_camera = center + view_vector * length * 2.4
        overview_up = np.array([0.0, 0.0, 1.0])
        plotter.camera_position = [overview_camera, center, overview_up]
        plotter.reset_camera_clipping_range()
        plotter.add_text("LOCATION ON VEHICLE", position="lower_left", color="#d9e1e8", font_size=9)
        plotter.screenshot(path)
        plotter.close()
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception as exc:
        print(json.dumps({"preview_warning": str(exc)}), flush=True)
        return None


def main(engine: ResidentDominoEngine | None = None, argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--density", type=float, default=1.225)
    parser.add_argument("--flow-axis", default="+x")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--controls", type=int, default=6)
    parser.add_argument("--control-points-json", default="")
    parser.add_argument("--control-points-file", type=Path)
    parser.add_argument("--shortlist", type=int, default=8)
    parser.add_argument("--coarse-reduction", type=float, default=0.9)
    parser.add_argument("--distinct-controls", action="store_true",
                        help="Keep only the best direction per surface control before refinement.")
    parser.add_argument("--move-ratio", type=float, default=0.002,
                        help="Maximum control displacement as a fraction of mesh diagonal.")
    parser.add_argument("--radius-ratio", type=float, default=0.08,
                        help="Smooth influence radius as a fraction of mesh diagonal.")
    parser.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    mesh = pv.read(args.stl).extract_surface().triangulate().clean()
    mesh = mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
    points = np.asarray(mesh.points, dtype=np.float64)
    normals = np.asarray(mesh.point_data["Normals"], dtype=np.float64)
    bounds_min, bounds_max = points.min(0), points.max(0)
    diagonal = max(float(np.linalg.norm(bounds_max - bounds_min)), 1e-9)
    if not 0 < args.move_ratio <= 0.01:
        parser.error("--move-ratio must be in (0, 0.01]")
    if not 0 < args.radius_ratio <= 0.25:
        parser.error("--radius-ratio must be in (0, 0.25]")
    if not args.top <= args.shortlist <= 32:
        parser.error("--shortlist must be between --top and 32")
    if not 0.5 <= args.coarse_reduction < 0.98:
        parser.error("--coarse-reduction must be in [0.5, 0.98)")
    radius = diagonal * args.radius_ratio
    move_length = diagonal * args.move_ratio
    lateral_axis = _lateral_axis(args.flow_axis)
    control_points_value = args.control_points_json
    if args.control_points_file:
        file_payload = json.loads(args.control_points_file.read_text(encoding="utf-8"))
        control_points_value = json.dumps(
            file_payload.get("points") if isinstance(file_payload, dict) else file_payload
        )
    if control_points_value:
        requested_points = np.asarray(json.loads(control_points_value), dtype=np.float64)
        if (
            requested_points.ndim != 2
            or requested_points.shape[1] != 3
            or not 1 <= len(requested_points) <= 72
            or not np.isfinite(requested_points).all()
        ):
            parser.error("--control-points-json must contain 1 to 72 finite XYZ points")
        # Workbench points originate on this same STL, but the cached inference
        # mesh may have been reduced. Snap each UI control to the closest
        # surviving surface vertex while preserving its Workbench control id.
        controls = np.asarray([
            int(np.argmin(np.einsum("ij,ij->i", points - point, points - point)))
            for point in requested_points
        ], dtype=np.int64)
    else:
        controls = _sample_controls(points, args.controls, lateral_axis)

    args.out.mkdir(parents=True, exist_ok=True)
    engine = engine or ResidentDominoEngine(args.checkpoint)
    started = time.perf_counter()
    baseline_path = args.out / "baseline.stl"
    mesh.save(baseline_path)
    baseline = {
        "label": "baseline",
        **engine.predict_prepared(engine.prepare(baseline_path, args.flow_axis), args.density, None),
    }
    coarse_baseline_mesh = (
        mesh.decimate_pro(args.coarse_reduction, preserve_topology=True)
        if mesh.n_cells >= 2000 else mesh.copy(deep=True)
    )
    coarse_baseline_path = args.out / "baseline.coarse.stl"
    coarse_baseline_mesh.save(coarse_baseline_path)
    coarse_baseline = engine.predict_prepared(
        engine.prepare(coarse_baseline_path, args.flow_axis), args.density, None
    )
    rows: list[dict] = []

    for control_number, point_index in enumerate(controls):
        origin = points[point_index]
        normal = normals[point_index]
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        for direction in (-1.0, 1.0):
            displacement = normal * move_length * direction
            candidate = _deform(mesh, origin, displacement, radius, lateral_axis, args.symmetric)
            label = f"point_{control_number:02d}_{'in' if direction < 0 else 'out'}"
            path = args.out / f"{label}.stl"
            candidate.save(path)
            coarse_path = args.out / f"{label}.coarse.stl"
            coarse_candidate = (
                candidate.decimate_pro(args.coarse_reduction, preserve_topology=True)
                if candidate.n_cells >= 2000 else candidate
            )
            coarse_candidate.save(coarse_path)
            row = {
                "label": label,
                "control_id": int(control_number),
                "position": np.round(origin, 7).tolist(),
                "surface_normal": np.round(normal, 7).tolist(),
                "displacement": np.round(displacement, 7).tolist(),
                "influence_radius": radius,
                "symmetric": args.symmetric,
                "lateral_axis": lateral_axis,
                "preview_points": _preview_points(points, origin, radius),
                "stl": path.name,
                "_path": path,
                "_coarse_path": coarse_path,
            }
            rows.append(row)

    # Stage 1: run the same model on aggressively reduced surfaces. This
    # considers every Workbench control while keeping preprocessing affordable.
    for row in rows:
        coarse = engine.predict_prepared(
            engine.prepare(row["_coarse_path"], args.flow_axis), args.density, None
        )
        row["coarse_cd"] = coarse["cd"]
        row["coarse_cl"] = coarse["cl"]
        row["coarse_delta_cd"] = coarse["cd"] - coarse_baseline["cd"]
        print(json.dumps({"stage": "coarse", "candidate": row["label"], "control_id": row["control_id"], "cd": coarse["cd"]}), flush=True)

    ranked_coarse = sorted(rows, key=lambda row: row["coarse_delta_cd"])
    if args.distinct_controls:
        distinct = []
        seen_controls = set()
        for row in ranked_coarse:
            if row["control_id"] in seen_controls:
                continue
            seen_controls.add(row["control_id"])
            distinct.append(row)
        ranked_coarse = distinct
    shortlist = ranked_coarse[: args.shortlist]
    # Stage 2: only the coarse shortlist receives full-resolution preprocessing
    # and inference. Final ranking never uses the coarse coefficient.
    for row in shortlist:
        prediction = engine.predict_prepared(
            engine.prepare(row["_path"], args.flow_axis), args.density, None
        )
        row.update(prediction)
        row["delta_cd"] = row["cd"] - baseline["cd"]
        row["delta_cl"] = row["cl"] - baseline["cl"]
        print(json.dumps({"stage": "refine", "candidate": row["label"], "control_id": row["control_id"], "cd": row["cd"]}), flush=True)

    for row in rows:
        row.pop("_path", None)
        row.pop("_coarse_path", None)
    recommendations = sorted(shortlist, key=lambda row: row["delta_cd"])[: args.top]
    for rank, row in enumerate(recommendations):
        row["preview_png"] = _render_recommendation(mesh, row, args.out / f"recommendation_{rank}.png")
    payload = {
        "format": "domino-surface-control-recommendation-v2",
        "baseline": baseline,
        "overview_points": _overview_points(points),
        "overview_bounds": {
            "min": np.round(bounds_min, 7).tolist(),
            "max": np.round(bounds_max, 7).tolist(),
        },
        "recommendations": recommendations,
        "candidates": rows,
        "shortlist_count": len(shortlist),
        "screening": "coarse-domino-then-full-domino",
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": ["Surrogate sensitivity screening; validate the applied shape with G2 CFD."],
    }
    (args.out / "recommendations.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"elapsed_seconds": payload["elapsed_seconds"], "recommendations": payload["recommendations"]}, indent=2))
    return payload


if __name__ == "__main__":
    main()
