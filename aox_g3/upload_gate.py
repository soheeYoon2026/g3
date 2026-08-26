"""Geometry-only gate: is an uploaded STL a complete car?

Classifies an upload as ``full_car`` / ``component`` / ``non_car_shape`` /
``unsure`` from cheap mesh features (size after unit normalization, aspect
ratios, projected frontal area). Runs before inference so the product can
warn that coefficients for non-car uploads are out of the model's domain —
the 2026-08-26 reaudit showed roughly half of "model failures" were wings,
mirrors, and wheels, not cars. No ML, no external calls; ambiguous cases are
labeled ``unsure`` for an optional second-stage (e.g. local CLIP) check.
"""

from __future__ import annotations

import numpy as np

# Complete road cars, in meters, after normalizing obvious mm uploads.
CAR_LENGTH_M = (2.4, 7.0)
CAR_WIDTH_M = (1.2, 2.6)
CAR_HEIGHT_M = (0.8, 2.4)
CAR_FRONTAL_M2 = (1.2, 6.0)   # bounding-box frontal (width x height)
LENGTH_OVER_WIDTH = (1.4, 4.0)
FLAT_RATIO = 0.12          # min/mid extent below this smells like a wing/panel


def _projected_areas(mesh) -> np.ndarray:
    normals = np.asarray(mesh.face_normals, dtype=float)
    areas = np.asarray(mesh.area_faces, dtype=float)
    return 0.5 * np.abs(normals * areas[:, None]).sum(axis=0)


def classify_mesh(mesh) -> dict:
    """Classify a trimesh mesh; returns verdict, reasons, and features."""
    extents = np.sort(np.asarray(mesh.extents, dtype=float))[::-1]
    scale = 1.0
    if extents[0] > 100.0:          # almost certainly millimeters
        scale = 1e-3
    length, width, height = extents * scale

    # Bounding-box frontal area, matching the reference area G2 reports. Summing
    # |n·A| instead would count every hidden layer (wheels behind bodywork, inner
    # panel faces) and inflate a NISMO-class car to ~2x its silhouette.
    frontal = float(width * height)
    projected = _projected_areas(mesh) * scale * scale
    flat = float(extents[2] / max(extents[1], 1e-12))

    features = {
        "unit_scale": scale,
        "length_m": round(float(length), 3),
        "width_m": round(float(width), 3),
        "height_m": round(float(height), 3),
        "frontal_area_m2": round(frontal, 3),
        "projected_area_min_m2": round(float(projected.min()), 3),
        "length_over_width": round(float(length / max(width, 1e-12)), 2),
        "flat_ratio": round(flat, 3),
    }

    reasons = []
    if not (CAR_LENGTH_M[0] <= length <= CAR_LENGTH_M[1]):
        reasons.append(f"length {length:.2f}m outside car range {CAR_LENGTH_M}")
    if not (CAR_WIDTH_M[0] <= width <= CAR_WIDTH_M[1]):
        reasons.append(f"width {width:.2f}m outside car range {CAR_WIDTH_M}")
    if not (CAR_HEIGHT_M[0] <= height <= CAR_HEIGHT_M[1]):
        reasons.append(f"height {height:.2f}m outside car range {CAR_HEIGHT_M}")
    if flat < FLAT_RATIO:
        reasons.append(f"flat ratio {flat:.3f} — wing/panel-like")
    ratio = length / max(width, 1e-12)
    if not (LENGTH_OVER_WIDTH[0] <= ratio <= LENGTH_OVER_WIDTH[1]):
        reasons.append(f"length/width {ratio:.2f} outside car range {LENGTH_OVER_WIDTH}")
    if not (CAR_FRONTAL_M2[0] <= frontal <= CAR_FRONTAL_M2[1]):
        reasons.append(f"frontal area {frontal:.2f}m² outside car range {CAR_FRONTAL_M2}")

    if not reasons:
        verdict = "full_car"
    elif length < 2.0 or frontal < 0.6:
        verdict = "component"
    elif flat < FLAT_RATIO or ratio > 5.0:
        verdict = "non_car_shape"
    elif len(reasons) >= 3:
        verdict = "non_car_shape"
    else:
        verdict = "unsure"
    return {"verdict": verdict, "reasons": reasons, "features": features}


def classify_stl(path: str) -> dict:
    import trimesh

    mesh = trimesh.load(path, force="mesh")
    result = classify_mesh(mesh)
    result["path"] = str(path)
    return result


# Road-car flow regime: incompressible, wind-tunnel-ish speeds, no incidence.
CAR_SPEED_MS = (5.0, 80.0)
MAX_ABS_AOA_DEG = 5.0


def classify_case(conditions: dict, geometry: dict | None = None) -> dict:
    """Combine geometry verdict with flow conditions for a training case.

    The 2026-08-26 reaudit found transonic wing cases (Mach 0.84, 285 m/s)
    sitting inside a "car" evaluation set; conditions catch those even when
    the geometry alone looks plausible.
    """
    reasons = []
    speed = float(conditions.get("speed") or 0.0)
    mach = conditions.get("mach")
    aoa = float(conditions.get("aoa") or 0.0)
    if not (CAR_SPEED_MS[0] <= speed <= CAR_SPEED_MS[1]):
        reasons.append(f"speed {speed:.1f} m/s outside road-car range {CAR_SPEED_MS}")
    if mach not in (None, 0) and float(mach) > 0.3:
        reasons.append(f"Mach {float(mach):.2f} is compressible")
    if abs(aoa) > MAX_ABS_AOA_DEG:
        reasons.append(f"AoA {aoa:.1f}° outside road-car range")

    geometry_verdict = (geometry or {}).get("verdict", "unsure")
    if reasons:
        case_class = "off_regime"
    elif geometry_verdict == "full_car":
        case_class = "car_case"
    else:
        case_class = geometry_verdict
    return {
        "case_class": case_class,
        "flow_reasons": reasons,
        "geometry_verdict": geometry_verdict,
        "geometry_reasons": (geometry or {}).get("reasons", []),
    }


if __name__ == "__main__":
    import json
    import sys

    for arg in sys.argv[1:]:
        print(json.dumps(classify_stl(arg), ensure_ascii=False))
