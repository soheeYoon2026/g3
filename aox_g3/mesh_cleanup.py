"""Small mesh repairs which do not move vertices or reconstruct surfaces."""
import numpy as np
import trimesh


def cleanup(mesh):
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("입력 좌표에 NaN/무한대가 있습니다.")
    result = mesh.copy()
    keep = result.nondegenerate_faces()
    removed = int((~keep).sum())
    if removed:
        result.update_faces(keep)
        result.remove_unreferenced_vertices()
    if not result.is_winding_consistent:
        trimesh.repair.fix_winding(result)
    return result


def validate(mesh):
    return {"watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "degenerate_faces": int((~mesh.nondegenerate_faces()).sum()),
            "finite_coordinates": bool(np.isfinite(mesh.vertices).all())}
