import numpy as np
import trimesh

from aox_g3.upload_gate import classify_mesh


def box(l, w, h):
    return trimesh.creation.box(extents=[l, w, h])


def test_car_sized_box_is_full_car():
    r = classify_mesh(box(4.5, 1.9, 1.4))
    assert r["verdict"] == "full_car", r
    assert abs(r["features"]["frontal_area_m2"] - 1.9 * 1.4) < 0.01


def test_millimeter_upload_is_normalized():
    r = classify_mesh(box(4500, 1900, 1400))
    assert r["features"]["unit_scale"] == 1e-3
    assert r["verdict"] == "full_car", r


def test_mirror_sized_object_is_component():
    r = classify_mesh(box(0.25, 0.2, 0.18))
    assert r["verdict"] == "component", r


def test_flat_wing_is_non_car():
    r = classify_mesh(box(1.2, 0.8, 0.05))
    assert r["verdict"] in ("non_car_shape", "component"), r
    assert any("flat" in reason or "outside" in reason for reason in r["reasons"])


def test_bus_sized_box_is_not_silently_accepted():
    r = classify_mesh(box(12.0, 2.5, 3.2))
    assert r["verdict"] != "full_car", r
