"""The frame guard must actually catch what it claims. Negative controls, not just positives.

A check that is never shown to fail is a check that might not fail: the LES session found a
position guard on 2026-09-16 that it had never tested, and our own regression suite was
vacuous until a negative control was added. So each case here is a mismatch that must warn.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import geometry_tools as gt


@pytest.fixture(scope="module")
def shapes(tmp_path_factory):
    d = tmp_path_factory.mktemp("frames")
    # anisotropic on purpose: a sphere is invariant under an axis swap, so it cannot test one
    # (the first run of this file failed for that reason, not because the guard was wrong)
    base = trimesh.creation.icosphere(subdivisions=3, radius=1000.0)
    base.vertices *= np.array([2.3, 1.0, 0.6])   # 4600 x 2000 x 1200 mm, a car-ish box
    paths = {}

    def put(name, mesh):
        p = d / f"{name}.stl"
        mesh.export(p)
        paths[name] = str(p)

    put("same", base)
    put("copy", base.copy())
    inch = base.copy(); inch.vertices *= 25.4
    put("units", inch)
    swap = base.copy(); swap.vertices = swap.vertices[:, [1, 0, 2]]   # 길이축을 y 로
    put("axes", swap)
    moved = base.copy(); moved.vertices += np.array([700.0, 0.0, 0.0])   # 길이의 15 %
    put("moved", moved)
    dented = base.copy()
    dented.vertices[dented.vertices[:, 0] > 1000] += np.array([0.0, 0.0, 40.0])  # 실제 형상 변화
    put("dented", dented)
    return paths


def warn(a, b):
    return gt.compare_to_reference(a, b, samples=20000)["frame_warning"]


def test_same_shape_passes(shapes):
    assert warn(shapes["copy"], shapes["same"]) == ""


def test_units_caught(shapes):
    assert "단위" in warn(shapes["units"], shapes["same"])


def test_axes_caught(shapes):
    # a swapped axis keeps the size, so it must be caught by overlap or by the median
    assert warn(shapes["axes"], shapes["same"]) != ""


def test_translation_caught(shapes):
    assert warn(shapes["moved"], shapes["same"]) != ""


def test_real_change_is_not_flagged(shapes):
    """국소 변형은 경고 대상이 아니다 — 그건 재려는 값이지 좌표계 문제가 아니다."""
    assert warn(shapes["dented"], shapes["same"]) == ""
