import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_domino_su2_v3.py"
SPEC = importlib.util.spec_from_file_location("prepare_domino_su2_v3", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_flow_frame_aligns_180_degree_aoa_with_positive_x():
    rotation = MODULE.flow_frame(180.0, 0.0)
    world_velocity = np.array([-10.0, 0.0, 0.0])
    assert np.allclose(rotation @ world_velocity, [10.0, 0.0, 0.0], atol=1e-12)


def test_flow_frame_rotates_vectors_consistently():
    rotation = MODULE.flow_frame(90.0, 0.0)
    assert np.allclose(rotation @ [0.0, 0.0, 5.0], [5.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(rotation @ [1.0, 2.0, 3.0], [3.0, 2.0, -1.0], atol=1e-12)


def test_read_flow_conditions_keeps_direction(tmp_path):
    cfg = tmp_path / "case.cfg"
    cfg.write_text(
        "MACH_NUMBER= 0.8395\n"
        "AOA= 180.0\n"
        "SIDESLIP_ANGLE= 0.0\n"
        "REF_AREA= 0.042924\n"
    )
    flow = MODULE.read_flow_conditions(cfg)
    assert np.isclose(flow["speed"], 0.8395 * 340.3)
    assert np.allclose(flow["velocity"], [-flow["speed"], 0.0, 0.0], atol=1e-10)
    assert flow["ref_area"] == 0.042924


def test_read_su2_coefficients_handles_quoted_spaced_headers(tmp_path):
    (tmp_path / "history.csv").write_text(
        '"Inner_Iter", "CD", "CL"\n'
        "99, 0.301, -0.012\n"
    )
    assert MODULE.read_su2_coefficients(tmp_path) == (0.301, -0.012)


def test_explicit_negative_velocity_is_not_reduced_to_speed(tmp_path):
    cfg = tmp_path / "reverse.cfg"
    cfg.write_text(
        "INC_VELOCITY_INIT= (-30.0, 0.0, 0.0)\n"
        "REF_AREA= 1.0\n"
    )
    flow = MODULE.read_flow_conditions(cfg)
    assert np.allclose(flow["velocity"], [-30.0, 0.0, 0.0])
    assert np.allclose(flow["rotation"] @ [-30.0, 0.0, 0.0], [30.0, 0.0, 0.0])
    assert np.isclose(abs(flow["aoa"]), 180.0)
