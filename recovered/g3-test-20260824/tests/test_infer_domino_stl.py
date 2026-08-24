import numpy as np

from scripts.infer_domino_stl import g2_reference_area


class Mesh:
    points = np.array([
        [-2.0, -1.2, 0.1],
        [3.0, 0.8, 1.6],
        [0.0, 1.3, -0.4],
    ])


def test_g2_reference_area_uses_yz_bounding_box():
    assert g2_reference_area(Mesh()) == 5.0


if __name__ == "__main__":
    test_g2_reference_area_uses_yz_bounding_box()
