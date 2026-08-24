from scripts.build_domino_s3_v4 import parse_rbf_objectives


def test_parse_rbf_objectives_maps_iteration_to_design_folder():
    text = """iteration,objective,max_displacement
1,0.26787487,0.0
2,0.26245703,0.1
bad,not-a-number,0.1
4,0.26172579,0.1
"""

    assert parse_rbf_objectives(text) == {
        "RBF_DSN_001": 0.26787487,
        "RBF_DSN_002": 0.26245703,
        "RBF_DSN_004": 0.26172579,
    }
