import unittest
import numpy as np
import trimesh
from aox_g3.mesh_cleanup import cleanup, validate


class MeshCleanupTests(unittest.TestCase):
    def test_orientation_only_repair_does_not_move_vertices(self):
        mesh = trimesh.creation.box()
        mesh.faces[0] = mesh.faces[0][::-1]
        before = mesh.vertices.copy()
        self.assertFalse(mesh.is_winding_consistent)
        result = cleanup(mesh)
        np.testing.assert_array_equal(result.vertices, before)
        self.assertTrue(result.is_winding_consistent)
        self.assertTrue(result.is_watertight)
        self.assertFalse(mesh.is_winding_consistent)

    def test_degenerate_face_removed_without_removing_valid_faces(self):
        mesh = trimesh.creation.box()
        mesh.faces = np.vstack([mesh.faces, [0, 0, 0]])
        result = cleanup(mesh)
        self.assertEqual(len(result.faces), 12)
        self.assertEqual(validate(result)["degenerate_faces"], 0)
        self.assertTrue(result.is_watertight)

    def test_invalid_coordinates_rejected(self):
        mesh = trimesh.creation.box()
        mesh.vertices[0, 0] = np.nan
        with self.assertRaises(ValueError):
            cleanup(mesh)
