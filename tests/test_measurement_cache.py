import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from aox_g3.measurement_cache import cached_thickness


class MeasurementCacheTests(unittest.TestCase):
    def test_controller_routes_and_resume(self):
        import trimesh
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            answers = base / "answers.json"
            answers.write_text(json.dumps({"units": "mm", "length_axis_now": "x", "closed_mesh_action": "resurface"}), encoding="utf-8")
            env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            for opened in (False, True):
                mesh = trimesh.creation.box(extents=[4000, 1800, 1400])
                if opened:
                    mesh.update_faces(mesh.face_normals[:, 2] > -0.5)
                else:
                    # A closed mesh with inconsistent face orientation needs repair.
                    mesh.faces[0] = mesh.faces[0][::-1]
                src = base / ("open.stl" if opened else "closed.stl")
                mesh.export(src)
                out = base / ("open" if opened else "closed")
                command = [sys.executable, str(root / "scripts" / "plan_geometry.py"),
                           "--in", str(src), "--out", str(out), "--answers", str(answers), "--no-render"]
                result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=env, timeout=60)
                self.assertEqual(result.returncode, 3 if opened else 0, result.stdout + result.stderr)
                cache = out / "thickness_measurement.json"
                self.assertFalse(cache.exists())
                if opened:
                    self.assertNotIn("[진행] 두께 측정: 측정 중", result.stdout)
                    self.assertIn("[진행] 두께 측정: 생략", result.stdout)
                    self.assertLess(result.stdout.index("[진단] 분류: 열린 메쉬"), result.stdout.index("[진행] 두께 측정: 생략"))
                else:
                    resumed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=env, timeout=60)
                    self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
                    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
                    self.assertEqual(plan["route"], "automatic-local-cleanup")
                    self.assertNotIn("closed_mesh_action", [q["id"] for q in plan["questions"]])
                    self.assertIn("[진단] 초기 검사: 저장된 결과 재사용", resumed.stdout)
                    self.assertNotIn("[진단] 초기 검사: 검사 중", resumed.stdout)
                    self.assertNotIn("[진행] 두께 측정: 측정 중", resumed.stdout)
                    self.assertFalse(cache.exists())

    def test_reuse_and_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            src, cache = Path(directory) / "input.stl", Path(directory) / "thickness.json"
            src.write_bytes(b"original")
            calls = []
            def measure():
                calls.append(1)
                return 16.123456
            self.assertEqual(cached_thickness(cache, src, {"seed": 0}, measure), (16.123456, False))
            self.assertEqual(cached_thickness(cache, src, {"seed": 0}, measure), (16.123456, True))
            src.write_bytes(b"changed")
            self.assertFalse(cached_thickness(cache, src, {"seed": 0}, measure)[1])
            self.assertFalse(cached_thickness(cache, src, {"seed": 1}, measure)[1])
            self.assertFalse(cached_thickness(cache, src, {"seed": 1}, measure, force=True)[1])
            self.assertEqual(len(calls), 4)

    def test_corruption_and_failed_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            src, cache = Path(directory) / "input.stl", Path(directory) / "thickness.json"
            src.write_bytes(b"input")
            cache.write_text("broken", encoding="utf-8")
            self.assertEqual(cached_thickness(cache, src, {}, lambda: 5), (5, False))
            before = cache.read_bytes()
            def fail():
                raise MemoryError("measurement interrupted")
            with self.assertRaises(MemoryError):
                cached_thickness(cache, src, {}, fail, force=True)
            self.assertEqual(cache.read_bytes(), before)
            with self.assertRaises(ValueError):
                cached_thickness(cache, src, {}, lambda: float("nan"), force=True)
            self.assertEqual(json.loads(cache.read_text(encoding="utf-8"))["status"], "done")


if __name__ == "__main__":
    unittest.main()
