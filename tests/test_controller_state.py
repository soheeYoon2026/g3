import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import trimesh

from aox_g3.controller_state import mesh_checkpoint, outcome, preparation_outcome


class ControllerStateTests(unittest.TestCase):
    def test_valid_step_preserves_original_without_heal_or_wrap(self):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "box.stp"
            writer = STEPControl_Writer()
            writer.Transfer(BRepPrimAPI_MakeBox(100, 50, 30).Shape(), STEPControl_AsIs)
            writer.Write(str(source))
            out = root / "run"
            script = Path(__file__).resolve().parents[1] / "scripts" / "plan_geometry.py"
            result = subprocess.run([sys.executable, str(script), "--in", str(source),
                                     "--out", str(out), "--no-render"],
                                    capture_output=True, encoding="utf-8", timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["route"], "preserve-step-basic-valid")
            self.assertEqual(plan["deliverables"], [str(source)])
            self.assertEqual(plan["runs"], [])

    def test_preparation_failure_is_nonzero_and_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "broken.stl"
            source.write_bytes(b"invalid mesh")
            out = root / "out"
            script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_geometry.py"
            result = subprocess.run([sys.executable, str(script), "--in", str(source),
                                     "--out", str(out), "--no-render", "--no-mirror"],
                                    capture_output=True, encoding="utf-8", timeout=60)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertIn("mesh", summary["failed_required_stages"])

    def test_checkpoint_reuse_invalidation_and_relocation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "first"
            root.mkdir()
            calls = []
            def build():
                calls.append(1)
                return trimesh.creation.box()
            mesh_checkpoint(root, "normalized", {"units": "mm"}, build, stl=True)
            stamp = (root / "input_mm.stl").stat().st_mtime_ns
            mesh, reused = mesh_checkpoint(root, "normalized", {"units": "mm"}, build, stl=True)
            self.assertTrue(reused)
            self.assertEqual(len(calls), 1)
            self.assertEqual(stamp, (root / "input_mm.stl").stat().st_mtime_ns)
            moved = Path(temp) / "moved"
            shutil.move(root, moved)
            self.assertTrue(mesh_checkpoint(moved, "normalized", {"units": "mm"}, build, stl=True)[1])
            self.assertFalse(mesh_checkpoint(moved, "normalized", {"units": "m"}, build, stl=True)[1])
            (moved / "input_mm.stl").write_bytes(b"corrupted")
            self.assertFalse(mesh_checkpoint(moved, "normalized", {"units": "m"}, build, stl=True)[1])

    def test_outcomes_are_not_process_success_only(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "result.stl"
            output.write_bytes(b"mesh")
            plan = {"questions": [], "checks": [], "deliverables": [str(output)]}
            self.assertEqual(outcome(plan), ("done", 0))
            plan["questions"] = [{"id": "wheel"}]
            self.assertEqual(outcome(plan), ("waiting_for_answers", 3))
            plan["questions"][0]["answer"] = "leave"
            plan["checks"] = [{"ok": False}]
            self.assertEqual(outcome(plan), ("needs_review", 4))
            plan["checks"][0]["resolved"] = "retry passed"
            output.unlink()
            self.assertEqual(outcome(plan), ("failed", 2))
            summary = {"stages": {"mesh": {"status": "ok"}, "render": {"status": "failed"}}}
            self.assertEqual(preparation_outcome(summary, ["mesh"]), 0)
            self.assertEqual(preparation_outcome(summary, ["mesh", "heal"]), 2)
            summary["stages"]["wrap"] = {"status": "hollow"}
            self.assertEqual(preparation_outcome(summary, ["mesh", "wrap"]), 4)

    def test_real_controller_pauses_and_reuses_normalization(self):
        # Stop before geometry repair: verifies the actual CLI across separate processes.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "car.stl"
            trimesh.creation.box(extents=[4000, 1600, 1200]).export(source)
            out = root / "run"
            answers = root / "answers.json"
            controller = Path(__file__).resolve().parents[1] / "scripts" / "plan_geometry.py"
            command = [sys.executable, str(controller), "--in", str(source), "--out", str(out), "--no-render"]
            env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            first = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=env, timeout=60)
            self.assertEqual(first.returncode, 3, first.stderr)
            self.assertEqual(json.loads((out / "plan.json").read_text(encoding="utf-8"))["status"], "waiting_for_answers")
            # Avoid thickness work while exercising the next question checkpoint.
            answers.write_text(json.dumps({"units": "mm", "length_axis_now": "x"}))
            second = subprocess.run(command + ["--answers", str(answers)], capture_output=True, text=True, encoding="utf-8", env=env, timeout=60)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["route"], "preserve-basic-valid")
            self.assertTrue(plan["initial_mesh_diagnosis"]["diagnosis"]["winding_consistent"])
            self.assertTrue(plan["initial_mesh_diagnosis"]["diagnosis"]["finite_coordinates"])
            self.assertEqual(plan["deliverables"], [str(source)])
            self.assertFalse((out / "thickness_measurement.json").exists())
            normalized = out / "input_mm.stl"
            stamp = normalized.stat().st_mtime_ns
            third = subprocess.run(command + ["--answers", str(answers)], capture_output=True, text=True, encoding="utf-8", env=env, timeout=60)
            self.assertEqual(third.returncode, 0, third.stdout + third.stderr)
            self.assertEqual(stamp, normalized.stat().st_mtime_ns)
            self.assertIn("STL 저장 생략", third.stdout)
            self.assertIn("원본 파싱", third.stdout)
            self.assertNotIn("mm 기준 퇴화 면 확인", third.stdout)


if __name__ == "__main__":
    unittest.main()
