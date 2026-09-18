import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from aox_g3.run_state import RunLock, reconcile, write_json


class RunStateTests(unittest.TestCase):
    def test_lock_blocks_other_process_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = RunLock(directory)
            command = [sys.executable, "-c", "from aox_g3.run_state import RunLock; import sys; RunLock(sys.argv[1])", directory]
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            lock.close()
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_abandoned_run_and_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            write_json(path, {"status": "running", "runtime": {"protocol": 1}, "answers": {"units": "mm"}})
            lock = RunLock(directory)
            self.assertTrue(reconcile(directory))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "running")
            lock.close()
            self.assertFalse(reconcile(directory))
            plan = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(plan["status"], "failed")
            self.assertEqual(plan["answers"], {"units": "mm"})
            write_json(path, {"status": "running"})
            reconcile(directory)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "running")


if __name__ == "__main__":
    unittest.main()
