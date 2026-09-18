"""Process-owned locks and atomic controller state (no geometry operations)."""
import json
import os
import tempfile
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=1)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class RunLock:
    def __init__(self, directory):
        path = Path(directory) / ".controller.lock"
        self.stream = open(path, "a+b")
        if path.stat().st_size == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError("같은 작업 폴더에서 이미 실행 중입니다.") from None

    def close(self):
        if self.stream.closed:
            return
        self.stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


def reconcile(directory):
    """Detect abandoned new-protocol runs; never guess about legacy processes."""
    try:
        lock = RunLock(directory)
    except RuntimeError:
        return True
    try:
        path = Path(directory) / "plan.json"
        if path.exists():
            plan = json.loads(path.read_text(encoding="utf-8"))
            if plan.get("status") == "running" and plan.get("runtime", {}).get("protocol") == 1:
                plan["status"] = "failed"
                plan["runtime"]["error"] = "실행 프로세스가 종료되어 중단을 감지했습니다."
                write_json(path, plan)
        return False
    finally:
        lock.close()
