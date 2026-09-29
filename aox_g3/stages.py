"""Ordered, checkpoint-persisted processing stages, independent of log wording."""
import time
import json
import threading
from contextlib import contextmanager


def record_stage(plan, stage_id, status="running"):
    if status not in {"running", "completed", "waiting", "paused", "failed", "review_required"}:
        raise ValueError(f"Invalid stage status: {status}")
    stages = plan.setdefault("processing_stages", [])
    previous = stages[-1] if stages else None
    if previous and previous["id"] == stage_id and previous["status"] in {"running", "waiting"}:
        previous["status"] = status
        previous["updated_at"] = time.time()
        return
    if previous and previous["status"] in {"running", "waiting"}:
        previous["status"] = "paused" if status == "waiting" else "completed"
        previous["updated_at"] = time.time()
    now = time.time()
    stages.append({"id": stage_id, "status": status, "started_at": now, "updated_at": now})


def end_current_stage(plan, status):
    stages = plan.get("processing_stages", [])
    if stages and stages[-1]["status"] == "running":
        stages[-1]["status"] = status
        stages[-1]["updated_at"] = time.time()


@contextmanager
def follow_preparation(plan, summary_path, lock, save):
    """Copy structured child stages while a preparation subprocess is running."""
    started = time.time()
    stopped = threading.Event()
    indices = []

    def sync():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if summary.get("started_at", 0) < started:
            return  # a previous attempt's summary
        with lock:
            for index, stage in enumerate(summary.get("processing_stages", [])):
                if index >= len(indices):
                    previous = plan.get("processing_stages", [])
                    if previous and previous[-1]["status"] in {"running", "waiting"}:
                        previous[-1]["status"] = "completed"
                    indices.append(len(plan.setdefault("processing_stages", [])))
                    plan["processing_stages"].append(dict(stage))
                else:
                    plan["processing_stages"][indices[index]].update(stage)
            plan["preparation"] = summary
            save()

    def watch():
        while not stopped.wait(.25):
            sync()

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()
        sync()
