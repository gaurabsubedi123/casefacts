"""Long work — downloads, loading a model, reading a PDF, answering — off the
request thread, so the page can poll progress, reload without losing it, and
press stop.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable

from .ollama import Stopped

KEEP = 60


class Job:
    def __init__(self, kind: str, label: str) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.label = label
        self.status = "running"
        self.progress: dict[str, Any] = {}
        self.result: Any = None
        self.error = ""
        self.started = time.time()
        self.finished: float | None = None
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def update(self, progress: dict[str, Any]) -> None:
        self.progress = {**self.progress, **progress}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "label": self.label, "status": self.status,
            "progress": self.progress, "result": self.result, "error": self.error,
            "elapsed": round((self.finished or time.time()) - self.started, 1),
        }


class Jobs:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, label: str, work: Callable[[Job], Any]) -> Job:
        job = Job(kind, label)
        with self._lock:
            self._jobs[job.id] = job
            finished = [j for j in self._jobs.values() if j.status != "running"]
            for old in sorted(finished, key=lambda j: j.started)[:-KEEP]:
                self._jobs.pop(old.id, None)

        def run() -> None:
            try:
                job.result = work(job)
                job.status = "stopped" if job.stopping else "done"
            except Stopped:
                job.status = "stopped"
            except Exception as exc:  # shown to the person; the thread must not die silently
                job.status = "error"
                job.error = str(exc) or exc.__class__.__name__
            finally:
                job.finished = time.time()

        threading.Thread(target=run, daemon=True, name=f"job-{kind}").start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def active(self, kind: str | None = None) -> list[Job]:
        return [j for j in self._jobs.values() if j.status == "running" and (kind is None or j.kind == kind)]

    def recent(self, kind: str) -> list[Job]:
        return sorted((j for j in self._jobs.values() if j.kind == kind), key=lambda j: j.started)
