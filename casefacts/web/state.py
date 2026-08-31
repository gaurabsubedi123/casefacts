"""Long jobs, run in a thread, watched from a browser.

Two things here take minutes rather than seconds: reading a folder in (which is
OCR plus embedding) and sweeping every page for a chronology. Both need to
report progress while they run, survive a closed tab, and stop when asked.

One job at a time, deliberately. This machine holds one model in memory; a
second job would not run alongside the first, it would swap against it.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

log = logging.getLogger(__name__)


class Stopped(BaseException):
    """Raised inside a job when someone asked it to stop.

    Deliberately a BaseException, like KeyboardInterrupt. Long jobs here wrap
    each document in `except Exception` so that one unreadable file does not
    lose the rest of the run — and an ordinary exception would be caught by
    that handler, logged as a per-document error, and the run would carry
    cheerfully on. Stopping has to pass straight through.
    """


@dataclass
class Job:
    """One background job and everything a watching browser needs from it."""

    kind: str
    label: str
    started: float = field(default_factory=time.time)
    finished: float | None = None
    status: str = "running"     # running | done | failed | stopped
    last: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    lines: list[str] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _watchers: list[queue.Queue] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def check(self) -> None:
        if self._stop.is_set():
            raise Stopped()

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        message = {"event": event, "at": time.time(), **payload}
        if event in ("ocr", "error"):
            # Keep a short tail of the noisy events so a tab opened late still
            # shows what has been happening, without keeping all of it.
            self.lines.append(str(payload.get("line") or payload.get("error") or ""))
            del self.lines[:-200]
        self.last = message
        with self._lock:
            watchers = list(self._watchers)
        for watcher in watchers:
            try:
                watcher.put_nowait(message)
            except queue.Full:
                pass

    def watch(self) -> Iterator[dict[str, Any]]:
        """Every event from now on, starting with where things stand."""
        channel: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._watchers.append(channel)
        try:
            yield {"event": "snapshot", **self.snapshot()}
            while True:
                try:
                    yield channel.get(timeout=15)
                except queue.Empty:
                    yield {"event": "ping", "at": time.time()}
                if self.status != "running" and channel.empty():
                    yield {"event": "closed", **self.snapshot()}
                    return
        finally:
            with self._lock:
                if channel in self._watchers:
                    self._watchers.remove(channel)

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "started": self.started,
            "finished": self.finished,
            "seconds": round((self.finished or time.time()) - self.started, 1),
            "last": self.last,
            "result": self.result,
            "error": self.error,
            "lines": self.lines[-40:],
        }


class Jobs:
    """The one running job, and the last one that finished."""

    def __init__(self) -> None:
        self.current: Job | None = None
        self.previous: Job | None = None
        self._lock = threading.Lock()

    def busy(self) -> bool:
        return self.current is not None and self.current.status == "running"

    def start(self, kind: str, label: str, work: Callable[[Job], dict[str, Any]]) -> Job:
        with self._lock:
            if self.busy():
                raise RuntimeError(
                    f"already {self.current.kind} — one job at a time, because only one "
                    "model fits in memory here"
                )
            job = Job(kind=kind, label=label)
            self.current = job

        def run() -> None:
            try:
                job.result = work(job)
                job.status = "stopped" if job.stopping else "done"
            except Stopped:
                job.status = "stopped"
            except Exception as exc:  # noqa: BLE001 - reported to the browser
                log.exception("%s failed", kind)
                job.status = "failed"
                job.error = str(exc)
            finally:
                job.finished = time.time()
                job.emit("finished", job.snapshot())
                with self._lock:
                    self.previous = job
                    if self.current is job:
                        self.current = None

        threading.Thread(target=run, name=f"casefacts-{kind}", daemon=True).start()
        return job

    def get(self) -> Job | None:
        return self.current or self.previous
