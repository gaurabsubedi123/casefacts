"""Talking to Ollama over HTTP with urllib, and starting it when it is not up.

Ollama does the two hard things — downloading models and running them on
whatever GPU the machine has — so this module is a thin client over its API:

    /api/tags    what is installed          /api/pull    download, streamed
    /api/ps      what is loaded, and where  /api/delete  remove a model
    /api/show    a model's shape and limits /api/chat    ask, streamed
    /api/embed   text to vectors            /api/generate  load or unload

The portal cannot bundle Ollama inside its own .exe (it ships GPU runtimes
measured in gigabytes), so on a fresh machine it is found or its absence is
reported with the download link. When it is installed but not running, it is
started here, so the person never needs a terminal.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator

DEFAULT_HOST = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
DOWNLOAD_URL = "https://ollama.com/download"

# Reasoning models wrap their scratch work in these when the server does not
# separate it out itself. It is not the answer, and it must never reach the
# citation checker as though it were.
THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
UNCLOSED_THINK = re.compile(r"^\s*<think>.*", re.DOTALL | re.IGNORECASE)


class OllamaError(RuntimeError):
    """Ollama could not be reached, or refused the request."""


class Stopped(Exception):
    """The person pressed stop."""


def strip_thinking(text: str) -> str:
    cleaned = THINK_BLOCK.sub("", text)
    cleaned = UNCLOSED_THINK.sub("", cleaned)
    return cleaned.strip()


def _no_window() -> int:
    """Creation flags that stop a console window flashing up on Windows."""
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if platform.system() == "Windows" else 0


def find_binary() -> str | None:
    """The ollama executable, from PATH or the places the installers put it."""
    found = shutil.which("ollama")
    if found:
        return found
    home = Path.home()
    candidates: list[Path] = []
    system = platform.system()
    if system == "Windows":
        local = os.environ.get("LOCALAPPDATA") or str(home / "AppData" / "Local")
        candidates += [
            Path(local) / "Programs" / "Ollama" / "ollama.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Ollama" / "ollama.exe",
        ]
    elif system == "Darwin":
        candidates += [
            Path("/Applications/Ollama.app/Contents/Resources/ollama"),
            Path("/usr/local/bin/ollama"),
            Path("/opt/homebrew/bin/ollama"),
        ]
    else:
        candidates += [
            Path("/usr/local/bin/ollama"),
            Path("/usr/bin/ollama"),
            home / ".local" / "bin" / "ollama",
            home / ".local" / "opt" / "ollama" / "bin" / "ollama",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


class Ollama:
    def __init__(self, host: str = DEFAULT_HOST) -> None:
        self.host = host.rstrip("/")
        self._show_cache: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------ plumbing

    def _request(self, path: str, payload: dict[str, Any] | None, method: str, timeout: float):
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            try:
                detail = json.loads(detail).get("error", detail)
            except ValueError:
                pass
            raise OllamaError(detail or f"{path} returned {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(f"cannot reach Ollama at {self.host} ({exc.reason})") from exc
        except TimeoutError as exc:
            raise OllamaError(f"{path} timed out after {timeout:.0f}s") from exc

    def _json(self, path: str, payload: dict[str, Any] | None = None, method: str = "GET",
              timeout: float = 30) -> dict[str, Any]:
        with self._request(path, payload, method, timeout) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw) if raw.strip() else {}

    def _stream(self, path: str, payload: dict[str, Any], timeout: float) -> Iterator[dict[str, Any]]:
        """One JSON object per line, as Ollama streams them."""
        with self._request(path, payload, "POST", timeout) as response:
            for line in response:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise OllamaError(str(data["error"]))
                yield data

    # ------------------------------------------------------------- server

    def is_up(self) -> bool:
        try:
            self._json("/api/version", timeout=3)
            return True
        except (OllamaError, ValueError):
            return False

    def version(self) -> str | None:
        try:
            return str(self._json("/api/version", timeout=3).get("version") or "") or None
        except (OllamaError, ValueError):
            return None

    def start(self, wait_seconds: float = 20) -> tuple[bool, str]:
        """Start `ollama serve` in the background if it is not already up."""
        if self.is_up():
            return True, "Ollama is already running."
        binary = find_binary()
        if not binary:
            return False, f"Ollama is not installed on this computer. Get it from {DOWNLOAD_URL}"
        kwargs: dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
        }
        if platform.system() == "Windows":
            kwargs["creationflags"] = _no_window() | getattr(subprocess, "DETACHED_PROCESS", 0)
        else:
            kwargs["start_new_session"] = True
        try:
            subprocess.Popen([binary, "serve"], **kwargs)
        except OSError as exc:
            return False, f"Could not start Ollama ({exc})."
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            if self.is_up():
                return True, "Started Ollama."
            time.sleep(0.5)
        return False, "Started Ollama, but it did not answer within 20 seconds."

    # ------------------------------------------------------------- models

    def installed(self) -> list[dict[str, Any]]:
        models = self._json("/api/tags").get("models") or []
        return [
            {
                "name": m.get("name", ""),
                "size": int(m.get("size") or 0),
                "family": (m.get("details") or {}).get("family") or "",
                "parameters": (m.get("details") or {}).get("parameter_size") or "",
                "quantization": (m.get("details") or {}).get("quantization_level") or "",
                "modified": m.get("modified_at") or "",
                "digest": m.get("digest") or "",
            }
            for m in models
        ]

    def running(self) -> list[dict[str, Any]]:
        """What is loaded right now, and how much of it sits on the GPU."""
        models = self._json("/api/ps").get("models") or []
        return [
            {
                "name": m.get("name", ""),
                "size": int(m.get("size") or 0),
                "size_vram": int(m.get("size_vram") or 0),
                "context": int(m.get("context_length") or 0),
            }
            for m in models
        ]

    def show(self, model: str) -> dict[str, Any]:
        """Capabilities and architecture of an installed model (cached)."""
        if model not in self._show_cache:
            self._show_cache[model] = self._json("/api/show", {"model": model}, "POST", timeout=60)
        return self._show_cache[model]

    def capabilities(self, model: str) -> list[str]:
        try:
            return list(self.show(model).get("capabilities") or [])
        except OllamaError:
            return []

    def delete(self, model: str) -> None:
        self._json("/api/delete", {"model": model}, "DELETE")
        self._show_cache.pop(model, None)

    def pull(
        self,
        model: str,
        on_progress: Callable[[dict[str, Any]], None],
        should_stop: Callable[[], bool],
    ) -> None:
        """Download a model, reporting bytes as they arrive.

        Ollama streams one status line per layer update. Stopping closes the
        stream; Ollama keeps the partial blobs, so pressing download again
        resumes rather than starting over.
        """
        layers: dict[str, tuple[int, int]] = {}
        for event in self._stream("/api/pull", {"model": model, "stream": True}, timeout=3600):
            if should_stop():
                raise Stopped()
            digest = event.get("digest")
            if digest and event.get("total"):
                layers[digest] = (int(event.get("completed") or 0), int(event["total"]))
            done = sum(c for c, _ in layers.values())
            total = sum(t for _, t in layers.values())
            on_progress({"status": event.get("status", ""), "completed": done, "total": total})
        self._show_cache.pop(model, None)

    # ----------------------------------------------------- loading models

    def load(self, model: str, num_ctx: int, keep_alive: str = "30m") -> None:
        """Put a model in memory now, at the context size it will be asked at.

        Loading happens on the first request anyway; doing it here means the
        wait happens when the person clicks "use", and /api/ps can then report
        how much of it landed on the GPU. A later request with a different
        num_ctx would force a reload, so the same value is used for both.
        """
        self._json(
            "/api/generate",
            {"model": model, "prompt": "", "keep_alive": keep_alive, "options": {"num_ctx": int(num_ctx)}},
            "POST",
            timeout=600,
        )

    def unload(self, model: str) -> None:
        try:
            self._json("/api/generate", {"model": model, "prompt": "", "keep_alive": 0}, "POST", timeout=60)
        except OllamaError:
            pass

    # ---------------------------------------------------------- answering

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        num_ctx: int,
        think: bool | None = None,
        json_format: bool = False,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        should_stop: Callable[[], bool] = lambda: False,
    ) -> dict[str, Any]:
        """Ask, streamed so that a stop button works and progress is visible."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": "30m",
            "options": {"num_ctx": int(num_ctx), "temperature": 0.1, "top_p": 0.9},
        }
        if think is not None:
            payload["think"] = think
        if json_format:
            payload["format"] = "json"
        started = time.monotonic()
        content: list[str] = []
        thinking: list[str] = []
        final: dict[str, Any] = {}
        pieces = 0
        for event in self._stream("/api/chat", payload, timeout=1800):
            if should_stop():
                raise Stopped()
            message = event.get("message") or {}
            if message.get("thinking"):
                thinking.append(message["thinking"])
            if message.get("content"):
                content.append(message["content"])
            pieces += 1
            if on_progress and pieces % 8 == 0:
                on_progress({
                    "phase": "writing" if content else ("thinking" if thinking else "reading"),
                    "tokens": pieces,
                    "seconds": round(time.monotonic() - started, 1),
                })
            if event.get("done"):
                final = event
        text = "".join(content)
        reasoning = "".join(thinking)
        # Some builds leave the reasoning inline instead of separating it.
        inline = THINK_BLOCK.search(text) or UNCLOSED_THINK.match(text)
        if inline:
            reasoning = reasoning or inline.group(0)
            text = strip_thinking(text)
        prompt_tokens = int(final.get("prompt_eval_count") or 0)
        return {
            "text": text.strip(),
            "thinking": reasoning.strip(),
            "seconds": round(time.monotonic() - started, 1),
            "prompt_tokens": prompt_tokens,
            "answer_tokens": int(final.get("eval_count") or 0),
            # Ollama never says it dropped the front of a prompt. Counting
            # right up to the window is the only sign it did.
            "truncated": bool(prompt_tokens and prompt_tokens >= int(num_ctx) - 8),
        }

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        batch = [t if t.strip() else " " for t in texts]
        if not batch:
            return []
        data = self._json("/api/embed", {"model": model, "input": batch}, "POST", timeout=600)
        return [[float(x) for x in v] for v in data.get("embeddings") or []]
