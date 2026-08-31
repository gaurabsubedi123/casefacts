"""Talking to Ollama over HTTP, with urllib and nothing else.

Four endpoints are used: /api/tags to see what is installed, /api/chat to ask
a question, /api/embed to turn text into a vector, and /api/show to find out
how much context a model will actually accept.

Timeouts are long on purpose. This machine has no GPU, and a 7B model reading
twelve pages of a claim file takes a minute or two of honest CPU work. A
timeout that fires mid-answer looks exactly like a broken tool.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

from .config import DEFAULT_NUM_CTX, DEFAULT_OLLAMA_HOST, DEFAULT_TEMPERATURE

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 900.0
EMBED_TIMEOUT = 300.0

# Reasoning models wrap their scratch work in these. It is not the answer, it
# routinely contains the model talking itself into and out of claims, and if it
# reached the citation checker it would be checked as though it were an answer.
THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
UNCLOSED_THINK = re.compile(r"^\s*<think>.*", re.DOTALL | re.IGNORECASE)


class OllamaError(RuntimeError):
    """Ollama could not be reached, or refused the request."""


@dataclass
class Reply:
    """One answer from one model, with what it cost."""

    text: str
    model: str
    seconds: float
    prompt_tokens: int = 0
    answer_tokens: int = 0
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "seconds": round(self.seconds, 1),
            "prompt_tokens": self.prompt_tokens,
            "answer_tokens": self.answer_tokens,
            "truncated": self.truncated,
        }


def strip_thinking(text: str) -> str:
    """Remove a reasoning model's scratch work, closed or not.

    An answer cut off by the context limit can end *inside* the think block,
    leaving no closing tag. Returning the raw text in that case would hand the
    caller the model's musings as though they were findings, so an unclosed
    block swallows the rest.
    """
    cleaned = THINK_BLOCK.sub("", text)
    cleaned = UNCLOSED_THINK.sub("", cleaned)
    return cleaned.strip()


class Ollama:
    def __init__(self, host: str = DEFAULT_OLLAMA_HOST, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------- plumbing

    def _post(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise OllamaError(f"{path} returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"cannot reach Ollama at {self.host} ({exc.reason}). Start it with: ollama serve"
            ) from exc
        except TimeoutError as exc:
            raise OllamaError(f"{path} timed out after {timeout or self.timeout:.0f}s") from exc

    def _get(self, path: str) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.host}{path}", timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"cannot reach Ollama at {self.host} ({exc.reason}). Start it with: ollama serve"
            ) from exc

    # ---------------------------------------------------------------- facts

    def is_up(self) -> bool:
        try:
            self._get("/api/tags")
            return True
        except OllamaError:
            return False

    def installed(self) -> list[dict[str, Any]]:
        data = self._get("/api/tags")
        models = data.get("models") or []
        return [
            {
                "name": m.get("name", ""),
                "size_bytes": int(m.get("size") or 0),
                "family": ((m.get("details") or {}).get("family") or ""),
                "parameters": ((m.get("details") or {}).get("parameter_size") or ""),
            }
            for m in models
        ]

    def context_limit(self, model: str) -> int | None:
        """The model's own maximum context, straight from its metadata.

        Asking for more than a model was trained for does not fail loudly; it
        degrades. Reading the real number lets the caller cap a whole-document
        prompt at something the model can actually hold.
        """
        try:
            data = self._post("/api/show", {"model": model}, timeout=60)
        except OllamaError:
            return None
        info = data.get("model_info") or {}
        for key, value in info.items():
            if key.endswith(".context_length"):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
        return None

    # --------------------------------------------------------------- asking

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        *,
        num_ctx: int = DEFAULT_NUM_CTX,
        temperature: float = DEFAULT_TEMPERATURE,
        json_format: bool = False,
        strip_think: bool = True,
    ) -> Reply:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "num_ctx": int(num_ctx),
                "temperature": float(temperature),
                # Nothing here should be creative. Narrowing the sampler as
                # well as the temperature keeps a long extraction run from
                # drifting into invention on page 400.
                "top_p": 0.9,
                "repeat_penalty": 1.05,
            },
        }
        if json_format:
            payload["format"] = "json"
        started = time.monotonic()
        data = self._post("/api/chat", payload)
        elapsed = time.monotonic() - started

        text = ((data.get("message") or {}).get("content") or "").strip()
        if strip_think:
            text = strip_thinking(text)
        prompt_tokens = int(data.get("prompt_eval_count") or 0)

        # Ollama does not say "I dropped the front of your prompt". The only
        # signal is that it counted fewer prompt tokens than the window it was
        # given while the prompt was plainly longer — surfaced here so an
        # answer built on a truncated record can be marked as such.
        truncated = bool(prompt_tokens and prompt_tokens >= int(num_ctx) - 8)
        return Reply(
            text=text,
            model=model,
            seconds=elapsed,
            prompt_tokens=prompt_tokens,
            answer_tokens=int(data.get("eval_count") or 0),
            truncated=truncated,
        )

    # ------------------------------------------------------------ embedding

    def embed(self, model: str, texts: Iterable[str]) -> list[list[float]]:
        """Vectors for a batch of texts.

        /api/embed takes a batch and is what current Ollama wants; older builds
        only have /api/embeddings, one text per call. Both are handled because
        the version installed on a given machine is not this tool's business.
        """
        batch = [t if t.strip() else " " for t in texts]
        if not batch:
            return []
        try:
            data = self._post("/api/embed", {"model": model, "input": batch}, timeout=EMBED_TIMEOUT)
            vectors = data.get("embeddings")
            if vectors:
                return [[float(x) for x in v] for v in vectors]
        except OllamaError as exc:
            if "404" not in str(exc):
                raise
        out: list[list[float]] = []
        for text in batch:
            data = self._post("/api/embeddings", {"model": model, "prompt": text}, timeout=EMBED_TIMEOUT)
            out.append([float(x) for x in (data.get("embedding") or [])])
        return out
