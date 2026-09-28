"""Saved conversations: each question and its checked answer, so a chat can be
reopened from History and continued.

One JSON file per chat under <data>/chats, next to the document library. Like
the documents, nothing here leaves the computer. Web pages a question found
are saved with its chat, so a follow-up can quote them without searching again.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

TITLE_CHARS = 70


def title_for(question: str) -> str:
    text = " ".join(question.split())
    return text if len(text) <= TITLE_CHARS else text[:TITLE_CHARS - 1].rstrip() + "…"


class Chats:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, chat_id: str) -> Path | None:
        # Ids come from the page; only ones this class could have made are used.
        if not re.fullmatch(r"[0-9a-f]{12}", chat_id or ""):
            return None
        return self.root / f"{chat_id}.json"

    def get(self, chat_id: str) -> dict[str, Any] | None:
        path = self._path(chat_id)
        if not path:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def all(self) -> list[dict[str, Any]]:
        """Every chat, newest first, without its answers."""
        listed = []
        for path in self.root.glob("*.json"):
            try:
                chat = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            listed.append({"id": chat["id"], "title": chat["title"], "updated": chat["updated"],
                           "questions": len(chat["turns"])})
        return sorted(listed, key=lambda c: c["updated"], reverse=True)

    def create(self, question: str) -> dict[str, Any]:
        now = time.time()
        chat = {"id": uuid.uuid4().hex[:12], "title": title_for(question), "created": now,
                "updated": now, "documents": [], "turns": []}
        self._save(chat)
        return chat

    def add_turn(self, chat_id: str, question: str, result: dict[str, Any], documents: list[str],
                 web_pages: list[dict[str, str]] = ()) -> None:
        with self._lock:
            chat = self.get(chat_id)
            if not chat:
                return
            chat["turns"].append({"question": question, "asked": time.time(), "result": result})
            # Web pages found for this question, kept so later questions can quote them unsearched.
            known = {p["url"] for p in chat.get("web_pages", [])}
            chat["web_pages"] = chat.get("web_pages", []) + [p for p in web_pages if p["url"] not in known]
            chat["documents"] = documents
            chat["updated"] = time.time()
            self._save(chat)

    def web_page(self, chat_id: str, page_id: str) -> dict[str, str] | None:
        chat = self.get(chat_id)
        return next((p for p in (chat or {}).get("web_pages", []) if p["id"] == page_id), None)

    def delete(self, chat_id: str) -> bool:
        path = self._path(chat_id)
        if not path or not path.exists():
            return False
        path.unlink()
        return True

    def _save(self, chat: dict[str, Any]) -> None:
        path = self.root / f"{chat['id']}.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(chat, indent=1), encoding="utf-8")
        temp.replace(path)


def history_of(chat: dict[str, Any] | None) -> list[dict[str, str]]:
    """The earlier questions and their answers, as the model is shown them."""
    if not chat:
        return []
    return [{"question": t["question"], "answer": t["result"].get("answer") or t["result"].get("missing") or ""}
            for t in chat["turns"]]
