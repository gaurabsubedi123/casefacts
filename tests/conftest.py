"""Fixtures: a small OCR output folder, and an Ollama that is not there.

The tests never talk to a model. Embeddings are a deterministic hash of the
text, which is enough to exercise the vector half of the search — two identical
texts get identical vectors, different texts get different ones — without
needing 274 MB of model on a CI machine, and without making the tests depend
on a model's actual behaviour.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from casefacts.config import Settings

PAGE_ONE = """GEICO000001
Export Type: email | Export ID: 99495788
GEICO Indemnity Company
Attn: Region IV Claims, PO Box 35
Macon, GA 31294-9643
Date: April 9, 2025
RE: Claim Documents 0613043540101028"""

PAGE_TWO = """GEICO000002
SPRING VALLEY HOSPITAL MEDICAL CENTER
Patient: JIANG, MING YAO
MRN: SVH35492594
Admit: 7/6/2018
Disch: 7/7/2018
Attending: Landis MD,Brandi R
Diagnosis: Cervical muscle strain; Low back sprain
Motor Vehicle Accident: No Serious Injury"""

PAGE_THREE = """GEICO000003
Jones Physical Therapy
Date of service: 12/03/2018
Lumbar Transforaminal Epidural Injection performed today.
Patient reports 6/10 low back pain radiating to the right leg."""


def _marker(number: int, source: str = "ocr") -> str:
    return f"----- page {number} ({source}) -----"


@pytest.fixture
def records(tmp_path: Path) -> Path:
    """A folder shaped exactly like one ocrtool wrote."""
    root = tmp_path / "ocr-output"
    (root / "txt").mkdir(parents=True)
    (root / "json").mkdir(parents=True)
    (root / "_previews" / "claim.pdf").mkdir(parents=True)

    pages = [PAGE_ONE, PAGE_TWO, PAGE_THREE]
    text = "\n\n".join(
        f"{_marker(i + 1, 'text-layer' if i == 0 else 'ocr')}\n{page}"
        for i, page in enumerate(pages)
    )
    (root / "txt" / "claim.txt").write_text(text, encoding="utf-8")

    (root / "json" / "claim.json").write_text(
        json.dumps(
            {
                "tool": "ocrtool",
                "document": {
                    "relpath": "claim.pdf",
                    "page_count": 3,
                    "pages": [
                        {
                            "page": i + 1,
                            "source": "text-layer" if i == 0 else "ocr",
                            "confidence": None if i == 0 else 55.0 + i * 20,
                            "needs_review": i == 1,
                            "preview": f"_previews/claim.pdf/p000{i + 1}.jpg",
                            "word_boxes": [{"text": "x", "box": [0, 0, 1, 1], "confidence": 90}],
                        }
                        for i in range(3)
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    for i in range(3):
        (root / "_previews" / "claim.pdf" / f"p000{i + 1}.jpg").write_bytes(b"\xff\xd8\xff\xe0 fake jpeg")
    return root


class FakeOllama:
    """Stands in for the real client. Records what it was asked."""

    replies: list[str] = []
    asked: list[dict] = []

    def __init__(self, host: str = "", timeout: float = 0) -> None:
        self.host = host

    def is_up(self) -> bool:
        return True

    def installed(self) -> list[dict]:
        return [
            {"name": "qwen2.5:7b-instruct", "size_bytes": 4_700_000_000, "family": "qwen2", "parameters": "7B"},
            {"name": "nomic-embed-text", "size_bytes": 274_000_000, "family": "nomic-bert", "parameters": "137M"},
        ]

    def context_limit(self, model: str) -> int:
        return 32768

    def embed(self, model: str, texts) -> list[list[float]]:
        out = []
        for text in texts:
            digest = hashlib.sha256(text.strip().lower().encode("utf-8")).digest()
            vector = np.frombuffer(digest * 8, dtype=np.uint8).astype(np.float32)[:64]
            out.append((vector / 255.0).tolist())
        return out

    def chat(self, model, system, user, **kwargs):
        from casefacts.ollama import Reply

        FakeOllama.asked.append({"model": model, "system": system, "user": user, **kwargs})
        text = FakeOllama.replies.pop(0) if FakeOllama.replies else '{"findings": [], "answer": "", "missing": ""}'
        return Reply(text=text, model=model, seconds=0.1, prompt_tokens=100, answer_tokens=20)


@pytest.fixture
def fake_ollama(monkeypatch):
    FakeOllama.replies = []
    FakeOllama.asked = []
    for module in ("casefacts.index", "casefacts.answer", "casefacts.chronology", "casefacts.web.app", "casefacts.cli"):
        monkeypatch.setattr(f"{module}.Ollama", FakeOllama, raising=False)
    return FakeOllama


@pytest.fixture
def settings(tmp_path: Path, records: Path) -> Settings:
    return Settings(records_dir=records, db_path=tmp_path / "index.db")


@pytest.fixture
def index(settings, fake_ollama):
    from casefacts.index import Index

    ix = Index(settings)
    ix.ingest_path(settings.records_dir, ocr=False)
    return ix
