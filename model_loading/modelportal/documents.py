"""The document library: files dropped into the portal, and their text by page.

Each upload is copied into the portal's own data folder, so the original is
never written to and a moved or renamed original does not break an answer
already given. Its text is pulled out once, page by page, and kept beside it.

What is read:

    .pdf   the text layer, page by page. An OCR'd PDF (ocrtool's output, or any
           "searchable" PDF) has one; a raw scan does not, and is flagged as
           needing OCR rather than quietly answering from nothing.
    .txt   ocrtool's text files keep their page markers
           ("----- page 14 (ocr) -----"), so pages stay real pages. Any other
           text file is split at form feeds, or into sections of about a page.
    .md    as .txt.

Nothing is sent anywhere. Reading a PDF happens in this process.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .paths import data_dir

SUPPORTED = {".pdf", ".txt", ".md"}

# ocrtool's page marker. The number is the page; what follows it in brackets
# (ocr / text) says how the page was read and is not needed here.
PAGE_MARKER = re.compile(r"^-{3,}\s*page\s+(\d+)\b[^\n]*-{3,}\s*$", re.IGNORECASE | re.MULTILINE)

# A page with less text than this is treated as having no text layer.
EMPTY_PAGE_CHARS = 25

# Plain text with no page structure is cut into sections of about a printed
# page, so a citation still points somewhere findable.
SECTION_CHARS = 3000


@dataclass
class Document:
    id: str
    name: str
    kind: str
    pages: int
    chars: int
    empty_pages: int
    added: float
    stored: str

    @property
    def needs_ocr(self) -> bool:
        return self.pages > 0 and self.empty_pages / self.pages > 0.5

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("stored")
        data["needs_ocr"] = self.needs_ocr
        return data


# ------------------------------------------------------------- extraction


def pdf_pages(path: Path, progress: Callable[[int, int], None] | None = None) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    total = len(reader.pages)
    if progress:
        progress(0, total)
    pages: list[str] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # a single malformed page must not lose the document
            text = ""
        pages.append(text)
        if progress:
            progress(number, total)
    return pages


def text_pages(text: str) -> list[str]:
    markers = list(PAGE_MARKER.finditer(text))
    if markers:
        by_number: dict[int, str] = {}
        for i, marker in enumerate(markers):
            end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            by_number[int(marker.group(1))] = text[marker.end():end].strip()
        last = max(by_number)
        return [by_number.get(n, "") for n in range(1, last + 1)]
    if "\f" in text:
        return [p.strip() for p in text.split("\f")]
    sections: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in re.split(r"\n\s*\n", text):
        if size and size + len(paragraph) > SECTION_CHARS:
            sections.append("\n\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph)
    if current:
        sections.append("\n\n".join(current))
    return [s.strip() for s in sections] or [""]


def read_pages(path: Path, progress: Callable[[int, int], None] | None = None) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return pdf_pages(path, progress)
    if suffix in (".txt", ".md"):
        return text_pages(path.read_text(encoding="utf-8", errors="replace"))
    raise ValueError(f"{path.name}: only PDF and text files can be read")


# ---------------------------------------------------------------- library


class Cancelled(Exception):
    """Raised by a progress callback to stop reading a file part-way."""


class NotOCRed(ValueError):
    """The file has no text to answer from — a raw scan, or an empty file."""


def ocr_problem(name: str, pages: list[str]) -> str | None:
    """Why this file cannot be answered from, or None when it can.

    A file is refused rather than accepted with a warning, because a model
    given empty pages does not say "I could not read this" — it answers
    anyway, from nothing.
    """
    if not pages or not any(p.strip() for p in pages):
        return (f"{name} has no text in it. If it is a scan, it has not been OCR'd yet — "
                "run it through ocrtool first, then add the OCR'd PDF or .txt it produces.")
    empty = sum(1 for p in pages if len(p.strip()) < EMPTY_PAGE_CHARS)
    if empty / len(pages) > 0.5:
        return (f"{name} is not OCR'd: {empty} of its {len(pages)} pages have no text layer. "
                "Run it through ocrtool first, then add the OCR'd PDF or .txt it produces.")
    return None


def empty_page_numbers(pages: list[str]) -> list[int]:
    return [n for n, p in enumerate(pages, start=1) if len(p.strip()) < EMPTY_PAGE_CHARS]


class Library:
    """Documents on disk under <data>/library, with a JSON index."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (data_dir() / "library")
        self.files = self.root / "files"
        self.text = self.root / "pages"
        self.vectors = self.root / "vectors"
        for folder in (self.files, self.text, self.vectors):
            folder.mkdir(parents=True, exist_ok=True)
        self.index_file = self.root / "index.json"
        self._lock = threading.Lock()
        self._pages_cache: dict[str, list[str]] = {}

    # ---------------------------------------------------------- index file

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            return json.loads(self.index_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save(self, index: dict[str, dict[str, Any]]) -> None:
        temp = self.index_file.with_suffix(".tmp")
        temp.write_text(json.dumps(index, indent=1), encoding="utf-8")
        temp.replace(self.index_file)

    def all(self) -> list[Document]:
        docs = [Document(**d) for d in self._load().values()]
        return sorted(docs, key=lambda d: d.added, reverse=True)

    def get(self, doc_id: str) -> Document | None:
        data = self._load().get(doc_id)
        return Document(**data) if data else None

    # --------------------------------------------------------------- adding

    def add(self, source: Path, name: str,
            progress: Callable[[int, int], None] | None = None) -> tuple[Document, bool]:
        """Copy a file in and read its pages. Returns (document, was_new).

        The id is the hash of the file's bytes, so dropping the same file twice
        — even under another name — is recognised rather than duplicated.
        """
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED:
            raise ValueError(f"{name}: only PDF and text files (.pdf, .txt, .md) can be added")
        digest = hashlib.sha256()
        with open(source, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        doc_id = digest.hexdigest()[:16]
        existing = self.get(doc_id)
        if existing:
            return existing, False

        stored = self.files / f"{doc_id}{suffix}"
        shutil.copyfile(source, stored)
        try:
            pages = read_pages(stored, progress)
        except Cancelled:
            stored.unlink(missing_ok=True)
            raise
        except Exception as exc:
            stored.unlink(missing_ok=True)
            raise ValueError(f"{name}: could not be read ({exc})") from exc
        problem = ocr_problem(Path(name).name, pages)
        if problem:
            stored.unlink(missing_ok=True)
            raise NotOCRed(problem)
        (self.text / f"{doc_id}.json").write_text(json.dumps(pages), encoding="utf-8")
        document = Document(
            id=doc_id,
            name=Path(name).name,
            kind=suffix.lstrip("."),
            pages=len(pages),
            chars=sum(len(p) for p in pages),
            empty_pages=sum(1 for p in pages if len(p.strip()) < EMPTY_PAGE_CHARS),
            added=time.time(),
            stored=stored.name,
        )
        with self._lock:
            index = self._load()
            index[doc_id] = asdict(document)
            self._save(index)
        return document, True

    def remove(self, doc_id: str) -> bool:
        with self._lock:
            index = self._load()
            data = index.pop(doc_id, None)
            if not data:
                return False
            self._save(index)
        (self.files / data["stored"]).unlink(missing_ok=True)
        (self.text / f"{doc_id}.json").unlink(missing_ok=True)
        for vector_file in self.vectors.glob(f"{doc_id}.*"):
            vector_file.unlink(missing_ok=True)
        self._pages_cache.pop(doc_id, None)
        return True

    # -------------------------------------------------------------- reading

    def pages(self, doc_id: str) -> list[str]:
        if doc_id not in self._pages_cache:
            path = self.text / f"{doc_id}.json"
            self._pages_cache[doc_id] = json.loads(path.read_text(encoding="utf-8"))
        return self._pages_cache[doc_id]

    def file_path(self, doc_id: str) -> Path | None:
        document = self.get(doc_id)
        if not document:
            return None
        path = self.files / document.stored
        return path if path.is_file() else None
