"""Reading what ocrtool wrote, without touching it.

ocrtool already solved the hard half of this. Its .txt files carry a marker at
the top of every page:

    ----- page 14 (ocr) -----

which means page numbers do not have to be guessed from form feeds or inferred
from headers — they are stated. And its .json files carry, per page, the OCR
confidence and the path to a JPEG of that page. Those two facts are what make a
citation in this tool checkable by eye instead of merely plausible.

So: text from the .txt, metadata from the .json beside it, and if the .json is
missing the tool still works with a little less to show.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .config import RESERVED_DIRS

log = logging.getLogger(__name__)

PAGE_MARKER = re.compile(r"^-{5} page (\d+) \(([^)]*)\) -{5}\s*$", re.MULTILINE)

# Bates numbers: the production stamp in the corner of a produced page, e.g.
# GEICO000472 or DEF-001234. In a claim file this is the citation opposing
# counsel will use, so it is worth pulling out and showing beside the page
# number.
#
# The pattern alone is not enough. "Macon, GA 31294" matches any reasonable
# stamp regex, and so does a letterhead with a phone number in it. What
# actually distinguishes a Bates stamp is that it is a *series*: one prefix,
# running across most of the document, with the number changing page to page.
# So candidates are collected first and the series is identified second, in
# `bates_series` — the regex only has to be generous.
BATES = re.compile(r"\b([A-Z][A-Z&._-]{2,15})[ _-]?(\d{4,10})\b")

# A prefix has to reach this share of a document's pages, with this many
# distinct numbers, before it is believed to be a production stamp.
BATES_MIN_PAGE_SHARE = 0.3
BATES_MIN_DISTINCT = 3


@dataclass
class Page:
    """One page of one document, as it will be indexed and cited."""

    page_no: int
    text: str
    source: str = "ocr"
    confidence: float | None = None
    needs_review: bool = False
    preview: str | None = None
    bates: str | None = None

    @property
    def chars(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page_no,
            "source": self.source,
            "confidence": self.confidence,
            "needs_review": self.needs_review,
            "preview": self.preview,
            "bates": self.bates,
            "chars": self.chars,
        }


@dataclass
class Document:
    """One OCR'd document: where it came from, and its pages."""

    doc_id: str
    title: str
    txt_path: Path
    rel_path: str = ""
    json_path: Path | None = None
    source_path: str | None = None
    mtime: float = 0.0
    size_bytes: int = 0
    pages: list[Page] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(p.chars for p in self.pages)

    @property
    def mean_confidence(self) -> float | None:
        values = [p.confidence for p in self.pages if p.confidence is not None]
        return round(sum(values) / len(values), 1) if values else None

    def page(self, page_no: int) -> Page | None:
        for page in self.pages:
            if page.page_no == page_no:
                return page
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "rel_path": self.rel_path,
            "source_path": self.source_path,
            "pages": len(self.pages),
            "chars": self.chars,
            "mean_confidence": self.mean_confidence,
        }


def _relative_to(path: Path, root: Path) -> str:
    """How to show this document's location, relative to what was plugged in."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _is_reserved(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return False
    return any(part in RESERVED_DIRS for part in parts)


def find_text_files(root: Path) -> list[Path]:
    """Every .txt ocrtool produced, in a stable order.

    All three of ocrtool's output layouts are handled by not caring about them:
    `mirror` puts a txt/ folder beside each source folder, `by-type` puts one at
    the top, `together` puts the .txt next to the .pdf. Globbing for .txt and
    skipping the tool's own folders covers all three without asking which was
    used.
    """
    if not root.is_dir():
        return []
    found = [p for p in root.rglob("*.txt") if p.is_file() and not _is_reserved(p, root)]
    return sorted(found, key=lambda p: str(p).lower())


def find_json_for(txt_path: Path, root: Path) -> Path | None:
    """The .json holding this document's per-page metadata, if it was written."""
    stem = txt_path.stem
    candidates = [
        txt_path.with_suffix(".json"),          # layout: together
        txt_path.parent.parent / "json" / f"{stem}.json",  # layout: mirror
        root / "json" / f"{stem}.json",         # layout: by-type
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def split_pages(text: str) -> list[tuple[int, str, str]]:
    """Cut a .txt into (page number, source, text) on ocrtool's page markers.

    A file with no markers at all is one page. That happens for a plain-text
    export and for anything a user drops into the folder by hand, and calling it
    page 1 is both true and better than refusing to index it.
    """
    matches = list(PAGE_MARKER.finditer(text))
    if not matches:
        stripped = text.strip()
        return [(1, "unknown", stripped)] if stripped else []

    pages: list[tuple[int, str, str]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip("\n")
        pages.append((int(match.group(1)), match.group(2).strip() or "unknown", body))
    return pages


def bates_series(pages: list[str]) -> set[str]:
    """Which stamp prefixes in this document are a real production series.

    A prefix that appears on most pages with a different number each time is a
    Bates stamp. A prefix that appears on most pages with the *same* number is
    letterhead. A prefix that appears twice is a coincidence.
    """
    numbers: dict[str, set[str]] = {}
    page_counts: dict[str, int] = {}
    for text in pages:
        on_this_page: set[str] = set()
        for prefix, number in BATES.findall(text):
            numbers.setdefault(prefix, set()).add(number)
            on_this_page.add(prefix)
        for prefix in on_this_page:
            page_counts[prefix] = page_counts.get(prefix, 0) + 1

    total = max(1, len(pages))
    series = set()
    for prefix, seen_numbers in numbers.items():
        if len(seen_numbers) < BATES_MIN_DISTINCT:
            continue
        if page_counts.get(prefix, 0) < max(BATES_MIN_DISTINCT, total * BATES_MIN_PAGE_SHARE):
            continue
        series.add(prefix)
    return series


def _pick_bates(page_text: str, series: set[str]) -> str | None:
    """This page's stamp, taken only from a prefix already shown to be a series."""
    for prefix, number in BATES.findall(page_text):
        if prefix in series:
            return f"{prefix}{number}"
    return None


def _load_page_metadata(json_path: Path) -> tuple[dict[int, dict[str, Any]], str | None]:
    """Per-page confidence and preview path out of ocrtool's JSON.

    The JSON also holds `word_boxes` — every word on every page with its
    pixel rectangle — which is most of the file's bulk and none of its use
    here. It is dropped as soon as it is parsed rather than carried into the
    index.
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("could not read %s: %s", json_path, exc)
        return {}, None
    document = data.get("document") or {}
    by_page: dict[int, dict[str, Any]] = {}
    for page in document.get("pages") or []:
        try:
            number = int(page.get("page"))
        except (TypeError, ValueError):
            continue
        by_page[number] = {
            "source": str(page.get("source") or "ocr"),
            "confidence": page.get("confidence"),
            "needs_review": bool(page.get("needs_review")),
            "preview": page.get("preview"),
        }
    return by_page, document.get("relpath") or document.get("source_path")


def load_document(txt_path: Path, root: Path) -> Document | None:
    """Everything about one document, ready to index."""
    try:
        raw = txt_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.warning("could not read %s: %s", txt_path, exc)
        return None

    split = split_pages(raw)
    if not split:
        return None

    json_path = find_json_for(txt_path, root)
    metadata, relpath = _load_page_metadata(json_path) if json_path else ({}, None)

    # Identify the production stamp series across the whole document before
    # any single page claims one.
    series = bates_series([body for _, _, body in split])

    stat = txt_path.stat()
    # The identity of a document is where its text actually is on disk.
    # A path relative to the folder you plugged in would collide the moment
    # two folders were plugged in that both contain "records/txt/notes.txt".
    document = Document(
        doc_id=txt_path.resolve().as_posix(),
        title=txt_path.stem,
        rel_path=_relative_to(txt_path, root),
        txt_path=txt_path,
        json_path=json_path,
        source_path=relpath,
        mtime=stat.st_mtime,
        size_bytes=stat.st_size,
    )
    for number, source, body in split:
        meta = metadata.get(number, {})
        document.pages.append(
            Page(
                page_no=number,
                text=body,
                source=str(meta.get("source") or source),
                confidence=meta.get("confidence"),
                needs_review=bool(meta.get("needs_review")),
                preview=meta.get("preview"),
                bates=_pick_bates(body, series),
            )
        )
    return document


def load_all(root: Path) -> Iterator[Document]:
    for txt_path in find_text_files(root):
        document = load_document(txt_path, root)
        if document is not None:
            yield document
