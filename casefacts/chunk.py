"""Cutting a page into retrievable pieces.

One rule governs everything here: **a chunk never crosses a page boundary.**

It would retrieve slightly better if it could — a finding split across a page
break is split in the index too. It is still not allowed, because every answer
this tool gives has to be checkable against a single page image. A chunk
spanning pages 14 and 15 produces a citation that is half right, and half right
is the worst kind of wrong in a medical record.

Within a page the cuts follow the text's own structure: blank lines first,
then single lines, and only then a hard character cut, so that a chunk usually
begins where a paragraph does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import CHUNK_CHARS, CHUNK_OVERLAP, MIN_CHUNK_CHARS


@dataclass
class Chunk:
    """A piece of one page, and enough to cite it."""

    doc_id: str
    page_no: int
    ord: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "page": self.page_no, "ord": self.ord, "text": self.text}


def _blocks(text: str) -> list[str]:
    """Paragraphs, falling back to lines, falling back to fixed slices."""
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    out: list[str] = []
    for block in blocks:
        if len(block) <= CHUNK_CHARS:
            out.append(block)
            continue
        # A "paragraph" longer than a whole chunk is usually a table or a
        # layout-preserved page with no blank lines in it at all. Cut on lines.
        line_buffer: list[str] = []
        length = 0
        for line in block.split("\n"):
            if length + len(line) + 1 > CHUNK_CHARS and line_buffer:
                out.append("\n".join(line_buffer))
                line_buffer, length = [], 0
            line_buffer.append(line)
            length += len(line) + 1
            # A single line longer than a chunk (a run-on OCR line) is cut by
            # character; there is nothing else left to cut on.
            while length > CHUNK_CHARS:
                joined = "\n".join(line_buffer)
                out.append(joined[:CHUNK_CHARS])
                remainder = joined[CHUNK_CHARS:]
                line_buffer = [remainder] if remainder else []
                length = len(remainder)
        if line_buffer:
            out.append("\n".join(line_buffer))
    return out


def chunk_page(doc_id: str, page_no: int, text: str) -> list[Chunk]:
    """The retrievable pieces of one page.

    A short page is kept whole. Most pages in a claim file are short — a fax
    cover, a signature page, an exhibit separator — and splitting them buys
    nothing while making the index bigger.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= CHUNK_CHARS + MIN_CHUNK_CHARS:
        return [Chunk(doc_id, page_no, 0, text)]

    pieces: list[str] = []
    buffer = ""
    for block in _blocks(text):
        if buffer and len(buffer) + len(block) + 2 > CHUNK_CHARS:
            pieces.append(buffer)
            # Carry the tail of the last piece into the next one so a sentence
            # cut in half is still whole in one of them.
            tail = buffer[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else ""
            buffer = (tail + "\n\n" + block).strip() if tail else block
        else:
            buffer = (buffer + "\n\n" + block).strip() if buffer else block
    if buffer.strip():
        pieces.append(buffer)

    # A final sliver — the last few words of a page landing alone — is folded
    # back into the piece before it rather than indexed as its own chunk.
    if len(pieces) > 1 and len(pieces[-1]) < MIN_CHUNK_CHARS:
        pieces[-2] = pieces[-2] + "\n\n" + pieces[-1]
        pieces.pop()

    return [Chunk(doc_id, page_no, i, piece) for i, piece in enumerate(pieces)]
