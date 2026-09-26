"""Choosing which pages the model gets to read.

Two rules carried over from casefacts, for the same reasons:

* **A chunk never crosses a page boundary.** Every answer has to be checkable
  against one page, and a citation spanning two pages is half right.
* **When everything selected fits in the model's context, nothing is
  searched.** The whole text goes in, labelled by page. For a twenty-page
  report, searching could only lose information.

Otherwise pages are ranked by BM25 over their words, and — when an embedding
model is installed in Ollama — by meaning as well, with the two rankings fused
(reciprocal rank fusion). Keywords find "L4-L5"; meaning finds "back injury"
when the page says "lumbar strain". Vectors are computed once per document and
kept on disk.
"""

from __future__ import annotations

import math
import re
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .documents import Library

CHUNK_CHARS = 1400
CHUNK_OVERLAP = 200

# Characters per token, conservatively low so a prompt is never larger than
# estimated. OCR text with numbers and broken words tokenises worse than prose.
CHARS_PER_TOKEN = 3.0

# Small models read a handful of well-chosen pages better than thirty
# marginal ones.
MAX_EXCERPTS = 14

STOPWORDS = set(
    "a an and are as at be but by for from had has have he her his i if in into is it its of on or "
    "she so that the their them there they this to was were what when where which who why will with "
    "you your did does do any all about how many much list tell me please".split()
)


@dataclass
class Chunk:
    doc_id: str
    title: str
    page: int
    text: str
    score: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.doc_id}:{self.page}:{hash(self.text) & 0xFFFF:x}"

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "title": self.title, "page": self.page,
                "text": self.text, "score": round(self.score, 4)}


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", text.lower()) if w not in STOPWORDS]


def chunk_page(doc_id: str, title: str, page: int, text: str) -> list[Chunk]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= CHUNK_CHARS:
        return [Chunk(doc_id, title, page, text)]
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_CHARS)
        if end < len(text):
            # Prefer to cut at a line or sentence end near the limit.
            cut = max(text.rfind("\n", start + CHUNK_CHARS // 2, end), text.rfind(". ", start + CHUNK_CHARS // 2, end))
            if cut > start:
                end = cut + 1
        chunks.append(Chunk(doc_id, title, page, text[start:end].strip()))
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def chunks_for(library: Library, doc_ids: Sequence[str]) -> list[Chunk]:
    out: list[Chunk] = []
    for doc_id in doc_ids:
        document = library.get(doc_id)
        if not document:
            continue
        for number, text in enumerate(library.pages(doc_id), start=1):
            out.extend(chunk_page(doc_id, document.name, number, text))
    return out


def bm25_rank(question: str, chunks: Sequence[Chunk], k1: float = 1.4, b: float = 0.75) -> list[tuple[int, float]]:
    terms = set(words(question))
    if not terms or not chunks:
        return []
    docs = [Counter(words(c.text)) for c in chunks]
    average = sum(sum(d.values()) for d in docs) / len(docs) or 1.0
    frequency = Counter(t for d in docs for t in terms if t in d)
    scored: list[tuple[int, float]] = []
    for i, counts in enumerate(docs):
        length = sum(counts.values())
        score = 0.0
        for term in terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (len(docs) - frequency[term] + 0.5) / (frequency[term] + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average))
        if score > 0:
            scored.append((i, score))
    scored.sort(key=lambda s: s[1], reverse=True)
    return scored


# ------------------------------------------------------------- embeddings


def _vector_file(library: Library, doc_id: str, model: str) -> Path:
    safe = re.sub(r"[^\w.-]+", "_", model)
    return library.vectors / f"{doc_id}.{safe}.f32"


def vectors_for(library: Library, doc_id: str, chunks: Sequence[Chunk], model: str,
                embed: Callable[[str, list[str]], list[list[float]]]) -> list[array]:
    """Unit-length vectors for one document's chunks, cached on disk."""
    path = _vector_file(library, doc_id, model)
    if path.is_file():
        flat = array("f")
        flat.frombytes(path.read_bytes())
        if chunks and len(flat) % len(chunks) == 0:
            width = len(flat) // len(chunks)
            return [flat[i * width:(i + 1) * width] for i in range(len(chunks))]
    vectors: list[array] = []
    for start in range(0, len(chunks), 32):
        batch = embed(model, [f"search_document: {c.text}" for c in chunks[start:start + 32]])
        for vector in batch:
            norm = math.sqrt(sum(x * x for x in vector)) or 1.0
            vectors.append(array("f", (x / norm for x in vector)))
    flat = array("f")
    for vector in vectors:
        flat.extend(vector)
    path.write_bytes(flat.tobytes())
    return vectors


def embedding_rank(question: str, chunks: Sequence[Chunk], vectors: Sequence[array],
                   model: str, embed: Callable[[str, list[str]], list[list[float]]]) -> list[tuple[int, float]]:
    query = embed(model, [f"search_query: {question}"])[0]
    norm = math.sqrt(sum(x * x for x in query)) or 1.0
    query = [x / norm for x in query]
    scored = [(i, sum(a * b for a, b in zip(query, v))) for i, v in enumerate(vectors)]
    scored.sort(key=lambda s: s[1], reverse=True)
    return scored


def fuse(*rankings: list[tuple[int, float]], k: int = 60) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for position, (index, _) in enumerate(ranking):
            fused[index] = fused.get(index, 0.0) + 1.0 / (k + position + 1)
    return sorted(fused.items(), key=lambda s: s[1], reverse=True)


# -------------------------------------------------------------- selecting


def budget_chars(context_tokens: int, reserve_tokens: int) -> int:
    return max(2000, int((context_tokens - reserve_tokens) * CHARS_PER_TOKEN))


def gather(library: Library, question: str, doc_ids: Sequence[str], *, context_tokens: int,
           reserve_tokens: int, embed_model: str | None = None,
           embed: Callable[[str, list[str]], list[list[float]]] | None = None,
           progress: Callable[[str], None] | None = None) -> tuple[list[Chunk], str]:
    """The excerpts to put in the prompt, and how they were chosen."""
    budget = budget_chars(context_tokens, reserve_tokens)

    whole: list[Chunk] = []
    for doc_id in doc_ids:
        document = library.get(doc_id)
        if not document:
            continue
        for number, text in enumerate(library.pages(doc_id), start=1):
            if text.strip():
                whole.append(Chunk(doc_id, document.name, number, text.strip()))
    total = sum(len(c.text) + 40 for c in whole)
    if whole and total <= budget:
        return whole, "whole document" if len(doc_ids) == 1 else "whole documents"

    chunks = chunks_for(library, doc_ids)
    if not chunks:
        return [], "search"
    rankings = [bm25_rank(question, chunks)]
    mode = "keyword search"
    if embed_model and embed:
        try:
            if progress:
                progress("Indexing pages for meaning search (first time only for each document)…")
            # chunks_for groups chunks by document in doc_ids order, so the
            # per-document vector files concatenate into the same order.
            vectors: list[array] = []
            for doc_id in doc_ids:
                own = [c for c in chunks if c.doc_id == doc_id]
                vectors.extend(vectors_for(library, doc_id, own, embed_model, embed))
            rankings.append(embedding_rank(question, chunks, vectors, embed_model, embed))
            mode = "keyword + meaning search"
        except Exception:  # meaning search is an improvement, never a requirement
            pass
    ranked = fuse(*rankings) if len(rankings) > 1 else rankings[0]
    if not ranked:
        return [], mode

    chosen: list[Chunk] = []
    used = 0
    for index, score in ranked:
        chunk = chunks[index]
        cost = len(chunk.text) + 40
        if used + cost > budget:
            continue
        chunk.score = score
        chosen.append(chunk)
        used += cost
        if len(chosen) >= MAX_EXCERPTS:
            break
    # Reading order, not score order: a model follows a record better when
    # page 12 comes before page 40.
    order = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    chosen.sort(key=lambda c: (order.get(c.doc_id, 0), c.page))
    return chosen, mode
