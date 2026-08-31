"""The index: SQLite for the text, numpy for the vectors, and both for search.

**Why not a vector database.** A case file is big for a person and small for a
computer. The 358-page production in the test folder is 747 chunks; a 10,000
page file would be about 21,000. At 768 dimensions that is 64 MB of float32 —
a matrix that fits in memory with room to spare, searched by one numpy dot
product in a few milliseconds. A vector store would add a dependency, a
process, and a second copy of the evidence, and would not be measurably faster
until this tool held more pages than a law firm produces in a year.

**Why both halves of the search.** Embeddings find "what did the orthopedist
say about the shoulder" when the note says "L glenohumeral joint". They are
bad at GEICO000472, at CPT 99213, at ICD M25.512, and at a surname — the exact
strings a medical-legal file is full of, where being close in meaning is worth
nothing and being the same string is worth everything. FTS5 handles those and
misses the paraphrase. Neither alone is good enough, so both run and the
rankings are fused.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .chunk import chunk_page
from .config import MIN_SIMILARITY, RRF_K, Settings
from .corpus import Document, load_document
from .sources import ResolvedDoc, Source, resolve as resolve_source
from .ollama import Ollama, OllamaError

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# How many chunks go to the embedding model in one HTTP call. Larger batches
# amortise the round trip; too large and a CPU-only machine holds the whole
# batch in memory at once. 32 was the point where throughput stopped improving.
EMBED_BATCH = 32

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    txt_path        TEXT NOT NULL,
    json_path       TEXT,
    source_path     TEXT,
    -- Which plugged-in file or folder this document arrived with, where it
    -- sits inside it, and what it originally was. `root` is what makes
    -- "ask this folder only" a WHERE clause; `original_path` is the PDF or
    -- scan a citation should name; `previews_root` is where its page images
    -- live, which differs between a folder ocrtool already processed and one
    -- this tool OCR'd into its own workspace.
    root            TEXT,
    rel_path        TEXT,
    origin          TEXT,
    original_path   TEXT,
    previews_root   TEXT,
    page_count      INTEGER NOT NULL,
    chars           INTEGER NOT NULL,
    mean_confidence REAL,
    mtime           REAL,
    size_bytes      INTEGER,
    content_hash    TEXT,
    duplicate_of    TEXT,
    ingested_at     TEXT
);

CREATE TABLE IF NOT EXISTS pages (
    doc_id       TEXT NOT NULL,
    page_no      INTEGER NOT NULL,
    source       TEXT,
    confidence   REAL,
    needs_review INTEGER DEFAULT 0,
    preview      TEXT,
    bates        TEXT,
    text         TEXT NOT NULL,
    chars        INTEGER NOT NULL,
    PRIMARY KEY (doc_id, page_no)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id   TEXT NOT NULL,
    page_no  INTEGER NOT NULL,
    ord      INTEGER NOT NULL,
    text     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_doc ON chunks (doc_id, page_no);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='chunk_id',
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS vectors (
    chunk_id INTEGER PRIMARY KEY,
    dim      INTEGER NOT NULL,
    vec      BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS bates_index (
    bates   TEXT NOT NULL,
    doc_id  TEXT NOT NULL,
    page_no INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS bates_lookup ON bates_index (bates);
"""

# FTS5 treats a bare user question as query syntax, so "patient's" or a stray
# hyphen is a syntax error rather than a search. Words are extracted and
# re-joined instead of escaped, which also drops the punctuation that carries
# no signal.
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'./-]*")

# Words too common in these documents to narrow anything. Kept short: an
# aggressive stop list throws away "no" and "not", which in a medical record
# are the entire meaning of the sentence.
STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in is it of on or that the
    to was were what when where which who why with does did do please tell me
    about any all""".split()
)


@dataclass
class Hit:
    """One retrieved chunk, with everything needed to cite and to check it."""

    chunk_id: int
    doc_id: str
    title: str
    page_no: int
    text: str
    score: float = 0.0
    similarity: float | None = None
    keyword_rank: int | None = None
    vector_rank: int | None = None
    confidence: float | None = None
    needs_review: bool = False
    preview: str | None = None
    bates: str | None = None
    source: str | None = None

    def label(self) -> str:
        """How this chunk is cited in a prompt and in an answer."""
        return f"{self.title} p.{self.page_no}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "page": self.page_no,
            "text": self.text,
            "score": round(self.score, 4),
            "similarity": round(self.similarity, 3) if self.similarity is not None else None,
            "keyword_rank": self.keyword_rank,
            "vector_rank": self.vector_rank,
            "confidence": self.confidence,
            "needs_review": self.needs_review,
            "preview": self.preview,
            "bates": self.bates,
            "source": self.source,
            "label": self.label(),
        }


@dataclass
class IngestReport:
    """What one ingest run did, in the terms a person would ask about."""

    documents: int = 0
    duplicates: int = 0
    unchanged: int = 0
    pages: int = 0
    chunks: int = 0
    embedded: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "duplicates": self.duplicates,
            "unchanged": self.unchanged,
            "pages": self.pages,
            "chunks": self.chunks,
            "embedded": self.embedded,
            "seconds": round(self.seconds, 1),
            "errors": self.errors,
        }


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), timeout=30.0)
    connection.row_factory = sqlite3.Row
    # WAL so the web interface can read the index while an ingest is still
    # writing to it — otherwise every question during a long ingest blocks.
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(SCHEMA)
    connection.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
    )
    connection.commit()
    return connection


def _pack(vector: Sequence[float]) -> bytes:
    """Store a unit vector, so search is a dot product and not a division.

    Normalising once at write time turns cosine similarity into a plain matrix
    multiply at query time, for every query, forever.
    """
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm > 0:
        array = array / norm
    return array.tobytes()


def _content_hash(document: Document) -> str:
    digest = hashlib.sha256()
    for page in document.pages:
        digest.update(page.text.encode("utf-8", "replace"))
        digest.update(b"\x00")
    return digest.hexdigest()


def fts_query(question: str) -> str:
    """A user's question, turned into something FTS5 will accept.

    Terms are OR'd rather than AND'd: a question mentioning six things should
    still find the page that answers four of them, and bm25 already ranks a
    page matching more of them higher.
    """
    words = []
    for word in WORD.findall(question):
        word = word.lower().strip("'.-/")
        # "patient's" should find "patient". The possessive is the one piece of
        # punctuation that changes which documents match, so it is removed
        # rather than quoted around.
        if word.endswith("'s"):
            word = word[:-2]
        if word:
            words.append(word)
    terms = [w for w in words if len(w) > 1 and w not in STOPWORDS]
    if not terms:
        terms = [w for w in words if w]
    if not terms:
        return ""
    # Double-quoted so that a term containing a hyphen or a slash — "x-ray",
    # "24/7", "GEICO000472" — is a literal, not an operator.
    return " OR ".join('"' + t.replace('"', "") + '"' for t in terms)


class Index:
    """The searchable copy of the case file."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db = connect(settings.db_path)
        self._matrix: np.ndarray | None = None
        self._chunk_ids: np.ndarray | None = None
        # Whether the last search matched any of the question's own words. See
        # the note at the end of `search`.
        self.last_search_had_keyword_match = True

    def close(self) -> None:
        self.db.close()

    # ---------------------------------------------------------------- facts

    def stats(self) -> dict[str, Any]:
        row = self.db.execute(
            """SELECT
                 (SELECT COUNT(*) FROM documents WHERE duplicate_of IS NULL) AS documents,
                 (SELECT COUNT(*) FROM documents WHERE duplicate_of IS NOT NULL) AS duplicates,
                 (SELECT COUNT(*) FROM pages) AS pages,
                 (SELECT COUNT(*) FROM chunks) AS chunks,
                 (SELECT COUNT(*) FROM vectors) AS vectors"""
        ).fetchone()
        stats = dict(row)
        stats["records_dir"] = str(self.settings.records_dir)
        stats["db_path"] = str(self.settings.db_path)
        stats["embed_model"] = self.meta("embed_model")
        stats["last_ingest"] = self.meta("last_ingest")
        return stats

    def meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )

    def documents(self, include_duplicates: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM documents"
        if not include_duplicates:
            sql += " WHERE duplicate_of IS NULL"
        sql += " ORDER BY title COLLATE NOCASE"
        return [dict(row) for row in self.db.execute(sql)]

    def document(self, doc_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def resolve_doc(self, needle: str) -> str | None:
        """Find a document from whatever the user typed.

        Exact id, then exact title, then a unique substring of the title — the
        titles in a claim file run to eighty characters and nobody is typing
        them in full.
        """
        if not needle:
            return None
        for sql in (
            "SELECT doc_id FROM documents WHERE doc_id = ?",
            "SELECT doc_id FROM documents WHERE title = ? COLLATE NOCASE",
        ):
            row = self.db.execute(sql, (needle,)).fetchone()
            if row:
                return row["doc_id"]
        rows = self.db.execute(
            "SELECT doc_id FROM documents WHERE title LIKE ? AND duplicate_of IS NULL",
            (f"%{needle}%",),
        ).fetchall()
        return rows[0]["doc_id"] if len(rows) == 1 else None

    def aliases(self, doc_id: str) -> list[dict[str, Any]]:
        """The other copies of this document in the file.

        A produced claim file contains the same records two and three times.
        Which copy a page was read from is an accident of filing; that there
        are three of them is sometimes the point.
        """
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT doc_id, title, rel_path, root FROM documents WHERE duplicate_of = ? ORDER BY title",
                (doc_id,),
            )
        ]

    def pages(self, doc_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT * FROM pages WHERE doc_id = ? ORDER BY page_no", (doc_id,)
            )
        ]

    def page(self, doc_id: str, page_no: int) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT * FROM pages WHERE doc_id = ? AND page_no = ?", (doc_id, page_no)
        ).fetchone()
        return dict(row) if row else None

    def by_bates(self, bates: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT doc_id, page_no FROM bates_index WHERE bates = ? COLLATE NOCASE LIMIT 1",
            (bates.strip(),),
        ).fetchone()
        return self.page(row["doc_id"], row["page_no"]) if row else None

    # -------------------------------------------------------------- ingest

    def ingest(
        self,
        source: Source,
        *,
        rebuild: bool = False,
        progress: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> IngestReport:
        """Read one plugged-in file or folder into the index.

        Incremental: a document whose .txt has not changed since it was last
        read is skipped without re-embedding it, and embedding is the entire
        cost of an ingest. `rebuild=True` forces the lot.
        """
        started = time.monotonic()
        report = IngestReport()
        client = Ollama(self.settings.ollama_host)

        def emit(event: str, **payload: Any) -> None:
            if progress:
                progress(event, payload)

        if rebuild:
            self.forget_source(source.root)

        emit("start", files=len(source.docs), root=str(source.root))

        # Hashes of what is already indexed, so the same document reaching the
        # index by two routes is embedded once.
        known_hashes: dict[str, str] = {
            row["content_hash"]: row["doc_id"]
            for row in self.db.execute(
                "SELECT content_hash, doc_id FROM documents "
                "WHERE duplicate_of IS NULL AND content_hash IS NOT NULL"
            )
        }

        previews_root = str(source.previews_dir) if source.previews_dir else None

        for position, resolved in enumerate(source.docs, start=1):
            txt_path = resolved.txt_path
            try:
                doc_id = txt_path.resolve().as_posix()
                existing = self.document(doc_id)
                stat = txt_path.stat()
                if (
                    existing
                    and not rebuild
                    and existing.get("mtime") == stat.st_mtime
                    and existing.get("size_bytes") == stat.st_size
                ):
                    report.unchanged += 1
                    emit("skip", file=txt_path.stem, position=position, total=len(source.docs))
                    continue

                document = load_document(txt_path, source.root)
                if document is None:
                    continue

                emit(
                    "document",
                    file=document.title,
                    pages=len(document.pages),
                    position=position,
                    total=len(source.docs),
                )
                digest = _content_hash(document)
                primary = known_hashes.get(digest)
                if primary and primary != document.doc_id:
                    self._write_duplicate(document, digest, primary, source, resolved, previews_root)
                    report.duplicates += 1
                    emit("duplicate", file=document.title, of=primary)
                    continue

                chunks_written, embedded = self._write_document(
                    document, digest, source, resolved, previews_root, client, emit
                )
                known_hashes[digest] = document.doc_id
                report.documents += 1
                report.pages += len(document.pages)
                report.chunks += chunks_written
                report.embedded += embedded
                self.db.commit()
            except OllamaError as exc:
                # No embedding model means no vector half of the search. The
                # keyword half still works, so this is reported and the run
                # continues rather than losing the whole ingest.
                report.errors.append(f"{txt_path.name}: {exc}")
                emit("error", file=txt_path.name, error=str(exc))
                self.db.commit()
            except Exception as exc:  # one unreadable document must not stop the rest
                log.exception("failed on %s", txt_path)
                report.errors.append(f"{txt_path.name}: {exc}")
                emit("error", file=txt_path.name, error=str(exc))

        self.set_meta("embed_model", self.settings.embed_model)
        self.set_meta("last_ingest", time.strftime("%Y-%m-%dT%H:%M:%S"))
        self._remember_source(source)
        self.db.commit()
        self._forget_matrix()
        report.seconds = time.monotonic() - started
        emit("done", **report.to_dict())
        return report

    def ingest_path(
        self,
        path: Path,
        *,
        ocr: bool = True,
        recursive: bool = True,
        rebuild: bool = False,
        progress: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> tuple[Source, IngestReport]:
        """Plug in a path and index it, OCR'ing anything that needs it first."""

        def relay(line: str) -> None:
            if progress:
                progress("ocr", {"line": line})

        source = resolve_source(path, recursive=recursive, ocr=ocr, progress=relay)
        return source, self.ingest(source, rebuild=rebuild, progress=progress)

    # ------------------------------------------------------------- sources

    def sources(self) -> list[dict[str, Any]]:
        """The files and folders that have been plugged in, most recent first."""
        rows = self.db.execute(
            """SELECT root,
                      COUNT(*) FILTER (WHERE duplicate_of IS NULL) AS documents,
                      SUM(page_count) FILTER (WHERE duplicate_of IS NULL) AS pages,
                      MAX(ingested_at) AS last_ingest
               FROM documents WHERE root IS NOT NULL
               GROUP BY root ORDER BY last_ingest DESC"""
        ).fetchall()
        return [dict(row) for row in rows]

    def _remember_source(self, source: Source) -> None:
        roots = json.loads(self.meta("roots") or "[]")
        root = str(source.root)
        roots = [r for r in roots if r != root]
        roots.insert(0, root)
        self.set_meta("roots", json.dumps(roots[:50]))

    def forget_source(self, root: Path | str) -> int:
        """Drop everything that came from one plugged-in path."""
        rows = self.db.execute(
            "SELECT doc_id FROM documents WHERE root = ?", (str(root),)
        ).fetchall()
        for row in rows:
            self._clear_document(row["doc_id"])
        self.db.commit()
        self._forget_matrix()
        return len(rows)

    def scope_doc_ids(self, *, doc: str | None = None, folder: str | None = None) -> list[str] | None:
        """Which documents a question is allowed to look at.

        None means "everything indexed". A document scope is one file; a folder
        scope is every document whose text lives under that folder, which is
        how "ask this folder" and "ask this one file" are the same code path
        with a different filter.
        """
        if doc:
            resolved = self.resolve_doc(doc)
            return [resolved] if resolved else []
        if folder:
            prefix = Path(folder).expanduser().resolve().as_posix().rstrip("/") + "/"
            rows = self.db.execute(
                "SELECT doc_id FROM documents WHERE doc_id LIKE ? OR root = ?",
                (prefix + "%", str(Path(folder).expanduser().resolve())),
            ).fetchall()
            return [row["doc_id"] for row in rows]
        return None

    def _clear_document(self, doc_id: str) -> None:
        rows = self.db.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()
        ids = [(row["chunk_id"],) for row in rows]
        if ids:
            self.db.executemany("DELETE FROM chunks_fts WHERE rowid = ?", ids)
            self.db.executemany("DELETE FROM vectors WHERE chunk_id = ?", ids)
        self.db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self.db.execute("DELETE FROM pages WHERE doc_id = ?", (doc_id,))
        self.db.execute("DELETE FROM bates_index WHERE doc_id = ?", (doc_id,))
        self.db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def _document_row(
        self,
        document: Document,
        digest: str,
        duplicate_of: str | None,
        source: Source,
        resolved: ResolvedDoc,
        previews_root: str | None,
    ) -> None:
        self.db.execute(
            """INSERT INTO documents
               (doc_id, title, txt_path, json_path, source_path, root, rel_path, origin,
                original_path, previews_root, page_count, chars,
                mean_confidence, mtime, size_bytes, content_hash, duplicate_of, ingested_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                document.doc_id,
                document.title,
                str(document.txt_path),
                str(document.json_path) if document.json_path else None,
                document.source_path,
                str(source.root),
                document.rel_path,
                resolved.origin,
                str(resolved.original_path),
                previews_root,
                len(document.pages),
                document.chars,
                document.mean_confidence,
                document.mtime,
                document.size_bytes,
                digest,
                duplicate_of,
                time.strftime("%Y-%m-%dT%H:%M:%S"),
            ),
        )

    def _write_duplicate(
        self,
        document: Document,
        digest: str,
        primary: str,
        source: Source,
        resolved: ResolvedDoc,
        previews_root: str | None,
    ) -> None:
        """Record a duplicate without indexing it twice.

        The document is still listed, so that a search hit can say "this page
        is also filed as ..." — in a produced claim file the same records
        arrive three times and knowing that is sometimes the point.
        """
        self._clear_document(document.doc_id)
        self._document_row(document, digest, primary, source, resolved, previews_root)

    def _write_document(
        self,
        document: Document,
        digest: str,
        source: Source,
        resolved: ResolvedDoc,
        previews_root: str | None,
        client: Ollama,
        emit: Callable[..., None],
    ) -> tuple[int, int]:
        self._clear_document(document.doc_id)
        self._document_row(document, digest, None, source, resolved, previews_root)

        self.db.executemany(
            """INSERT INTO pages (doc_id, page_no, source, confidence, needs_review, preview, bates, text, chars)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            [
                (
                    document.doc_id,
                    page.page_no,
                    page.source,
                    page.confidence,
                    int(page.needs_review),
                    page.preview,
                    page.bates,
                    page.text,
                    page.chars,
                )
                for page in document.pages
            ],
        )
        self.db.executemany(
            "INSERT INTO bates_index (bates, doc_id, page_no) VALUES (?,?,?)",
            [(p.bates, document.doc_id, p.page_no) for p in document.pages if p.bates],
        )

        chunks = [c for page in document.pages for c in chunk_page(document.doc_id, page.page_no, page.text)]
        chunk_ids: list[int] = []
        for chunk in chunks:
            cursor = self.db.execute(
                "INSERT INTO chunks (doc_id, page_no, ord, text) VALUES (?,?,?,?)",
                (chunk.doc_id, chunk.page_no, chunk.ord, chunk.text),
            )
            chunk_ids.append(int(cursor.lastrowid))
        self.db.executemany(
            "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)",
            list(zip(chunk_ids, [c.text for c in chunks])),
        )

        embedded = 0
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start : start + EMBED_BATCH]
            ids = chunk_ids[start : start + EMBED_BATCH]
            vectors = client.embed(self.settings.embed_model, [c.text for c in batch])
            self.db.executemany(
                "INSERT OR REPLACE INTO vectors (chunk_id, dim, vec) VALUES (?,?,?)",
                [(cid, len(vec), _pack(vec)) for cid, vec in zip(ids, vectors)],
            )
            embedded += len(vectors)
            emit(
                "embedding",
                file=document.title,
                done=min(start + EMBED_BATCH, len(chunks)),
                total=len(chunks),
            )
        return len(chunks), embedded

    # -------------------------------------------------------------- search

    def _forget_matrix(self) -> None:
        self._matrix = None
        self._chunk_ids = None

    def _load_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Every vector in one array, built once and kept.

        Rebuilt only after an ingest. Reading 21,000 vectors out of SQLite takes
        about a second; doing it per question would dominate the search.
        """
        if self._matrix is not None and self._chunk_ids is not None:
            return self._matrix, self._chunk_ids
        rows = self.db.execute("SELECT chunk_id, vec FROM vectors ORDER BY chunk_id").fetchall()
        if not rows:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            self._chunk_ids = np.zeros((0,), dtype=np.int64)
            return self._matrix, self._chunk_ids
        self._chunk_ids = np.array([row["chunk_id"] for row in rows], dtype=np.int64)
        self._matrix = np.vstack([np.frombuffer(row["vec"], dtype=np.float32) for row in rows])
        return self._matrix, self._chunk_ids

    def keyword_search(self, question: str, limit: int, doc_ids: Sequence[str] | None = None) -> list[tuple[int, float]]:
        query = fts_query(question)
        if not query:
            return []
        sql = """SELECT c.chunk_id AS chunk_id, bm25(chunks_fts) AS score
                 FROM chunks_fts JOIN chunks c ON c.chunk_id = chunks_fts.rowid
                 WHERE chunks_fts MATCH ?"""
        params: list[Any] = [query]
        if doc_ids:
            sql += " AND c.doc_id IN (%s)" % ",".join("?" for _ in doc_ids)
            params.extend(doc_ids)
        # bm25() is negative and more negative is better, so ascending is best
        # first. This surprises everyone once.
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        try:
            return [(int(row["chunk_id"]), float(row["score"])) for row in self.db.execute(sql, params)]
        except sqlite3.OperationalError as exc:
            log.warning("FTS query failed for %r: %s", query, exc)
            return []

    def vector_search(
        self, question: str, limit: int, doc_ids: Sequence[str] | None = None
    ) -> list[tuple[int, float]]:
        matrix, ids = self._load_matrix()
        if matrix.size == 0:
            return []
        client = Ollama(self.settings.ollama_host)
        try:
            vectors = client.embed(self.settings.embed_model, [question])
        except OllamaError as exc:
            log.warning("no vector search this time: %s", exc)
            return []
        if not vectors or not vectors[0]:
            return []
        query = np.asarray(vectors[0], dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0 or query.shape[0] != matrix.shape[1]:
            # A different embedding model was used for the query than for the
            # index. Silently returning nothing would look like "no results".
            log.warning(
                "embedding size %d does not match the index's %d — reingest after changing the embedding model",
                query.shape[0],
                matrix.shape[1],
            )
            return []
        similarities = matrix @ (query / norm)

        allowed: np.ndarray | None = None
        if doc_ids:
            rows = self.db.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id IN (%s)" % ",".join("?" for _ in doc_ids),
                list(doc_ids),
            ).fetchall()
            allowed = np.isin(ids, np.array([r["chunk_id"] for r in rows], dtype=np.int64))
            similarities = np.where(allowed, similarities, -1.0)

        take = min(limit, similarities.shape[0])
        top = np.argpartition(-similarities, take - 1)[:take] if take > 1 else np.array([int(np.argmax(similarities))])
        top = top[np.argsort(-similarities[top])]
        return [
            (int(ids[i]), float(similarities[i]))
            for i in top
            if similarities[i] >= MIN_SIMILARITY
        ]

    def search(
        self,
        question: str,
        *,
        top_k: int | None = None,
        candidates: int | None = None,
        doc_ids: Sequence[str] | None = None,
    ) -> list[Hit]:
        """Both searches, fused by reciprocal rank.

        RRF rather than a weighted sum of the two scores, because bm25 and
        cosine similarity are not on the same scale and any weighting between
        them would be a number invented to look principled. Rank position is
        comparable; the raw scores are not.
        """
        top_k = top_k or self.settings.top_k
        candidates = candidates or self.settings.candidates

        keyword = self.keyword_search(question, candidates, doc_ids)
        vector = self.vector_search(question, candidates, doc_ids)

        # Dense retrieval always returns its nearest neighbours, however far
        # away they are — ask about a knee replacement in a file that has none
        # and it still hands back twelve pages, ranked. The honest signal that
        # nothing matched is the other half: if none of the question's own
        # distinctive words appear anywhere, what follows is the closest text
        # found, not an answer. Recorded before the early return below so the
        # value is never left over from the previous search.
        self.last_search_had_keyword_match = bool(keyword)

        scores: dict[int, float] = {}
        keyword_rank: dict[int, int] = {}
        vector_rank: dict[int, int] = {}
        similarity: dict[int, float] = {}

        for rank, (chunk_id, _) in enumerate(keyword, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            keyword_rank[chunk_id] = rank
        for rank, (chunk_id, score) in enumerate(vector, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            vector_rank[chunk_id] = rank
            similarity[chunk_id] = score

        ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        if not ordered:
            return []
        return self._hydrate(ordered, keyword_rank, vector_rank, similarity)

    def _hydrate(
        self,
        ordered: list[tuple[int, float]],
        keyword_rank: dict[int, int],
        vector_rank: dict[int, int],
        similarity: dict[int, float],
    ) -> list[Hit]:
        ids = [chunk_id for chunk_id, _ in ordered]
        rows = self.db.execute(
            """SELECT c.chunk_id, c.doc_id, c.page_no, c.text,
                      d.title, p.confidence, p.needs_review, p.preview, p.bates, p.source
               FROM chunks c
               JOIN documents d ON d.doc_id = c.doc_id
               LEFT JOIN pages p ON p.doc_id = c.doc_id AND p.page_no = c.page_no
               WHERE c.chunk_id IN (%s)""" % ",".join("?" for _ in ids),
            ids,
        ).fetchall()
        by_id = {int(row["chunk_id"]): row for row in rows}
        hits: list[Hit] = []
        for chunk_id, score in ordered:
            row = by_id.get(chunk_id)
            if row is None:
                continue
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    doc_id=row["doc_id"],
                    title=row["title"],
                    page_no=int(row["page_no"]),
                    text=row["text"],
                    score=score,
                    similarity=similarity.get(chunk_id),
                    keyword_rank=keyword_rank.get(chunk_id),
                    vector_rank=vector_rank.get(chunk_id),
                    confidence=row["confidence"],
                    needs_review=bool(row["needs_review"]),
                    preview=row["preview"],
                    bates=row["bates"],
                    source=row["source"],
                )
            )
        return hits

    def whole_document_pages(self, doc_id: str) -> list[dict[str, Any]]:
        """Every page of one document, for the ask-about-this-file-only mode."""
        return self.pages(doc_id)
