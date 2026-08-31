"""Building a medical chronology: every dated event in the file, with its page.

Question-and-answer is how you check a fact you already suspect. A chronology
is how you find the facts you did not know to ask about, and in an injury case
it is the document everything else is built on — the treatment timeline, the
gaps in treatment the other side will point at, the date the complaint of pain
first appears in a record.

This is deliberately *not* retrieval. Retrieval answers a question by finding
the best few pages; a chronology has to be right about all of them, so every
page is read once, in order. That costs one model call per page, so:

* Pages with no date on them are skipped without a model call. An event needs a
  date to be an event, and this removes most of the exhibit sheets, fax covers
  and signature pages in a produced file.
* Results are written to the index as they are produced, keyed by page and
  model. A sweep that is interrupted — or a machine that goes to sleep — picks
  up where it stopped instead of starting again.
* Every extracted event carries the quote it came from and is checked against
  the page, exactly as an answer's findings are.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Sequence

from .answer import is_checkable, quote_coverage, CLOSE_ENOUGH
from .config import model_spec
from .index import Index
from .ollama import Ollama, OllamaError

log = logging.getLogger(__name__)

EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id       TEXT NOT NULL,
    page_no      INTEGER NOT NULL,
    date_text    TEXT,
    date_iso     TEXT,
    provider     TEXT,
    facility     TEXT,
    description  TEXT NOT NULL,
    category     TEXT,
    quote        TEXT,
    verdict      TEXT,
    model        TEXT,
    extracted_at TEXT
);
CREATE INDEX IF NOT EXISTS events_by_date ON events (date_iso);
CREATE INDEX IF NOT EXISTS events_by_page ON events (doc_id, page_no);

CREATE TABLE IF NOT EXISTS page_extractions (
    doc_id       TEXT NOT NULL,
    page_no      INTEGER NOT NULL,
    model        TEXT NOT NULL,
    events       INTEGER DEFAULT 0,
    extracted_at TEXT,
    PRIMARY KEY (doc_id, page_no, model)
);
"""

# A page with no date cannot contribute a dated event. Recognising that without
# a model call is what makes a sweep of a 2,000 page file affordable.
DATE_HINT = re.compile(
    r"""(\b\d{1,2}\s*[/-]\s*\d{1,2}\s*[/-]\s*\d{2,4}\b)          # 7/6/2018, 07-06-18
      | (\b\d{4}-\d{2}-\d{2}\b)                                   # 2018-07-06
      | (\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}\b)
      | (\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{2,4}\b)
    """,
    re.IGNORECASE | re.VERBOSE,
)

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# A page needs some text before it is worth a model call; a page holding only a
# stamp and a date is a separator sheet.
MIN_PAGE_CHARS = 80

# How long a silence between two events is worth calling a gap. Thirty days is
# the interval a defence medical examiner starts describing as a break in
# treatment, which is exactly why you want to see it before they do.
GAP_DAYS = 30

SYSTEM_PROMPT = """You extract dated events from one page of a medical or legal record.

The page is OCR'd, so expect broken words and columns run together.

Extract only events that happened on a date **written on this page**. A visit, \
an examination, an imaging study, a procedure, a prescription, a referral, a \
report, a letter, a phone call, an accident. If the page states a date but no \
event, extract nothing.

For each event give the exact words from the page that establish it. Copy them \
as they appear, OCR errors included. Never write a date that is not on the page.

Reply with JSON only:

{"events": [
  {"date": "the date exactly as written on the page",
   "event": "what happened, in a short phrase",
   "provider": "the clinician named, or empty",
   "facility": "the hospital or clinic named, or empty",
   "category": "one of: accident, emergency, visit, imaging, therapy, procedure, prescription, referral, report, correspondence, billing, legal, other",
   "quote": "the exact words from the page"}
]}

If nothing on the page is a dated event, reply {"events": []}."""


@dataclass
class Event:
    """One dated thing that happened, and the page that says so."""

    doc_id: str
    page_no: int
    date_text: str
    date_iso: str | None
    description: str
    provider: str = ""
    facility: str = ""
    category: str = "other"
    quote: str = ""
    verdict: str = "unverified"
    title: str = ""
    bates: str | None = None
    preview: str | None = None
    confidence: float | None = None

    @property
    def citation(self) -> str:
        base = f"{self.title} p.{self.page_no}" if self.title else f"p.{self.page_no}"
        return f"{base} ({self.bates})" if self.bates else base

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "page": self.page_no,
            "date_text": self.date_text,
            "date": self.date_iso,
            "event": self.description,
            "provider": self.provider,
            "facility": self.facility,
            "category": self.category,
            "quote": self.quote,
            "verdict": self.verdict,
            "citation": self.citation,
            "title": self.title,
            "bates": self.bates,
            "preview": self.preview,
            "confidence": self.confidence,
        }


@dataclass
class Gap:
    """A stretch of time with no record in the file."""

    after: str
    before: str
    days: int

    def to_dict(self) -> dict[str, Any]:
        return {"after": self.after, "before": self.before, "days": self.days}


@dataclass
class SweepReport:
    pages_read: int = 0
    pages_skipped: int = 0
    pages_cached: int = 0
    events: int = 0
    unverified: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages_read": self.pages_read,
            "pages_skipped": self.pages_skipped,
            "pages_cached": self.pages_cached,
            "events": self.events,
            "unverified": self.unverified,
            "seconds": round(self.seconds, 1),
            "errors": self.errors,
        }


def has_date(text: str) -> bool:
    return bool(DATE_HINT.search(text))


def parse_date(raw: str, *, today: date | None = None) -> str | None:
    """A date written any of the ways a clinic writes one, as YYYY-MM-DD.

    Returns None rather than guessing when the text is not a date. Sorting a
    chronology on a wrong date is worse than leaving the row unsorted, so an
    ambiguous string is left as text and shown as written.

    Two-digit years are read as 19xx when they would otherwise be in the
    future — a record dated 7/6/99 is 1999, not 2099.
    """
    if not raw:
        return None
    text = raw.strip()
    today = today or date.today()

    match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if match:
        return _iso(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    # US order: these are US medical records, where 7/6/2018 is 6 July.
    match = re.search(r"\b(\d{1,2})\s*[/-]\s*(\d{1,2})\s*[/-]\s*(\d{2,4})\b", text)
    if match:
        month, day, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if month > 12 and day <= 12:
            month, day = day, month  # written the other way round
        return _iso(_full_year(year, today), month, day)

    match = re.search(
        r"\b([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{2,4})\b", text, re.IGNORECASE
    )
    if match:
        month = MONTHS.get(match.group(1)[:4].lower()) or MONTHS.get(match.group(1)[:3].lower())
        if month:
            return _iso(_full_year(int(match.group(3)), today), month, int(match.group(2)))

    match = re.search(
        r"\b(\d{1,2})\s+([a-z]{3,9})\.?,?\s+(\d{2,4})\b", text, re.IGNORECASE
    )
    if match:
        month = MONTHS.get(match.group(2)[:4].lower()) or MONTHS.get(match.group(2)[:3].lower())
        if month:
            return _iso(_full_year(int(match.group(3)), today), month, int(match.group(1)))
    return None


def _full_year(year: int, today: date) -> int:
    if year >= 100:
        return year
    century = today.year - today.year % 100
    candidate = century + year
    return candidate - 100 if candidate > today.year else candidate


def _iso(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def ensure_schema(index: Index) -> None:
    index.db.executescript(EVENTS_SCHEMA)
    index.db.commit()


def sweep(
    index: Index,
    *,
    model: str | None = None,
    doc: str | None = None,
    folder: str | None = None,
    rebuild: bool = False,
    limit: int | None = None,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> SweepReport:
    """Read every page once and record what happened on it.

    Resumable: a page already read by this model is not read again unless
    `rebuild` is set. Interrupt it with Ctrl-C and run it again.
    """
    ensure_schema(index)
    settings = index.settings
    model = model or settings.chat_model
    spec = model_spec(model)
    client = Ollama(settings.ollama_host)
    report = SweepReport()
    started = time.monotonic()

    def emit(event: str, **payload: Any) -> None:
        if progress:
            progress(event, payload)

    doc_ids = index.scope_doc_ids(doc=doc, folder=folder)
    pages = _pages_to_read(index, doc_ids)
    if rebuild:
        ids = doc_ids or [d["doc_id"] for d in index.documents()]
        for doc_id in ids:
            index.db.execute("DELETE FROM events WHERE doc_id = ? AND model = ?", (doc_id, model))
            index.db.execute(
                "DELETE FROM page_extractions WHERE doc_id = ? AND model = ?", (doc_id, model)
            )
        index.db.commit()

    done = {
        (row["doc_id"], row["page_no"])
        for row in index.db.execute(
            "SELECT doc_id, page_no FROM page_extractions WHERE model = ?", (model,)
        )
    }

    emit("start", pages=len(pages), model=model)
    read = 0
    for position, page in enumerate(pages, start=1):
        key = (page["doc_id"], page["page_no"])
        text = page["text"] or ""

        if key in done and not rebuild:
            report.pages_cached += 1
            continue
        if len(text) < MIN_PAGE_CHARS or not has_date(text):
            report.pages_skipped += 1
            _mark_read(index, key, model, 0)
            continue
        if limit is not None and read >= limit:
            break

        emit(
            "page",
            title=page["title"],
            page=page["page_no"],
            position=position,
            total=len(pages),
            read=read,
        )
        try:
            events = _read_page(client, model, spec, page, index.settings)
        except OllamaError as exc:
            report.errors.append(f"p.{page['page_no']}: {exc}")
            emit("error", page=page["page_no"], error=str(exc))
            continue

        _store(index, key, model, events)
        report.pages_read += 1
        report.events += len(events)
        report.unverified += sum(1 for e in events if e.verdict == "unverified")
        read += 1
        index.db.commit()

    report.seconds = time.monotonic() - started
    emit("done", **report.to_dict())
    return report


def _pages_to_read(index: Index, doc_ids: Sequence[str] | None) -> list[dict[str, Any]]:
    sql = """SELECT p.doc_id, p.page_no, p.text, p.bates, p.preview, p.confidence, d.title
             FROM pages p JOIN documents d ON d.doc_id = p.doc_id
             WHERE d.duplicate_of IS NULL"""
    params: list[Any] = []
    if doc_ids:
        sql += " AND p.doc_id IN (%s)" % ",".join("?" for _ in doc_ids)
        params.extend(doc_ids)
    sql += " ORDER BY d.title, p.page_no"
    return [dict(row) for row in index.db.execute(sql, params)]


def _read_page(
    client: Ollama, model: str, spec: Any, page: dict[str, Any], settings: Any
) -> list[Event]:
    text = page["text"] or ""
    reply = client.chat(
        model,
        SYSTEM_PROMPT,
        f"Page {page['page_no']} of {page['title']}:\n\n{text}",
        num_ctx=settings.num_ctx,
        temperature=0.0,
        json_format=True,
        strip_think=bool(spec.strips_thinking) if spec else True,
    )
    try:
        data = json.loads(reply.text)
    except ValueError:
        log.debug("page %s: reply was not JSON: %.200s", page["page_no"], reply.text)
        return []
    raw_events = data.get("events") if isinstance(data, dict) else None
    if not isinstance(raw_events, list):
        return []

    events: list[Event] = []
    for item in raw_events:
        if not isinstance(item, dict):
            continue
        description = str(item.get("event") or "").strip()
        date_text = str(item.get("date") or "").strip()
        if not description or not date_text:
            continue
        quote = str(item.get("quote") or "").strip()

        # Two checks, both cheap. The quote has to be on the page, and the date
        # has to be on the page — a model that invents one invents the other,
        # and a chronology sorted on an invented date is worse than no
        # chronology.
        coverage = quote_coverage(quote, text) if is_checkable(quote) else 0.0
        date_on_page = bool(date_text) and normalised_contains(text, date_text)
        if coverage >= CLOSE_ENOUGH and date_on_page:
            verdict = "verified"
        elif coverage >= CLOSE_ENOUGH:
            verdict = "date not on page"
        else:
            verdict = "unverified"

        events.append(
            Event(
                doc_id=page["doc_id"],
                page_no=int(page["page_no"]),
                date_text=date_text,
                date_iso=parse_date(date_text),
                description=description,
                provider=str(item.get("provider") or "").strip(),
                facility=str(item.get("facility") or "").strip(),
                category=str(item.get("category") or "other").strip().lower(),
                quote=quote,
                verdict=verdict,
                title=page["title"],
                bates=page["bates"],
                preview=page["preview"],
                confidence=page["confidence"],
            )
        )
    return events


def normalised_contains(page_text: str, needle: str) -> bool:
    """Is this date on the page, allowing for how OCR spaces things out?"""
    from .answer import normalise

    return normalise(needle) in normalise(page_text) if needle.strip() else False


def _mark_read(index: Index, key: tuple[str, int], model: str, count: int) -> None:
    index.db.execute(
        """INSERT INTO page_extractions (doc_id, page_no, model, events, extracted_at)
           VALUES (?,?,?,?,?)
           ON CONFLICT(doc_id, page_no, model) DO UPDATE SET
             events = excluded.events, extracted_at = excluded.extracted_at""",
        (key[0], key[1], model, count, time.strftime("%Y-%m-%dT%H:%M:%S")),
    )


def _store(index: Index, key: tuple[str, int], model: str, events: Sequence[Event]) -> None:
    index.db.execute(
        "DELETE FROM events WHERE doc_id = ? AND page_no = ? AND model = ?", (key[0], key[1], model)
    )
    index.db.executemany(
        """INSERT INTO events
           (doc_id, page_no, date_text, date_iso, provider, facility, description,
            category, quote, verdict, model, extracted_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                e.doc_id, e.page_no, e.date_text, e.date_iso, e.provider, e.facility,
                e.description, e.category, e.quote, e.verdict, model,
                time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            for e in events
        ],
    )
    _mark_read(index, key, model, len(events))


def timeline(
    index: Index,
    *,
    model: str | None = None,
    doc: str | None = None,
    folder: str | None = None,
    include_unverified: bool = False,
) -> list[Event]:
    """Every extracted event, in date order, undated ones last."""
    ensure_schema(index)
    model = model or index.settings.chat_model
    doc_ids = index.scope_doc_ids(doc=doc, folder=folder)

    sql = """SELECT e.*, d.title, p.bates, p.preview, p.confidence
             FROM events e
             JOIN documents d ON d.doc_id = e.doc_id
             LEFT JOIN pages p ON p.doc_id = e.doc_id AND p.page_no = e.page_no
             WHERE e.model = ?"""
    params: list[Any] = [model]
    if not include_unverified:
        sql += " AND e.verdict != 'unverified'"
    if doc_ids:
        sql += " AND e.doc_id IN (%s)" % ",".join("?" for _ in doc_ids)
        params.extend(doc_ids)

    events = [
        Event(
            doc_id=row["doc_id"],
            page_no=int(row["page_no"]),
            date_text=row["date_text"] or "",
            date_iso=row["date_iso"],
            description=row["description"],
            provider=row["provider"] or "",
            facility=row["facility"] or "",
            category=row["category"] or "other",
            quote=row["quote"] or "",
            verdict=row["verdict"] or "unverified",
            title=row["title"] or "",
            bates=row["bates"],
            preview=row["preview"],
            confidence=row["confidence"],
        )
        for row in index.db.execute(sql, params)
    ]
    events.sort(key=lambda e: (e.date_iso is None, e.date_iso or "", e.page_no))
    return _dedupe(events)


def _dedupe(events: list[Event]) -> list[Event]:
    """Collapse the same event extracted twice.

    A discharge summary repeats the admission date on every page, so the same
    event legitimately appears many times. Same date and same description is
    one row; the page it keeps is the first one, which is where a reader should
    look.
    """
    seen: set[tuple[str, str]] = set()
    out: list[Event] = []
    for event in events:
        key = (event.date_iso or event.date_text, event.description.lower().strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(event)
    return out


def gaps(events: Sequence[Event], *, days: int = GAP_DAYS) -> list[Gap]:
    """Stretches with no record between two dated events."""
    dated = [e for e in events if e.date_iso]
    found: list[Gap] = []
    for earlier, later in zip(dated, dated[1:]):
        delta = (
            datetime.fromisoformat(later.date_iso).date()
            - datetime.fromisoformat(earlier.date_iso).date()
        ).days
        if delta >= days:
            found.append(Gap(after=earlier.date_iso, before=later.date_iso, days=delta))
    return found


def to_csv(events: Sequence[Event]) -> str:
    """The chronology as a spreadsheet, because that is where it ends up."""
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["Date", "As written", "Event", "Provider", "Facility", "Category",
         "Document", "Page", "Bates", "Checked", "Quote"]
    )
    for e in events:
        writer.writerow(
            [e.date_iso or "", e.date_text, e.description, e.provider, e.facility,
             e.category, e.title, e.page_no, e.bates or "", e.verdict, e.quote]
        )
    return buffer.getvalue()
