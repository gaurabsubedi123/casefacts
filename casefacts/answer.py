"""Asking a model a question about the records, and checking what it says.

The design here follows from one fact: a 7B model running on a laptop will
sometimes state something the record does not say, and it will state it in the
same confident tone as everything it got right. In a medical-legal file that is
not an inconvenience, it is the whole risk.

So the model is never asked for prose. It is asked for a list of findings, and
**every finding must carry a verbatim quote from the page it cites**. Then this
module goes and checks each quote against the page text it was supposedly taken
from, before anyone reads it:

    verified    the quote is on the page it cites
    close       the quote is on the page, modulo OCR noise
    joined      every word of the quote is on the page and in that order, but
                not as one run of text — the model read across two columns of a
                layout-preserved page and joined them. Not a fabrication; also
                not something to paste into a brief as a quotation.
    wrong page  the quote is real, but it is on a different retrieved page
                (the citation is corrected and the finding is kept)
    unverified  the quote is nowhere in the retrieved pages

Nothing is thrown away silently — an unverified finding is shown, marked, and
sorted to the bottom, because knowing the model made something up is more
useful than not seeing it. The check costs no tokens and catches the failure
that matters most.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Sequence

from .config import MAX_NUM_CTX, WHOLE_DOC_MAX_CHARS, model_spec
from .index import Hit, Index
from .ollama import Ollama, OllamaError

log = logging.getLogger(__name__)

# How much of a quote has to be found on the cited page before the finding is
# called "close" rather than unverified. OCR turns "glenohumeral" into
# "glenohurneral" often enough that demanding an exact match on every quote
# would reject true findings; three quarters of a quote matching contiguously
# is well beyond what a fabrication produces.
CLOSE_ENOUGH = 0.75

# How much of a quote has to be present *in order but not contiguously* before
# it is called "joined" rather than unverified. Set high: a model reading down
# a column and across to the next produces every word in order, while an
# invented sentence does not.
JOINED_ENOUGH = 0.90

# Matching fragments shorter than this are noise — any two English texts share
# plenty of three-character runs, and counting those would let a fabrication
# accumulate coverage it did not earn.
MIN_MATCH_RUN = 4

# Quotes shorter than this cannot be checked meaningfully — "the patient" is on
# every page, so finding it proves nothing. The model is told to quote a full
# clause for this reason.
#
# Length alone is the wrong test, though: an MRN, a Bates stamp, a date or a
# dose is short *and* unique, and "SVH35492594" is exactly the kind of fact
# worth checking. So a short quote is still checkable when it carries a number,
# which is what makes it specific. The trade is that a very short numeric quote
# could match by luck; the quote is always shown next to the finding, so that
# luck is visible rather than hidden.
MIN_QUOTE_CHARS = 12
MIN_NUMERIC_QUOTE_CHARS = 5

SYSTEM_PROMPT = """You read medical and legal records and answer questions about them.

You are given numbered excerpts from OCR'd pages. The OCR is imperfect: expect \
broken words, missing punctuation, and columns run together. Read through that.

Rules, in order of importance:

1. Use ONLY the excerpts provided. You have medical knowledge; do not use it to \
add facts. If the records do not answer the question, say so.
2. Every finding must quote the excerpt it came from, word for word, copied \
exactly as it appears including any OCR errors. Do not tidy up a quote.
3. Cite the excerpt number the quote came from. Never cite an excerpt you did \
not quote.
4. A date, a name, a dose or a measurement must appear in your quote. Do not \
state one from memory.
5. If the excerpts disagree with each other, report both and say they disagree.

Reply with JSON only, in exactly this shape:

{
  "findings": [
    {"statement": "one fact, in plain English",
     "quote": "the exact words from the excerpt that establish it",
     "excerpt": 3}
  ],
  "answer": "two or three sentences answering the question from the findings",
  "missing": "what the question asked for that the excerpts do not contain, or empty"
}

If the excerpts contain nothing relevant, return an empty findings list, an \
empty answer, and say what is missing."""


@dataclass
class Finding:
    """One claim the model made, and what happened when it was checked."""

    statement: str
    quote: str
    doc_id: str = ""
    title: str = ""
    page_no: int = 0
    verdict: str = "unverified"
    coverage: float = 0.0
    confidence: float | None = None
    needs_review: bool = False
    preview: str | None = None
    bates: str | None = None

    @property
    def citation(self) -> str:
        if not self.title:
            return ""
        base = f"{self.title} p.{self.page_no}"
        return f"{base} ({self.bates})" if self.bates else base

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "quote": self.quote,
            "doc_id": self.doc_id,
            "title": self.title,
            "page": self.page_no,
            "citation": self.citation,
            "verdict": self.verdict,
            "coverage": round(self.coverage, 2),
            "confidence": self.confidence,
            "needs_review": self.needs_review,
            "preview": self.preview,
            "bates": self.bates,
        }


@dataclass
class Answer:
    """Everything one question produced, including how it was arrived at."""

    question: str
    model: str
    answer: str = ""
    missing: str = ""
    findings: list[Finding] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    mode: str = "search"
    seconds: float = 0.0
    prompt_tokens: int = 0
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def verified_count(self) -> int:
        return sum(
            1 for f in self.findings if f.verdict in ("verified", "close", "joined", "wrong page")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "model": self.model,
            "answer": self.answer,
            "missing": self.missing,
            "findings": [f.to_dict() for f in self.findings],
            "sources": [h.to_dict() for h in self.hits],
            "mode": self.mode,
            "seconds": round(self.seconds, 1),
            "prompt_tokens": self.prompt_tokens,
            "truncated": self.truncated,
            "warnings": self.warnings,
            "verified": self.verified_count,
            "claimed": len(self.findings),
        }


# ------------------------------------------------------------ quote checking


def normalise(text: str) -> str:
    """Reduce text to what two versions of the same words have in common.

    OCR disagrees with itself about punctuation, capitals, and how many spaces
    are between two words. None of that is what makes a quote true, so it is
    all removed before comparing.
    """
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def quote_coverage(quote: str, page_text: str) -> float:
    """How much of the quote is present in the page as one unbroken run, 0 to 1."""
    needle = normalise(quote)
    haystack = normalise(page_text)
    if not needle or not haystack:
        return 0.0
    if needle in haystack:
        return 1.0
    matcher = SequenceMatcher(None, needle, haystack, autojunk=False)
    match = matcher.find_longest_match(0, len(needle), 0, len(haystack))
    return match.size / len(needle)


def ordered_coverage(quote: str, page_text: str) -> float:
    """How much of the quote is on the page in order, allowing gaps, 0 to 1.

    This is what separates a model reading across a two-column page from a
    model making something up. A layout-preserved OCR page puts a column of
    room names beside a column of prices; a model quoting "Standard Double ...
    $120" has quoted the page truthfully even though those words are nowhere
    adjacent in the text. Every word is there, in that order, with the rest of
    the column in between.

    Fragments shorter than MIN_MATCH_RUN are ignored so that scattered
    coincidental characters cannot add up to a passing score.
    """
    needle = normalise(quote)
    haystack = normalise(page_text)
    if not needle or not haystack:
        return 0.0
    matcher = SequenceMatcher(None, needle, haystack, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks() if block.size >= MIN_MATCH_RUN)
    return min(1.0, matched / len(needle))


def is_checkable(quote: str) -> bool:
    """Whether this quote is specific enough that finding it means anything."""
    normalised = normalise(quote)
    if len(normalised) >= MIN_QUOTE_CHARS:
        return True
    return bool(re.search(r"\d", normalised)) and len(normalised) >= MIN_NUMERIC_QUOTE_CHARS


def check_finding(finding: Finding, cited: Hit | None, all_hits: Sequence[Hit]) -> Finding:
    """Decide whether this finding's quote is really on the page it cites."""
    if not is_checkable(finding.quote):
        finding.verdict = "unverified"
        finding.coverage = 0.0
        return finding

    if cited is not None:
        coverage = quote_coverage(finding.quote, cited.text)
        if coverage >= CLOSE_ENOUGH:
            finding.verdict = "verified" if coverage >= 0.999 else "close"
            finding.coverage = coverage
            _attach(finding, cited)
            return finding
        joined = ordered_coverage(finding.quote, cited.text)
        if joined >= JOINED_ENOUGH:
            finding.verdict = "joined"
            finding.coverage = joined
            _attach(finding, cited)
            return finding

    # The quote may be real but attributed to the wrong excerpt — a small model
    # miscounting the numbered list. That is a citation error, not a
    # fabrication, and it is worth correcting rather than discarding.
    best: tuple[float, Hit] | None = None
    for hit in all_hits:
        if cited is not None and hit.chunk_id == cited.chunk_id:
            continue
        coverage = quote_coverage(finding.quote, hit.text)
        if best is None or coverage > best[0]:
            best = (coverage, hit)
    if best and best[0] >= CLOSE_ENOUGH:
        finding.verdict = "wrong page"
        finding.coverage = best[0]
        _attach(finding, best[1])
        return finding

    finding.verdict = "unverified"
    finding.coverage = best[0] if best else 0.0
    if cited is not None:
        _attach(finding, cited)
    return finding


def _attach(finding: Finding, hit: Hit) -> None:
    finding.doc_id = hit.doc_id
    finding.title = hit.title
    finding.page_no = hit.page_no
    finding.confidence = hit.confidence
    finding.needs_review = hit.needs_review
    finding.preview = hit.preview
    finding.bates = hit.bates


# ------------------------------------------------------------------ prompting


def build_context(hits: Sequence[Hit]) -> str:
    """The excerpts, numbered, each labelled with where it came from.

    The label goes above the text rather than after it because a model that
    runs out of context loses the end of the prompt first, and losing the label
    is worse than losing the excerpt.
    """
    parts: list[str] = []
    for number, hit in enumerate(hits, start=1):
        header = f"[excerpt {number}] {hit.title}, page {hit.page_no}"
        if hit.bates:
            header += f", stamped {hit.bates}"
        if hit.confidence is not None and hit.confidence < 70:
            header += f" (OCR confidence {hit.confidence:.0f}%, read with care)"
        parts.append(f"{header}\n{hit.text.strip()}")
    return "\n\n".join(parts)


def parse_reply(text: str) -> tuple[list[dict[str, Any]], str, str]:
    """Pull findings out of the model's reply, JSON or not.

    Asking for JSON does not guarantee JSON — a model can wrap it in prose or
    stop mid-object. The fenced-block and brace-scan fallbacks exist because
    losing a good answer to a stray backtick is a bad trade.
    """
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    if not candidate.startswith("{"):
        brace = candidate.find("{")
        if brace >= 0:
            candidate = candidate[brace:]
    for attempt in (candidate, candidate + "}", candidate + "]}"):
        try:
            data = json.loads(attempt)
        except ValueError:
            continue
        if isinstance(data, dict):
            findings = data.get("findings")
            return (
                findings if isinstance(findings, list) else [],
                str(data.get("answer") or ""),
                str(data.get("missing") or ""),
            )
    return [], text.strip(), ""


# --------------------------------------------------------------------- asking


def ask(
    index: Index,
    question: str,
    *,
    model: str | None = None,
    doc: str | None = None,
    folder: str | None = None,
    top_k: int | None = None,
    whole: bool | None = None,
) -> Answer:
    """Answer one question, over everything or over one file or folder.

    `whole=True` skips retrieval and puts the entire document in the prompt —
    the "attach this file and ask about it" behaviour. It is chosen
    automatically for a document small enough to fit, because for a twenty-page
    report retrieval can only lose information.
    """
    settings = index.settings
    model = model or settings.chat_model
    spec = model_spec(model)
    client = Ollama(settings.ollama_host)
    answer = Answer(question=question, model=model)

    doc_ids = index.scope_doc_ids(doc=doc, folder=folder)
    if doc_ids is not None and not doc_ids:
        candidates = index.documents_matching(doc) if doc else []
        if len(candidates) > 1:
            names = ", ".join(c["rel_path"] or c["title"] for c in candidates[:5])
            answer.warnings.append(
                f"{doc!r} matches {len(candidates)} documents ({names}) — "
                "name one exactly, or give its full path"
            )
        else:
            answer.warnings.append(
                f"nothing indexed matches {doc or folder!r} — check the name, or plug the path in first"
            )
        return answer

    hits, mode, num_ctx = _gather(index, question, doc, doc_ids, top_k, whole, client, model)
    answer.hits = hits
    answer.mode = mode
    if not hits:
        answer.warnings.append("no page in the records matched this question")
        return answer
    if mode == "search (no word match)":
        answer.warnings.append(
            "none of the words in this question appear anywhere in the indexed pages — "
            "the excerpts below are only the closest text found, so read the citations "
            "before believing the answer"
        )

    context = build_context(hits)
    user = (
        f"Question: {question}\n\n"
        f"Excerpts from the records:\n\n{context}\n\n"
        "Answer the question using only these excerpts, in the JSON shape you were given."
    )

    try:
        reply = client.chat(
            model,
            SYSTEM_PROMPT,
            user,
            num_ctx=num_ctx,
            temperature=settings.temperature,
            json_format=True,
            strip_think=bool(spec.strips_thinking) if spec else True,
        )
    except OllamaError as exc:
        answer.warnings.append(str(exc))
        return answer

    answer.seconds = reply.seconds
    answer.prompt_tokens = reply.prompt_tokens
    answer.truncated = reply.truncated
    answer.raw = reply.text
    if reply.truncated:
        answer.warnings.append(
            "the prompt filled the model's context window, so some excerpts may not have been read — "
            "ask a narrower question or lower the number of excerpts"
        )

    raw_findings, prose, missing = parse_reply(reply.text)
    answer.answer = prose.strip()
    answer.missing = missing.strip()

    for item in raw_findings:
        if not isinstance(item, dict):
            continue
        statement = str(item.get("statement") or item.get("fact") or "").strip()
        quote = str(item.get("quote") or "").strip()
        if not statement and not quote:
            continue
        cited = _hit_for(item.get("excerpt"), hits)
        answer.findings.append(
            check_finding(Finding(statement=statement, quote=quote), cited, hits)
        )

    # Verified first, and within that in the order the model gave them. An
    # unverified finding stays visible, at the bottom, marked.
    order = {"verified": 0, "close": 1, "joined": 2, "wrong page": 3, "unverified": 4}
    answer.findings.sort(key=lambda f: order.get(f.verdict, 4))

    unverified = sum(1 for f in answer.findings if f.verdict == "unverified")
    if unverified:
        answer.warnings.append(
            f"{unverified} of {len(answer.findings)} findings quote text that is not on the page they cite — "
            "treat those as unsupported"
        )
    if not answer.findings and answer.answer:
        answer.warnings.append(
            "the model answered in prose without quoting the records, so nothing here has been checked"
        )
    return answer


def _hit_for(excerpt: Any, hits: Sequence[Hit]) -> Hit | None:
    """The excerpt the model says it quoted, if that number makes sense."""
    try:
        number = int(str(excerpt).strip().lstrip("#[").rstrip("]"))
    except (TypeError, ValueError):
        return None
    if 1 <= number <= len(hits):
        return hits[number - 1]
    return None


def _gather(
    index: Index,
    question: str,
    doc: str | None,
    doc_ids: Sequence[str] | None,
    top_k: int | None,
    whole: bool | None,
    client: Ollama,
    model: str,
) -> tuple[list[Hit], str, int]:
    """The excerpts to answer from, and how much context they need.

    Two ways in. If the question is about one document that fits in the model's
    window, the whole document goes in and nothing is retrieved — this is what
    makes asking about a single report as reliable as reading it. Otherwise the
    hybrid search picks the pages.
    """
    settings = index.settings
    single = doc_ids[0] if (doc and doc_ids and len(doc_ids) == 1) else None

    if single:
        pages = index.whole_document_pages(single)
        total = sum(len(p["text"] or "") for p in pages)
        fits = total <= WHOLE_DOC_MAX_CHARS
        if whole or (whole is None and fits):
            if not fits and whole:
                log.warning("document is %d chars; it will not fit and will be truncated", total)
            record = index.document(single) or {}
            hits = [
                Hit(
                    chunk_id=-(i + 1),
                    doc_id=single,
                    title=record.get("title") or single,
                    page_no=int(page["page_no"]),
                    text=page["text"] or "",
                    confidence=page["confidence"],
                    needs_review=bool(page["needs_review"]),
                    preview=page["preview"],
                    bates=page["bates"],
                    source=page["source"],
                )
                for i, page in enumerate(pages)
                if (page["text"] or "").strip()
            ]
            # About 3.6 OCR'd characters per token, plus the instructions and
            # room for the answer, rounded up to something the model will
            # actually allocate.
            needed = int(total / 3.2) + 1200
            limit = client.context_limit(model) or MAX_NUM_CTX
            num_ctx = max(settings.num_ctx, min(needed, MAX_NUM_CTX, limit))
            return hits, "whole document", num_ctx

    hits = index.search(question, top_k=top_k, doc_ids=doc_ids)
    hits = _pin_bates(index, question, hits, doc_ids)
    return hits, "search" if index.last_search_had_keyword_match else "search (no word match)", settings.num_ctx


# A production stamp typed into a question is a request for that exact page,
# and no amount of semantic similarity substitutes for looking it up.
BATES_IN_QUESTION = re.compile(r"\b([A-Z][A-Z&._-]{2,15}[ _-]?\d{3,10})\b")


def _pin_bates(
    index: Index, question: str, hits: list[Hit], doc_ids: Sequence[str] | None
) -> list[Hit]:
    """Put any page named by its stamp in the question at the top of the pile."""
    for candidate in BATES_IN_QUESTION.findall(question.upper()):
        page = index.by_bates(candidate.replace(" ", "").replace("_", ""))
        if not page:
            continue
        if doc_ids is not None and page["doc_id"] not in doc_ids:
            continue
        if any(h.doc_id == page["doc_id"] and h.page_no == page["page_no"] for h in hits):
            continue
        record = index.document(page["doc_id"]) or {}
        hits.insert(
            0,
            Hit(
                chunk_id=-9000 - len(hits),
                doc_id=page["doc_id"],
                title=record.get("title") or page["doc_id"],
                page_no=int(page["page_no"]),
                text=page["text"] or "",
                score=1.0,
                confidence=page["confidence"],
                needs_review=bool(page["needs_review"]),
                preview=page["preview"],
                bates=page["bates"],
                source=page["source"],
            ),
        )
    return hits
