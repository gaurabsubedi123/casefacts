"""Asking the chosen model about the chosen documents, and checking what it says.

The same discipline as casefacts, because the risk is the same: a small local
model will sometimes state something the document does not say, in exactly
the tone of everything it got right. So it is never asked for prose. It is
asked for findings, each carrying a verbatim quote from a numbered excerpt,
and each quote is looked for on that page before anyone reads it:

    verified    the quote is on the page it cites, word for word
    close       on that page, allowing for OCR mangling the characters
    joined      every word is on the page in order, but not as one run — the
                model read across two columns. True to the page; not a quotation.
    wrong page  the quote is real but on a different excerpt; citation corrected
    unverified  the quote is in none of the excerpts. Shown, marked, sorted last.

The check costs no tokens and catches the failure that matters most.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any, Callable, Sequence

from .documents import Library
from .ollama import Ollama
from .retrieve import Chunk, gather

CLOSE_ENOUGH = 0.75
JOINED_ENOUGH = 0.90
MIN_MATCH_RUN = 4
MIN_QUOTE_CHARS = 12
MIN_NUMERIC_QUOTE_CHARS = 5

# Tokens held back from the context window for the reply. Reasoning models
# spend a lot before they answer; running out mid-thought produces nothing.
REPLY_TOKENS = 1600
THINKING_REPLY_TOKENS = 4500
SYSTEM_TOKENS = 450

# How much of the conversation so far goes back in with a follow-up. Enough to
# know who "he" or "that visit" is; not so much it crowds out the pages.
HISTORY_TURNS = 4
HISTORY_ANSWER_CHARS = 600

SYSTEM_PROMPT = """You answer questions about documents using only what they say.

You are given numbered excerpts from the pages of one or more documents. Many \
were OCR'd from scans: expect broken words, missing punctuation and columns run \
together. Read through that.

Rules, in order of importance:

1. Use ONLY the excerpts. Do not add facts from your own knowledge. If the \
excerpts do not answer the question, say so.
2. Every finding must quote the excerpt it came from, word for word, exactly as \
it appears, including OCR errors. Do not tidy up a quote.
3. Cite the number of the excerpt you quoted. Never cite one you did not quote.
4. Any date, name, number or amount you state must appear in your quote.
5. If excerpts disagree, report both and say that they disagree.
6. You may be shown earlier questions and answers from this conversation. Use \
them only to understand what a follow-up refers to ("he", "that visit", "the \
second one"). They are not evidence: every fact still comes from the excerpts.

Reply with JSON only, in exactly this shape:

{
  "answer": "two to four sentences answering the question from what the excerpts say",
  "findings": [
    {"statement": "one fact, in plain English",
     "quote": "the exact words from the excerpt that establish it",
     "excerpt": 3}
  ],
  "missing": "what the question asked for that the excerpts do not contain, or empty"
}

Every fact in the answer must be backed by one of the findings you list after it.

If nothing in the excerpts is relevant, return an empty findings list, an empty \
answer, and say what is missing."""


class AskError(ValueError):
    """The question cannot be asked as given — shown to the person as-is."""


# ------------------------------------------------------------ quote checks


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def quote_coverage(quote: str, page: str) -> float:
    needle, haystack = normalise(quote), normalise(page)
    if not needle or not haystack:
        return 0.0
    if needle in haystack:
        return 1.0
    matcher = SequenceMatcher(None, needle, haystack, autojunk=False)
    return matcher.find_longest_match(0, len(needle), 0, len(haystack)).size / len(needle)


def ordered_coverage(quote: str, page: str) -> float:
    needle, haystack = normalise(quote), normalise(page)
    if not needle or not haystack:
        return 0.0
    matcher = SequenceMatcher(None, needle, haystack, autojunk=False)
    matched = sum(b.size for b in matcher.get_matching_blocks() if b.size >= MIN_MATCH_RUN)
    return min(1.0, matched / len(needle))


def is_checkable(quote: str) -> bool:
    text = normalise(quote)
    if len(text) >= MIN_QUOTE_CHARS:
        return True
    return bool(re.search(r"\d", text)) and len(text) >= MIN_NUMERIC_QUOTE_CHARS


def check(finding: dict[str, Any], excerpts: Sequence[Chunk], page_text: Callable[[Chunk], str]) -> dict[str, Any]:
    """Set verdict, coverage and the page actually cited on one finding."""
    quote = str(finding.get("quote") or "")
    try:
        number = int(finding.get("excerpt"))
    except (TypeError, ValueError):
        number = 0
    cited = excerpts[number - 1] if 1 <= number <= len(excerpts) else None
    result = {"statement": str(finding.get("statement") or "").strip(), "quote": quote.strip(),
              "verdict": "unverified", "coverage": 0.0, "doc_id": "", "title": "", "page": 0}

    def attach(chunk: Chunk) -> None:
        result.update(doc_id=chunk.doc_id, title=chunk.title, page=chunk.page)

    if not is_checkable(quote):
        if cited:
            attach(cited)
        return result
    if cited:
        # Checked against the whole page, not only the excerpt, so a quote that
        # ran just past the chunk's edge is still found where it really is.
        text = page_text(cited)
        coverage = quote_coverage(quote, text)
        if coverage >= CLOSE_ENOUGH:
            attach(cited)
            result.update(verdict="verified" if coverage >= 0.999 else "close", coverage=coverage)
            return result
        joined = ordered_coverage(quote, text)
        if joined >= JOINED_ENOUGH:
            attach(cited)
            result.update(verdict="joined", coverage=joined)
            return result
    best: tuple[float, Chunk] | None = None
    for chunk in excerpts:
        if chunk is cited:
            continue
        coverage = quote_coverage(quote, page_text(chunk))
        if best is None or coverage > best[0]:
            best = (coverage, chunk)
    if best and best[0] >= CLOSE_ENOUGH:
        attach(best[1])
        result.update(verdict="wrong page", coverage=best[0])
        return result
    if cited:
        attach(cited)
    result["coverage"] = best[0] if best else 0.0
    return result


VERDICT_ORDER = {"verified": 0, "close": 1, "wrong page": 2, "joined": 3, "unverified": 4}


# ---------------------------------------------------------------- prompting


def build_context(excerpts: Sequence[Chunk]) -> str:
    return "\n\n".join(
        f"[excerpt {n}] {c.title}, page {c.page}\n{c.text}" for n, c in enumerate(excerpts, start=1)
    )


PARTIAL_ANSWER = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)')


def partial_answer(text: str) -> str:
    """The answer so far, out of a reply that is still being written.

    The reply is JSON that is not finished yet, so it cannot be parsed; the
    answer string is picked out as far as it goes and its escapes undone.
    """
    match = PARTIAL_ANSWER.search(text)
    if not match:
        return ""
    raw = match.group(1)
    if raw.endswith("\\") and not raw.endswith("\\\\"):
        raw = raw[:-1]  # half an escape: wait for the rest
    try:
        return json.loads(f'"{raw}"')
    except ValueError:
        return raw.replace('\\"', '"').replace("\\n", "\n")


def history_block(history: Sequence[dict[str, str]]) -> str:
    """Earlier turns, as plain text in the prompt.

    Not as chat messages: earlier answers are prose, and a model shown its own
    prose replies tends to drop the JSON shape it was asked for.
    """
    lines = []
    for turn in list(history)[-HISTORY_TURNS:]:
        answer = " ".join((turn.get("answer") or "(no answer found)").split())
        if len(answer) > HISTORY_ANSWER_CHARS:
            answer = answer[:HISTORY_ANSWER_CHARS - 1] + "…"
        lines.append(f"Q: {' '.join(turn['question'].split())}\nA: {answer}")
    return "\n\n".join(lines)


def search_text(question: str, history: Sequence[dict[str, str]]) -> str:
    """What to look for in the pages. A follow-up such as "and the second
    visit?" shares few words with the pages on its own, so the last question
    before it goes in too."""
    earlier = [t["question"] for t in list(history)[-1:]]
    return " ".join(earlier + [question])


def parse_reply(text: str) -> tuple[list[dict[str, Any]], str, str]:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    if not candidate.startswith("{"):
        brace = candidate.find("{")
        if brace >= 0:
            candidate = candidate[brace:]
    for attempt in (candidate, candidate + "}", candidate + "]}", candidate + '"}]}'):
        try:
            data = json.loads(attempt)
        except ValueError:
            continue
        if isinstance(data, dict):
            findings = data.get("findings")
            return (
                [f for f in findings if isinstance(f, dict)] if isinstance(findings, list) else [],
                str(data.get("answer") or ""),
                str(data.get("missing") or ""),
            )
    return [], text.strip(), ""


# ------------------------------------------------------------------ asking


def validate(library: Library, question: str, doc_ids: Sequence[str], model: str):
    """Refuse a question that cannot be answered, with a message saying why.

    Called before any work starts, so the person sees the problem at once
    rather than after a model has loaded.
    """
    question = (question or "").strip()
    if not question:
        raise AskError("Type a question first.")
    if not model:
        raise AskError("No model is plugged in. Pick one on the Models tab, or download one there first.")
    doc_ids = list(dict.fromkeys(d for d in doc_ids if d))
    if not doc_ids:
        raise AskError("No document is attached. Add an OCR'd PDF or text file on the Documents tab, "
                       "then tick it in the list before asking.")
    documents = [library.get(d) for d in doc_ids]
    missing = [d for d, doc in zip(doc_ids, documents) if doc is None]
    if missing:
        raise AskError("A selected document is no longer in the library. Refresh the page and select again.")
    readable = [doc for doc in documents if doc and doc.chars and not doc.needs_ocr]
    if not readable:
        raise AskError("None of the selected documents has OCR text to answer from. "
                       "Run the scans through ocrtool first, then add the OCR'd files.")

    return documents, readable


def ask(
    client: Ollama,
    library: Library,
    question: str,
    doc_ids: Sequence[str],
    model: str,
    context_tokens: int,
    *,
    embed_model: str | None = None,
    history: Sequence[dict[str, str]] = (),
    on_progress: Callable[[dict[str, Any]], None] = lambda _: None,
    should_stop: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    question = question.strip()
    documents, readable = validate(library, question, doc_ids, model)

    capabilities = client.capabilities(model)
    thinking = "thinking" in capabilities
    earlier = history_block(history)
    reserve = SYSTEM_TOKENS + (THINKING_REPLY_TOKENS if thinking else REPLY_TOKENS) + len(earlier) // 3

    on_progress({"phase": "finding pages"})
    excerpts, mode = gather(
        library, search_text(question, history), [d.id for d in readable],
        context_tokens=context_tokens, reserve_tokens=reserve,
        embed_model=embed_model, embed=client.embed if embed_model else None,
        progress=lambda message: on_progress({"phase": message}),
    )
    result: dict[str, Any] = {"question": question, "model": model, "mode": mode, "findings": [],
                              "answer": "", "missing": "", "warnings": [], "excerpts": [],
                              "thinking": ""}
    if not excerpts:
        result["warnings"].append(
            "No page in the selected documents shares a word with this question. "
            "Try the words the document itself would use."
            + ("" if embed_model else " (Installing an embedding model such as nomic-embed-text "
                                       "on the Models tab adds search by meaning.)")
        )
        return result

    user = (f"Earlier in this conversation:\n\n{earlier}\n\n" if earlier else "") + (
            f"Question: {question}\n\nExcerpts:\n\n{build_context(excerpts)}\n\n"
            "Answer the question using only these excerpts, in the JSON shape you were given.")
    on_progress({"phase": "reading", "excerpts": len(excerpts)})

    def relay(progress: dict[str, Any]) -> None:
        text = progress.pop("text", "")
        # Reasoning left inline may draft the JSON too; only what follows it counts.
        if "<think>" in text:
            text = text.split("</think>", 1)[1] if "</think>" in text else ""
        progress["partial"] = partial_answer(text)
        on_progress(progress)

    reply = client.chat(
        model,
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        num_ctx=context_tokens,
        think=True if thinking else None,
        # A JSON grammar and free reasoning fight each other; reasoning models
        # are asked for JSON in words and parsed leniently instead.
        json_format=not thinking,
        on_progress=relay,
        should_stop=should_stop,
    )
    findings_raw, answer_text, missing_text = parse_reply(reply["text"])
    pages_cache: dict[tuple[str, int], str] = {}

    def page_text(chunk: Chunk) -> str:
        key = (chunk.doc_id, chunk.page)
        if key not in pages_cache:
            pages = library.pages(chunk.doc_id)
            pages_cache[key] = pages[chunk.page - 1] if 0 < chunk.page <= len(pages) else chunk.text
        return pages_cache[key]

    findings = [check(f, excerpts, page_text) for f in findings_raw if f.get("statement") or f.get("quote")]
    findings.sort(key=lambda f: VERDICT_ORDER.get(f["verdict"], 9))
    for f in findings:
        f["coverage"] = round(f["coverage"], 2)

    result.update(
        findings=findings,
        answer=answer_text.strip(),
        missing=missing_text.strip(),
        thinking=reply["thinking"],
        seconds=reply["seconds"],
        prompt_tokens=reply["prompt_tokens"],
        excerpts=[c.to_dict() for c in excerpts],
        verified=sum(1 for f in findings if f["verdict"] != "unverified"),
    )
    if reply["truncated"]:
        result["warnings"].append("The prompt filled the model's whole context window, so the start of it "
                                  "may have been cut off. Select fewer documents or ask a narrower question.")
    if not findings and not answer_text and reply["text"] and not missing_text:
        result["warnings"].append("The model did not reply in the expected format. Its raw reply is shown.")
        result["answer"] = reply["text"][:4000]
    unreadable = [doc.name for doc in documents if doc and doc not in readable]
    if unreadable:
        result["warnings"].append("Skipped, no OCR text: " + ", ".join(unreadable))
    return result
