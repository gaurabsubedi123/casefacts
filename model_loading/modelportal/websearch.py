"""Searching the web, when a question asks for it.

Off unless the person ticks "Search the web" for a question. What leaves the
computer is one search query — never a document — and the query is shown with
the answer, so it is plain what was sent. The model writes the query from the
question, told to leave out names and other personal details.

Results come from Ollama's web search (a free ollama.com account key). Each
result is kept as a page: saved with the chat, so a follow-up can quote it
again without searching again, and checked quote by quote like a document
page.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from typing import Any, Callable
from urllib.parse import urlparse

SEARCH_URL = "https://ollama.com/api/web_search"
FETCH_URL = "https://ollama.com/api/web_fetch"
KEY_PAGE = "https://ollama.com/settings/keys"
SIGNUP_PAGE = "https://ollama.com/signup"

RESULTS = 5
# A search result shorter than this is a snippet; the page itself is fetched.
SNIPPET_CHARS = 800
PAGE_CHARS = 12000
QUERY_WORDS = 14


class WebError(RuntimeError):
    """Web search failed — shown to the person as-is."""


def page_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def site_of(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def is_web_url(url: str) -> bool:
    return urlparse(url).scheme in ("http", "https") and bool(urlparse(url).netloc)


class WebSearch:
    def __init__(self, key: Callable[[], str | None], timeout: float = 20,
                 counted: Callable[[str], None] = lambda kind: None) -> None:
        self._key = key
        self.timeout = timeout
        # Told "search" or "fetch" for each call Ollama answered: its free tier
        # has a limit it does not publish, so the portal keeps its own count.
        self._counted = counted

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        key = (self._key() or "").strip()
        if not key:
            raise WebError(f"No web search key is set. Add one on the Models tab (get a free key at {KEY_PAGE}).")
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                self._counted("search" if url == SEARCH_URL else "fetch")
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise WebError(f"The web search key was refused. Check it on the Models tab, or make a new one at {KEY_PAGE}.") from exc
            if exc.code == 429:
                raise WebError("Ollama's web search says too many searches; wait a minute and try again.") from exc
            raise WebError(f"Ollama's web search answered {exc.code}.") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise WebError("Could not reach Ollama's web search. Is this computer online?") from exc
        except ValueError as exc:
            raise WebError("Ollama's web search sent back something unreadable.") from exc

    def check(self) -> None:
        """Raise WebError unless the key works. Costs one page fetch."""
        self._post(FETCH_URL, {"url": "https://ollama.com"})

    def search(self, query: str, results: int = RESULTS) -> list[dict[str, str]]:
        """Pages for a query: [{id, url, title, site, text}], text as found."""
        data = self._post(SEARCH_URL, {"query": query, "max_results": results})
        pages: list[dict[str, str]] = []
        for item in data.get("results") or []:
            url = str(item.get("url") or "")
            if not is_web_url(url):
                continue
            text = str(item.get("content") or "").strip()
            if len(text) < SNIPPET_CHARS:
                try:
                    fetched = self._post(FETCH_URL, {"url": url})
                    text = str(fetched.get("content") or "").strip() or text
                except WebError:
                    pass  # the snippet is still something to read
            if not text:
                continue
            pages.append({"id": page_id(url), "url": url, "title": str(item.get("title") or site_of(url)).strip(),
                          "site": site_of(url), "text": text[:PAGE_CHARS]})
        return pages


QUERY_PROMPT = """Write one web search query for the latest question below.

- Use what the conversation says to make it stand alone ("that injury" -> the injury named).
- Ask about the general topic only. Leave out people's names, dates of birth, addresses, \
case or record numbers and any other personal details.
- Write it in the same language as the question.
- At most 12 words. Reply with the query alone: no quotes, no explanation."""


# Chinese, Japanese and Korean. Qwen models sometimes drift into Chinese
# mid-query when the question was in English.
CJK = re.compile(r"[\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]+")


def clean_query(text: str, fallback: str) -> str:
    line = next((l for l in text.strip().splitlines() if l.strip()), "")
    if not CJK.search(fallback):
        line = CJK.sub(" ", line)
    line = re.sub(r"^(search query|query)\s*:\s*", "", line.strip(), flags=re.I).strip(" \"'`*")
    words = line.split()
    return " ".join(words[:QUERY_WORDS]) if words else " ".join(fallback.split()[:QUERY_WORDS])


# Capitalised words that are not names, so a query keeps "What" and "How".
NOT_NAMES = set("""a an and are as at be but by can could did do does for from had has have he her his how i if
in is it its may might of on or q she should so that the their they this to was were what when where which
who why will with would you your after before during""".split())
DATE_OR_NUMBER = re.compile(r"\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}|\d{4,}")


def scrub(query: str, conversation: str) -> str:
    """The query without the names, dates and long numbers the conversation
    mentions. A small model told to leave them out does not always manage to;
    this is what makes sure."""
    # Capitalised mid-sentence is a name; at the start of a sentence it is just a word.
    names = {m.group(1).lower() for m in re.finditer(r"(?<![.?!:\n])[ \t(\"']([A-Z][a-z]+(?:'s)?)\b", conversation)
             if m.group(1).lower().removesuffix("'s") not in NOT_NAMES}
    names |= {n.removesuffix("'s") for n in names}
    kept = [w for w in query.split()
            if w.strip(",.?!:;\"'").lower().removesuffix("'s") not in names and not DATE_OR_NUMBER.search(w)]
    return " ".join(kept)
