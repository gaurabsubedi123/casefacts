import io
import json
import urllib.error

import pytest

from modelportal import websearch
from modelportal.documents import Library
from modelportal.retrieve import gather
from modelportal.websearch import WebError, WebSearch, clean_query, scrub

LONG = "Lumbar strain usually improves within two to six weeks with rest and gentle activity. " * 12


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_urlopen(calls, results):
    def urlopen(request, timeout):
        calls.append((request.full_url, json.loads(request.data), request.headers.get("Authorization")))
        if request.full_url == websearch.SEARCH_URL:
            return FakeResponse(json.dumps({"results": results}).encode())
        return FakeResponse(json.dumps({"title": "Fetched", "content": LONG}).encode())
    return urlopen


def test_search_keeps_long_results_and_fetches_snippets(monkeypatch):
    calls, counted = [], []
    monkeypatch.setattr(websearch.urllib.request, "urlopen", fake_urlopen(calls, [
        {"title": "Back pain", "url": "https://www.example.org/back", "content": LONG},
        {"title": "Short", "url": "https://example.com/short", "content": "a snippet"},
        {"title": "Bad", "url": "javascript:alert(1)", "content": LONG},
    ]))
    pages = WebSearch(lambda: "key123", counted=counted.append).search("lumbar strain recovery time")
    assert [p["site"] for p in pages] == ["example.org", "example.com"]
    assert pages[1]["text"].startswith("Lumbar strain")  # the snippet was replaced by the fetched page
    assert calls[0][1] == {"query": "lumbar strain recovery time", "max_results": 5}
    assert calls[0][2] == "Bearer key123"
    assert counted == ["search", "fetch"]


def test_no_key_and_refused_key_say_what_to_do(monkeypatch):
    with pytest.raises(WebError, match="No web search key"):
        WebSearch(lambda: "").search("x")

    def refused(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "no", {}, None)
    monkeypatch.setattr(websearch.urllib.request, "urlopen", refused)
    with pytest.raises(WebError, match="refused"):
        WebSearch(lambda: "bad").search("x")

    def limited(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 429, "slow down", {}, None)
    monkeypatch.setattr(websearch.urllib.request, "urlopen", limited)
    with pytest.raises(WebError, match="too many searches"):
        WebSearch(lambda: "k").search("x")


def test_clean_query():
    assert clean_query('Search query: "lumbar strain recovery time"\nbecause…', "q") == "lumbar strain recovery time"
    assert clean_query("", "what is the recovery time") == "what is the recovery time"
    assert clean_query("Recovery time for lumbar strain通常需要多久", "How long?") == "Recovery time for lumbar strain"


def test_web_pages_join_the_reading_list(tmp_path):
    library = Library(tmp_path / "lib")
    source = tmp_path / "record.txt"
    source.write_text("Diagnosis: lumbar strain.", encoding="utf-8")
    doc, _ = library.add(source, "record.txt")
    page = {"id": "abc123abc123", "url": "https://example.org/back", "title": "Back pain", "site": "example.org",
            "text": LONG}
    chunks, mode = gather(library, "recovery time", [doc.id], context_tokens=8000, reserve_tokens=1000,
                          web_pages=[page])
    assert mode == "whole document + web pages"
    assert [c.url for c in chunks] == ["", "https://example.org/back"]  # the document first


CONVERSATION = ("Q: What was Maria Lopez diagnosed with after her accident on 07/06/2018?\n"
                "A: Maria Lopez was diagnosed with a lumbar strain. Recovery was slow.")


def test_names_dates_and_numbers_are_kept_out_of_the_query():
    assert scrub("recovery time for Maria Lopez's lumbar strain", CONVERSATION) == "recovery time for lumbar strain"
    assert scrub("lumbar strain 07/06/2018 record 4471902", CONVERSATION) == "lumbar strain record"


def test_ordinary_words_that_start_a_sentence_are_kept():
    assert scrub("Recovery time for lumbar strain", CONVERSATION) == "Recovery time for lumbar strain"
