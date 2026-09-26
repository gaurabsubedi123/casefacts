import pytest

from modelportal.answer import AskError, check, parse_reply, validate
from modelportal.documents import Library
from modelportal.retrieve import Chunk, bm25_rank, gather

PAGE = "Emergency department admission after a motor vehicle accident on 07/06/2018. Diagnosis: lumbar strain."


def excerpts():
    return [Chunk("d", "Record", 1, "Unrelated billing page with totals."), Chunk("d", "Record", 2, PAGE)]


def by_text(chunk):
    return chunk.text


def test_verified_quote():
    f = check({"statement": "s", "quote": "Diagnosis: lumbar strain", "excerpt": 2}, excerpts(), by_text)
    assert f["verdict"] == "verified" and f["page"] == 2


def test_wrong_page_is_corrected():
    f = check({"statement": "s", "quote": "motor vehicle accident on 07/06/2018", "excerpt": 1}, excerpts(), by_text)
    assert f["verdict"] == "wrong page" and f["page"] == 2


def test_invented_quote_is_unverified():
    f = check({"statement": "s", "quote": "fractured femur requiring surgery", "excerpt": 2}, excerpts(), by_text)
    assert f["verdict"] == "unverified"


def test_ocr_noise_is_close():
    f = check({"statement": "s", "quote": "motor vehicle accldent on 07/06/2018", "excerpt": 2}, excerpts(), by_text)
    assert f["verdict"] in ("close", "joined", "verified")


def test_parse_reply_tolerates_fences_and_truncation():
    findings, answer, _ = parse_reply('```json\n{"findings": [{"statement": "a", "quote": "b", "excerpt": 1}], "answer": "x"}\n```')
    assert findings[0]["statement"] == "a" and answer == "x"
    findings, _, _ = parse_reply('{"findings": [{"statement": "a", "quote": "b", "excerpt": 1}')
    assert findings and findings[0]["excerpt"] == 1


def make_library(tmp_path, pages):
    source = tmp_path / "doc.txt"
    source.write_text("".join(f"----- page {n} (ocr) -----\n{text}\n" for n, text in enumerate(pages, 1)))
    library = Library(tmp_path / "lib")
    doc, _ = library.add(source, "doc.txt")
    return library, doc


def test_validate_refuses_missing_things(tmp_path):
    library, doc = make_library(tmp_path, [PAGE])
    with pytest.raises(AskError, match="No document"):
        validate(library, "what happened?", [], "qwen2.5:7b")
    with pytest.raises(AskError, match="question"):
        validate(library, "  ", [doc.id], "qwen2.5:7b")
    with pytest.raises(AskError, match="No model"):
        validate(library, "what happened?", [doc.id], "")
    validate(library, "what happened?", [doc.id], "qwen2.5:7b")


def test_small_document_is_read_whole(tmp_path):
    library, doc = make_library(tmp_path, [PAGE, "Second page text about follow up visits."])
    chosen, mode = gather(library, "diagnosis", [doc.id], context_tokens=8192, reserve_tokens=2000)
    assert mode == "whole document" and [c.page for c in chosen] == [1, 2]


def test_large_document_is_searched_and_ranked(tmp_path):
    filler = "Routine billing statement with codes and totals for services. " * 20
    pages = [filler] * 30
    pages[17] = PAGE
    library, doc = make_library(tmp_path, pages)
    chosen, mode = gather(library, "what was the diagnosis after the accident?", [doc.id],
                          context_tokens=2048, reserve_tokens=500)
    assert mode == "keyword search"
    assert 18 in [c.page for c in chosen]


def test_bm25_prefers_the_matching_chunk():
    chunks = excerpts()
    ranked = bm25_rank("lumbar strain diagnosis", chunks)
    assert ranked[0][0] == 1
