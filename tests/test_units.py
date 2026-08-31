"""The parts that have to be right before anything else can be."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from casefacts.chunk import chunk_page
from casefacts.chronology import has_date, parse_date
from casefacts.config import CHUNK_CHARS, CHUNK_OVERLAP
from casefacts.corpus import bates_series, load_document, split_pages
from casefacts.index import fts_query
from casefacts.answer import (
    is_checkable,
    normalise,
    ordered_coverage,
    quote_coverage,
)
from casefacts.ollama import strip_thinking


class TestPageSplitting:
    def test_markers_give_page_numbers(self):
        text = "----- page 1 (ocr) -----\nfirst\n\n----- page 2 (text-layer) -----\nsecond"
        assert split_pages(text) == [(1, "ocr", "first"), (2, "text-layer", "second")]

    def test_a_file_with_no_markers_is_one_page(self):
        assert split_pages("just some text") == [(1, "unknown", "just some text")]

    def test_empty_file_has_no_pages(self):
        assert split_pages("   \n  ") == []

    def test_page_numbers_are_taken_not_counted(self):
        """ocrtool states the number; a resumed run may not start at 1."""
        text = "----- page 340 (ocr) -----\nlate page"
        assert split_pages(text)[0][0] == 340


class TestBates:
    def test_a_running_series_is_found(self):
        pages = [f"GEICO{n:06d}\nsome text" for n in range(1, 12)]
        assert "GEICO" in bates_series(pages)

    def test_a_zip_code_is_not_a_bates_stamp(self):
        pages = ["Macon, GA 31294-9643\ntext"] * 10
        assert "GA" not in bates_series(pages)

    def test_letterhead_repeating_one_number_is_not_a_series(self):
        pages = ["CALL US 8005551212\ncontent here"] * 10
        assert bates_series(pages) == set()

    def test_two_occurrences_are_a_coincidence(self):
        pages = ["REF 1001", "REF 1002", "nothing", "nothing"]
        assert bates_series(pages) == set()


class TestChunking:
    def test_a_short_page_is_kept_whole(self):
        chunks = chunk_page("d", 1, "a short page")
        assert len(chunks) == 1
        assert chunks[0].text == "a short page"

    def test_a_long_page_is_split(self):
        page = "\n\n".join(["paragraph number %d %s" % (i, "x" * 200) for i in range(20)])
        chunks = chunk_page("d", 1, page)
        assert len(chunks) > 1

    def test_no_chunk_crosses_a_page(self):
        """The rule the citations depend on."""
        page = "\n\n".join("word " * 100 for _ in range(20))
        for chunk in chunk_page("d", 7, page):
            assert chunk.page_no == 7

    def test_a_line_longer_than_a_chunk_is_still_cut(self):
        chunks = chunk_page("d", 1, "x" * (CHUNK_CHARS * 3))
        assert len(chunks) >= 3
        # The overlap deliberately carries the tail of one chunk into the next,
        # so a chunk is allowed to exceed CHUNK_CHARS by that much and no more.
        assert all(len(c.text) <= CHUNK_CHARS + CHUNK_OVERLAP + 10 for c in chunks)

    def test_an_empty_page_produces_nothing(self):
        assert chunk_page("d", 1, "   ") == []


class TestDates:
    @pytest.mark.parametrize(
        "written,expected",
        [
            ("7/6/2018", "2018-07-06"),
            ("07-06-18", "2018-07-06"),
            ("2018-07-06", "2018-07-06"),
            ("July 6, 2018", "2018-07-06"),
            ("Jul 6 2018", "2018-07-06"),
            ("6 July 2018", "2018-07-06"),
            ("Date of service: 12/03/2018", "2018-12-03"),
        ],
    )
    def test_the_ways_a_clinic_writes_a_date(self, written, expected):
        assert parse_date(written, today=date(2026, 8, 30)) == expected

    def test_two_digit_years_do_not_land_in_the_future(self):
        assert parse_date("7/6/99", today=date(2026, 8, 30)) == "1999-07-06"

    def test_an_impossible_date_is_not_invented(self):
        assert parse_date("2/30/2018") is None

    def test_text_that_is_not_a_date(self):
        assert parse_date("no date here") is None
        assert parse_date("") is None

    def test_the_page_filter_agrees_with_the_parser(self):
        assert has_date("seen on 7/6/2018")
        assert not has_date("no dates on this page at all")


class TestQuoteChecking:
    def test_an_exact_quote_is_complete(self):
        assert quote_coverage("the patient fell", "note: the patient fell today") == 1.0

    def test_ocr_punctuation_does_not_break_a_quote(self):
        assert quote_coverage("Attending: Landis MD,Brandi R", "Attending:  Landis  MD, Brandi R") == 1.0

    def test_an_invented_quote_scores_low(self):
        assert quote_coverage("underwent a total knee replacement", "the patient fell today") < 0.5

    def test_words_read_across_a_layout_count_as_joined(self):
        page = "Standard Double Room          Breakfast included          $120 per night"
        quote = "Standard Double Room $120 per night"
        assert quote_coverage(quote, page) < 0.75
        assert ordered_coverage(quote, page) >= 0.9

    def test_scattered_coincidence_does_not_pass_as_joined(self):
        page = "the quick brown fox jumps over the lazy dog again and again"
        assert ordered_coverage("underwent bilateral knee arthroplasty", page) < 0.9

    def test_an_identifier_is_checkable_even_though_it_is_short(self):
        assert is_checkable("SVH35492594")
        assert is_checkable("7/6/2018")

    def test_a_generic_phrase_is_not_checkable(self):
        assert not is_checkable("the patient")
        assert not is_checkable("")

    def test_normalise_ignores_case_and_punctuation(self):
        assert normalise("Attending: Landis MD,") == normalise("attending  landis  md")


class TestFtsQuery:
    def test_punctuation_cannot_break_the_query(self):
        query = fts_query("what did the patient's x-ray show?")
        assert '"patient"' in query and '"x-ray"' in query
        assert "?" not in query and "'" not in query

    def test_stopwords_are_dropped_but_negations_are_not(self):
        query = fts_query("was there no fracture")
        assert '"no"' in query
        assert '"was"' not in query

    def test_a_question_of_only_stopwords_still_searches(self):
        assert fts_query("what is the") != ""

    def test_an_empty_question_gives_an_empty_query(self):
        assert fts_query("   ") == ""


class TestThinking:
    def test_a_closed_think_block_is_removed(self):
        assert strip_thinking("<think>hmm, maybe</think>The answer.") == "The answer."

    def test_an_unclosed_block_swallows_the_rest(self):
        """An answer cut off inside its own reasoning is not an answer."""
        assert strip_thinking("<think>I am still thinking and then I ran out") == ""

    def test_ordinary_text_is_untouched(self):
        assert strip_thinking("  The answer.  ") == "The answer."


class TestCorpus:
    def test_a_document_is_read_with_its_metadata(self, records: Path):
        document = load_document(records / "txt" / "claim.txt", records)
        assert document is not None
        assert len(document.pages) == 3
        assert document.pages[0].source == "text-layer"
        assert document.pages[1].confidence == 75.0
        assert document.pages[1].needs_review is True
        assert document.pages[1].preview.endswith("p0002.jpg")

    def test_the_bates_stamp_is_attached_to_each_page(self, records: Path):
        document = load_document(records / "txt" / "claim.txt", records)
        assert [p.bates for p in document.pages] == ["GEICO000001", "GEICO000002", "GEICO000003"]

    def test_identity_is_the_path_so_two_folders_cannot_collide(self, records: Path):
        document = load_document(records / "txt" / "claim.txt", records)
        assert document.doc_id == (records / "txt" / "claim.txt").resolve().as_posix()
        assert document.rel_path == "txt/claim.txt"
