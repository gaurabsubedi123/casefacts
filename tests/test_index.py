"""Indexing: what gets stored, what gets skipped, and what search finds."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from casefacts.index import Index


class TestIngest:
    def test_a_folder_becomes_pages_chunks_and_vectors(self, index: Index):
        stats = index.stats()
        assert stats["documents"] == 1
        assert stats["pages"] == 3
        assert stats["chunks"] >= 3
        assert stats["vectors"] == stats["chunks"]

    def test_reading_the_same_folder_again_re_embeds_nothing(self, index: Index, settings):
        _, report = index.ingest_path(settings.records_dir, ocr=False)
        assert report.unchanged == 1
        assert report.embedded == 0

    def test_a_changed_document_is_read_again(self, index: Index, settings):
        path = settings.records_dir / "txt" / "claim.txt"
        path.write_text(path.read_text() + "\n\n----- page 4 (ocr) -----\nnew page 12/25/2018", encoding="utf-8")
        _, report = index.ingest_path(settings.records_dir, ocr=False)
        assert report.documents == 1
        assert index.stats()["pages"] == 4

    def test_the_same_document_in_two_places_is_embedded_once(self, index: Index, settings, tmp_path: Path):
        second = tmp_path / "copy"
        shutil.copytree(settings.records_dir, second)
        _, report = index.ingest_path(second, ocr=False)
        assert report.duplicates == 1
        assert report.embedded == 0
        assert index.stats()["documents"] == 1

    def test_a_duplicate_is_listed_as_an_alias_of_the_original(self, index: Index, settings, tmp_path: Path):
        second = tmp_path / "copy"
        shutil.copytree(settings.records_dir, second)
        index.ingest_path(second, ocr=False)
        primary = index.documents()[0]["doc_id"]
        assert len(index.aliases(primary)) == 1

    def test_forgetting_a_source_removes_only_that_source(self, index: Index, settings, tmp_path: Path):
        second = tmp_path / "other"
        shutil.copytree(settings.records_dir, second)
        (second / "txt" / "claim.txt").write_text(
            "----- page 1 (ocr) -----\nan entirely different document about a boat", encoding="utf-8"
        )
        index.ingest_path(second, ocr=False)
        assert index.stats()["documents"] == 2
        removed = index.forget_source(second.resolve())
        assert removed == 1
        assert index.stats()["documents"] == 1

    def test_the_page_image_path_survives_into_the_index(self, index: Index):
        doc_id = index.documents()[0]["doc_id"]
        page = index.page(doc_id, 2)
        assert page["preview"].endswith("p0002.jpg")
        assert index.document(doc_id)["previews_root"]


class TestSearch:
    def test_an_exact_identifier_is_found_by_the_keyword_half(self, index: Index):
        assert index.keyword_search("SVH35492594", 5)

    def test_a_bates_stamp_resolves_to_its_page(self, index: Index):
        page = index.by_bates("GEICO000002")
        assert page and page["page_no"] == 2

    def test_a_stamp_that_does_not_exist_resolves_to_nothing(self, index: Index):
        assert index.by_bates("GEICO999999") is None

    def test_search_returns_hits_that_can_be_cited(self, index: Index):
        hits = index.search("physical therapy injection low back")
        assert hits
        assert all(hit.page_no and hit.title for hit in hits)
        assert "p." in hits[0].label()

    def test_a_question_matching_no_words_is_flagged_as_such(self, index: Index):
        """Dense search always returns its nearest neighbours, however far away.

        So "nothing matched" cannot mean "no hits". It means none of the
        question's own words are anywhere in the file, which is what gets
        recorded and warned about.
        """
        index.search("zzzz nonexistent qqqq")
        assert index.last_search_had_keyword_match is False
        index.search("physical therapy")
        assert index.last_search_had_keyword_match is True

    def test_scoping_to_one_document(self, index: Index):
        doc_id = index.documents()[0]["doc_id"]
        assert index.scope_doc_ids(doc="claim") == [doc_id]

    def test_a_file_is_found_by_the_path_it_came_from(self, index: Index, settings):
        """Plugging in one file must resolve to it exactly, not by name."""
        txt = settings.records_dir / "txt" / "claim.txt"
        assert index.doc_for_original(txt) == txt.resolve().as_posix()
        assert index.scope_doc_ids(doc=str(txt)) == [txt.resolve().as_posix()]

    def test_two_files_with_the_same_name_do_not_confuse_each_other(self, index: Index, settings, tmp_path):
        other = tmp_path / "elsewhere"
        (other / "txt").mkdir(parents=True)
        (other / "txt" / "claim.txt").write_text(
            "----- page 1 (ocr) -----\na different claim entirely, seen 5/5/2020", encoding="utf-8"
        )
        index.ingest_path(other, ocr=False)
        first = settings.records_dir / "txt" / "claim.txt"
        second = other / "txt" / "claim.txt"
        assert index.scope_doc_ids(doc=str(first)) == [first.resolve().as_posix()]
        assert index.scope_doc_ids(doc=str(second)) == [second.resolve().as_posix()]
        # by bare name it is genuinely ambiguous: refuse rather than guess
        assert index.scope_doc_ids(doc="claim") == []
        assert len(index.documents_matching("claim")) == 2

    def test_scoping_to_a_name_that_matches_nothing(self, index: Index):
        assert index.scope_doc_ids(doc="no such document") == []

    def test_scoping_to_a_folder(self, index: Index, settings):
        assert index.scope_doc_ids(folder=str(settings.records_dir))

    def test_no_scope_means_everything(self, index: Index):
        assert index.scope_doc_ids() is None

    def test_a_scoped_search_stays_inside_its_scope(self, index: Index, settings, tmp_path: Path):
        other = tmp_path / "other"
        (other / "txt").mkdir(parents=True)
        (other / "txt" / "boat.txt").write_text(
            "----- page 1 (ocr) -----\nthe boat has a low back sprain of the hull", encoding="utf-8"
        )
        index.ingest_path(other, ocr=False)
        scoped = index.scope_doc_ids(doc="boat")
        hits = index.search("low back sprain", doc_ids=scoped)
        assert hits and all(hit.doc_id in scoped for hit in hits)


class TestResilience:
    def test_an_unreadable_document_does_not_stop_the_others(self, index: Index, tmp_path: Path):
        folder = tmp_path / "mixed"
        (folder).mkdir()
        (folder / "good.txt").write_text("----- page 1 (ocr) -----\nreadable content here", encoding="utf-8")
        (folder / "bad.txt").write_bytes(b"\xff\xfe\x00\x00 not utf-8 at all")
        _, report = index.ingest_path(folder, ocr=False)
        assert report.documents >= 1

    def test_an_empty_folder_is_not_an_error(self, index: Index, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        _, report = index.ingest_path(empty, ocr=False)
        assert report.documents == 0
        assert report.errors == []


class TestStopping:
    def test_a_stop_is_not_swallowed_by_the_per_document_handler(self, index, settings, tmp_path):
        """Stopping must not look like one unreadable document.

        Ingest wraps each document in `except Exception` so that one bad file
        does not lose the run. If the stop signal were an ordinary exception it
        would be caught there, logged as an error, and the run would carry on —
        which is exactly what the stop button is for preventing.
        """
        from casefacts.web.state import Stopped

        folder = tmp_path / "several"
        folder.mkdir()
        for i in range(4):
            (folder / f"doc{i}.txt").write_text(
                f"----- page 1 (ocr) -----\ndocument number {i} seen on 4/1/2019", encoding="utf-8"
            )

        def progress(event, payload):
            if event == "document":
                raise Stopped()

        with pytest.raises(Stopped):
            index.ingest_path(folder, ocr=False, progress=progress)
