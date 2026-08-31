"""The checking. These are the tests that matter most.

Everything here is about what happens when the model is wrong, because that is
the case the tool exists to survive.
"""

from __future__ import annotations

import json

from casefacts.answer import ask
from casefacts.chronology import sweep, timeline, gaps


def reply(findings, answer="", missing=""):
    return json.dumps({"findings": findings, "answer": answer, "missing": missing})


class TestVerdicts:
    def test_a_real_quote_is_verified(self, index, fake_ollama):
        fake_ollama.replies = [reply(
            [{"statement": "The attending was Landis.", "quote": "Attending: Landis MD,Brandi R", "excerpt": 1}],
            answer="Landis was the attending.",
        )]
        result = ask(index, "who was the attending physician?")
        assert [f.verdict for f in result.findings] == ["verified"]
        assert result.findings[0].page_no == 2
        assert result.findings[0].bates == "GEICO000002"

    def test_an_invented_quote_is_marked_unverified(self, index, fake_ollama):
        fake_ollama.replies = [reply(
            [{"statement": "The patient had a total knee replacement.",
              "quote": "the patient underwent a total knee replacement on 3 March", "excerpt": 1}],
            answer="There was a knee replacement.",
        )]
        result = ask(index, "was there surgery?")
        assert result.findings[0].verdict == "unverified"
        assert any("unsupported" in w for w in result.warnings)

    def test_a_quote_from_the_wrong_excerpt_is_moved_not_discarded(self, index, fake_ollama):
        fake_ollama.replies = [reply(
            [{"statement": "The MRN is SVH35492594.", "quote": "MRN: SVH35492594", "excerpt": 99}]
        )]
        result = ask(index, "what is the MRN?")
        finding = result.findings[0]
        assert finding.verdict == "wrong page"
        assert finding.page_no == 2

    def test_a_finding_with_no_quote_at_all_is_unverified(self, index, fake_ollama):
        fake_ollama.replies = [reply([{"statement": "Something happened.", "quote": "", "excerpt": 1}])]
        assert ask(index, "what happened?").findings[0].verdict == "unverified"

    def test_checked_findings_sort_above_unsupported_ones(self, index, fake_ollama):
        """Whatever survived checking comes first; the invented one goes last.

        The real quote here may land as "verified" or as "wrong page"
        depending on which excerpt the search put first — both mean the words
        are in the records, which is the distinction being tested.
        """
        fake_ollama.replies = [reply([
            {"statement": "invented", "quote": "a total knee replacement was performed", "excerpt": 1},
            {"statement": "real", "quote": "Attending: Landis MD,Brandi R", "excerpt": 1},
        ])]
        findings = ask(index, "anything?").findings
        assert findings[0].statement == "real"
        assert findings[0].verdict != "unverified"
        assert findings[-1].verdict == "unverified"


class TestRefusal:
    def test_an_empty_findings_list_is_reported_honestly(self, index, fake_ollama):
        fake_ollama.replies = [reply([], answer="", missing="nothing about a knee in these records")]
        result = ask(index, "what did the knee surgeon say?")
        assert result.findings == []
        assert "knee" in result.missing

    def test_a_question_whose_words_appear_nowhere_is_flagged(self, index, fake_ollama):
        fake_ollama.replies = [reply([])]
        result = ask(index, "zzzqqq wibble")
        assert any("closest text found" in w for w in result.warnings)

    def test_prose_with_no_quotes_is_marked_as_unchecked(self, index, fake_ollama):
        fake_ollama.replies = ["The patient was fine."]
        result = ask(index, "how was the patient?")
        assert any("nothing here has been checked" in w for w in result.warnings)

    def test_a_model_that_returns_broken_json_does_not_crash(self, index, fake_ollama):
        fake_ollama.replies = ['{"findings": [{"statement": "cut off mid']
        result = ask(index, "anything?")
        assert result.findings == [] or result.answer


class TestScope:
    def test_asking_about_a_document_that_is_not_there(self, index, fake_ollama):
        result = ask(index, "anything?", doc="no such file")
        assert result.findings == []
        assert any("nothing indexed matches" in w for w in result.warnings)

    def test_one_small_document_is_read_whole_instead_of_searched(self, index, fake_ollama):
        fake_ollama.replies = [reply([])]
        result = ask(index, "summarise this", doc="claim")
        assert result.mode == "whole document"
        # every page, not a selection
        assert {hit.page_no for hit in result.hits} == {1, 2, 3}

    def test_the_whole_document_prompt_contains_every_page(self, index, fake_ollama):
        fake_ollama.replies = [reply([])]
        ask(index, "summarise this", doc="claim")
        prompt = fake_ollama.asked[-1]["user"]
        assert "GEICO000001" in prompt and "GEICO000003" in prompt

    def test_a_bates_stamp_in_the_question_pins_that_page(self, index, fake_ollama):
        fake_ollama.replies = [reply([])]
        result = ask(index, "what does GEICO000003 say?")
        assert result.hits[0].bates == "GEICO000003"


class TestChronology:
    def test_events_are_extracted_verified_and_dated(self, index, fake_ollama):
        fake_ollama.replies = [
            json.dumps({"events": []}),
            json.dumps({"events": [{
                "date": "7/6/2018", "event": "Emergency department admission",
                "provider": "Landis MD", "facility": "Spring Valley Hospital",
                "category": "emergency", "quote": "Admit: 7/6/2018"}]}),
            json.dumps({"events": [{
                "date": "12/03/2018", "event": "Lumbar epidural injection",
                "provider": "", "facility": "Jones Physical Therapy",
                "category": "procedure", "quote": "Lumbar Transforaminal Epidural Injection performed today."}]}),
        ]
        report = sweep(index)
        assert report.events == 2
        assert report.unverified == 0

        events = timeline(index)
        assert [e.date_iso for e in events] == ["2018-07-06", "2018-12-03"]
        assert events[0].citation.endswith("(GEICO000002)")

    def test_an_event_dated_off_page_is_flagged(self, index, fake_ollama):
        fake_ollama.replies = [
            json.dumps({"events": []}),
            json.dumps({"events": [{
                "date": "1/1/2020", "event": "invented follow-up",
                "category": "visit", "quote": "Admit: 7/6/2018"}]}),
            json.dumps({"events": []}),
        ]
        sweep(index)
        assert timeline(index, include_unverified=True)[0].verdict == "date not on page"

    def test_a_second_sweep_reads_nothing_again(self, index, fake_ollama):
        fake_ollama.replies = [json.dumps({"events": []})] * 3
        sweep(index)
        second = sweep(index)
        assert second.pages_read == 0
        assert second.pages_cached + second.pages_skipped == 3

    def test_pages_with_no_date_are_never_sent_to_a_model(self, index, fake_ollama, settings):
        (settings.records_dir / "txt" / "nodate.txt").write_text(
            "----- page 1 (ocr) -----\n" + "a page about nothing in particular " * 5, encoding="utf-8"
        )
        index.ingest_path(settings.records_dir, ocr=False)
        fake_ollama.replies = [json.dumps({"events": []})] * 10
        before = len(fake_ollama.asked)
        report = sweep(index, doc="nodate")
        assert report.pages_read == 0
        assert len(fake_ollama.asked) == before

    def test_gaps_are_measured_between_dated_events(self, index, fake_ollama):
        fake_ollama.replies = [
            json.dumps({"events": []}),
            json.dumps({"events": [{"date": "7/6/2018", "event": "ED visit", "category": "emergency",
                                    "quote": "Admit: 7/6/2018"}]}),
            json.dumps({"events": [{"date": "12/03/2018", "event": "injection", "category": "procedure",
                                    "quote": "Lumbar Transforaminal Epidural Injection performed today."}]}),
        ]
        sweep(index)
        found = gaps(timeline(index), days=30)
        assert len(found) == 1 and found[0].days == 150
