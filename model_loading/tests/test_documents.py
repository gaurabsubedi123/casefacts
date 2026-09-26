import pytest
from pypdf import PdfWriter

from modelportal.documents import Library, NotOCRed, ocr_problem, text_pages


def test_ocrtool_page_markers_become_pages():
    text = "----- page 1 (ocr) -----\nfirst\n----- page 3 (text) -----\nthird\n"
    assert text_pages(text) == ["first", "", "third"]


def test_plain_text_is_split_into_sections():
    pages = text_pages("\n\n".join(["word " * 200] * 10))
    assert len(pages) > 1


def test_no_text_at_all_is_refused():
    assert "not been OCR'd" in ocr_problem("scan.pdf", ["", "  ", ""])


def test_mostly_empty_pages_are_refused():
    assert "not OCR'd" in ocr_problem("scan.pdf", ["real text on this page " * 3, "", "", ""])


def test_a_few_empty_pages_are_accepted():
    assert ocr_problem("ok.pdf", ["text on the page " * 5] * 9 + [""]) is None


def test_raw_scan_pdf_is_rejected_and_not_kept(tmp_path):
    scan = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(612, 792)
    writer.write(str(scan))
    library = Library(tmp_path / "lib")
    with pytest.raises(NotOCRed):
        library.add(scan, "scan.pdf")
    assert library.all() == []
    assert list((tmp_path / "lib" / "files").iterdir()) == []


def test_text_file_is_added_once(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("----- page 1 (ocr) -----\nThe patient was seen on 2018-07-06.\n")
    library = Library(tmp_path / "lib")
    doc, new = library.add(source, "notes.txt")
    again, new_again = library.add(source, "renamed.txt")
    assert new and not new_again and again.id == doc.id
    assert library.pages(doc.id) == ["The patient was seen on 2018-07-06."]
    assert library.remove(doc.id) and library.all() == []


def test_unsupported_type_is_refused(tmp_path):
    source = tmp_path / "a.docx"
    source.write_bytes(b"x")
    with pytest.raises(ValueError):
        Library(tmp_path / "lib").add(source, "a.docx")
