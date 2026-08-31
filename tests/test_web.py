"""The web interface: the routes, and the one that serves files off disk."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

from casefacts.web.app import create_app


@pytest.fixture
def client(settings, index, fake_ollama):
    app = create_app(settings)
    app.config["TESTING"] = True
    return app.test_client()


class TestPages:
    def test_the_home_page_renders(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"Plug in a file or folder" in response.data

    def test_state_lists_what_is_indexed(self, client):
        data = client.get("/api/state").get_json()
        assert data["stats"]["pages"] == 3
        assert len(data["documents"]) == 1

    def test_models_are_offered_with_their_notes(self, client):
        data = client.get("/api/models").get_json()
        names = [m["name"] for m in data["models"]]
        assert "qwen2.5:7b-instruct" in names
        assert any(m["notes"] for m in data["models"])

    def test_search_returns_citable_hits(self, client):
        data = client.get("/api/search?q=physical therapy&k=3").get_json()
        assert data["hits"]
        assert data["hits"][0]["label"]


class TestAsking:
    def test_a_question_comes_back_with_checked_findings(self, client, fake_ollama):
        fake_ollama.replies = [json.dumps({
            "findings": [{"statement": "Landis was the attending.",
                          "quote": "Attending: Landis MD,Brandi R", "excerpt": 1}],
            "answer": "Landis.", "missing": "",
        })]
        data = client.post("/api/ask", json={"question": "who was the attending?"}).get_json()
        assert data["findings"][0]["verdict"] == "verified"
        assert data["findings"][0]["page"] == 2

    def test_an_empty_question_is_refused(self, client):
        assert client.post("/api/ask", json={"question": "  "}).status_code == 400

    def test_two_models_produce_a_comparison(self, client, fake_ollama):
        fake_ollama.replies = [json.dumps({"findings": [], "answer": "one", "missing": ""}),
                               json.dumps({"findings": [], "answer": "two", "missing": ""})]
        data = client.post("/api/ask", json={
            "question": "anything?", "models": ["qwen2.5:7b-instruct", "medgemma:4b"],
        }).get_json()
        assert len(data["answers"]) == 2
        assert "agreement" in data


class TestPageViewer:
    def test_a_page_comes_back_with_what_is_needed_to_check_it(self, client, index):
        doc_id = index.documents()[0]["doc_id"]
        data = client.get(f"/api/page?doc={quote(doc_id)}&page=2").get_json()
        assert data["bates"] == "GEICO000002"
        assert data["has_image"] is True
        assert "SVH35492594" in data["text"]

    def test_a_page_that_does_not_exist(self, client, index):
        doc_id = index.documents()[0]["doc_id"]
        assert client.get(f"/api/page?doc={quote(doc_id)}&page=99").status_code == 404

    def test_the_page_image_is_served(self, client, index):
        doc_id = index.documents()[0]["doc_id"]
        response = client.get(f"/api/preview?doc={quote(doc_id)}&page=2")
        assert response.status_code == 200
        assert response.data.startswith(b"\xff\xd8")

    def test_a_preview_path_cannot_escape_its_folder(self, client, index, settings):
        """The preview path comes out of a file on disk, so it is input."""
        doc_id = index.documents()[0]["doc_id"]
        secret = settings.db_path.parent / "secret.txt"
        secret.write_text("not yours", encoding="utf-8")
        index.db.execute(
            "UPDATE pages SET preview = ? WHERE doc_id = ? AND page_no = 2",
            ("../" * 12 + "secret.txt", doc_id),
        )
        index.db.commit()
        assert client.get(f"/api/preview?doc={quote(doc_id)}&page=2").status_code in (403, 404)


class TestJobs:
    def test_reading_a_folder_starts_a_job(self, client, settings, tmp_path):
        folder = tmp_path / "more"
        (folder).mkdir()
        (folder / "note.txt").write_text("----- page 1 (ocr) -----\nseen on 4/1/2019", encoding="utf-8")
        data = client.post("/api/add", json={"path": str(folder), "ocr": False}).get_json()
        assert data["status"] in ("running", "done")

    def test_a_path_that_does_not_exist_is_refused(self, client):
        response = client.post("/api/add", json={"path": "/nowhere/at/all"})
        assert response.status_code == 400

    def test_the_chronology_endpoint_answers_before_anything_is_swept(self, client):
        data = client.get("/api/chronology").get_json()
        assert data["events"] == [] and data["gaps"] == []

    def test_the_csv_download_has_a_header_row(self, client):
        response = client.get("/api/chronology.csv")
        assert response.status_code == 200
        assert response.data.splitlines()[0].startswith(b"Date,As written,Event")
