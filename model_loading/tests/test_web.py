import io

import pytest

from modelportal.documents import Library
from modelportal.web import create_app


class FakeOllama:
    def is_up(self):
        return True

    def version(self):
        return "test"

    def installed(self):
        return [{"name": "qwen2.5:7b", "size": 4_700_000_000, "digest": "845dbda0ea48", "family": "qwen2",
                 "parameters": "7.6B", "quantization": "Q4_K_M", "modified": ""}]

    def running(self):
        return []

    def capabilities(self, model):
        return ["completion"]

    def show(self, model):
        return {"model_info": {}}


@pytest.fixture
def client(tmp_path):
    app = create_app(FakeOllama(), Library(tmp_path / "lib"))
    return app.test_client()


def test_page_loads(client):
    assert client.get("/").status_code == 200


def test_other_hosts_are_refused(client):
    assert client.get("/api/documents", headers={"Host": "evil.example"}).status_code == 403


def test_other_origins_cannot_post(client):
    response = client.post("/api/pull", json={"name": "x"}, headers={"Origin": "http://evil.example"})
    assert response.status_code == 403


def test_upload_with_no_file_is_an_error(client):
    response = client.post("/api/documents", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "No file" in response.get_json()["error"]


def test_upload_of_wrong_type_is_an_error(client):
    data = {"files": (io.BytesIO(b"hi"), "letter.docx")}
    response = client.post("/api/documents", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["rejected"][0]["name"] == "letter.docx"


def test_ask_without_documents_is_an_error(client):
    response = client.post("/api/ask", json={"question": "what?", "documents": [], "model": "qwen2.5:7b"})
    assert response.status_code == 400
    assert "No document" in response.get_json()["error"]


def test_cloud_models_are_refused(client):
    response = client.post("/api/pull", json={"name": "deepseek-v4-pro:cloud"})
    assert response.status_code == 400
    assert "cloud" in response.get_json()["error"]
