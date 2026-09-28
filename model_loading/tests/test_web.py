import io
import time

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


class ChattyOllama(FakeOllama):
    """Answers every question from page 1, and keeps each prompt it was sent."""

    def __init__(self):
        self.prompts = []

    def chat(self, model, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        kwargs["on_progress"]({"phase": "writing", "text": '{"answer": "Lumbar'})
        return {"text": '{"answer": "Lumbar strain.", "findings": [{"statement": "Lumbar strain", '
                        '"quote": "Diagnosis: lumbar strain", "excerpt": 1}], "missing": ""}',
                "thinking": "", "seconds": 0.1, "prompt_tokens": 10, "answer_tokens": 5, "truncated": False}


def finished(client, job):
    for _ in range(200):
        state = client.get(f"/api/jobs/{job}").get_json()
        if state["status"] != "running":
            return state
        time.sleep(0.02)
    raise AssertionError("the job never finished")


def test_a_chat_is_saved_and_continued(tmp_path):
    fake = ChattyOllama()
    library = Library(tmp_path / "lib")
    source = tmp_path / "record.txt"
    source.write_text("Emergency admission after a motor vehicle accident. Diagnosis: lumbar strain.", encoding="utf-8")
    doc, _ = library.add(source, "record.txt")
    client = create_app(fake, library).test_client()
    ask = {"documents": [doc.id], "model": "qwen2.5:7b"}

    first = client.post("/api/ask", json={**ask, "question": "What was the diagnosis?"}).get_json()
    assert finished(client, first["job"])["status"] == "done"
    second = client.post("/api/ask", json={**ask, "question": "Was it serious?", "chat": first["chat"]}).get_json()
    assert second["chat"] == first["chat"]
    assert finished(client, second["job"])["result"]["findings"][0]["verdict"] == "verified"

    assert "Q: What was the diagnosis?" in fake.prompts[1] and "A: Lumbar strain." in fake.prompts[1]
    listed = client.get("/api/chats").get_json()["chats"]
    assert [(c["title"], c["questions"]) for c in listed] == [("What was the diagnosis?", 2)]
    saved = client.get(f"/api/chats/{first['chat']}").get_json()
    assert [t["question"] for t in saved["turns"]] == ["What was the diagnosis?", "Was it serious?"]
    assert saved["documents"] == [doc.id]

    assert client.delete(f"/api/chats/{first['chat']}").get_json()["ok"]
    assert client.get("/api/chats").get_json()["chats"] == []


def test_unknown_or_malformed_chat_ids(client):
    assert client.get("/api/chats/0123456789ab").status_code == 404
    assert client.get("/api/chats/..%2Fsettings").status_code == 404
    assert not client.delete("/api/chats/../../x").get_json(silent=True)
