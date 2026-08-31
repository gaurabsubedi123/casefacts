"""The local web interface.

Flask, server-rendered shell, plain JavaScript. No build step, no bundler,
nothing loaded from a CDN — the same constraint ocrtool works under, and for
the same reason: this reads medical records, and a tool that reaches the
network to render a page is a tool that could send one somewhere.

The interface exists for one thing the command line cannot do. An answer here
carries a citation, and clicking the citation puts the scanned page on screen
beside the quote. Reading an answer is optional; checking it should take one
click.
"""

from __future__ import annotations

import json
import logging
import mimetypes
from pathlib import Path
from typing import Any, Iterator

from flask import (
    Blueprint,
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
)

from .. import __version__
from ..answer import ask
from ..chronology import gaps, sweep, timeline, to_csv
from ..compare import compare
from ..config import KNOWN_MODELS, Settings, model_spec
from ..index import Index
from ..ollama import Ollama, OllamaError
from ..sources import SourceError
from .state import Jobs

log = logging.getLogger(__name__)

bp = Blueprint("casefacts", __name__)
jobs = Jobs()


def _index() -> Index:
    """A connection for this request.

    SQLite connections belong to one thread, and Flask serves each request on
    whichever thread is free, so a new one is opened per request rather than
    shared. The expensive part of a question is the model, not the connect.
    """
    from flask import current_app

    return Index(current_app.config["CASEFACTS_SETTINGS"])


def _settings() -> Settings:
    from flask import current_app

    return current_app.config["CASEFACTS_SETTINGS"]


def _sse(events: Iterator[dict[str, Any]]) -> Response:
    def stream() -> Iterator[str]:
        for event in events:
            yield f"data: {json.dumps(event)}\n\n"

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------- pages


@bp.route("/")
def home() -> str:
    index = _index()
    return render_template(
        "index.html",
        version=__version__,
        stats=index.stats(),
        documents=index.documents(),
        sources=index.sources(),
        settings=_settings(),
    )


# --------------------------------------------------------------------- api


@bp.route("/api/state")
def api_state() -> Response:
    index = _index()
    job = jobs.get()
    return jsonify(
        {
            "stats": index.stats(),
            "documents": [
                {**d, "aliases": index.aliases(d["doc_id"])} for d in index.documents()
            ],
            "sources": index.sources(),
            "job": job.snapshot() if job else None,
            "busy": jobs.busy(),
        }
    )


@bp.route("/api/models")
def api_models() -> Response:
    settings = _settings()
    client = Ollama(settings.ollama_host)
    try:
        installed = client.installed()
    except OllamaError as exc:
        return jsonify({"error": str(exc), "models": []})
    models = []
    for model in installed:
        spec = model_spec(model["name"])
        kind = spec.kind if spec else ("embed" if "embed" in model["name"].lower() else "chat")
        models.append(
            {
                "name": model["name"],
                "label": spec.label if spec else model["name"],
                "notes": spec.notes if spec else "",
                "kind": kind,
                "size_gb": round(model["size_bytes"] / 1e9, 1),
                "default": model["name"] == settings.chat_model,
            }
        )
    known = {s.name for s in KNOWN_MODELS}
    missing = [
        {"name": s.name, "label": s.label, "notes": s.notes, "kind": s.kind}
        for s in KNOWN_MODELS
        if s.name not in {m["name"] for m in installed}
    ]
    return jsonify({"models": models, "missing": missing, "known": sorted(known)})


@bp.route("/api/add", methods=["POST"])
def api_add() -> Response:
    """Plug in a file or folder: OCR what needs it, then index it."""
    data = request.get_json(silent=True) or {}
    raw = str(data.get("path") or "").strip()
    if not raw:
        return jsonify({"error": "give a file or folder to read"}), 400
    path = Path(raw).expanduser()
    if not path.exists():
        return jsonify({"error": f"no such file or folder: {path}"}), 400

    ocr = bool(data.get("ocr", True))
    rebuild = bool(data.get("rebuild", False))
    settings = _settings()

    def work(job: Any) -> dict[str, Any]:
        index = Index(settings)

        def progress(event: str, payload: dict[str, Any]) -> None:
            job.check()
            job.emit(event, payload)

        try:
            source, report = index.ingest_path(
                path, ocr=ocr, rebuild=rebuild, progress=progress
            )
        except SourceError as exc:
            raise RuntimeError(str(exc)) from exc
        return {"root": str(source.root), **report.to_dict()}

    try:
        job = jobs.start("reading", str(path), work)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify(job.snapshot())


@bp.route("/api/ask", methods=["POST"])
def api_ask() -> Response:
    data = request.get_json(silent=True) or {}
    question = str(data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "ask something"}), 400

    index = _index()
    scope = {
        "doc": (data.get("doc") or None),
        "folder": (data.get("folder") or None),
        "top_k": int(data["top_k"]) if data.get("top_k") else None,
        "whole": True if data.get("whole") else None,
    }
    models = [m for m in (data.get("models") or []) if m]
    try:
        if len(models) > 1:
            return jsonify(compare(index, question, models, **scope).to_dict())
        return jsonify(ask(index, question, model=models[0] if models else None, **scope).to_dict())
    except OllamaError as exc:
        return jsonify({"error": str(exc)}), 502


@bp.route("/api/chronology", methods=["POST"])
def api_chronology_start() -> Response:
    data = request.get_json(silent=True) or {}
    settings = _settings()
    scope = {"doc": data.get("doc") or None, "folder": data.get("folder") or None}
    model = data.get("model") or settings.chat_model
    rebuild = bool(data.get("rebuild"))
    limit = int(data["limit"]) if data.get("limit") else None

    def work(job: Any) -> dict[str, Any]:
        index = Index(settings)

        def progress(event: str, payload: dict[str, Any]) -> None:
            job.check()
            job.emit(event, payload)

        report = sweep(
            index, model=model, rebuild=rebuild, limit=limit, progress=progress, **scope
        )
        return report.to_dict()

    try:
        job = jobs.start("chronology", model, work)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify(job.snapshot())


@bp.route("/api/chronology")
def api_chronology() -> Response:
    index = _index()
    events = timeline(
        index,
        model=request.args.get("model") or None,
        doc=request.args.get("doc") or None,
        folder=request.args.get("folder") or None,
        include_unverified=request.args.get("unverified") == "1",
    )
    days = int(request.args.get("gap_days") or 30)
    return jsonify(
        {
            "events": [e.to_dict() for e in events],
            "gaps": [g.to_dict() for g in gaps(events, days=days)],
        }
    )


@bp.route("/api/chronology.csv")
def api_chronology_csv() -> Response:
    index = _index()
    events = timeline(
        index,
        model=request.args.get("model") or None,
        doc=request.args.get("doc") or None,
        folder=request.args.get("folder") or None,
        include_unverified=request.args.get("unverified") == "1",
    )
    return Response(
        to_csv(events),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=chronology.csv"},
    )


@bp.route("/api/job")
def api_job() -> Response:
    job = jobs.get()
    return jsonify(job.snapshot() if job else {"status": "idle"})


@bp.route("/api/job/stream")
def api_job_stream() -> Response:
    job = jobs.get()
    if job is None:
        return _sse(iter([{"event": "idle"}]))
    return _sse(job.watch())


@bp.route("/api/job/stop", methods=["POST"])
def api_job_stop() -> Response:
    job = jobs.current
    if job is None:
        return jsonify({"status": "idle"})
    job.stop()
    return jsonify({"status": "stopping"})


@bp.route("/api/page")
def api_page() -> Response:
    """One page, as text, with everything needed to show and check it."""
    index = _index()
    doc_id = request.args.get("doc") or ""
    try:
        page_no = int(request.args.get("page") or 0)
    except ValueError:
        abort(400)
    page = index.page(doc_id, page_no)
    if not page:
        abort(404)
    document = index.document(doc_id) or {}
    return jsonify(
        {
            **page,
            "title": document.get("title"),
            "original_path": document.get("original_path"),
            "page_count": document.get("page_count"),
            "has_image": bool(page.get("preview") and document.get("previews_root")),
        }
    )


@bp.route("/api/preview")
def api_preview() -> Response:
    """The scanned page itself — the thing a citation is checked against."""
    index = _index()
    doc_id = request.args.get("doc") or ""
    try:
        page_no = int(request.args.get("page") or 0)
    except ValueError:
        abort(400)

    page = index.page(doc_id, page_no)
    document = index.document(doc_id)
    if not page or not document or not page.get("preview"):
        abort(404)
    root = document.get("previews_root")
    if not root:
        abort(404)

    # The stored preview path comes from ocrtool's own JSON, but it is still
    # joined and re-checked here: a path out of a file on disk is input, and a
    # "..\\..\\" in one must not be able to serve an arbitrary file.
    base = Path(root).resolve()
    target = (base / str(page["preview"])).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        abort(403)
    if not target.is_file():
        abort(404)
    return send_file(target, mimetype=mimetypes.guess_type(target.name)[0] or "image/jpeg")


@bp.route("/api/search")
def api_search() -> Response:
    index = _index()
    query = request.args.get("q") or ""
    if not query.strip():
        return jsonify({"hits": []})
    doc_ids = index.scope_doc_ids(
        doc=request.args.get("doc") or None, folder=request.args.get("folder") or None
    )
    hits = index.search(query, top_k=int(request.args.get("k") or 15), doc_ids=doc_ids)
    return jsonify({"hits": [h.to_dict() for h in hits]})


@bp.route("/api/forget", methods=["POST"])
def api_forget() -> Response:
    data = request.get_json(silent=True) or {}
    root = str(data.get("root") or "")
    if not root:
        return jsonify({"error": "which source?"}), 400
    removed = _index().forget_source(root)
    return jsonify({"removed": removed})


def create_app(settings: Settings) -> Flask:
    app = Flask(__name__)
    app.config["CASEFACTS_SETTINGS"] = settings
    app.config["JSON_SORT_KEYS"] = False
    app.register_blueprint(bp)
    return app
