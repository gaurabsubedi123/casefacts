"""The browser interface: one Flask app on 127.0.0.1, nothing reachable from
outside this computer.

Because any web page the person has open could try to send requests to a port
on localhost, every request is checked: the Host must be this machine, and a
state-changing request must come from this page (its Origin, when the browser
sends one, has to match). That is what stops some other site from quietly
triggering a 40 GB download or deleting a model.
"""

from __future__ import annotations

import json
import os
import platform
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from flask import Flask, Response, abort, jsonify, render_template, request, send_file

from . import __version__, catalog, hardware
from .answer import AskError, ask, quote_spans, validate
from .chats import Chats, history_of
from .documents import SUPPORTED, Cancelled, Library, NotOCRed, empty_page_numbers
from .jobs import Job, Jobs
from .ollama import DOWNLOAD_URL, Ollama, OllamaError, Stopped, find_binary
from .paths import data_dir, resource_dir
from .websearch import KEY_PAGE, SIGNUP_PAGE, WebError, WebSearch

LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
OLLAMA_WINDOWS_INSTALLER = "https://ollama.com/download/OllamaSetup.exe"


class Settings:
    """The few choices that should survive a restart."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        try:
            self.data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, **values: Any) -> None:
        with self._lock:
            self.data.update(values)
            self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")


def is_cloud(model: dict[str, Any] | str) -> bool:
    name = model if isinstance(model, str) else model.get("name", "")
    return name.endswith("cloud") or name.endswith("-cloud:latest") or ":cloud" in name


def is_embedding(name: str, capabilities: list[str]) -> bool:
    return "embedding" in capabilities or "embed" in name.lower()


def create_app(client: Ollama | None = None, library: Library | None = None) -> Flask:
    root = resource_dir()
    app = Flask(__name__, static_folder=str(root / "static"), template_folder=str(root / "templates"))
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024**3
    client = client or Ollama()
    library = library or Library()
    chats = Chats(library.root.parent / "chats")
    jobs = Jobs()
    settings = Settings(data_dir() / "settings.json")
    incoming = data_dir() / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------- guards

    @app.before_request
    def only_this_computer() -> None:
        host = (request.host or "").rsplit(":", 1)[0]
        if host not in LOCAL_HOSTS:
            abort(403)
        if request.method in ("POST", "DELETE", "PUT"):
            origin = request.headers.get("Origin")
            if origin and origin.split("://", 1)[-1] != request.host:
                abort(403)

    @app.errorhandler(AskError)
    @app.errorhandler(NotOCRed)
    def refused(exc: Exception):
        return jsonify(error=str(exc)), 400

    @app.errorhandler(OllamaError)
    def ollama_down(exc: Exception):
        return jsonify(error=f"Ollama: {exc}"), 502

    @app.errorhandler(catalog.CatalogError)
    def catalog_failed(exc: Exception):
        return jsonify(error=str(exc)), 502

    def body() -> dict[str, Any]:
        return request.get_json(silent=True) or {}

    # ------------------------------------------------------------ pages

    @app.get("/")
    def home() -> str:
        return render_template("index.html", version=__version__)

    # -------------------------------------------------------- web search

    def web_key() -> str:
        return settings.get("web_key") or os.environ.get("OLLAMA_API_KEY", "")

    def count_web_call(kind: str) -> None:
        today = time.strftime("%Y-%m-%d")
        usage = settings.get("web_usage") or {}
        if usage.get("date") != today:
            usage = {"date": today, "search": 0, "fetch": 0}
        usage[kind] = usage.get(kind, 0) + 1
        settings.set(web_usage=usage)

    def web_state() -> dict[str, Any]:
        usage = settings.get("web_usage") or {}
        today = usage if usage.get("date") == time.strftime("%Y-%m-%d") else {}
        return {"key_set": bool(web_key()), "from_env": not settings.get("web_key") and bool(web_key()),
                "key_page": KEY_PAGE, "signup_page": SIGNUP_PAGE, "searches_today": today.get("search", 0),
                "fetches_today": today.get("fetch", 0)}

    @app.get("/api/web")
    def web_get():
        return jsonify(web_state())

    @app.post("/api/web/key")
    def web_set_key():
        # Kept in settings.json on this computer; never sent back to the page.
        key = "".join(str(body().get("key") or "").split())
        if not key:
            settings.set(web_key=None)
            return jsonify(web_state())
        # Tried before it is kept, so a key copied half-way is caught now, not
        # at the first question. Offline is not the key's fault: kept anyway.
        note = ""
        try:
            WebSearch(lambda: key, counted=count_web_call).check()
        except WebError as exc:
            if "refused" in str(exc):
                return jsonify(error="Ollama did not accept that key. Copy it again from the key page "
                                     "(the whole key, nothing else) and paste it here."), 400
            note = f"Saved, but it could not be tried: {exc}"
        settings.set(web_key=key)
        return jsonify({**web_state(), "note": note})

    # ----------------------------------------------------------- status

    def embed_model(installed: list[dict[str, Any]] | None = None) -> str | None:
        """The embedding model to use for meaning search, if one is installed."""
        if not settings.get("meaning_search", True):
            return None
        try:
            installed = installed if installed is not None else client.installed()
        except OllamaError:
            return None
        names = [m["name"] for m in installed if not is_cloud(m)]
        chosen = settings.get("embed_model")
        if chosen in names:
            return chosen
        for name in names:
            if "embed" in name.lower():
                return name
        return None

    @app.get("/api/status")
    def status():
        machine = hardware.detect()
        up = client.is_up()
        active = settings.get("active_model")
        info: dict[str, Any] = {
            "ollama": {"up": up, "version": client.version() if up else None,
                       "installed": bool(find_binary()) or up, "download": DOWNLOAD_URL,
                       "platform": platform.system(),
                       "installing": [j.to_dict() for j in jobs.recent("install")][-1:]},
            "machine": machine.to_dict(),
            "active_model": active,
            "context": (settings.get("contexts") or {}).get(active) if active else None,
            "loading": [j.to_dict() for j in jobs.active("load")],
            "web": web_state(),
        }
        if up and active:
            try:
                for m in client.running():
                    if m["name"] == active:
                        info["gpu_share"] = round(100 * m["size_vram"] / m["size"]) if m["size"] else 0
            except OllamaError:
                pass
        return jsonify(info)

    @app.post("/api/ollama/start")
    def start_ollama():
        ok, message = client.start()
        return jsonify(ok=ok, message=message), (200 if ok else 503)

    @app.post("/api/ollama/install")
    def install_ollama():
        """Fetch Ollama's official Windows installer and open it.

        Only Windows gets a one-click install: the installer is a normal
        setup program the person clicks through. On macOS and Linux the page
        gives the official download link and command instead.
        """
        if platform.system() != "Windows":
            return jsonify(error="One-click install is for Windows. Use the download link instead."), 400
        for job in jobs.active("install"):
            return jsonify(job=job.id)

        def work(job: Job) -> dict[str, Any]:
            target = Path(tempfile.gettempdir()) / "OllamaSetup.exe"
            request_ = urllib.request.Request(OLLAMA_WINDOWS_INSTALLER, headers={"User-Agent": catalog.USER_AGENT})
            with urllib.request.urlopen(request_, timeout=60) as response, open(target, "wb") as out:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    if job.stopping:
                        raise Stopped()
                    block = response.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    job.update({"completed": done, "total": total})
            os.startfile(str(target))  # type: ignore[attr-defined]  # Windows only
            return {"installer": str(target)}

        job = jobs.start("install", "Install Ollama", work)
        return jsonify(job=job.id)

    @app.get("/api/recommended")
    def recommended():
        """For each good model family, the largest size that runs well here."""
        machine = hardware.detect()
        have, digests = installed_names()
        picks = []
        for family, blurb, fallback in catalog.RECOMMENDED:
            sizes = catalog.family_sizes(family, fallback)
            ordered = sorted(((tag, size) for tag, (size, _) in sizes.items()), key=lambda kv: kv[1])
            best = None
            for tag, size in ordered:
                fit = hardware.assess(size, machine)
                if machine.gpu_memory:
                    good = fit["level"] == "gpu"
                else:
                    # On the CPU, speed falls off a cliff above ~3B parameters.
                    good = fit["level"] == "cpu" and size <= 2.6e9
                if good:
                    best = (tag, size, fit)
            if not best and ordered:
                tag, size = ordered[0]
                best = (tag, size, hardware.assess(size, machine))
            if best:
                tag, size, fit = best
                name = f"{family}:{tag}"
                digest = sizes[tag][1]
                picks.append({"family": family, "name": name, "tag": tag, "size": size, "fit": fit,
                              "description": blurb,
                              "installed": name in have or bool(digest and digest[:12] in digests)})
        embed_name, embed_blurb, embed_size = catalog.EMBEDDING
        installed_embed = any(n.startswith(embed_name) for n in have)
        picks.append({"family": embed_name, "name": embed_name, "tag": "latest", "size": embed_size,
                      "fit": hardware.assess(embed_size, machine, context=2048), "description": embed_blurb,
                      "installed": installed_embed, "embedding": True})
        return jsonify(machine=machine.summary(), picks=picks)

    # ----------------------------------------------------------- models

    @app.get("/api/models")
    def models():
        machine = hardware.detect()
        installed = client.installed()
        running = {m["name"]: m for m in client.running()}
        active = settings.get("active_model")
        embedder = embed_model(installed)
        out = []
        for m in installed:
            caps = client.capabilities(m["name"])
            entry = dict(m, capabilities=caps, cloud=is_cloud(m), active=m["name"] == active,
                         embedding=is_embedding(m["name"], caps), embed_in_use=m["name"] == embedder)
            try:
                shape = client.show(m["name"]).get("model_info") or {}
            except OllamaError:
                shape = {}
            ctx = hardware.context_for(m["size"], shape, machine)
            entry["fit"] = hardware.assess(m["size"], machine, ctx["kv_per_token"], ctx["context"])
            # Once plugged in, the context that was actually verified on this
            # GPU replaces the estimate.
            entry["context"] = (settings.get("contexts") or {}).get(m["name"]) or ctx["context"]
            if m["name"] in running:
                r = running[m["name"]]
                entry["loaded"] = {"gpu_share": round(100 * r["size_vram"] / r["size"]) if r["size"] else 0,
                                   "context": r["context"]}
            out.append(entry)
        out.sort(key=lambda e: (not e["active"], e["embedding"], e["name"]))
        return jsonify(models=out, embed_model=embedder, meaning_search=settings.get("meaning_search", True))

    @app.post("/api/models/use")
    def use_model():
        name = str(body().get("name") or "")
        if is_cloud(name):
            return jsonify(error="Cloud models run on Ollama's servers, so your documents would leave this "
                                 "computer. Pick a model that runs locally."), 400
        installed = {m["name"]: m for m in client.installed()}
        if name not in installed:
            return jsonify(error=f"{name} is not downloaded yet."), 404
        if is_embedding(name, client.capabilities(name)):
            return jsonify(error=f"{name} is an embedding model — it helps search, it cannot answer. "
                                 "It is used automatically."), 400

        def work(job: Job) -> dict[str, Any]:
            machine = hardware.detect(max_age=0)
            size = installed[name]["size"]
            shape = client.show(name).get("model_info") or {}
            ctx = hardware.context_for(size, shape, machine)["context"]
            embedder = embed_model()
            for other in client.running():
                if other["name"] not in (name, embedder):
                    job.update({"phase": f"Unloading {other['name']}"})
                    client.unload(other["name"])
            # Load, then look at where Ollama actually put it. If part spilled
            # to the CPU and a smaller context would have avoided it, shrink the
            # context and load again — the estimate is good, Ollama is the truth.
            share = 0
            for attempt in range(4):
                job.update({"phase": f"Loading {name} with a {ctx:,}-token context", "attempt": attempt + 1})
                client.load(name, ctx)
                loaded = next((m for m in client.running() if m["name"] == name), None)
                share = round(100 * loaded["size_vram"] / loaded["size"]) if loaded and loaded["size"] else 0
                if share >= 99 or not machine.gpu_memory or ctx <= hardware.MIN_CONTEXT:
                    break
                if size + hardware.RUNTIME_OVERHEAD > machine.gpu_memory:
                    break  # the weights alone do not fit; a smaller context cannot fix that
                client.unload(name)
                ctx = max(hardware.MIN_CONTEXT, ctx * 3 // 4 // 1024 * 1024)
            contexts = dict(settings.get("contexts") or {}, **{name: ctx})
            settings.set(active_model=name, contexts=contexts)
            note = ""
            if machine.gpu_memory and share < 99:
                note = (f"Only {share}% of {name} fits in GPU memory; the rest runs on the CPU, so answers "
                        "will be slower. A smaller size of this model would run fully on the GPU.")
            elif not machine.gpu_memory:
                note = "No usable GPU was found, so this model runs on the CPU. Answers will be slow."
            return {"model": name, "context": ctx, "gpu_share": share, "note": note}

        job = jobs.start("load", f"Plug in {name}", work)
        return jsonify(job=job.id)

    @app.post("/api/models/delete")
    def delete_model():
        name = str(body().get("name") or "")
        client.unload(name)
        client.delete(name)
        if settings.get("active_model") == name:
            settings.set(active_model=None)
        return jsonify(ok=True)

    @app.post("/api/models/meaning")
    def meaning_search():
        settings.set(meaning_search=bool(body().get("on")))
        return jsonify(ok=True)

    # ---------------------------------------------------------- catalog

    @app.get("/api/catalog/search")
    def catalog_search():
        query = request.args.get("q", "").strip()
        source = request.args.get("source", "ollama")
        if source == "hf":
            if not query:
                return jsonify(results=[], warning="Type part of a model name to search Hugging Face.")
            return jsonify(catalog.search_hf(query))
        return jsonify(catalog.search_ollama(query))

    def installed_names() -> tuple[set[str], set[str]]:
        """Installed model names, and their digests' first 12 characters.

        One file often has several names (qwen2.5:7b is qwen2.5:7b-instruct),
        so "already downloaded" is decided by digest as well as by name.
        """
        try:
            models = client.installed()
        except OllamaError:
            return set(), set()
        return {m["name"] for m in models}, {m["digest"][:12] for m in models if m.get("digest")}

    def with_fit(data: dict[str, Any]) -> dict[str, Any]:
        machine = hardware.detect()
        have, digests = installed_names()
        for tag in data["tags"]:
            tag["fit"] = hardware.assess(tag["size"], machine) if tag["size"] else None
            tag["installed"] = (tag["name"] in have or f"{tag['name']}:latest" in have
                                or bool(tag.get("digest") and tag["digest"][:12] in digests))
        return data

    @app.get("/api/catalog/tags")
    def catalog_tags():
        return jsonify(with_fit(catalog.ollama_tags(request.args.get("name", "").strip())))

    @app.get("/api/catalog/hf")
    def catalog_hf():
        return jsonify(with_fit(catalog.hf_files(request.args.get("repo", "").strip())))

    @app.post("/api/pull")
    def pull():
        data = body()
        name = str(data.get("name") or "").strip()
        if not name:
            return jsonify(error="Type or pick a model name first."), 400
        if is_cloud(name):
            return jsonify(error="That is a cloud model: it runs on Ollama's servers, so your documents "
                                 "would leave this computer. Pick a size that runs locally."), 400
        size = int(data.get("size") or 0)
        if size and not data.get("force"):
            verdict = hardware.assess(size, hardware.detect(max_age=0))
            if verdict.get("disk_warning"):
                return jsonify(error=verdict["disk_warning"]), 400
        for job in jobs.active("pull"):
            if job.label == name:
                return jsonify(job=job.id)

        def work(job: Job) -> dict[str, Any]:
            client.pull(name, job.update, lambda: job.stopping)
            return {"model": name}

        job = jobs.start("pull", name, work)
        return jsonify(job=job.id)

    # -------------------------------------------------------- documents

    @app.get("/api/documents")
    def documents():
        return jsonify(documents=[d.to_dict() for d in library.all()],
                       adding=[j.to_dict() for j in jobs.active("add")])

    @app.post("/api/documents")
    def add_documents():
        files = [f for f in request.files.getlist("files") if f and f.filename]
        if not files:
            return jsonify(error="No file was given. Choose one or more OCR'd PDF or text files."), 400
        staged: list[tuple[Path, str]] = []
        rejected: list[dict[str, str]] = []
        for upload in files:
            name = Path(upload.filename.replace("\\", "/")).name
            if Path(name).suffix.lower() not in SUPPORTED:
                rejected.append({"name": name, "error": "Only PDF and text files (.pdf, .txt, .md) can be added."})
                continue
            handle, temp = tempfile.mkstemp(dir=incoming, suffix=Path(name).suffix.lower())
            with open(handle, "wb") as out:
                upload.save(out)
            staged.append((Path(temp), name))
        if not staged:
            return jsonify(error="None of these files can be added.", rejected=rejected), 400

        def work(job: Job) -> dict[str, Any]:
            added, skipped = [], list(rejected)
            for number, (temp, name) in enumerate(staged, start=1):
                if job.stopping:
                    break
                job.update({"phase": f"Reading {name}", "file": number, "files": len(staged),
                            "page": 0, "pages": 0, "seconds_left": None})
                started = time.monotonic()

                def on_page(page: int, pages: int) -> None:
                    if job.stopping:
                        raise Cancelled()
                    # Pages take about as long as each other, so the pace so
                    # far is a fair guess at the rest. Too few pages is noise.
                    left = None
                    if page >= 3:
                        left = round((time.monotonic() - started) / page * (pages - page))
                    job.update({"page": page, "pages": pages, "seconds_left": left})

                try:
                    doc, new = library.add(temp, name, on_page)
                    entry = doc.to_dict()
                    entry["new"] = new
                    empty = empty_page_numbers(library.pages(doc.id))
                    if empty:
                        shown = ", ".join(map(str, empty[:12])) + ("…" if len(empty) > 12 else "")
                        entry["warning"] = f"{len(empty)} page(s) have no OCR text and cannot be cited: {shown}"
                    added.append(entry)
                except Cancelled:
                    break
                except ValueError as exc:
                    skipped.append({"name": name, "error": str(exc)})
                finally:
                    temp.unlink(missing_ok=True)
            for temp, _ in staged:  # the ones never reached, after a stop
                temp.unlink(missing_ok=True)
            return {"added": added, "skipped": skipped}

        job = jobs.start("add", f"Add {len(staged)} file(s)", work)
        return jsonify(job=job.id, rejected=rejected)

    @app.delete("/api/documents/<doc_id>")
    def remove_document(doc_id: str):
        return jsonify(ok=library.remove(doc_id))

    @app.get("/api/documents/<doc_id>/file")
    def document_file(doc_id: str):
        path = library.file_path(doc_id)
        if not path:
            abort(404)
        document = library.get(doc_id)
        return send_file(path, download_name=document.name if document else path.name,
                         mimetype="application/pdf" if path.suffix == ".pdf" else "text/plain; charset=utf-8")

    @app.get("/api/documents/<doc_id>/page/<int:number>")
    def document_page(doc_id: str, number: int):
        if not library.get(doc_id):
            abort(404)
        pages = library.pages(doc_id)
        if not 1 <= number <= len(pages):
            abort(404)
        text = pages[number - 1]
        # With ?quote=, where that quote is on the page, for the viewer to mark.
        quote = request.args.get("quote", "")
        return jsonify(page=number, pages=len(pages), text=text, spans=quote_spans(quote, text) if quote else [])

    # ------------------------------------------------------------ asking

    @app.post("/api/ask")
    def api_ask():
        data = body()
        question = str(data.get("question") or "")
        doc_ids = [str(d) for d in data.get("documents") or []]
        model = str(data.get("model") or settings.get("active_model") or "")
        validate(library, question, doc_ids, model)
        chat = chats.get(str(data.get("chat") or ""))
        search_web = bool(data.get("web"))
        if search_web and not web_key():
            raise AskError(f"Web search needs a key. Add one on the Models tab (a free key from {KEY_PAGE}), "
                           "or untick Search the web.")
        if is_cloud(model):
            raise AskError("That is a cloud model; your documents would leave this computer.")
        if jobs.active("load"):
            raise AskError("A model is still being plugged in. Ask again when it says ready.")
        context = (settings.get("contexts") or {}).get(model)
        if not context:
            size = next((m["size"] for m in client.installed() if m["name"] == model), 0)
            shape = client.show(model).get("model_info") or {}
            context = hardware.context_for(size, shape, hardware.detect())["context"]

        if chat is None:
            chat = chats.create(question)
        history = history_of(chat)
        saved_pages = chat.get("web_pages", [])
        searcher = WebSearch(web_key, counted=count_web_call) if search_web else None

        def work(job: Job) -> dict[str, Any]:
            result = ask(client, library, question, doc_ids, model, context, embed_model=embed_model(),
                         history=history, web_pages=saved_pages, web=searcher,
                         on_progress=job.update, should_stop=lambda: job.stopping)
            # Saved only once answered: a stopped or failed question is not part of the chat.
            found = result.pop("new_web_pages", [])
            chats.add_turn(chat["id"], question.strip(), result, doc_ids, web_pages=found)
            return result

        job = jobs.start("ask", question[:80], work)
        return jsonify(job=job.id, chat=chat["id"])

    # ------------------------------------------------------------- chats

    @app.get("/api/chats")
    def chat_list():
        # A chat whose first question was stopped has nothing in it to show.
        return jsonify(chats=[c for c in chats.all() if c["questions"]])

    @app.get("/api/chats/<chat_id>")
    def chat_get(chat_id: str):
        chat = chats.get(chat_id)
        if not chat:
            return jsonify(error="That chat is no longer in History."), 404
        return jsonify(chat)

    @app.get("/api/chats/<chat_id>/web/<page_id>")
    def chat_web_page(chat_id: str, page_id: str):
        page = chats.web_page(chat_id, page_id)
        if not page:
            return jsonify(error="That web page is no longer saved with this chat."), 404
        quote = request.args.get("quote", "")
        return jsonify(url=page["url"], title=page["title"], site=page["site"], text=page["text"],
                       spans=quote_spans(quote, page["text"]) if quote else [])

    @app.delete("/api/chats/<chat_id>")
    def chat_delete(chat_id: str):
        return jsonify(ok=chats.delete(chat_id))

    # -------------------------------------------------------------- jobs

    @app.get("/api/jobs/<job_id>")
    def job_state(job_id: str):
        job = jobs.get(job_id)
        if not job:
            return jsonify(error="That task is no longer known (the portal was restarted)."), 404
        return jsonify(job.to_dict())

    @app.get("/api/jobs")
    def job_list():
        kind = request.args.get("kind", "pull")
        return jsonify(jobs=[j.to_dict() for j in jobs.recent(kind)])

    @app.post("/api/jobs/<job_id>/stop")
    def job_stop(job_id: str):
        job = jobs.get(job_id)
        if job:
            job.stop()
        return jsonify(ok=bool(job))

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status=204)

    return app
