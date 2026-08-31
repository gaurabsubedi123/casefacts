"""Command line: plug something in, ask about it, build a chronology.

argparse rather than a CLI framework, for the same reason as ocrtool: one
fewer dependency, and this is a handful of commands.

The shape most of the work takes is:

    casefacts ask "when was the MRI?" --in ~/Desktop/records
    casefacts ask "what did the ED find?" --file "ED report"
    casefacts chronology --in ~/Desktop/records --csv timeline.csv

`--in` does whatever is needed to make that path answerable — OCR it if it is
a stack of scans, index it if it is new, nothing if it has not changed since
last time — and then limits the question to it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .answer import ask
from .chronology import GAP_DAYS, gaps, sweep, timeline, to_csv
from .compare import compare
from .config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    KNOWN_MODELS,
    Settings,
    config_path,
    default_records_dir,
    model_spec,
    save_config,
    settings_from_args,
)
from .index import Index
from .ollama import Ollama, OllamaError
from .sources import SourceError, ocrtool_command, workspace_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="casefacts",
        description="Ask questions of an OCR'd case file, on this machine, with a page citation on every answer.",
    )
    parser.add_argument("--version", action="version", version=f"casefacts {__version__}")
    parser.add_argument("--db", default=None, help="index file to use (default ~/.casefacts/index.db)")
    parser.add_argument("--host", dest="ollama_host", default=None, help="Ollama address (default 127.0.0.1:11434)")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="plug in a file or folder and index it")
    add.add_argument("path", help="a document, or a folder of them")
    add.add_argument("--no-ocr", action="store_true", help="index only what is already text")
    add.add_argument("--no-recursive", action="store_true", help="only the top level of the folder")
    add.add_argument("--rebuild", action="store_true", help="re-read it from scratch")
    add.add_argument("--embed-model", default=None, help=f"embedding model (default {DEFAULT_EMBED_MODEL})")

    ask_cmd = sub.add_parser("ask", help="ask a question of the records")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--in", dest="source", default=None, help="a file or folder to plug in and ask about")
    ask_cmd.add_argument("--file", default=None, help="ask about one indexed document (name or part of it)")
    ask_cmd.add_argument("--folder", default=None, help="ask about one indexed folder")
    ask_cmd.add_argument("-m", "--model", default=None, help=f"model to answer with (default {DEFAULT_CHAT_MODEL})")
    ask_cmd.add_argument("--compare", default=None, help="comma-separated models to answer the same question")
    ask_cmd.add_argument("--all-models", action="store_true", help="compare every installed chat model")
    ask_cmd.add_argument("-k", "--top-k", type=int, default=None, help="how many excerpts to give the model")
    ask_cmd.add_argument("--whole", action="store_true", help="put the whole document in the prompt, no retrieval")
    ask_cmd.add_argument("--json", action="store_true", help="print the full result as JSON")
    ask_cmd.add_argument("--show-sources", action="store_true", help="print the excerpts the model was given")

    chron = sub.add_parser("chronology", help="build a dated timeline of the whole file")
    chron.add_argument("--in", dest="source", default=None, help="a file or folder to plug in first")
    chron.add_argument("--file", default=None, help="one indexed document")
    chron.add_argument("--folder", default=None, help="one indexed folder")
    chron.add_argument("-m", "--model", default=None, help=f"model to extract with (default {DEFAULT_CHAT_MODEL})")
    chron.add_argument("--rebuild", action="store_true", help="read every page again")
    chron.add_argument("--limit", type=int, default=None, help="stop after this many pages (for a trial run)")
    chron.add_argument("--csv", default=None, help="write the timeline to this file")
    chron.add_argument("--gap-days", type=int, default=GAP_DAYS, help=f"call a silence a gap after this many days (default {GAP_DAYS})")
    chron.add_argument("--include-unverified", action="store_true", help="include events whose quote was not found")
    chron.add_argument("--show-only", action="store_true", help="print what was already extracted, read no pages")

    search = sub.add_parser("search", help="find pages without asking a model anything")
    search.add_argument("query")
    search.add_argument("--file", default=None)
    search.add_argument("--folder", default=None)
    search.add_argument("-k", "--top-k", type=int, default=10)

    page_cmd = sub.add_parser("page", help="print one page as it was read")
    page_cmd.add_argument("document", help="document name, or part of it")
    page_cmd.add_argument("page", type=int)

    sub.add_parser("sources", help="what has been plugged in so far")
    docs = sub.add_parser("documents", help="what is indexed")
    docs.add_argument("--duplicates", action="store_true", help="include the copies too")

    forget = sub.add_parser("forget", help="drop everything that came from one path")
    forget.add_argument("path")

    sub.add_parser("models", help="which models are installed, and what each is for")
    sub.add_parser("doctor", help="check that everything this tool needs is present")

    ui_cmd = sub.add_parser("ui", help="start the local web interface")
    ui_cmd.add_argument("--host", default="127.0.0.1", help="default 127.0.0.1 — this machine only")
    ui_cmd.add_argument("--port", type=int, default=5002, help="default 5002 (ocrtool uses 5001)")
    ui_cmd.add_argument("--debug", action="store_true")

    setup = sub.add_parser("config", help="show or set the defaults")
    setup.add_argument("--records", default=None, help="folder to answer from when none is given")
    setup.add_argument("--model", default=None, help="default model")
    setup.add_argument("--embed-model", default=None, help="default embedding model")

    args = parser.parse_args(argv)
    handler = {
        "add": cmd_add,
        "ask": cmd_ask,
        "chronology": cmd_chronology,
        "search": cmd_search,
        "page": cmd_page,
        "sources": cmd_sources,
        "documents": cmd_documents,
        "forget": cmd_forget,
        "models": cmd_models,
        "doctor": cmd_doctor,
        "ui": cmd_ui,
        "config": cmd_config,
    }[args.command]
    try:
        return handler(args)
    except SourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\nstopped. Nothing was lost — run the same command again to carry on.", file=sys.stderr)
        return 130


# ------------------------------------------------------------------ helpers


def _settings(args: argparse.Namespace) -> Settings:
    return settings_from_args(
        db=getattr(args, "db", None),
        model=getattr(args, "model", None),
        embed_model=getattr(args, "embed_model", None),
        ollama_host=getattr(args, "ollama_host", None),
        top_k=getattr(args, "top_k", None),
    )


def _progress(event: str, payload: dict[str, Any]) -> None:
    if event == "ocr":
        print(f"  ocr: {payload['line']}")
    elif event == "document":
        print(f"  [{payload['position']}/{payload['total']}] {payload['file']} — {payload['pages']} pages")
    elif event == "duplicate":
        print(f"      already indexed under another name; not read again")
    elif event == "embedding":
        print(f"      indexing {payload['done']}/{payload['total']}", end="\r", flush=True)
    elif event == "error":
        print(f"  ! {payload.get('file', '')}: {payload['error']}")
    elif event == "done":
        print(" " * 60, end="\r")


def _plug_in(index: Index, path_text: str, *, ocr: bool = True, recursive: bool = True, rebuild: bool = False) -> str:
    """Make a path answerable, and return it as the folder scope to use."""
    path = Path(path_text).expanduser()
    print(f"reading {path}")
    source, report = index.ingest_path(
        path, ocr=ocr, recursive=recursive, rebuild=rebuild, progress=_progress
    )
    data = report.to_dict()
    print(
        f"  {data['documents']} document(s), {data['pages']} pages, {data['chunks']} pieces indexed"
        + (f", {data['duplicates']} duplicate(s) skipped" if data["duplicates"] else "")
        + (f", {data['unchanged']} unchanged" if data["unchanged"] else "")
        + f" in {data['seconds']}s"
    )
    for error in data["errors"]:
        print(f"  ! {error}")
    return str(source.root)


def _scope(args: argparse.Namespace, index: Index) -> dict[str, Any]:
    scope: dict[str, Any] = {"doc": getattr(args, "file", None), "folder": getattr(args, "folder", None)}
    source = getattr(args, "source", None)
    if source:
        root = _plug_in(index, source)
        path = Path(source).expanduser().resolve()
        # A single file plugged in is a document scope, not a folder scope, so
        # that it can go into the prompt whole.
        if path.is_file():
            scope["doc"] = scope["doc"] or path.stem
        else:
            scope["folder"] = scope["folder"] or root
        print()
    return scope


def _chat_models(client: Ollama) -> list[str]:
    installed = [m["name"] for m in client.installed()]
    return [
        name
        for name in installed
        if not (model_spec(name) and model_spec(name).kind == "embed")
        and "embed" not in name.lower()
    ]


# ----------------------------------------------------------------- commands


def cmd_add(args: argparse.Namespace) -> int:
    index = Index(_settings(args))
    _plug_in(
        index,
        args.path,
        ocr=not args.no_ocr,
        recursive=not args.no_recursive,
        rebuild=args.rebuild,
    )
    stats = index.stats()
    print(f"\nindex now holds {stats['documents']} document(s), {stats['pages']} pages")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    settings = _settings(args)
    index = Index(settings)
    if not index.stats()["pages"] and not args.source:
        print(
            "nothing is indexed yet. Plug something in first:\n"
            "  casefacts add <a file or folder>\n"
            "or ask about it directly:\n"
            '  casefacts ask "your question" --in <a file or folder>',
            file=sys.stderr,
        )
        return 1

    scope = _scope(args, index)
    models: list[str] = []
    if args.all_models:
        models = _chat_models(Ollama(settings.ollama_host))
    elif args.compare:
        models = [m.strip() for m in args.compare.split(",") if m.strip()]

    if models:
        result = compare(
            index, args.question, models,
            doc=scope["doc"], folder=scope["folder"],
            top_k=args.top_k, whole=True if args.whole else None,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
            return 0
        _print_comparison(result)
        return 0

    answer = ask(
        index, args.question,
        model=args.model, doc=scope["doc"], folder=scope["folder"],
        top_k=args.top_k, whole=True if args.whole else None,
    )
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2))
        return 0
    _print_answer(answer, show_sources=args.show_sources)
    return 0 if answer.findings or answer.answer else 1


def _print_answer(answer: Any, *, show_sources: bool = False) -> None:
    print(f"Q: {answer.question}")
    print(f"   {answer.model}, {answer.mode}, {answer.seconds:.1f}s\n")

    if answer.answer:
        print(answer.answer)
        print()

    if answer.findings:
        print("Findings — every one checked against the page it cites:\n")
        marks = {
            "verified": "OK   ", "close": "OK~  ", "joined": "JOIN ",
            "wrong page": "MOVED", "unverified": "  ?? ",
        }
        for finding in answer.findings:
            print(f"  {marks.get(finding.verdict, '  ?? ')} {finding.statement}")
            print(f"        {finding.citation or '(no page)'}", end="")
            if finding.confidence is not None and finding.confidence < 70:
                print(f"   [OCR {finding.confidence:.0f}% — read the page image]", end="")
            print()
            print(f'        "{finding.quote[:160]}"')
            if finding.verdict == "joined":
                print("        (every word is on the page in this order, but read across the layout —")
                print("         true to the page, not usable as a verbatim quotation)")
            elif finding.verdict == "wrong page":
                print("        (the model cited a different page; corrected to the one holding the quote)")
            elif finding.verdict == "unverified":
                print("        (this quote is not on any page retrieved — treat it as unsupported)")
            print()

    if answer.missing:
        print(f"Not in the records: {answer.missing}\n")

    for warning in answer.warnings:
        print(f"! {warning}")

    if show_sources:
        print("\nExcerpts given to the model:")
        for number, hit in enumerate(answer.hits, start=1):
            print(f"\n  [{number}] {hit.label()}" + (f" ({hit.bates})" if hit.bates else ""))
            for line in hit.text.strip().splitlines()[:8]:
                print(f"      {line[:100]}")


def _print_comparison(result: Any) -> None:
    print(f"Q: {result.question}\n")
    for answer in result.answers:
        unverified = len(answer.findings) - answer.verified_count
        print(f"--- {answer.model}  ({answer.seconds:.1f}s, "
              f"{answer.verified_count} checked, {unverified} unsupported)")
        print(f"    {answer.answer or '(no answer)'}")
        for finding in answer.findings:
            mark = "OK " if finding.verdict != "unverified" else "?? "
            print(f"      {mark} {finding.citation}: {finding.statement[:90]}")
        for warning in answer.warnings:
            print(f"      ! {warning}")
        print()

    agreement = result.agreement()
    if agreement:
        print(f"Pages every model cited: {', '.join(agreement['pages_all_models_cited']) or 'none'}")
        print(f"Agreement on pages: {agreement['agreement']:.0%}")
        for model, only in agreement["only_one_model"].items():
            if only:
                print(f"  only {model}: {', '.join(only)}")
        print("\nThe pages every model landed on are the ones to read first.")


def cmd_chronology(args: argparse.Namespace) -> int:
    index = Index(_settings(args))
    scope = _scope(args, index)

    if not args.show_only:
        state = {"last": ""}

        def progress(event: str, payload: dict[str, Any]) -> None:
            if event == "start":
                print(f"reading {payload['pages']} pages with {payload['model']}")
                print("(interrupt with Ctrl-C at any time; what has been read is kept)\n")
            elif event == "page":
                state["last"] = f"  page {payload['position']}/{payload['total']}: {payload['title'][:40]} p.{payload['page']}"
                print(state["last"][:100], end="\r", flush=True)
            elif event == "done":
                print(" " * 100, end="\r")
                print(
                    f"read {payload['pages_read']} pages "
                    f"({payload['pages_skipped']} had no date, {payload['pages_cached']} already done) "
                    f"in {payload['seconds']}s\n"
                )

        sweep(
            index, model=args.model, doc=scope["doc"], folder=scope["folder"],
            rebuild=args.rebuild, limit=args.limit, progress=progress,
        )

    events = timeline(
        index, model=args.model, doc=scope["doc"], folder=scope["folder"],
        include_unverified=args.include_unverified,
    )
    if not events:
        print("no dated events found yet.")
        return 1

    print(f"{len(events)} events\n")
    for event in events:
        when = event.date_iso or f"({event.date_text})"
        who = event.provider or event.facility
        line = f"  {when:12s} {event.description[:60]}"
        if who:
            line += f"  — {who[:28]}"
        print(line)
        print(f"               {event.citation}" + ("" if event.verdict == "verified" else f"  [{event.verdict}]"))

    found = gaps(events, days=args.gap_days)
    if found:
        print(f"\nGaps of {args.gap_days}+ days with no record:")
        for gap in found:
            print(f"  {gap.days:4d} days   {gap.after} -> {gap.before}")
        print("\nThese are what a defence examiner will call a break in treatment.")

    if args.csv:
        path = Path(args.csv).expanduser()
        path.write_text(to_csv(events), encoding="utf-8")
        print(f"\nwritten to {path}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    index = Index(_settings(args))
    doc_ids = index.scope_doc_ids(doc=args.file, folder=args.folder)
    hits = index.search(args.query, top_k=args.top_k, doc_ids=doc_ids)
    if not hits:
        print("nothing matched.")
        return 1
    for hit in hits:
        found_by = []
        if hit.keyword_rank:
            found_by.append(f"words #{hit.keyword_rank}")
        if hit.vector_rank:
            found_by.append(f"meaning #{hit.vector_rank}")
        print(f"\n{hit.label()}" + (f" ({hit.bates})" if hit.bates else "") + f"   [{', '.join(found_by)}]")
        for line in hit.text.strip().splitlines()[:5]:
            print(f"    {line[:100]}")
    return 0


def cmd_page(args: argparse.Namespace) -> int:
    index = Index(_settings(args))
    doc_id = index.resolve_doc(args.document)
    if not doc_id:
        print(f"no document matching {args.document!r}", file=sys.stderr)
        return 1
    page = index.page(doc_id, args.page)
    if not page:
        print(f"no page {args.page} in that document", file=sys.stderr)
        return 1
    record = index.document(doc_id) or {}
    print(f"{record.get('title')} — page {page['page_no']}"
          + (f" ({page['bates']})" if page["bates"] else ""))
    if page["confidence"] is not None:
        print(f"OCR confidence {page['confidence']:.0f}%" + ("  — flagged for review" if page["needs_review"] else ""))
    if page["preview"]:
        print(f"page image: {Path(record.get('previews_root') or '') / page['preview']}")
    print("-" * 70)
    print(page["text"])
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    index = Index(_settings(args))
    rows = index.sources()
    if not rows:
        print("nothing plugged in yet. Try:  casefacts add <a file or folder>")
        return 1
    for row in rows:
        print(f"  {row['documents']:4d} docs  {row['pages'] or 0:6d} pages  {row['last_ingest']}  {row['root']}")
    return 0


def cmd_documents(args: argparse.Namespace) -> int:
    index = Index(_settings(args))
    rows = index.documents(include_duplicates=args.duplicates)
    if not rows:
        print("nothing indexed yet.")
        return 1
    for row in rows:
        confidence = f"{row['mean_confidence']:.0f}%" if row["mean_confidence"] is not None else "  -"
        print(f"  {row['page_count']:5d}p  OCR {confidence:>5}  {row['title'][:60]}")
        for alias in index.aliases(row["doc_id"]):
            print(f"                        also filed as {alias['rel_path']}")
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    index = Index(_settings(args))
    removed = index.forget_source(Path(args.path).expanduser().resolve())
    print(f"removed {removed} document(s) from the index. Your files were not touched.")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    settings = _settings(args)
    client = Ollama(settings.ollama_host)
    try:
        installed = client.installed()
    except OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    names = {m["name"] for m in installed}

    print("Installed:\n")
    for model in installed:
        spec = model_spec(model["name"])
        size = model["size_bytes"] / 1e9
        default = "  (default)" if model["name"] == settings.chat_model else ""
        print(f"  {model['name']:26s} {size:5.1f} GB{default}")
        if spec:
            for line in _wrap(spec.notes, 74):
                print(f"      {line}")
        print()

    missing = [s for s in KNOWN_MODELS if s.name not in names]
    if missing:
        print("Worth having, not installed:\n")
        for spec in missing:
            print(f"  ollama pull {spec.name}")
            for line in _wrap(spec.notes, 74):
                print(f"      {line}")
            print()
    print("Compare them on your own records rather than trusting any list:")
    print('  casefacts ask "your question" --all-models')
    return 0


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width)


def cmd_doctor(args: argparse.Namespace) -> int:
    settings = _settings(args)
    ok = True
    print(f"casefacts {__version__}\n")

    client = Ollama(settings.ollama_host)
    if client.is_up():
        installed = {m["name"] for m in client.installed()}
        print(f"  Ollama          running at {settings.ollama_host}")
        for name, role in ((settings.chat_model, "answering"), (settings.embed_model, "the meaning half of search")):
            if name in installed or f"{name}:latest" in installed:
                print(f"  {name:15s} installed, used for {role}")
            else:
                print(f"  {name:15s} NOT INSTALLED — run: ollama pull {name}")
                ok = False
    else:
        print(f"  Ollama          NOT RUNNING at {settings.ollama_host} — start it with: ollama serve")
        ok = False

    command = ocrtool_command()
    if command:
        print(f"  ocrtool         {' '.join(command)}")
    else:
        print("  ocrtool         not found — files that are not already OCR'd cannot be read")

    index = Index(settings)
    stats = index.stats()
    print(f"\n  index           {stats['db_path']}")
    print(f"  holding         {stats['documents']} documents, {stats['pages']} pages, {stats['chunks']} pieces, {stats['vectors']} vectors")
    if stats["chunks"] and stats["vectors"] < stats["chunks"]:
        print("                  some pieces have no vector — search is running on words alone")
        print("                  re-run: casefacts add <path> --rebuild")
        ok = False
    print(f"  OCR workspace   {workspace_root() / 'ocr'}")
    print(f"  settings        {config_path()}")

    print("\n  everything needed is here." if ok else "\n  something above needs attention.")
    return 0 if ok else 1


def cmd_config(args: argparse.Namespace) -> int:
    values = {}
    if args.records:
        values["records_dir"] = str(Path(args.records).expanduser().resolve())
    if args.model:
        values["chat_model"] = args.model
    if args.embed_model:
        values["embed_model"] = args.embed_model
    if values:
        path = save_config(values)
        print(f"saved to {path}")
    settings = settings_from_args()
    print(f"  default folder  {default_records_dir()}")
    print(f"  answering with  {settings.chat_model}")
    print(f"  embedding with  {settings.embed_model}")
    print(f"  index           {settings.db_path}")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from .web.app import create_app

    settings = _settings(args)
    app = create_app(settings)
    print(f"casefacts {__version__} — http://{args.host}:{args.port}")
    print("Nothing leaves this machine.\n")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
