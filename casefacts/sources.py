"""Turning whatever the user plugged in into text this tool can index.

The point of this module is that you should not have to prepare anything. You
point at a thing — one PDF, a folder of scans, a folder ocrtool has already
been through, a single .txt — and it becomes answerable.

    casefacts ask "when was the MRI?" --in ~/Desktop/records/MRI-report.pdf
    casefacts ask "list every provider" --in "/mnt/c/.../Smith medical records"

Three cases, decided per file:

* **Already text** (.txt, .md) — read it, no work.
* **Already OCR'd by ocrtool** — a document with a .txt beside it, or in a
  txt/ folder beside it, is done. Its text is used and the original is left
  alone. This is what makes pointing at an ocr-output folder free.
* **Not OCR'd yet** — ocrtool is run over it, into a workspace under
  ~/.casefacts/ocr/, and the .txt it writes is what gets indexed.

The workspace is keyed by the path you plugged in, and ocrtool keeps its own
ledger of what it has already read, so plugging the same folder in a second
time re-OCRs nothing.

Nothing is ever written next to your documents.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .config import RESERVED_DIRS

log = logging.getLogger(__name__)

TEXT_SUFFIXES = {".txt", ".md", ".text"}

# What ocrtool can read. Kept here rather than imported so this tool does not
# depend on ocrtool being importable — it is invoked as a command, not linked.
OCRABLE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class SourceError(RuntimeError):
    """The path cannot be turned into text."""


@dataclass
class ResolvedDoc:
    """One document ready to index, and where it really came from."""

    txt_path: Path
    origin: str          # "text" | "ocr-output" | "ocr'd here"
    original_path: Path  # the PDF or scan behind the text, when there is one


@dataclass
class Source:
    """A path the user plugged in, resolved to its documents."""

    root: Path
    docs: list[ResolvedDoc]
    ocr_output_dir: Path | None = None
    ocr_ran: bool = False

    @property
    def previews_dir(self) -> Path | None:
        """Where the page images for these documents live, if there are any."""
        return self.ocr_output_dir


def workspace_root() -> Path:
    return Path(os.environ.get("CASEFACTS_HOME") or (Path.home() / ".casefacts"))


def workspace_for(path: Path) -> Path:
    """A stable, private folder to hold the OCR of one plugged-in path.

    Named after the path so it is recognisable in a file manager, and hashed so
    that two folders called "records" in different places never collide.
    """
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:10]
    stem = SAFE_NAME.sub("-", resolved.name or "root").strip("-")[:48] or "source"
    return workspace_root() / "ocr" / f"{stem}-{digest}"


def ocrtool_command() -> list[str] | None:
    """How to run ocrtool on this machine, or None if it is not installed.

    Checked in the order of who is most likely to be right: an explicit
    setting, then the sibling checkout's virtualenv (which is how it is
    installed here), then PATH.
    """
    explicit = os.environ.get("CASEFACTS_OCRTOOL")
    if explicit:
        return [explicit]

    here = Path(__file__).resolve().parent.parent
    for candidate in (
        here.parent / "ocr" / ".venv" / "bin" / "ocrtool",
        here.parent / "ocr" / ".venv" / "Scripts" / "ocrtool.exe",
    ):
        if candidate.is_file():
            return [str(candidate)]

    found = shutil.which("ocrtool")
    if found:
        return [found]

    # Installed as a library in this interpreter but not on PATH.
    try:
        import ocrtool  # noqa: F401

        return [sys.executable, "-m", "ocrtool.cli"]
    except ImportError:
        return None


def _is_reserved(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return False
    return any(part in RESERVED_DIRS for part in parts)


def existing_text_for(document: Path) -> Path | None:
    """The .txt ocrtool already wrote for this document, if it did.

    Covers all three ocrtool output layouts: the .txt beside the file, in a
    txt/ folder beside it, or in a txt/ folder at the top of the tree.
    """
    stem = document.stem
    candidates = [
        document.with_suffix(".txt"),
        document.parent / "txt" / f"{stem}.txt",
        document.parent.parent / "txt" / f"{stem}.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _walk(path: Path, recursive: bool = True) -> list[Path]:
    if path.is_file():
        return [path]
    walker = path.rglob("*") if recursive else path.glob("*")
    return sorted(
        p
        for p in walker
        if p.is_file() and not p.name.startswith(".") and not _is_reserved(p, path)
    )


def run_ocr(
    input_dir: Path,
    output_dir: Path,
    *,
    progress: Callable[[str], None] | None = None,
    extra_args: Iterable[str] = (),
) -> None:
    """Hand a folder to ocrtool and relay what it says.

    The searchable PDFs are skipped (`--no-pdf`): they are the slowest part of
    an OCR run and this tool never reads them. Page images are *not* skipped —
    they are what a citation is checked against.
    """
    command = ocrtool_command()
    if command is None:
        raise SourceError(
            "ocrtool is not installed, so files that are not already OCR'd cannot be read. "
            "Install it from the sibling folder, or point CASEFACTS_OCRTOOL at it."
        )
    argv = [*command, "run", str(input_dir), "-o", str(output_dir), "--no-pdf", *extra_args]
    log.info("running: %s", " ".join(argv))
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if line and progress:
            progress(line)
    code = process.wait()
    if code != 0:
        raise SourceError(f"ocrtool exited with status {code} — see the output above")


def resolve(
    path: Path,
    *,
    recursive: bool = True,
    ocr: bool = True,
    progress: Callable[[str], None] | None = None,
) -> Source:
    """Everything answerable under `path`, OCR'ing whatever needs it.

    `ocr=False` indexes only what is already text, which is the right setting
    when pointing at an ocr-output folder and the wrong one when pointing at a
    folder of scans.
    """
    path = path.expanduser()
    if not path.exists():
        raise SourceError(f"no such file or folder: {path}")
    path = path.resolve()

    files = _walk(path, recursive)
    docs: list[ResolvedDoc] = []
    needs_ocr: list[Path] = []

    # Text first, then the documents behind it. Two passes rather than one
    # because a folder is walked in name order, which puts pdf/ before txt/ —
    # so on a single pass a PDF is examined before the .txt that already
    # covers it exists to be found.
    by_txt: dict[Path, ResolvedDoc] = {}
    for candidate in files:
        if candidate.suffix.lower() in TEXT_SUFFIXES:
            doc = ResolvedDoc(txt_path=candidate, origin="text", original_path=candidate)
            docs.append(doc)
            by_txt[candidate] = doc

    for candidate in files:
        if candidate.suffix.lower() not in OCRABLE_SUFFIXES:
            continue
        already = existing_text_for(candidate)
        if already is not None and already in by_txt:
            # Its text is already indexed; recording the original here is what
            # lets an answer say which PDF the cited page came from.
            doc = by_txt[already]
            doc.origin = "ocr-output"
            doc.original_path = candidate
        elif already is None:
            needs_ocr.append(candidate)

    output_dir: Path | None = None
    ocr_ran = False
    if needs_ocr and ocr:
        output_dir = workspace_for(path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ocrtool reads a folder, so the files needing OCR are gathered into
        # one — by link, not by copy. Only the ones that need it: a folder
        # where half the documents already have their text would otherwise be
        # handed to ocrtool whole and read from the beginning.
        #
        # Links rather than copies because these are medical records, and there
        # is no reason to make a second copy of one on disk.
        input_dir = output_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        for original in needs_ocr:
            relative = original.relative_to(path) if path.is_dir() else Path(original.name)
            link = input_dir / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.exists() or link.is_symlink():
                continue
            try:
                link.symlink_to(original)
            except OSError:
                shutil.copy2(original, link)

        if progress:
            progress(f"OCR needed for {len(needs_ocr)} file(s) — running ocrtool into {output_dir}")
        run_ocr(input_dir, output_dir, progress=progress)
        ocr_ran = True

        for original in needs_ocr:
            relative = original.relative_to(path) if path.is_dir() else Path(original.name)
            produced = _find_produced_text(output_dir, relative, original.stem)
            if produced is not None:
                docs.append(ResolvedDoc(txt_path=produced, origin="ocr'd here", original_path=original))
            else:
                log.warning("ocrtool produced no text for %s", original)
    elif needs_ocr and not ocr:
        log.info("%d file(s) are not OCR'd and --no-ocr was set; skipping them", len(needs_ocr))

    # An ocr-output folder that was pointed at directly has its previews in
    # itself; one this tool OCR'd has them in the workspace.
    previews = output_dir
    if previews is None and path.is_dir() and (path / "_previews").is_dir():
        previews = path
    if previews is None and path.is_dir():
        for parent in [path, *path.parents]:
            if (parent / "_previews").is_dir():
                previews = parent
                break

    # Shallowest path first, so that when the same document appears twice the
    # copy kept as the primary is the one nearer the top of the tree. In a
    # produced file the nested copies are the ones named "a.pdf" and "c.pdf";
    # the top-level copy is the one with the name a person gave it, and that
    # name is what every citation will carry.
    docs.sort(key=lambda d: (len(d.txt_path.parts), str(d.txt_path).lower()))
    return Source(root=path, docs=docs, ocr_output_dir=previews, ocr_ran=ocr_ran)


def _find_produced_text(output_dir: Path, relative: Path, stem: str) -> Path | None:
    """Where ocrtool put the .txt for a document, whichever layout it used."""
    candidates = [
        output_dir / relative.parent / "txt" / f"{stem}.txt",
        output_dir / "txt" / f"{stem}.txt",
        output_dir / relative.parent / f"{stem}.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(output_dir.rglob(f"{stem}.txt"))
    return matches[0] if matches else None
