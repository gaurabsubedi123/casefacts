"""Where things live, whether running from source or from a frozen .exe.

Two different questions with two different answers:

* the program's own files (HTML, CSS, JS) sit next to the code — or, inside a
  PyInstaller bundle, in the temporary folder it unpacks to (sys._MEIPASS);
* the user's data (uploaded documents, extracted text, settings) must survive
  the program being replaced, so it goes in the per-user application-data
  folder of whatever OS this is, never next to the .exe.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


def resource_dir() -> Path:
    """The folder holding static/ and templates/."""
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        return Path(bundle) / "modelportal"
    return Path(__file__).resolve().parent


def data_dir() -> Path:
    """The per-user folder for documents and settings, created on first use."""
    override = os.environ.get("MODELPORTAL_HOME")
    if override:
        root = Path(override).expanduser()
    elif platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        root = Path(base) / "ModelPortal"
    elif platform.system() == "Darwin":
        root = Path.home() / "Library" / "Application Support" / "ModelPortal"
    else:
        root = Path.home() / ".modelportal"
    root.mkdir(parents=True, exist_ok=True)
    return root


def ollama_models_dir() -> Path:
    """Where Ollama keeps downloaded models — what a download's disk check is about.

    OLLAMA_MODELS wins when set. Otherwise the per-user default; a Linux
    service install keeps them under the ollama user's home instead, so that is
    tried when the per-user folder does not exist.
    """
    override = os.environ.get("OLLAMA_MODELS")
    if override:
        return Path(override).expanduser()
    candidates = [Path.home() / ".ollama" / "models"]
    if platform.system() == "Linux":
        candidates.append(Path("/usr/share/ollama/.ollama/models"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
