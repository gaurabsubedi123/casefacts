"""Every knob in one place, with the reasoning for each default.

Two rules shape this file:

* The OCR output folder is **read-only** to this tool. ocrtool owns it, may be
  running against it right now, and the documents in it are evidence. Nothing
  here ever writes inside it.
* The index therefore lives somewhere else, and that somewhere is on the Linux
  filesystem rather than under /mnt/c. SQLite on a drvfs mount pays a syscall
  round trip per page fetch; the same index built on ext4 is faster by more
  than an order of magnitude. The documents stay where they are on C:, the
  derived index does not.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- models ---

# Ollama's own default context is 2048 tokens, which silently truncates the
# retrieved pages out of the prompt: the model answers from whatever survived
# and looks confident doing it. That failure is invisible and it is exactly
# the failure this tool must not have, so every call sets num_ctx explicitly.
DEFAULT_NUM_CTX = 8192

# The largest window worth asking for on a machine this size. Qwen2.5 and
# Gemma 3 both handle far more, but the KV cache is RAM, and RAM here is 7 GB
# total with a 4.7 GB model already in it. Past this the machine swaps and a
# question takes minutes instead of seconds.
MAX_NUM_CTX = 32768

# Answers are extraction, not composition. Temperature 0 so the same question
# on the same records gives the same answer twice — a report you cannot
# reproduce is not evidence of anything.
DEFAULT_TEMPERATURE = 0.0


@dataclass(frozen=True)
class ModelSpec:
    """One model this tool knows how to talk to, and what it is good for.

    `notes` is shown in the UI's model picker. The whole point of carrying
    several models is that you compare them on your own records rather than
    trusting a benchmark, so the picker says what to watch for in each.
    """

    name: str
    label: str
    notes: str
    kind: str = "chat"  # "chat" or "embed"
    strips_thinking: bool = False


# The registry is a starting list, not a whitelist: any model `ollama list`
# reports can be used by name. These are the ones with something to say about
# them.
KNOWN_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="qwen2.5:7b-instruct",
        label="Qwen2.5 7B Instruct",
        notes=(
            "The default. Follows the citation format reliably and refuses to "
            "answer off-record more often than the others, which is the "
            "behaviour that matters here."
        ),
    ),
    ModelSpec(
        name="qwen3:8b",
        label="Qwen3 8B",
        notes=(
            "Better at multi-step questions ('did anyone connect the headache "
            "to the collision?'). Slower, and it thinks before answering — the "
            "thinking is stripped before the answer is parsed."
        ),
        strips_thinking=True,
    ),
    ModelSpec(
        name="medgemma:4b",
        label="MedGemma 4B",
        notes=(
            "Google's medical model. Knows clinical vocabulary and abbreviations "
            "the general models guess at. Being medically tuned also makes it "
            "readier to supply textbook knowledge that is not in the record — "
            "check its citations harder, not less."
        ),
    ),
    ModelSpec(
        name="nomic-embed-text",
        label="Nomic Embed Text",
        notes="Turns pages into vectors for the meaning half of the search.",
        kind="embed",
    ),
)

DEFAULT_CHAT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_EMBED_MODEL = "nomic-embed-text"

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


def model_spec(name: str) -> ModelSpec | None:
    for spec in KNOWN_MODELS:
        if spec.name == name or spec.name.split(":")[0] == name:
            return spec
    return None


# -------------------------------------------------------------- chunking ---

# How much page text goes in one retrievable piece.
#
# A chunk is never allowed to cross a page boundary, because a citation that
# spans two pages cannot be checked against one page image, and checking
# against the image is the whole safety story. Within a page, ~1200 characters
# is about a paragraph and a half of a clinical note: big enough to keep a
# finding with its date, small enough that ten of them still leave the model
# room to think.
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200

# A page shorter than this is kept whole rather than chunked. Most pages in a
# claim file are short — a fax header, a signature page, an exhibit sheet.
MIN_CHUNK_CHARS = 120

# ------------------------------------------------------------- retrieval ---

# How many chunks each half of the search returns before they are fused.
DEFAULT_CANDIDATES = 40

# How many survive fusion and go to the model. Twelve chunks at 1200 chars is
# roughly 4k tokens, which leaves half of an 8k window for the question, the
# instructions and the answer.
DEFAULT_TOP_K = 12

# Below this cosine similarity a chunk is noise — included only if the keyword
# half also asked for it. Measured on this corpus: unrelated pages sit around
# 0.35-0.45 against a typical question, related ones above 0.55.
MIN_SIMILARITY = 0.45

# Reciprocal rank fusion's damping constant. 60 is the value from the original
# paper and it is not sensitive; it exists to stop rank 1 dominating rank 2.
RRF_K = 60

# ------------------------------------------------------- whole-document ---

# A document this small skips retrieval entirely and goes into the prompt whole
# — the "attach the file and ask about it" behaviour. Retrieval can only lose
# information here, and for a 20-page discharge summary there is no reason to
# pay that cost.
#
# Characters, not tokens: OCR'd clinical text runs about 3.6 characters per
# token, so 60k characters is roughly 17k tokens and fits a 32k window with the
# instructions and an answer.
WHOLE_DOC_MAX_CHARS = 60000

# --------------------------------------------------------------- folders ---

RESERVED_DIRS = ("_previews", "_runs", "_cache")


def ocrtool_output_dir() -> Path | None:
    """Where ocrtool is currently writing, if it has been configured.

    Read straight out of ocrtool's own config rather than duplicated here, so
    that changing the output folder over there does not silently leave this
    tool indexing a stale one.
    """
    for var in ("OCRTOOL_OUTPUT_DIR",):
        value = os.environ.get(var)
        if value:
            return Path(value).expanduser()
    path = Path.home() / ".ocrtool" / "config.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = data.get("output_dir") or data.get("output")
    return Path(str(value)).expanduser() if value else None


def config_path() -> Path:
    return Path(os.environ.get("CASEFACTS_CONFIG") or (Path.home() / ".casefacts" / "config.json"))


def default_db_path() -> Path:
    value = os.environ.get("CASEFACTS_DB")
    if value:
        return Path(value).expanduser()
    return Path.home() / ".casefacts" / "index.db"


@dataclass
class Settings:
    """What a run of this tool was told to do."""

    records_dir: Path
    db_path: Path = field(default_factory=default_db_path)
    chat_model: str = DEFAULT_CHAT_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    ollama_host: str = DEFAULT_OLLAMA_HOST
    top_k: int = DEFAULT_TOP_K
    candidates: int = DEFAULT_CANDIDATES
    num_ctx: int = DEFAULT_NUM_CTX
    temperature: float = DEFAULT_TEMPERATURE

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["records_dir"] = str(self.records_dir)
        data["db_path"] = str(self.db_path)
        return data


def load_config() -> dict[str, Any]:
    try:
        return json.loads(config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_config(values: dict[str, Any]) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    current = load_config()
    current.update({k: v for k, v in values.items() if v is not None})
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    return path


def default_records_dir() -> Path:
    """The OCR output folder to read, in order of who gets the last word."""
    value = os.environ.get("CASEFACTS_RECORDS")
    if value:
        return Path(value).expanduser()
    saved = load_config().get("records_dir")
    if saved:
        return Path(str(saved)).expanduser()
    from_ocrtool = ocrtool_output_dir()
    if from_ocrtool:
        return from_ocrtool
    return Path.home() / "ocr-output"


def settings_from_args(records: str | None = None, db: str | None = None, **kwargs: Any) -> Settings:
    saved = load_config()
    host = kwargs.get("ollama_host") or saved.get("ollama_host") or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
    if not host.startswith("http"):
        host = "http://" + host
    return Settings(
        records_dir=Path(records).expanduser() if records else default_records_dir(),
        db_path=Path(db).expanduser() if db else Path(str(saved.get("db_path") or default_db_path())),
        chat_model=kwargs.get("model") or saved.get("chat_model") or DEFAULT_CHAT_MODEL,
        embed_model=kwargs.get("embed_model") or saved.get("embed_model") or DEFAULT_EMBED_MODEL,
        ollama_host=host.rstrip("/"),
        top_k=int(kwargs.get("top_k") or saved.get("top_k") or DEFAULT_TOP_K),
        candidates=int(kwargs.get("candidates") or saved.get("candidates") or DEFAULT_CANDIDATES),
        num_ctx=int(kwargs.get("num_ctx") or saved.get("num_ctx") or DEFAULT_NUM_CTX),
        temperature=float(kwargs.get("temperature", DEFAULT_TEMPERATURE)),
    )
