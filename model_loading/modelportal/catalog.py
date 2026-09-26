"""Finding models to download: the Ollama library and Hugging Face.

Ollama has no public search API, so its library is read from the same HTML
pages a browser gets (ollama.com/search and /library/<name>/tags). That markup
can change without notice; when parsing finds nothing, a short built-in list
of well-known models is returned instead, and the "download by exact name" box
always works regardless, because pulling goes through Ollama itself.

Hugging Face does have an API. Only GGUF repositories are offered, because
that is the only format Ollama can pull from there (as hf.co/<repo>:<quant>).
Models split across several .gguf files are left out: Ollama cannot pull them.
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

USER_AGENT = "ModelPortal/0.1 (+local model manager)"
CACHE_SECONDS = 600

# Shown when ollama.com cannot be read. Names only; sizes come from the tags
# page once it can be reached.
FALLBACK = [
    {"name": "qwen2.5", "description": "Alibaba's Qwen 2.5 — strong general model, good at following formats.",
     "sizes": ["0.5b", "1.5b", "3b", "7b", "14b", "32b", "72b"], "capabilities": ["tools"]},
    {"name": "qwen3", "description": "Qwen 3 — newer Qwen with optional thinking.",
     "sizes": ["0.6b", "1.7b", "4b", "8b", "14b", "30b", "32b"], "capabilities": ["tools", "thinking"]},
    {"name": "deepseek-r1", "description": "DeepSeek-R1 reasoning models; the small sizes are distilled into Qwen/Llama.",
     "sizes": ["1.5b", "7b", "8b", "14b", "32b", "70b", "671b"], "capabilities": ["thinking"]},
    {"name": "llama3.1", "description": "Meta Llama 3.1.", "sizes": ["8b", "70b", "405b"], "capabilities": ["tools"]},
    {"name": "gemma3", "description": "Google Gemma 3.", "sizes": ["1b", "4b", "12b", "27b"], "capabilities": ["vision"]},
    {"name": "mistral", "description": "Mistral 7B.", "sizes": ["7b"], "capabilities": ["tools"]},
    {"name": "phi4", "description": "Microsoft Phi-4.", "sizes": ["14b"], "capabilities": []},
    {"name": "nomic-embed-text", "description": "Embedding model — improves document search. Small (274 MB).",
     "sizes": [], "capabilities": ["embedding"]},
]

_cache: dict[str, tuple[float, Any]] = {}


class CatalogError(RuntimeError):
    pass


def _fetch(url: str, timeout: float = 20) -> str:
    cached = _cache.get(url)
    if cached and time.monotonic() - cached[0] < CACHE_SECONDS:
        return cached[1]
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        raise CatalogError(f"could not reach {urllib.parse.urlparse(url).netloc} ({reason})") from exc
    _cache[url] = (time.monotonic(), text)
    return text


def parse_size(text: str) -> int:
    """'5.2GB' -> bytes. Ollama's site uses decimal units."""
    match = re.match(r"\s*([\d.]+)\s*([KMGT]?)B", text, re.IGNORECASE)
    if not match:
        return 0
    scale = {"": 1, "K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12}[match.group(2).upper()]
    return int(float(match.group(1)) * scale)


def _text(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


# ------------------------------------------------------------- ollama.com


def parse_search(page: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for block in re.split(r"<li\b", page)[1:]:
        link = re.search(r'href="/library/([\w.\-]+)"', block)
        if not link:
            continue
        name = link.group(1)
        description = re.search(r"<p[^>]*break-words[^>]*>(.*?)</p>", block, re.DOTALL)
        badges = re.findall(r'<span[^>]*class="[^"]*inline-flex[^"]*"[^>]*>([^<]+)</span>', block)
        sizes = [b.strip() for b in badges if re.fullmatch(r"(e?[\d.]+[bmk](-[\w]+)?|[\dx.]+b)", b.strip(), re.I)]
        capabilities = [b.strip() for b in badges if b.strip() not in sizes]
        # "cloud" models run on Ollama's servers, not this computer: a question
        # about a document would send the document there. They are listed so
        # nobody wonders where they went, and refused at download time.
        cloud = "cloud" in capabilities and not sizes
        pulls = re.search(r"<span[^>]*>([\d.,]+[KMB]?)</span>\s*<span[^>]*>&nbsp;Pulls", block)
        updated = re.search(r"Updated&nbsp;</span>\s*<span[^>]*>([^<]+)</span>", block)
        results.append({
            "name": name,
            "description": _text(description.group(1)) if description else "",
            "sizes": sizes,
            "capabilities": capabilities,
            "pulls": pulls.group(1) if pulls else "",
            "updated": updated.group(1).strip() if updated else "",
            "cloud": cloud,
        })
    return results


def search_ollama(query: str) -> dict[str, Any]:
    url = "https://ollama.com/search?" + urllib.parse.urlencode({"q": query})
    try:
        results = parse_search(_fetch(url))
    except CatalogError as exc:
        return {"results": _fallback(query), "warning": f"{exc}. Showing a built-in list instead."}
    if not results and query.strip():
        return {"results": [], "warning": ""}
    if not results:
        return {"results": _fallback(query),
                "warning": "ollama.com's page could not be read. Showing a built-in list instead."}
    return {"results": results, "warning": ""}


def _fallback(query: str) -> list[dict[str, Any]]:
    q = query.lower().strip()
    return [m for m in FALLBACK if not q or q in m["name"] or q in m["description"].lower()]


QUANT_SUFFIX = re.compile(r"-(q\d|fp16|bf16|fp32|f16|iq\d)", re.IGNORECASE)


def parse_tags(page: str, name: str) -> list[dict[str, Any]]:
    tags: list[dict[str, Any]] = []
    seen: set[str] = set()
    for block in re.split(r'<div class="group px-4 py-3">', page)[1:]:
        tag = re.search(r'<span class="group-hover:underline">\s*([^<\s]+)\s*</span>', block)
        detail = re.search(
            r'font-mono">\s*([0-9a-f]{6,})\s*</span>\s*•\s*([\d.]+\s*[KMGT]?B)\s*•\s*([^•<]+?)\s*context window',
            block,
        )
        if not tag or not detail:
            continue
        full = tag.group(1)
        if full in seen or not full.startswith(name + ":") or full.endswith("cloud"):
            continue
        seen.add(full)
        label = full.split(":", 1)[1]
        inputs = re.search(r"context window\s*•\s*<span[^>]*>\s*([^•<]+?)\s*•", block)
        tags.append({
            "name": full,
            "tag": label,
            "digest": detail.group(1),
            "size": parse_size(detail.group(2)),
            "context": detail.group(3).strip(),
            "input": inputs.group(1).strip() if inputs else "",
            "latest": "latest</span>" in block,
            # Tags like 7b-instruct-q8_0 are other quantizations of the same
            # model. They are real choices, but a wall of forty of them hides
            # the half-dozen sizes a person is actually choosing between.
            "variant": bool(QUANT_SUFFIX.search(label)) or label.count("-") >= 2,
        })
    return tags


def ollama_tags(name: str) -> dict[str, Any]:
    if not re.fullmatch(r"[\w.\-/]+", name):
        raise CatalogError("that is not a model name")
    url = f"https://ollama.com/library/{urllib.parse.quote(name)}/tags"
    tags = parse_tags(_fetch(url), name)
    if not tags:
        raise CatalogError(
            f"{name} has no version that runs on this computer — it is either cloud-only "
            "(runs on Ollama's servers, so documents would leave this machine) or its page could not be read"
        )
    return {"name": name, "tags": tags}


# ----------------------------------------------------------- Hugging Face

QUANT = re.compile(r"(IQ\d_[A-Z]+|Q\d_K(?:_[SML])?|Q\d_\d|Q\d|F16|BF16|F32)", re.IGNORECASE)
SHARD = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)


def search_hf(query: str) -> dict[str, Any]:
    params = urllib.parse.urlencode({
        "search": query, "filter": "gguf", "sort": "downloads", "direction": "-1", "limit": "25",
    })
    try:
        data = json.loads(_fetch("https://huggingface.co/api/models?" + params))
    except CatalogError as exc:
        return {"results": [], "warning": str(exc)}
    results = [
        {
            "name": item.get("id", ""),
            "downloads": int(item.get("downloads") or 0),
            "likes": int(item.get("likes") or 0),
            "description": ", ".join(t for t in item.get("tags", []) if not t.startswith(("arxiv:", "base_model:", "region:", "license:")))[:160],
        }
        for item in data if item.get("id")
    ]
    return {"results": results, "warning": ""}


def parse_hf_tree(repo: str, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in files:
        path = str(item.get("path") or "")
        if item.get("type") != "file" or not path.lower().endswith(".gguf"):
            continue
        filename = path.rsplit("/", 1)[-1]
        if SHARD.search(filename) or "mmproj" in filename.lower():
            continue
        quant = QUANT.search(filename)
        if not quant:
            continue
        size = int((item.get("lfs") or {}).get("size") or item.get("size") or 0)
        label = quant.group(1).upper()
        out.append({"name": f"hf.co/{repo}:{label}", "tag": label, "file": filename, "size": size,
                    "context": "", "variant": False, "latest": False})
    out.sort(key=lambda t: t["size"])
    return out


def hf_files(repo: str) -> dict[str, Any]:
    if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", repo):
        raise CatalogError("a Hugging Face repository looks like owner/name")
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    try:
        files = json.loads(_fetch(url))
    except ValueError as exc:
        raise CatalogError("Hugging Face returned something unreadable") from exc
    tags = parse_hf_tree(repo, files if isinstance(files, list) else [])
    if not tags:
        raise CatalogError("this repository has no single-file GGUF that Ollama can pull")
    return {"name": repo, "tags": tags}


# ------------------------------------------------------------ recommended

# The families worth offering someone who just wants to ask questions of
# documents, with approximate download sizes from ollama.com (Q4 builds) so a
# recommendation can still be made when the site cannot be read. Live sizes
# replace these whenever they can be fetched.
RECOMMENDED = [
    ("qwen2.5", "Best all-rounder for questions about documents; follows the citation format well.",
     {"0.5b": 0.40, "1.5b": 0.99, "3b": 1.9, "7b": 4.7, "14b": 9.0, "32b": 20, "72b": 47}),
    ("deepseek-r1", "Reasons step by step before answering. Slower, good on tricky questions.",
     {"1.5b": 1.1, "7b": 4.7, "8b": 5.2, "14b": 9.0, "32b": 20, "70b": 43}),
    ("qwen3", "Newer Qwen that can think before answering.",
     {"0.6b": 0.52, "1.7b": 1.4, "4b": 2.5, "8b": 5.2, "14b": 9.3, "32b": 20}),
    ("llama3.1", "Meta's Llama, a solid second opinion.", {"8b": 4.9, "70b": 43}),
    ("gemma3", "Google's Gemma.", {"1b": 0.82, "4b": 3.3, "12b": 8.1, "27b": 17}),
]

EMBEDDING = ("nomic-embed-text", "Not a chat model: makes document search understand meaning, "
             "so a question about 'back injury' finds 'lumbar strain'. Small.", 274_000_000)

SIZE_TAG = re.compile(r"^[\d.]+b$")


def family_sizes(name: str, fallback: dict[str, float]) -> dict[str, tuple[int, str]]:
    """Size in bytes and digest of each plain parameter-size tag (7b, 14b…)."""
    try:
        tags = ollama_tags(name)["tags"]
        live = {t["tag"]: (t["size"], t["digest"]) for t in tags if SIZE_TAG.match(t["tag"]) and t["size"]}
        if live:
            return live
    except CatalogError:
        pass
    return {tag: (int(gb * 1e9), "") for tag, gb in fallback.items()}
