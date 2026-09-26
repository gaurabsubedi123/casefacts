"""What this computer can run, worked out on whatever computer this is.

The portal is meant to be copied to machines nobody has looked at, so nothing
here assumes a GPU. Each source is tried and silently skipped when it does not
apply:

    NVIDIA         nvidia-smi — ships with the driver on Windows and Linux
    AMD (Linux)    /sys/class/drm/card*/device/mem_info_vram_total
    Windows, any   the display-adapter registry keys, which hold the true
                   64-bit VRAM size (WMI's AdapterRAM is capped at 4 GB)
    Apple Silicon  unified memory: the GPU may use about two thirds of RAM

Intel integrated graphics are reported but not counted, because Ollama does
not run models on them — pretending otherwise would promise speed that never
arrives.

Then `assess` turns a model's download size into one of four plain verdicts,
and `context_for` picks the largest context window that still keeps the model
entirely in GPU memory. That second one matters more than it looks: ask for a
context bigger than the VRAM left over and Ollama quietly spills part of the
model into system RAM, where the same question takes minutes instead of
seconds. Nothing reports that but the clock.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .paths import ollama_models_dir

GB = 1024**3

# VRAM the driver, the desktop and the browser showing this page keep for
# themselves. Counting it as available is how a model that "fits" ends up
# half on the CPU.
VRAM_RESERVE = int(0.6 * GB)

# Ollama's own working buffers beyond the weights and the KV cache.
RUNTIME_OVERHEAD = int(0.5 * GB)

# When a model's architecture is unknown (before it is downloaded), the KV
# cache is estimated per token from typical 7-8B models at f16.
DEFAULT_KV_PER_TOKEN = 110 * 1024
ASSUMED_CONTEXT = 8192

MIN_CONTEXT = 2048
MAX_CONTEXT = 32768

USABLE_VENDORS = {"nvidia", "amd", "apple"}


@dataclass
class GPU:
    name: str
    vendor: str
    vram_total: int
    vram_free: int | None = None
    unified: bool = False

    @property
    def usable(self) -> bool:
        return self.vendor in USABLE_VENDORS and self.vram_total >= 2 * GB


@dataclass
class Machine:
    os: str
    cpu: str
    cpu_cores: int
    ram_total: int
    ram_free: int | None
    disk_free: int | None
    models_dir: str
    gpus: list[GPU] = field(default_factory=list)

    @property
    def usable_gpus(self) -> list[GPU]:
        return [g for g in self.gpus if g.usable]

    @property
    def gpu_memory(self) -> int:
        """Memory a model can occupy on the GPU(s), after the reserve.

        Ollama splits a model across several GPUs of one vendor, so their
        memory adds up.
        """
        usable = self.usable_gpus
        if not usable:
            return 0
        if any(g.unified for g in usable):
            return max(g.vram_total for g in usable)
        return max(0, sum(g.vram_total for g in usable) - VRAM_RESERVE * len(usable))

    def summary(self) -> str:
        usable = self.usable_gpus
        if usable:
            names = ", ".join(f"{g.name} ({g.vram_total / GB:.0f} GB)" for g in usable)
            return f"{names} · {self.ram_total / GB:.0f} GB RAM"
        return f"No usable GPU — models run on the CPU · {self.ram_total / GB:.0f} GB RAM"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["gpus"] = [dict(asdict(g), usable=g.usable) for g in self.gpus]
        data["gpu_memory"] = self.gpu_memory
        data["summary"] = self.summary()
        return data


# --------------------------------------------------------------- running


def _run(command: list[str], timeout: float = 6) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if platform.system() == "Windows" else 0,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


# ------------------------------------------------------------------ GPUs


def _nvidia_smi() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for candidate in (
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        "/usr/lib/wsl/lib/nvidia-smi",
        "/usr/bin/nvidia-smi",
    ):
        if Path(candidate).is_file():
            return candidate
    return None


def parse_nvidia_smi(output: str) -> list[GPU]:
    """Rows of `name, memory.total, memory.free` in MiB, no header, no units."""
    gpus: list[GPU] = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            total = int(float(parts[1])) * 1024**2
            free = int(float(parts[2])) * 1024**2
        except ValueError:
            continue
        gpus.append(GPU(name=parts[0], vendor="nvidia", vram_total=total, vram_free=free))
    return gpus


def nvidia_gpus() -> list[GPU]:
    binary = _nvidia_smi()
    if not binary:
        return []
    output = _run([binary, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"])
    return parse_nvidia_smi(output)


def linux_amd_gpus() -> list[GPU]:
    gpus: list[GPU] = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        device = card / "device"
        total_file = device / "mem_info_vram_total"
        if not total_file.is_file():
            continue
        try:
            total = int(total_file.read_text().strip())
            used_file = device / "mem_info_vram_used"
            used = int(used_file.read_text().strip()) if used_file.is_file() else None
        except (OSError, ValueError):
            continue
        name_file = device / "product_name"
        name = name_file.read_text().strip() if name_file.is_file() else "AMD Radeon GPU"
        gpus.append(GPU(name=name or "AMD Radeon GPU", vendor="amd", vram_total=total,
                        vram_free=total - used if used is not None else None))
    return gpus


def vendor_of(name: str) -> str:
    lowered = name.lower()
    if "nvidia" in lowered or "geforce" in lowered or "quadro" in lowered or "rtx" in lowered:
        return "nvidia"
    if "amd" in lowered or "radeon" in lowered:
        return "amd"
    if "intel" in lowered:
        return "intel"
    return "other"


def windows_registry_gpus() -> list[GPU]:
    """Every display adapter Windows knows about, with its real VRAM size."""
    try:
        import winreg  # type: ignore[import-not-found]
    except ImportError:
        return []
    root = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    gpus: list[GPU] = []
    try:
        base = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, root)
    except OSError:
        return []
    index = 0
    while True:
        try:
            sub = winreg.EnumKey(base, index)
        except OSError:
            break
        index += 1
        if not sub.isdigit():
            continue
        try:
            key = winreg.OpenKey(base, sub)
        except OSError:
            continue
        try:
            name = str(winreg.QueryValueEx(key, "DriverDesc")[0])
        except OSError:
            continue
        size = 0
        for value_name in ("HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
            try:
                value = winreg.QueryValueEx(key, value_name)[0]
            except OSError:
                continue
            if isinstance(value, bytes):
                value = int.from_bytes(value[:8], "little")
            try:
                size = int(value)
            except (TypeError, ValueError):
                continue
            if size:
                break
        if size and name not in {g.name for g in gpus}:
            gpus.append(GPU(name=name, vendor=vendor_of(name), vram_total=size))
    return gpus


def apple_gpu(ram_total: int) -> list[GPU]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []
    # macOS lets Metal use about two thirds of memory on smaller machines and
    # three quarters on larger ones; the rest stays with the system.
    share = 0.75 if ram_total > 36 * GB else 0.67
    return [GPU(name="Apple Silicon GPU", vendor="apple", vram_total=int(ram_total * share), unified=True)]


def detect_gpus(ram_total: int) -> list[GPU]:
    gpus = nvidia_gpus()
    system = platform.system()
    if system == "Windows":
        # The registry sees every adapter; nvidia-smi's numbers win where both
        # report the same card, because they include what is free right now.
        for gpu in windows_registry_gpus():
            if gpu.vendor == "nvidia" and any(g.vendor == "nvidia" for g in gpus):
                continue
            gpus.append(gpu)
    elif system == "Linux":
        gpus += linux_amd_gpus()
    elif system == "Darwin":
        gpus += apple_gpu(ram_total)
    return gpus


# ------------------------------------------------------------------- RAM


def memory() -> tuple[int, int | None]:
    """Total and currently available system RAM, in bytes."""
    system = platform.system()
    if system == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
        return 0, None
    if system == "Darwin":
        total = _run(["sysctl", "-n", "hw.memsize"]).strip()
        free: int | None = None
        stats = _run(["vm_stat"])
        page = re.search(r"page size of (\d+)", stats)
        if page:
            pages = sum(
                int(m) for m in re.findall(r"Pages (?:free|inactive|speculative):\s+(\d+)", stats)
            )
            free = pages * int(page.group(1))
        return (int(total) if total.isdigit() else 0), free
    try:
        info = Path("/proc/meminfo").read_text()
    except OSError:
        return 0, None
    values = {k: int(v) * 1024 for k, v in re.findall(r"^(\w+):\s+(\d+) kB", info, re.MULTILINE)}
    return values.get("MemTotal", 0), values.get("MemAvailable")


def cpu_name() -> str:
    system = platform.system()
    if system == "Linux":
        try:
            match = re.search(r"^model name\s*:\s*(.+)$", Path("/proc/cpuinfo").read_text(), re.MULTILINE)
            if match:
                return match.group(1).strip()
        except OSError:
            pass
    if system == "Darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        if name:
            return name
    return platform.processor() or platform.machine()


def disk_free(path: Path) -> int | None:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


# ------------------------------------------------------------- detection

_cache: tuple[float, Machine] | None = None


def detect(max_age: float = 20.0) -> Machine:
    """This computer, re-read at most every `max_age` seconds."""
    global _cache
    if _cache and time.monotonic() - _cache[0] < max_age:
        return _cache[1]
    ram_total, ram_free = memory()
    models = ollama_models_dir()
    machine = Machine(
        os=f"{platform.system()} {platform.release()}",
        cpu=cpu_name(),
        cpu_cores=os.cpu_count() or 1,
        ram_total=ram_total,
        ram_free=ram_free,
        disk_free=disk_free(models),
        models_dir=str(models),
        gpus=detect_gpus(ram_total),
    )
    _cache = (time.monotonic(), machine)
    return machine


# ------------------------------------------------------------- verdicts


def _gb(n: float) -> str:
    return f"{n / GB:.1f} GB"


def assess(size: int, machine: Machine, kv_per_token: int = DEFAULT_KV_PER_TOKEN,
           context: int = ASSUMED_CONTEXT) -> dict[str, Any]:
    """Will a model of this download size run here, and how well?

    Levels, best first:
        gpu      entirely in GPU memory — fast
        partial  split between GPU and system RAM — works, several times slower
        cpu      no usable GPU, fits in RAM — works, slow
        no       bigger than GPU and RAM together — will not load
    """
    need = size + RUNTIME_OVERHEAD + kv_per_token * context
    gpu = machine.gpu_memory
    ram = machine.ram_total
    unified = any(g.unified for g in machine.usable_gpus)
    # With unified memory the GPU share *is* RAM, so the two must not be added.
    ceiling = gpu if unified else gpu + int(ram * 0.8)
    result: dict[str, Any] = {"need": need, "size": size}
    disk = machine.disk_free
    if disk is not None and size > disk - 1 * GB:
        result["disk_warning"] = (
            f"Not enough disk space: the download is {_gb(size)} and only {_gb(disk)} is free "
            f"where Ollama stores models ({machine.models_dir})."
        )

    if gpu and need <= gpu:
        result.update(level="gpu", label="Fits on GPU",
                      message=f"Runs entirely on your GPU (needs about {_gb(need)} of {_gb(gpu)}). Fast.")
    elif need > (ceiling if gpu else int(ram * 0.8)):
        have = f"{_gb(gpu)} GPU + {_gb(ram)} RAM" if gpu and not unified else _gb(ceiling or ram)
        result.update(level="no", label="Too big",
                      message=f"Needs about {_gb(need)} of memory; this computer has {have}. "
                              "It would not load, or would be unusably slow.")
    elif gpu:
        share = max(0, min(99, int(100 * gpu / need)))
        result.update(level="partial", label="Slow — partly on CPU", gpu_share=share,
                      message=f"Too big for GPU memory ({_gb(need)} needed, {_gb(gpu)} available). "
                              f"About {share}% would run on the GPU and the rest on the CPU, so answers "
                              "will be several times slower. A smaller size of the same model is usually the better choice.")
    else:
        result.update(level="cpu", label="CPU only",
                      message=f"No usable GPU found, so it would run on the CPU from RAM "
                              f"(needs about {_gb(need)} of {_gb(ram)}). Expect slow answers.")
    return result


def kv_bytes_per_token(model_info: dict[str, Any]) -> int | None:
    """KV cache cost of one token of context, from /api/show's model_info.

    Each layer keeps a key and a value vector per KV head, at 2 bytes (f16).
    """
    def pick(suffix: str) -> int | None:
        for key, value in model_info.items():
            if key.endswith(suffix):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
        return None

    layers = pick(".block_count")
    heads = pick(".attention.head_count")
    kv_heads = pick(".attention.head_count_kv") or heads
    key_len = pick(".attention.key_length")
    value_len = pick(".attention.value_length")
    if not key_len and heads:
        width = pick(".embedding_length")
        key_len = width // heads if width else None
    value_len = value_len or key_len
    if not (layers and kv_heads and key_len and value_len):
        return None
    return layers * kv_heads * (key_len + value_len) * 2


def model_context_limit(model_info: dict[str, Any]) -> int | None:
    for key, value in model_info.items():
        if key.endswith(".context_length"):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def context_for(size: int, model_info: dict[str, Any], machine: Machine) -> dict[str, Any]:
    """The largest context window that keeps this model wholly on the GPU.

    Bounded below by MIN_CONTEXT (too little to hold a question and a few
    pages) and above by the model's own limit and MAX_CONTEXT (beyond which
    small models read worse, not better). With no GPU, RAM is the budget.
    """
    per_token = kv_bytes_per_token(model_info) or DEFAULT_KV_PER_TOKEN
    limit = min(model_context_limit(model_info) or MAX_CONTEXT, MAX_CONTEXT)
    budget = machine.gpu_memory or int(machine.ram_total * 0.5)
    spare = budget - size - RUNTIME_OVERHEAD
    fits = spare // per_token if spare > 0 else 0
    context = int(max(MIN_CONTEXT, min(limit, fits)) // 1024 * 1024)
    return {"context": max(MIN_CONTEXT, context), "kv_per_token": per_token, "limit": limit,
            "fits_on_gpu": bool(machine.gpu_memory) and fits >= MIN_CONTEXT}
