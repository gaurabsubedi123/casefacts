from modelportal import hardware
from modelportal.hardware import GB, GPU, Machine


def machine(vram_gb=0.0, ram_gb=16.0, disk_gb=500.0, vendor="nvidia", unified=False):
    gpus = [GPU("Test GPU", vendor, int(vram_gb * GB), unified=unified)] if vram_gb else []
    return Machine("Test", "cpu", 8, int(ram_gb * GB), None, int(disk_gb * GB), "/models", gpus)


def test_parse_nvidia_smi():
    gpus = hardware.parse_nvidia_smi("NVIDIA GeForce RTX 4070 Laptop GPU, 8188, 7948\n")
    assert gpus[0].name.startswith("NVIDIA GeForce")
    assert gpus[0].vram_total == 8188 * 1024**2
    assert gpus[0].vram_free == 7948 * 1024**2


def test_parse_nvidia_smi_ignores_junk():
    assert hardware.parse_nvidia_smi("No devices were found\n") == []


def test_verdicts_on_an_8gb_gpu():
    m = machine(vram_gb=8)
    assert hardware.assess(int(4.7e9), m)["level"] == "gpu"
    assert hardware.assess(int(9e9), m)["level"] == "partial"
    assert hardware.assess(int(43e9), m)["level"] == "no"


def test_no_gpu_means_cpu_verdicts():
    m = machine(vram_gb=0, ram_gb=16)
    assert hardware.assess(int(2e9), m)["level"] == "cpu"
    assert hardware.assess(int(40e9), m)["level"] == "no"


def test_intel_integrated_graphics_is_not_counted():
    m = machine(vram_gb=2, vendor="intel")
    assert m.gpu_memory == 0
    assert hardware.assess(int(2e9), m)["level"] == "cpu"


def test_apple_unified_memory_is_not_double_counted():
    m = machine(vram_gb=10.7, ram_gb=16, vendor="apple", unified=True)
    assert hardware.assess(int(20e9), m)["level"] == "no"


def test_disk_warning():
    verdict = hardware.assess(int(20e9), machine(vram_gb=24, disk_gb=10))
    assert "disk" in verdict["disk_warning"]


QWEN3_8B = {
    "qwen3.block_count": 36, "qwen3.attention.head_count": 32, "qwen3.attention.head_count_kv": 8,
    "qwen3.attention.key_length": 128, "qwen3.attention.value_length": 128, "qwen3.context_length": 40960,
}


def test_kv_bytes_per_token():
    assert hardware.kv_bytes_per_token(QWEN3_8B) == 36 * 8 * 256 * 2


def test_context_shrinks_with_vram_and_respects_limits():
    size = int(5.2e9)
    big = hardware.context_for(size, QWEN3_8B, machine(vram_gb=24))
    small = hardware.context_for(size, QWEN3_8B, machine(vram_gb=8))
    assert big["context"] == hardware.MAX_CONTEXT
    assert hardware.MIN_CONTEXT <= small["context"] < big["context"]
    assert small["context"] % 1024 == 0
