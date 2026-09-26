from pathlib import Path

from modelportal import catalog

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_search_finds_models_sizes_and_cloud():
    results = catalog.parse_search((FIXTURES / "ollama_search_deepseek.html").read_text())
    by_name = {r["name"]: r for r in results}
    r1 = by_name["deepseek-r1"]
    assert "7b" in r1["sizes"] and "671b" in r1["sizes"]
    assert "thinking" in r1["capabilities"]
    assert not r1["cloud"]
    assert any(r["cloud"] for r in results)


def test_parse_tags_sizes_and_variants():
    tags = catalog.parse_tags((FIXTURES / "ollama_tags_deepseek-r1.html").read_text(), "deepseek-r1")
    by_tag = {t["tag"]: t for t in tags}
    assert by_tag["8b"]["size"] == 5_200_000_000
    assert not by_tag["8b"]["variant"]
    assert by_tag["7b-qwen-distill-q8_0"]["variant"]
    assert all(t["name"].startswith("deepseek-r1:") for t in tags)


def test_parse_size():
    assert catalog.parse_size("5.2GB") == 5_200_000_000
    assert catalog.parse_size("274MB") == 274_000_000


def test_hf_tree_keeps_single_file_ggufs_only():
    files = [
        {"type": "file", "path": "qwen2.5-7b-instruct-q4_k_m.gguf", "lfs": {"size": 4_683_073_536}},
        {"type": "file", "path": "qwen2.5-7b-instruct-q8_0-00001-of-00002.gguf", "size": 1},
        {"type": "file", "path": "mmproj-f16.gguf", "size": 1},
        {"type": "file", "path": "README.md", "size": 1},
    ]
    tags = catalog.parse_hf_tree("Qwen/Qwen2.5-7B-Instruct-GGUF", files)
    assert [t["name"] for t in tags] == ["hf.co/Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M"]
    assert tags[0]["size"] == 4_683_073_536
