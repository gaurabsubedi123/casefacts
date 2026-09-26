import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test gets its own empty data folder, never the real one."""
    monkeypatch.setenv("MODELPORTAL_HOME", str(tmp_path / "home"))
    return tmp_path / "home"
