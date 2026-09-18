import pytest

from sgs import config


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Redirect every pipeline path into a temporary directory."""
    monkeypatch.setattr(config, "ROOT", tmp_path)
    return tmp_path
