import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

from sgs import config

REAL_DATA = Path(config.ROOT) / "data"        # captured at import, before any monkeypatch


def _real_data_snapshot() -> dict:
    if not REAL_DATA.exists():
        return {}
    return {str(p.relative_to(REAL_DATA)): (p.stat().st_mode & 0o777, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(REAL_DATA.rglob("*")) if p.is_file() and p.name != ".DS_Store"}


@pytest.fixture(scope="session", autouse=True)
def real_data_untouched():
    """xslope results are not in git and cost hours to reproduce: the suite must leave
    every file under the real data/ exactly as it found it, contents and permissions."""
    before = _real_data_snapshot()
    yield
    after = _real_data_snapshot()
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    assert not changed, f"the test suite changed the real data/: {changed[:10]}"


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Redirect every pipeline path into a temporary directory."""
    monkeypatch.setattr(config, "ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def tiny_protocol(root, monkeypatch):
    """Small grids and three small training sets, for fast fitting tests."""
    from tests.helpers import shrink_roles, synthetic_final, write_final

    monkeypatch.setattr(config, "GRIDS", {"rf": {"min_samples_leaf": [1, 5]},
                                          "gbm": {"max_iter": [20], "min_samples_leaf": [3, 10]}})
    monkeypatch.setattr(config, "ESTIMATOR_FIXED", {"rf": {"n_estimators": 20},
                                                    "gbm": {"early_stopping": False}})
    monkeypatch.setattr(config, "CV_FOLDS", 3)
    shrink_roles(monkeypatch, {"seed": 60, "lhs-extension": 60, "guided": 60})
    for i, role in enumerate(("seed", "lhs-extension", "guided")):
        write_final(role, synthetic_final(60, 10 + i))


@pytest.fixture
def pool_roles(root, monkeypatch):
    """Two sub-pools, five bins, a small cap and chunk, and synthetic seed and test sets."""
    import pandas as pd

    from tests.helpers import shrink_roles, synthetic_final, write_final

    monkeypatch.setattr(config, "REP_REALIZATIONS", 2)
    monkeypatch.setattr(config, "N_BINS", 5)
    monkeypatch.setattr(config, "POOL_CAP", 400)
    monkeypatch.setattr(config, "POOL_CHUNK", 40)
    shrink_roles(monkeypatch, {"seed": 300, "test-natural": 60, "test-stratified": 60})
    for i, role in enumerate(("seed", *config.TEST_ROLES)):
        frame = synthetic_final(300 if role == "seed" else 60, 50 + i)
        write_final(role, frame)
        path = config.design_path(role)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame[list(config.PARAMS)].to_csv(path, index=False)
    return root
