import pytest

from sgs import config


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
