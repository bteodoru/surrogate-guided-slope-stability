import numpy as np
import pandas as pd
import pytest

from sgs import config, data, design, replicate
from tests.helpers import fake_analyse, shrink_roles, synthetic_final, write_final


def _write_design(role, frame):
    path = config.design_path(role)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[list(config.PARAMS)].to_csv(path, index=False)


@pytest.fixture
def small_reps(root, monkeypatch):
    """Tiny replications on synthetic seed/test finals; the solver is fake_analyse."""
    monkeypatch.setattr(config, "REP_REALIZATIONS", 2)
    monkeypatch.setattr(config, "REP_EXTENSION", 20)
    monkeypatch.setattr(config, "REP_SEQ_BATCHES", (10, 10))
    monkeypatch.setattr(config, "N_GUIDED_POOL", 6000)
    monkeypatch.setattr(config, "N_BINS", 5)
    monkeypatch.setitem(config.SCREENER, "n_estimators", 20)
    monkeypatch.setitem(config.SCREENER, "cv_folds", 3)
    shrink_roles(monkeypatch, {"seed": 300, "test-natural": 60, "test-stratified": 60})
    for i, role in enumerate(("seed", *config.TEST_ROLES)):
        frame = synthetic_final(300 if role == "seed" else 60, 50 + i)
        write_final(role, frame)
        _write_design(role, frame)


def test_rep_seed_is_derived_and_distinct():
    a = replicate.rep_seed("lhs", 1)
    assert a == replicate.rep_seed("lhs", 1) and isinstance(a, int)
    assert len({a, replicate.rep_seed("lhs", 2), replicate.rep_seed("guided-log", 1),
                replicate.rep_seed("guided-seq", 1, 1), replicate.rep_seed("guided-seq", 1, 2)}) == 5


def test_set_keys_order(small_reps):
    assert replicate.set_keys(1) == [("lhs_r1", "lhs", 0), ("guided_log_r1", "guided-log", 0),
                                     ("guided_seq_r1_b1", "guided-seq", 1),
                                     ("guided_seq_r1_b2", "guided-seq", 2)]


def test_generate_all_builds_every_set_in_order(small_reps):
    replicate.generate_all(workers=1, fn=fake_analyse)
    meta = data.read_metadata()["replications"]
    for r in (1, 2):
        for key, strategy, batch in replicate.set_keys(r):
            frame = replicate.load_set(key)
            expected = (config.REP_EXTENSION if strategy != "guided-seq"
                        else config.REP_SEQ_BATCHES[batch - 1])
            assert len(frame) == expected and frame["row_id"].iloc[0] == f"rep:{key}:0"
            assert meta[key]["seed"] == replicate.rep_seed(strategy, r, batch)
    b2 = meta["guided_seq_r1_b2"]
    assert b2["screener"]["trained_on"] == ["seed", "guided_seq_r1_b1"]
    assert b2["screener"]["n_train"] == 300 + config.REP_SEQ_BATCHES[0]
    assert sum(b2["allocation"]["added"]) == config.REP_SEQ_BATCHES[1]
    tests = pd.concat([design.read_design(r) for r in ("seed", *config.TEST_ROLES)])
    for key in ("lhs_r1", "guided_log_r2", "guided_seq_r2_b2"):
        assert design.nearest_distance(replicate.load_set(key), tests).min() > config.MIN_DISTANCE


def test_generate_set_reuses_and_refuses(small_reps, monkeypatch):
    replicate.generate_all(workers=1, fn=fake_analyse)
    calls = []
    monkeypatch.setattr(replicate.solver, "run_set", lambda *a, **k: calls.append(1))
    replicate.generate_set("lhs_r1", "lhs", 1, 0, workers=1, fn=fake_analyse)
    assert calls == []
    monkeypatch.setitem(config.SEEDS, "replication_design", 1)
    with pytest.raises(replicate.ReplicationError, match="--force"):
        replicate.generate_set("lhs_r1", "lhs", 1, 0, workers=1, fn=fake_analyse)
