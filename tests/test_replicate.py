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


@pytest.fixture
def small_fit(small_reps, monkeypatch):
    monkeypatch.setattr(config, "ESTIMATOR_FIXED", {"rf": {"n_estimators": 20}, "gbm": {"early_stopping": False}})
    monkeypatch.setitem(config.LC_PARAMS, "gbm", {**config.LC_PARAMS["gbm"], "max_iter": 20})
    # r0 roles: the existing main-pipeline sets, synthetic here
    shrink_roles(monkeypatch, {"lhs-extension": 20, "guided-log": 20, "guided-seq-1": 10, "guided-seq-2": 10,
                               "guided-seq-3": 10})
    monkeypatch.setattr(config, "REP_SEQ_BATCHES", (10, 10, 10))
    for i, role in enumerate(("lhs-extension", "guided-log", "guided-seq-1", "guided-seq-2", "guided-seq-3")):
        write_final(role, synthetic_final(config.ROLES[role].n, 80 + i))
    replicate.generate_all(workers=1, fn=fake_analyse)


def test_arm_frames_compose_seed_and_added_points(small_fit):
    r0, r1 = replicate.arm_frames(0), replicate.arm_frames(1)
    assert set(r0) == set(replicate.REP_ARMS) and set(r1) == set(replicate.REP_ARMS) - {"A"}
    assert len(r1["B"]) == 300 + config.REP_EXTENSION and len(r1["C0-seq"]) == sum(config.REP_SEQ_BATCHES)
    assert r1["C-log"]["row_id"].str.startswith("rep:guided_log_r1").sum() == config.REP_EXTENSION
    assert len(r0["C-seq"]) == 300 + 30 and len(r0["A"]) == 300


def test_fit_all_is_cached_and_refuses_stale_state(small_fit, monkeypatch):
    replicate.fit_all()
    units = replicate.load_units()
    assert set(units) == {(f, r) for f in config.REP_FAMILIES for r in range(0, config.REP_REALIZATIONS + 1)}
    stored = units[("gbm", 1)]
    pred = stored["arms"]["C-seq"]["predictions"]["test-natural"]
    assert len(pred) == 60 and (pred > 0).all()
    replicate.fit_all()   # up to date: no refit
    monkeypatch.setattr(config, "REP_EXTENSION", 21)
    with pytest.raises(replicate.models.StaleFit):
        replicate.load_units()
