import json

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
    monkeypatch.setitem(config.SCREENERS["rf"], "n_estimators", 20)
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


def test_the_gbm_strategy_shares_the_c_log_stream():
    for r in range(1, config.REP_REALIZATIONS + 1):
        assert replicate.rep_seed("guided-log-g", r) == replicate.rep_seed("guided-log", r)


def test_existing_replication_seeds_are_untouched():
    # Appending a strategy must not move the existing indices.
    assert config.REP_STRATEGIES[:3] == ("lhs", "guided-log", "guided-seq")
    assert replicate.rep_seed("lhs", 3) != replicate.rep_seed("guided-log", 3)


def test_the_two_screeners_see_the_same_pool():
    for r in (1, 3):
        pools = [design.uniform_pool(np.random.default_rng(replicate.rep_seed(s, r)),
                                     config.N_GUIDED_POOL)
                 for s in ("guided-log", "guided-log-g")]
        pd.testing.assert_frame_equal(pools[0], pools[1])   # row for row


def test_set_keys_include_the_gbm_strategy():
    keys = replicate.set_keys(2)
    assert keys[2] == ("guided_log_g_r2", "guided-log-g", 0)
    assert keys[1] == ("guided_log_r2", "guided-log", 0)


def test_strategy_family():
    assert replicate.strategy_family("guided-log") == "rf"
    assert replicate.strategy_family("guided-log-g") == "gbm"
    assert replicate.strategy_family("guided-seq") == "rf"


def test_set_keys_order(small_reps):
    assert replicate.set_keys(1) == [("lhs_r1", "lhs", 0), ("guided_log_r1", "guided-log", 0),
                                     ("guided_log_g_r1", "guided-log-g", 0),
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


def test_generate_set_records_non_converged_rows(small_reps):
    def last_stops_early(combo, _seen=[]):
        row = fake_analyse(combo)
        _seen.append(combo)
        if len(_seen) == config.REP_EXTENSION:
            row["Search_Converged"] = False
        return row

    entry = replicate.generate_set("lhs_r1", "lhs", 1, 0, workers=1, fn=last_stops_early)
    assert entry["non_converged"] == [config.REP_EXTENSION - 1]


# --- resuming an interrupted generation ------------------------------------------------------

def _generate_one(key="lhs_r1", strategy="lhs", batch=0, realization=1, **kw):
    return replicate.generate_set(key, strategy, realization, batch, workers=1, fn=fake_analyse, **kw)


def _count_solves(monkeypatch):
    """Count solver calls while still solving."""
    calls = []
    original = replicate.solver.run_set

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(replicate.solver, "run_set", counted)
    return calls


def _write_metadata(meta):
    config.metadata_path().write_text(json.dumps(meta))


def test_generate_set_resolves_when_the_metadata_entry_is_missing(small_reps, monkeypatch, capsys):
    entry = _generate_one()
    meta = data.read_metadata()
    del meta["replications"]["lhs_r1"]
    _write_metadata(meta)
    calls = _count_solves(monkeypatch)
    again = _generate_one()
    assert calls == [1]
    assert again["final_sha256"] == entry["final_sha256"]
    assert data.read_metadata()["replications"]["lhs_r1"]["final_sha256"] == entry["final_sha256"]
    assert "lhs_r1" in capsys.readouterr().out


def test_generate_set_resolves_after_each_partial_state(small_reps, monkeypatch, capsys):
    entry = _generate_one()
    paths = {stage: config.replication_path(stage, "lhs_r1")
             for stage in ("designs", "results", "final")}
    calls = _count_solves(monkeypatch)
    for missing, named in ((("results", "final"), "results"), (("final",), "final"),
                           ((), "results")):
        for stage in missing:
            paths[stage].unlink()
        if not missing:                      # stale results and final of an interrupted run
            paths["results"].write_text("angle\n1.0\n")
            paths["final"].write_text("angle\n1.0\n")
        calls.clear()
        capsys.readouterr()
        again = _generate_one()
        assert calls == [1], f"not re-solved after {missing or 'a stale results/final pair'}"
        assert again["final_sha256"] == entry["final_sha256"]
        assert len(replicate.load_set("lhs_r1")) == config.REP_EXTENSION
        reason = capsys.readouterr().out.split("solving it again")[0]
        assert named in reason and paths[named].name not in reason


def test_generate_set_overwrites_a_torn_design_when_nothing_was_solved(small_reps, monkeypatch):
    path = config.replication_path("designs", "lhs_r1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")                      # killed while the design was being written
    calls = _count_solves(monkeypatch)
    entry = _generate_one()
    assert calls == [1] and len(replicate.load_set("lhs_r1")) == config.REP_EXTENSION
    assert entry["design_sha256"] == data.sha256_file(path)
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_generate_set_resolves_a_torn_design_of_a_complete_set(small_reps, monkeypatch):
    entry = _generate_one()
    _generate_one("guided_log_r1", "guided-log", 0)
    untouched = data.read_metadata()["replications"]["guided_log_r1"]
    config.replication_path("designs", "lhs_r1").write_text("")
    calls = _count_solves(monkeypatch)
    again = _generate_one()
    assert calls == [1]
    assert again["design_sha256"] == entry["design_sha256"]
    assert again["final_sha256"] == entry["final_sha256"]
    assert data.read_metadata()["replications"]["guided_log_r1"] == untouched


def test_reuse_re_records_stale_screener_provenance(small_reps, monkeypatch):
    _generate_one("guided_seq_r1_b1", "guided-seq", 1)
    b2 = _generate_one("guided_seq_r1_b2", "guided-seq", 2)
    final = config.replication_path("final", "guided_seq_r1_b1")
    table = data.read_table(final)           # b1 re-solved: same FOS, another runtime
    table["Runtime_s"] = table["Runtime_s"].astype(float) + 1.0
    table.to_csv(final, index=False)
    meta = data.read_metadata()
    meta["replications"]["guided_seq_r1_b1"]["final_sha256"] = data.sha256_file(final)
    _write_metadata(meta)
    calls = _count_solves(monkeypatch)
    again = _generate_one("guided_seq_r1_b2", "guided-seq", 2)
    assert calls == []
    assert again["final_sha256"] == b2["final_sha256"]
    fresh = data.sha256_file(final)
    assert again["screener"]["trained_on_sha256"]["guided_seq_r1_b1"] == fresh
    recorded = data.read_metadata()["replications"]["guided_seq_r1_b2"]
    assert recorded["screener"]["trained_on_sha256"]["guided_seq_r1_b1"] == fresh
    assert recorded["final_sha256"] == b2["final_sha256"]


def test_build_design_reports_a_missing_earlier_batch_design(small_reps):
    with pytest.raises(FileNotFoundError, match="sgs replicate"):
        replicate.build_design("guided_seq_r1_b2", "guided-seq", 1, 2)


def test_generate_set_resolves_when_the_final_violates_the_contract(small_reps, monkeypatch):
    entry = _generate_one()
    final = config.replication_path("final", "lhs_r1")
    data.read_table(final).iloc[:-1].to_csv(final, index=False)
    meta = data.read_metadata()
    meta["replications"]["lhs_r1"]["final_sha256"] = data.sha256_file(final)
    _write_metadata(meta)
    calls = _count_solves(monkeypatch)
    again = _generate_one()
    assert calls == [1] and again["final_sha256"] == entry["final_sha256"]
    assert len(replicate.load_set("lhs_r1")) == config.REP_EXTENSION


def test_generate_set_refuses_other_solver_settings(small_reps, monkeypatch):
    entry = _generate_one()
    assert entry["solver_fingerprint"] == config.solver_fingerprint()
    monkeypatch.setattr(config, "REPAIR_VARIANTS", ((2.0, None),))
    with pytest.raises(replicate.ReplicationError, match="--force"):
        _generate_one()


def test_build_design_checks_the_size(small_reps, monkeypatch):
    monkeypatch.setattr(design, "guided_log_deficit",
                        lambda *a, **k: pd.DataFrame({p: [0.0] for p in config.PARAMS}))
    with pytest.raises(design.DesignError, match="expected"):
        replicate.build_design("guided_log_r1", "guided-log", 1, 0)


def test_screener_records_the_training_hashes(small_reps):
    b1 = _generate_one("guided_seq_r1_b1", "guided-seq", 1)
    b2 = _generate_one("guided_seq_r1_b2", "guided-seq", 2)
    seed_sha = data.sha256_file(config.final_path("seed"))
    assert b1["screener"]["trained_on_sha256"] == {"seed": seed_sha}
    assert b2["screener"]["trained_on_sha256"] == {
        "seed": seed_sha,
        "guided_seq_r1_b1": data.sha256_file(config.replication_path("final", "guided_seq_r1_b1"))}


@pytest.fixture
def small_fit(small_reps, monkeypatch):
    monkeypatch.setattr(config, "ESTIMATOR_FIXED", {"rf": {"n_estimators": 20}, "gbm": {"early_stopping": False}})
    monkeypatch.setitem(config.LC_PARAMS, "gbm", {**config.LC_PARAMS["gbm"], "max_iter": 20})
    # r0 roles: the existing main-pipeline sets, synthetic here
    shrink_roles(monkeypatch, {"lhs-extension": 20, "guided-log": 20, "guided-log-g": 20, "guided-seq-1": 10,
                               "guided-seq-2": 10, "guided-seq-3": 10})
    monkeypatch.setattr(config, "REP_SEQ_BATCHES", (10, 10, 10))
    for i, role in enumerate(("lhs-extension", "guided-log", "guided-log-g", "guided-seq-1", "guided-seq-2",
                              "guided-seq-3")):
        write_final(role, synthetic_final(config.ROLES[role].n, 80 + i))
    replicate.generate_all(workers=1, fn=fake_analyse)


def test_arm_frames_compose_seed_and_added_points(small_fit):
    r0, r1 = replicate.arm_frames(0), replicate.arm_frames(1)
    assert set(r0) == set(replicate.REP_ARMS) and set(r1) == set(replicate.REP_ARMS) - {"A"}
    assert len(r1["B"]) == 300 + config.REP_EXTENSION and len(r1["C0-seq"]) == sum(config.REP_SEQ_BATCHES)
    assert r1["C-log"]["row_id"].str.startswith("rep:guided_log_r1").sum() == config.REP_EXTENSION
    assert len(r0["C-seq"]) == 300 + 30 and len(r0["A"]) == 300


def test_arm_frames_carry_the_gbm_arms(monkeypatch, root):
    # Eight arms at r0 (A included), seven at r1..r5.
    monkeypatch.setattr(replicate, "load_set", lambda key: synthetic_final(10, hash(key) % 97))
    monkeypatch.setattr(data, "load_role", lambda role: synthetic_final(10, 1))
    assert set(replicate.arm_frames(1)) == {"B", "C-log", "C0-log", "C-seq", "C0-seq",
                                            "C-logG", "C0-logG"}
    assert set(replicate.arm_frames(0)) == {"A", "B", "C-log", "C0-log", "C-seq", "C0-seq",
                                            "C-logG", "C0-logG"}


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
