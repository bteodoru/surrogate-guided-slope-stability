import os

import numpy as np
import pandas as pd
import pytest

from sgs import config, data, design, pool
from tests.helpers import FormulaModel, fake_analyse, tamper, tamper_frame


def _raw():
    return design.uniform_pool(np.random.default_rng(config.SEEDS["pool_design"]), config.POOL_CAP)


def test_full_design_is_deterministic_and_assigns_round_robin(pool_roles):
    a, n_a = pool.full_design()
    b, n_b = pool.full_design()
    pd.testing.assert_frame_equal(a, b)
    assert n_a == n_b == 0
    assert list(a.columns) == [*config.PARAMS, "pool_row", "subpool"]
    assert a["pool_row"].tolist() == list(range(len(a)))
    assert a["subpool"].tolist() == [i % 2 + 1 for i in range(len(a))]
    pd.testing.assert_frame_equal(a[list(config.PARAMS)], _raw())


def test_full_design_filters_rows_near_the_seed_and_tests(pool_roles):
    raw = _raw()
    seed_design = pd.concat([design.read_design("seed"), raw.iloc[[3]]], ignore_index=True)
    seed_design.to_csv(config.design_path("seed"), index=False)
    frame, n_filtered = pool.full_design()
    assert n_filtered == 1 and len(frame) == len(raw) - 1
    pd.testing.assert_frame_equal(frame[list(config.PARAMS)],
                                  raw.drop(index=3).reset_index(drop=True))
    assert frame["pool_row"].tolist() == list(range(len(frame)))   # renumbered after the filter


def test_chunks_are_consecutive_and_bounded(pool_roles):
    frame, _ = pool.full_design()
    assert pool.n_chunks(frame) == 10
    c2 = pool.chunk_design(frame, 2)
    assert c2["pool_row"].tolist() == list(range(40, 80))
    assert (c2["subpool"].value_counts() == 20).all()
    for k in (0, 11):
        with pytest.raises(pool.PoolError):
            pool.chunk_design(frame, k)


def test_chunk_must_divide_into_the_subpools(pool_roles, monkeypatch):
    monkeypatch.setattr(config, "POOL_CHUNK", 41)
    with pytest.raises(pool.PoolError, match="multiple"):
        pool.full_design()


# --- solving one chunk, with resume ------------------------------------------------------------

def _broken(combo):
    raise KeyboardInterrupt


def test_solve_chunk_writes_three_files_and_records_them(pool_roles):
    frame, _ = pool.full_design()
    entry = pool.solve_chunk(frame, 1, workers=1, fn=fake_analyse)
    for stage in ("designs", "results", "final"):
        assert config.pool_path(stage, 1).exists()
    stored = data.read_metadata()["pool"]["c001"]
    assert stored == entry and stored["n"] == 40 and stored["chunk"] == 1
    assert stored["platform"] == {"system": pool.platform.system(), "machine": pool.platform.machine()}
    assert stored["solver_fingerprint"] == config.solver_fingerprint()
    assert list(pd.read_csv(config.pool_path("designs", 1)).columns) == [*config.PARAMS, "pool_row", "subpool"]


def test_solve_chunk_reuses_a_complete_chunk(pool_roles, monkeypatch):
    frame, _ = pool.full_design()
    pool.solve_chunk(frame, 1, workers=1, fn=fake_analyse)
    monkeypatch.setattr(pool.solver, "run_set", lambda *a, **k: pytest.fail("solved again"))
    pool.solve_chunk(frame, 1, workers=1, fn=fake_analyse)


def _no_solver(monkeypatch):
    monkeypatch.setattr(pool.solver, "run_set", lambda *a, **k: pytest.fail("the solver was called"))


def _quarantined(name):
    return sorted((config.ROOT / "data" / "quarantine").rglob(name))


def test_recorded_chunk_files_are_read_only(pool_roles):
    frame, _ = pool.full_design()
    pool.solve_chunk(frame, 1, workers=1, fn=fake_analyse)
    for path in pool.chunk_paths(1).values():
        assert not os.access(path, os.W_OK)
        with pytest.raises(PermissionError):
            path.write_text("overwritten by accident")


def test_a_torn_final_is_rebuilt_from_intact_results_without_the_solver(pool_roles, monkeypatch, capsys):
    frame, _ = pool.full_design()
    pool.solve_chunk(frame, 1, workers=1, fn=fake_analyse)
    results_before = config.pool_path("results", 1).read_bytes()
    tamper(config.pool_path("final", 1), "angle,phi\n1,")
    _no_solver(monkeypatch)
    pool.solve_chunk(frame, 1, workers=1, fn=fake_analyse)
    out = capsys.readouterr().out
    assert "the solved results are intact" in out and "solving it again" not in out
    assert config.pool_path("results", 1).read_bytes() == results_before
    assert data.stored_set_problem(pool.chunk_paths(1), data.read_metadata()["pool"]["c001"], 40) is None
    [kept] = _quarantined("pool_c001.csv")                       # the torn final, moved aside
    assert kept.read_text() == "angle,phi\n1,"


def test_a_missing_design_is_rewritten_without_the_solver(pool_roles, monkeypatch, capsys):
    # The recorded results hash already proves the results are the chunk's.
    frame, _ = pool.full_design()
    pool.solve_chunk(frame, 1, workers=1, fn=fake_analyse)
    results_before = config.pool_path("results", 1).read_bytes()
    config.pool_path("designs", 1).unlink()
    _no_solver(monkeypatch)
    pool.solve_chunk(frame, 1, workers=1, fn=fake_analyse)
    assert "the solved results are intact" in capsys.readouterr().out
    assert config.pool_path("results", 1).read_bytes() == results_before
    assert config.pool_path("designs", 1).read_text() == pool.chunk_design(frame, 1).to_csv(index=False)
    assert data.stored_set_problem(pool.chunk_paths(1), data.read_metadata()["pool"]["c001"], 40) is None
    assert not os.access(config.pool_path("designs", 1), os.W_OK)


def test_a_damaged_design_is_quarantined_and_rewritten_without_the_solver(pool_roles, monkeypatch):
    frame, _ = pool.full_design()
    pool.solve_chunk(frame, 1, workers=1, fn=fake_analyse)
    tamper(config.pool_path("designs", 1), "damaged design")
    _no_solver(monkeypatch)
    pool.solve_chunk(frame, 1, workers=1, fn=fake_analyse)
    assert config.pool_path("designs", 1).read_text() == pool.chunk_design(frame, 1).to_csv(index=False)
    assert any(p.read_text() == "damaged design" for p in _quarantined("pool_c001.csv"))   # moved, never deleted
    assert data.stored_set_problem(pool.chunk_paths(1), data.read_metadata()["pool"]["c001"], 40) is None


def test_results_without_metadata_are_salvaged_without_the_solver(pool_roles, monkeypatch, capsys):
    # Killed after the results were written and before the chunk was recorded:
    # 48 min of solving are on disk, complete (write_rows is atomic). Keep them.
    frame, _ = pool.full_design()
    pool.solve_chunk(frame, 1, workers=1, fn=fake_analyse)
    meta = data.read_metadata()
    del meta["pool"]["c001"]
    config.metadata_path().write_text(__import__("json").dumps(meta))
    _no_solver(monkeypatch)
    pool.solve_chunk(frame, 1, workers=1, fn=fake_analyse)
    assert "the solved results are intact" in capsys.readouterr().out
    assert "c001" in data.read_metadata()["pool"]


def test_damaged_results_are_quarantined_then_solved_again(pool_roles, capsys):
    frame, _ = pool.full_design()
    pool.solve_chunk(frame, 1, workers=1, fn=fake_analyse)
    tamper(config.pool_path("results", 1), "damaged")
    pool.solve_chunk(frame, 1, workers=1, fn=fake_analyse)
    assert "solving it again" in capsys.readouterr().out
    assert any(p.read_text() == "damaged" for p in _quarantined("pool_c001.csv"))   # moved, never deleted
    assert data.stored_set_problem(pool.chunk_paths(1), data.read_metadata()["pool"]["c001"], 40) is None


def test_results_that_do_not_match_the_design_are_never_salvaged(pool_roles, monkeypatch):
    frame, _ = pool.full_design()
    pool.solve_chunk(frame, 1, workers=1, fn=fake_analyse)
    rows, _ = pool.solver.read_rows(config.pool_path("results", 1))
    rows[0]["angle"] = str(float(rows[0]["angle"]) + 1.0)
    config.pool_path("results", 1).chmod(0o644)
    pool.solver.write_rows(config.pool_path("results", 1), rows, data.RESULT_COLUMNS)
    meta = data.read_metadata()
    del meta["pool"]["c001"]
    config.metadata_path().write_text(__import__("json").dumps(meta))
    calls = []
    real = pool.solver.run_set
    monkeypatch.setattr(pool.solver, "run_set", lambda *a, **k: calls.append(1) or real(*a, **k))
    pool.solve_chunk(frame, 1, workers=1, fn=fake_analyse)
    assert calls == [1]


def test_solve_chunk_refuses_a_changed_design(pool_roles, monkeypatch):
    frame, _ = pool.full_design()
    pool.solve_chunk(frame, 1, workers=1, fn=fake_analyse)
    monkeypatch.setitem(config.SEEDS, "pool_design", 1)
    changed, _ = pool.full_design()
    with pytest.raises(pool.PoolError, match="declared"):
        pool.solve_chunk(changed, 1, workers=1, fn=fake_analyse)


def test_an_interrupted_solve_records_nothing(pool_roles):
    frame, _ = pool.full_design()
    with pytest.raises(KeyboardInterrupt):
        pool.solve_chunk(frame, 1, workers=1, fn=_broken)
    assert "c001" not in data.read_metadata().get("pool", {})


# --- eligibility, the stopping rule, status and the run loop ------------------------------------

@pytest.fixture
def controlled(pool_roles, monkeypatch):
    """A fixed allocation, and the formula as the screener."""
    monkeypatch.setattr(pool, "allocation", lambda: np.array([0, 0, 4, 4, 0]))
    monkeypatch.setattr(pool, "seed_counts", lambda: np.array([9, 9, 5, 5, 9]))
    monkeypatch.setattr(pool, "screener", lambda: (FormulaModel(), {"n_train": 300}))
    return pool_roles


def _table(solved, predicted, a):
    return {"allocation": a, "solved": solved, "predicted": predicted}


def test_rule_is_met_exactly_at_equality():
    a = [1, 0, 2]
    assert pool.rule_met(_table([[1, 0, 2], [3, 1, 2]], [[1, 0, 2], [1, 0, 5]], a))


@pytest.mark.parametrize("side", ["solved", "predicted"])
def test_rule_is_unmet_when_one_subpool_misses_one_bin(side):
    a = [1, 0, 2]
    ok = [[1, 0, 2], [1, 0, 2]]
    short = [[1, 0, 2], [1, 0, 1]]
    table = _table(short if side == "solved" else ok, short if side == "predicted" else ok, a)
    assert not pool.rule_met(table)


def test_eligible_drops_excluded_rows_and_rule_ignores_them(controlled):
    frame, _ = pool.full_design()
    pool.solve_chunk(frame, 1, workers=1, fn=fake_analyse)
    path = config.pool_path("final", 1)
    final = data.read_table(path)
    final.loc[[0, 1], "excluded"] = "unresolved"
    tamper_frame(path, final)
    rows = pool.eligible(frame, 1, FormulaModel())
    assert len(rows) == 38 and not {0, 1} & set(rows["pool_row"])
    assert set(rows.columns) >= {"pool_row", "subpool", "chunk", "chunk_row", "FOS", "FOS_pred"}
    table = pool.rule_table(rows, np.array([0, 0, 4, 4, 0]))
    assert int(np.asarray(table["solved"]).sum()) <= 38


def test_eligible_refuses_a_misaligned_final(controlled):
    frame, _ = pool.full_design()
    pool.solve_chunk(frame, 1, workers=1, fn=fake_analyse)
    path = config.pool_path("final", 1)
    final = data.read_table(path)
    final.loc[0, "angle"] = final.loc[0, "angle"] + 1.0
    tamper_frame(path, final)
    with pytest.raises(pool.PoolError, match="aligned"):
        pool.eligible(frame, 1, FormulaModel())


def test_run_stops_at_the_first_chunk_that_meets_the_rule(controlled):
    stop = pool.run(chunks=10, workers=1, fn=fake_analyse)
    frame, _ = pool.full_design()
    assert stop is not None and stop == pool.stop_record()
    k = stop["k"]
    tables = [pool.rule_table(pool.eligible(frame, j, FormulaModel()), pool.allocation())
              for j in range(1, k + 1)]
    assert pool.rule_met(tables[-1]) and not any(pool.rule_met(t) for t in tables[:-1])
    assert pool.solved_prefix(frame) == k
    assert not config.pool_path("final", k + 1).exists()
    assert set(stop["chunks_sha256"]) == {pool.chunk_key(j) for j in range(1, k + 1)}


def test_run_respects_the_chunk_budget(controlled, monkeypatch):
    monkeypatch.setattr(pool, "allocation", lambda: np.array([0, 0, 10 ** 6, 0, 0]))
    assert pool.run(chunks=2, workers=1, fn=fake_analyse) is None
    frame, _ = pool.full_design()
    assert pool.solved_prefix(frame) == 2 and pool.stop_record() is None


def test_run_raises_at_the_cap_without_relaxing(controlled, monkeypatch):
    monkeypatch.setattr(pool, "allocation", lambda: np.array([0, 0, 10 ** 6, 0, 0]))
    with pytest.raises(pool.PoolError, match="cap"):
        pool.run(chunks=50, workers=1, fn=fake_analyse)
    assert pool.stop_record() is None


def test_run_after_the_stop_solves_nothing(controlled, monkeypatch, capsys):
    stop = pool.run(chunks=10, workers=1, fn=fake_analyse)
    monkeypatch.setattr(pool.solver, "run_set", lambda *a, **k: pytest.fail("solved after the stop"))
    assert pool.run(chunks=10, workers=1, fn=fake_analyse) == stop
    assert f"met at chunk {stop['k']}" in capsys.readouterr().out


def test_run_resolves_a_torn_chunk_before_counting_it(controlled, monkeypatch):
    monkeypatch.setattr(pool, "allocation", lambda: np.array([0, 0, 10 ** 6, 0, 0]))
    pool.run(chunks=2, workers=1, fn=fake_analyse)
    tamper(config.pool_path("final", 2), "torn")
    frame, _ = pool.full_design()
    assert pool.solved_prefix(frame) == 1
    pool.run(chunks=1, workers=1, fn=fake_analyse)
    assert pool.solved_prefix(frame) == 2


def test_status_lists_the_platforms_of_the_solved_chunks(controlled, monkeypatch):
    monkeypatch.setattr(pool, "allocation", lambda: np.array([0, 0, 10 ** 6, 0, 0]))
    pool.run(chunks=1, workers=1, fn=fake_analyse)
    monkeypatch.setattr(pool, "_platform", lambda: {"system": "Linux", "machine": "x86_64"})
    pool.run(chunks=1, workers=1, fn=fake_analyse)
    st = pool.status()
    assert st["solved_chunks"] == 2 and len(st["platforms"]) == 2
    text = pool.format_status(st)
    assert "Linux/x86_64" in text and "not met" in text


def test_verified_stop_refuses_before_the_stop(controlled):
    frame, _ = pool.full_design()
    with pytest.raises(pool.PoolError, match="sgs pool"):
        pool.verified_stop(frame)


def test_verified_stop_accepts_an_untouched_pool(controlled):
    stop = pool.run(chunks=10, workers=1, fn=fake_analyse)
    frame, _ = pool.full_design()
    assert pool.verified_stop(frame) == stop
    assert stop["platform"] == pool._platform()


def test_verified_stop_refuses_a_chunk_changed_after_the_stop(controlled):
    stop = pool.run(chunks=10, workers=1, fn=fake_analyse)
    frame, _ = pool.full_design()
    path = config.pool_path("final", stop["k"])
    final = data.read_table(path)
    final.loc[0, "FOS"] = final.loc[0, "FOS"] * 1.5
    tamper_frame(path, final)
    with pytest.raises(pool.PoolError, match="changed after the stop"):
        pool.verified_stop(frame)


def test_verified_stop_refuses_a_chunk_rerecorded_after_the_stop(controlled):
    # The chunk's own metadata agrees with a new final, but the stop recorded another one.
    stop = pool.run(chunks=10, workers=1, fn=fake_analyse)
    frame, _ = pool.full_design()
    key = pool.chunk_key(1)
    path = config.pool_path("final", 1)
    final = data.read_table(path)
    final.loc[0, "FOS"] = final.loc[0, "FOS"] * 1.5
    tamper_frame(path, final)
    data.record("pool", key, {**data.read_metadata()["pool"][key], "final_sha256": data.sha256_file(path)})
    with pytest.raises(pool.PoolError, match="not the one the stop recorded"):
        pool.verified_stop(frame)


def test_verified_stop_refuses_another_platform(controlled, monkeypatch):
    pool.run(chunks=10, workers=1, fn=fake_analyse)
    frame, _ = pool.full_design()
    monkeypatch.setattr(pool, "_platform", lambda: {"system": "Linux", "machine": "x86_64"})
    with pytest.raises(pool.PoolError, match="platform"):
        pool.verified_stop(frame)


def test_status_before_any_chunk(controlled):
    st = pool.status()
    assert st["solved_chunks"] == 0 and st["table"] is None and not st["met"]
    assert "0 of 10" in pool.format_status(st)
