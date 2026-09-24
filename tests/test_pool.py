import os

import numpy as np
import pandas as pd
import pytest

from sgs import config, data, design, pool
from tests.helpers import fake_analyse, tamper, tamper_frame


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
