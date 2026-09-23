import numpy as np
import pandas as pd
import pytest

from sgs import config, design, pool


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
