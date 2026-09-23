import hashlib

import numpy as np
import pandas as pd
import pytest

from sgs import config, data, design
from tests.helpers import FormulaModel, shrink_roles, synthetic_final, write_final


def _hash(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def _write_design(role, frame):
    path = config.design_path(role)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


# --- LHS and disjointness -------------------------------------------------------------

def test_lhs_uses_the_seed_keyword_semantics():
    # First row of the published seed design. Pins SciPy's `seed=` stream; a
    # change of keyword (`rng=`) or of SciPy's LHS draws fails here.
    _, frame = design.lhs(300, 42)
    np.testing.assert_allclose(
        frame.iloc[0].to_numpy(),
        [24.70810805776474, 33.64879751536226, 12.628212513088526,
         17.341403519225082, 4.832465812168988], rtol=0, atol=1e-12)


def test_lhs_is_stratified_and_in_domain():
    unit, frame = design.lhs(50, 7)
    assert design.lhs_strata_complete(unit)
    for p, (lo, hi) in config.DOMAIN.items():
        assert frame[p].between(lo, hi).all()


def test_strata_check_detects_a_collapsed_column():
    unit, _ = design.lhs(20, 7)
    unit[:, 2] = 0.5
    assert not design.lhs_strata_complete(unit)


def test_assert_disjoint_names_the_offending_set():
    _, a = design.lhs(20, 1)
    with pytest.raises(design.DesignError, match="'twin'"):
        design.assert_disjoint(a, {"twin": a.copy()}, 1e-3)


def test_assert_disjoint_catches_internal_duplicates():
    _, a = design.lhs(20, 1)
    doubled = pd.concat([a, a.iloc[:1]], ignore_index=True)
    with pytest.raises(design.DesignError, match="inside the set"):
        design.assert_disjoint(doubled, {}, 1e-3)


def test_build_lhs_roles(root):
    frame, info = design.build("seed")
    assert len(frame) == 300 and info == {"seeds": {"seed_lhs_design": 42}}
    assert _hash(design.build("seed")[0]) == _hash(frame)
    with pytest.raises(FileNotFoundError, match="seed_lhs_300.csv"):
        design.build("lhs-extension")
    _write_design("seed", frame)
    extension, _ = design.build("lhs-extension")
    assert design.nearest_distance(extension, frame).min() > config.MIN_DISTANCE


# --- screener -------------------------------------------------------------------------

@pytest.fixture
def small_screener(monkeypatch):
    monkeypatch.setitem(config.SCREENERS["rf"], "n_estimators", 20)
    monkeypatch.setitem(config.SCREENER, "cv_folds", 3)


def test_screener_uses_named_seeds(small_screener):
    model = design.screener_model("rf", oob=True)
    assert model.random_state == config.SEEDS["screener_fit"] and model.oob_score


def test_fit_screener_is_deterministic_and_reports_diagnostics(small_screener):
    frame = data.add_features(synthetic_final(80, 5))
    a, diagnostics = design.fit_screener(frame, "rf")
    b, _ = design.fit_screener(frame, "rf")
    X = data.add_features(synthetic_final(10, 9))[list(config.FEATURES)]
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    assert diagnostics["n_train"] == 80 and 0 < diagnostics["oob_r2_log"] <= 1
    assert {"oof_rmse", "oof_rmse_selection_range", "oof_spearman"} <= set(diagnostics)


def test_screener_model_dispatches_on_family():
    rf = design.screener_model("rf", oob=True)
    gbm = design.screener_model("gbm", oob=False)
    assert type(rf).__name__ == "RandomForestRegressor"
    assert rf.n_estimators == 500 and rf.n_jobs == 1 and rf.oob_score is True
    assert rf.random_state == config.SEEDS["screener_fit"]
    params = gbm.get_params()
    assert type(gbm).__name__ == "HistGradientBoostingRegressor"
    assert params["early_stopping"] is False
    assert params["max_iter"] == 8000 and params["max_leaf_nodes"] == 3
    assert params["learning_rate"] == 0.06 and params["min_samples_leaf"] == 3
    assert params["random_state"] == config.SEEDS["screener_fit"]


def test_fit_screener_diagnostics_are_family_aware(monkeypatch):
    monkeypatch.setitem(config.SCREENERS, "gbm",
                        {**config.SCREENERS["gbm"], "max_iter": 20})
    frame = data.add_features(synthetic_final(60, 3))
    common = {"n_train", "oof_rmse", "oof_rmse_selection_range", "oof_spearman"}
    _, rf_diag = design.fit_screener(frame, "rf")
    _, gbm_diag = design.fit_screener(frame, "gbm")
    assert common <= set(rf_diag) and common <= set(gbm_diag)
    assert "oob_r2_log" in rf_diag        # RF only: GBM has no out-of-bag score
    assert "oob_r2_log" not in gbm_diag
    assert rf_diag["n_train"] == 60 and gbm_diag["n_train"] == 60


# --- guided max-min ---------------------------------------------------------------------

def _guided(rng_seed=42):
    return design.guided_maxmin(FormulaModel(), rng=np.random.default_rng(rng_seed), n_pool=6000,
                                n_bins=5, per_bin=4, drawn_per_bin=6, selection_range=(0.5, 6.0))


def test_guided_fills_every_bin_with_predicted_fos_inside_it():
    frame = _guided()
    assert frame["bin"].value_counts().sort_index().tolist() == [4] * 5
    edges = design.bin_edges(5, (0.5, 6.0))
    for b, group in frame.groupby("bin"):
        assert group["FOS_pred"].gt(edges[b]).all() and group["FOS_pred"].le(edges[b + 1]).all()


def test_guided_draws_the_pool_first_from_the_stream():
    # The order contract of `guided_stream`: every selected point is a row of
    # the pool drawn first from a fresh generator with the same seed.
    frame = _guided(42)
    pool = design.uniform_pool(np.random.default_rng(42), 6000)
    merged = frame.merge(pool, on=list(config.PARAMS), how="left", indicator=True)
    assert (merged["_merge"] == "both").all()


def test_guided_is_deterministic_and_seed_sensitive():
    assert _hash(_guided(42)) == _hash(_guided(42))
    assert _hash(_guided(42)) != _hash(_guided(43))


def test_guided_refuses_thin_bins():
    with pytest.raises(design.DesignError, match="fewer than"):
        design.guided_maxmin(FormulaModel(), rng=np.random.default_rng(0), n_pool=200, n_bins=5,
                             per_bin=4, drawn_per_bin=60, selection_range=(0.5, 6.0))


def test_greedy_maxmin_picks_spread_points():
    X = np.array([[0.0, 0.0], [0.01, 0.0], [1.0, 1.0], [0.0, 1.0]])
    chosen = design.greedy_maxmin(X, 3, np.random.default_rng(0))
    assert len(set(chosen.tolist())) == 3


# --- stratified random ------------------------------------------------------------------

def _stratified(pool_seed=1, selection_seed=2, references=None):
    return design.stratified_random(FormulaModel(), pool_rng=np.random.default_rng(pool_seed),
                                    selection_rng=np.random.default_rng(selection_seed), n_pool=6000,
                                    n_bins=5, per_bin=3, selection_range=(0.5, 6.0),
                                    references=references or {}, min_distance=1e-3)


def test_stratified_fills_bins_with_closed_first_bin():
    frame = _stratified()
    assert frame["bin"].value_counts().sort_index().tolist() == [3] * 5
    assert frame["FOS_pred"].between(0.5, 6.0).all()


def test_stratified_uses_both_streams_independently():
    base = _hash(_stratified(1, 2))
    assert _hash(_stratified(1, 2)) == base
    assert _hash(_stratified(1, 3)) != base
    assert _hash(_stratified(9, 2)) != base


def test_stratified_filters_candidates_near_references():
    first = _stratified()
    again = _stratified(references={"previous": first})
    assert design.nearest_distance(again, first).min() > 1e-3


# --- guided log-deficit -------------------------------------------------------------------

def test_log_edges_are_geometric_on_the_range():
    edges = design.bin_edges_log(4, (0.5, 8.0))
    np.testing.assert_allclose(edges, [0.5, 1.0, 2.0, 4.0, 8.0])


def test_bin_counts_close_the_first_bin_and_ignore_outside_points():
    edges = np.array([0.5, 1.0, 2.0, 4.0])
    fos = np.array([0.5, 0.7, 1.0, 1.5, 4.0, 5.0, 0.4])
    assert design.bin_counts(fos, edges).tolist() == [3, 1, 1]


def test_deficit_allocation_fills_the_emptiest_bins_lowest_first():
    assert design.deficit_allocation(np.array([1, 4, 7]), 6).tolist() == [5, 1, 0]
    assert design.deficit_allocation(np.array([0, 100]), 10).tolist() == [10, 0]
    assert design.deficit_allocation(np.array([3, 3]), 1).tolist() == [1, 0]


ALLOCATION = np.array([4, 0, 3, 2, 1])


def _guided_log(rng_seed=42, allocation=ALLOCATION, n_pool=6000):
    return design.guided_log_deficit(FormulaModel(), rng=np.random.default_rng(rng_seed),
                                     allocation=allocation, n_pool=n_pool,
                                     selection_range=(0.5, 6.0))


def test_guided_log_follows_the_allocation_inside_log_bins():
    frame = _guided_log()
    counts = frame["bin"].value_counts().reindex(range(len(ALLOCATION)), fill_value=0)
    assert counts.tolist() == ALLOCATION.tolist()
    edges = design.bin_edges_log(len(ALLOCATION), (0.5, 6.0))
    for b, group in frame.groupby("bin"):
        assert group["FOS_pred"].ge(edges[b]).all() and group["FOS_pred"].le(edges[b + 1]).all()
        if b > 0:
            assert group["FOS_pred"].gt(edges[b]).all()


def test_guided_log_draws_the_pool_first_from_the_stream():
    frame = _guided_log(42)
    pool = design.uniform_pool(np.random.default_rng(42), 6000)
    merged = frame.merge(pool, on=list(config.PARAMS), how="left", indicator=True)
    assert (merged["_merge"] == "both").all()


def test_guided_log_is_deterministic_and_seed_sensitive():
    assert _hash(_guided_log(42)) == _hash(_guided_log(42))
    assert _hash(_guided_log(42)) != _hash(_guided_log(43))


def test_guided_log_refuses_thin_bins():
    with pytest.raises(design.DesignError, match="fewer than"):
        _guided_log(0, allocation=np.array([60, 0, 0, 0, 0]), n_pool=200)


def test_guided_log_does_not_push_points_to_the_domain_edges():
    allocation = np.array([40, 40, 40, 40, 40])
    frame = _guided_log(7, allocation=allocation, n_pool=20000)
    pool = design.uniform_pool(np.random.default_rng(7), 20000)

    def near_edge(f):
        u = data.normalize(f)
        return float(np.mean((np.minimum(u, 1 - u) < 0.05).sum(1) >= 2))

    assert near_edge(frame) < 0.2 and near_edge(frame) < 2 * near_edge(pool) + 0.05


def test_guided_log_selects_from_the_bin_at_random():
    # Stream contract: pool, then per bin (skipping empty allocations) one
    # rng.choice over the bin's candidate indices, without replacement.
    frame = _guided_log(42)
    rng = np.random.default_rng(42)
    pool = design.uniform_pool(rng, 6000)
    pred = design.predict_fos(FormulaModel(), pool)
    edges = design.bin_edges_log(len(ALLOCATION), (0.5, 6.0))
    expected = []
    for b, k in enumerate(ALLOCATION):
        if k == 0:
            continue
        lower = pred >= edges[b] if b == 0 else pred > edges[b]
        candidates = np.flatnonzero(lower & (pred <= edges[b + 1]))
        expected.extend(candidates[rng.choice(len(candidates), k, replace=False)].tolist())
    np.testing.assert_allclose(frame[list(config.PARAMS)].to_numpy(),
                               pool.loc[expected, list(config.PARAMS)].to_numpy())


# --- screener roles through build() -------------------------------------------------------

@pytest.fixture
def small_designs(root, monkeypatch, small_screener):
    monkeypatch.setattr(config, "N_GUIDED_POOL", 6000)
    monkeypatch.setattr(config, "N_TEST_POOL", 6000)
    monkeypatch.setattr(config, "N_BINS", 5)
    monkeypatch.setattr(config, "GUIDED_PER_BIN", 4)
    monkeypatch.setattr(config, "GUIDED_DRAWN_PER_BIN", 6)
    monkeypatch.setattr(config, "TEST_PER_BIN", 3)
    shrink_roles(monkeypatch, {"guided": 20, "test-stratified": 15})
    seed_design, _ = design.build("seed")
    _write_design("seed", seed_design)
    write_final("seed", synthetic_final(300, 1))


def test_screener_roles_are_deterministic_and_record_the_screener(small_designs):
    guided, info = design.build("guided")
    assert _hash(design.build("guided")[0]) == _hash(guided)
    assert info["seeds"] == {"guided_stream": 42}
    assert info["screener"]["trained_on_sha256"] == data.sha256_file(config.final_path("seed"))
    for role in ("guided", "lhs-extension", "test-natural"):
        _write_design(role, design.build(role)[0])
    stratified, _ = design.build("test-stratified")
    assert len(stratified) == 15 and set(stratified.columns) == {*config.PARAMS, "FOS_pred", "bin"}


def test_guided_log_role_fills_the_seed_deficit(small_designs, monkeypatch):
    shrink_roles(monkeypatch, {"guided-log": 20})
    for role in ("guided", "lhs-extension", "test-natural", "test-stratified"):
        _write_design(role, design.build(role)[0])
    frame, info = design.build("guided-log")
    assert info["seeds"] == {"guided_log_stream": config.SEEDS["guided_log_stream"]}
    allocation = info["allocation"]
    seed_fos = data.load_role("seed")["FOS"].to_numpy(float)
    edges = design.bin_edges_log(config.N_BINS, config.SELECTION_RANGE)
    assert allocation["seed_counts"] == design.bin_counts(seed_fos, edges).tolist()
    assert sum(allocation["added"]) == 20 and len(frame) == 20
    counts = frame["bin"].value_counts().reindex(range(config.N_BINS), fill_value=0)
    assert counts.tolist() == allocation["added"]
    assert _hash(design.build("guided-log")[0]) == _hash(frame)


# --- sequential batches: a screener role that learns from several solved sets ---------------

def _add_roles(monkeypatch, **new):
    roles = dict(config.ROLES)
    roles.update(new)
    monkeypatch.setattr(config, "ROLES", roles)


def test_screener_roles_default_to_the_seed_and_follow_trained_on(monkeypatch):
    assert config.screener_roles("guided-log") == ("seed",)
    _add_roles(monkeypatch, **{"batch-x": config.SetRole(
        n=20, design="guided-log-deficit", seeds=("guided_log_stream",), stem="bx", screener=True,
        trained_on=("seed", "lhs-extension"))})
    assert config.screener_roles("batch-x") == ("seed", "lhs-extension")


def test_a_batch_role_uses_every_trained_on_role(small_designs, monkeypatch):
    write_final("lhs-extension", synthetic_final(300, 2))
    _add_roles(monkeypatch, **{"batch-x": config.SetRole(
        n=20, design="guided-log-deficit", seeds=("guided_log_stream",), stem="bx", screener=True,
        trained_on=("seed", "lhs-extension"))})
    frame, info = design.build("batch-x")
    both = np.concatenate([data.load_role(r)["FOS"].to_numpy(float) for r in ("seed", "lhs-extension")])
    edges = design.bin_edges_log(config.N_BINS, config.SELECTION_RANGE)
    assert info["allocation"]["trained_counts"] == design.bin_counts(both, edges).tolist()
    assert info["screener"]["trained_on"] == ["seed", "lhs-extension"]
    assert set(info["screener"]["trained_on_sha256"]) == {"seed", "lhs-extension"}
    assert info["screener"]["n_train"] == 600 and len(frame) == 20
    assert _hash(design.build("batch-x")[0]) == _hash(frame)


def test_single_role_allocation_keeps_the_original_record_format(small_designs, monkeypatch):
    shrink_roles(monkeypatch, {"guided-log": 20})
    for role in ("guided", "lhs-extension", "test-natural", "test-stratified"):
        _write_design(role, design.build(role)[0])
    _, info = design.build("guided-log")
    assert set(info["allocation"]) == {"edges", "seed_counts", "added"}
    assert info["screener"]["trained_on"] == "seed"


def test_an_explicit_trained_on_of_just_the_seed_still_writes_trained_counts(small_designs, monkeypatch):
    # The key follows the role's own declaration, not how many roles it
    # resolves to: guided-seq-1 names trained_on=("seed",) explicitly (it is
    # the first of a sequential family) and must keep "trained_counts", unlike
    # guided-log, which has no trained_on and keeps the legacy "seed_counts".
    _add_roles(monkeypatch, **{"batch-y": config.SetRole(
        n=20, design="guided-log-deficit", seeds=("guided_log_stream",), stem="by", screener=True,
        trained_on=("seed",))})
    _, info = design.build("batch-y")
    assert set(info["allocation"]) == {"edges", "trained_counts", "added"}
    assert info["screener"]["trained_on"] == "seed"
