import joblib
import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV

from sgs import config, data, models
from tests.helpers import synthetic_final, write_final


def test_estimators_are_single_threaded_and_seeded():
    rf = models.estimator("rf")
    assert rf.n_jobs == 1 and rf.random_state == config.SEEDS["protocol_estimator"]
    assert models.estimator("gbm").random_state == config.SEEDS["protocol_estimator"]


def test_search_is_parallel_and_uses_the_named_split():
    search = models.searcher("rf", "log-rmse")
    assert isinstance(search, GridSearchCV) and search.n_jobs == -1
    assert search.cv.random_state == config.SEEDS["protocol_cv_split"]
    assert search.cv.n_splits == config.CV_FOLDS
    assert search.scoring == "neg_root_mean_squared_error"


def test_fos_scale_scorer():
    assert models._fos_rmse(np.log([1.0, 2.0]), np.log([1.0, 4.0])) == pytest.approx(np.sqrt(2.0))
    assert models.scorer("fos-rmse") is not None
    with pytest.raises(ValueError):
        models.scorer("r2")


def test_fit_arm_artifact(tiny_protocol):
    art = models.fit_arm("rf", "C", "log-rmse")
    assert (art.family, art.arm, art.scoring, art.target) == ("rf", "C", "log-rmse", "log")
    assert len(art.train_ids) == 120 and art.train_ids[0] == "seed:0" and art.train_ids[-1] == "guided:59"
    assert set(art.fold) == {0, 1, 2}
    assert set(art.best_params) == {"min_samples_leaf"}
    assert len(art.cv_results["params"]) == 2
    assert np.all(art.scores >= 0) and art.scores.shape == (120,)


def test_oof_residuals_come_from_the_search_object(tiny_protocol, monkeypatch):
    seen = []
    real = models.cross_val_predict

    def spy(estimator, *args, **kwargs):
        seen.append(estimator)
        return real(estimator, *args, **kwargs)

    monkeypatch.setattr(models, "cross_val_predict", spy)
    models.fit_arm("gbm", "A", "log-rmse")
    assert len(seen) == 1 and isinstance(seen[0], GridSearchCV)


def test_fit_arm_is_deterministic(tiny_protocol):
    a = models.fit_arm("gbm", "C0", "log-rmse")
    b = models.fit_arm("gbm", "C0", "log-rmse")
    np.testing.assert_array_equal(a.oof_pred, b.oof_pred)
    assert a.best_params == b.best_params


def test_saved_fit_loads_while_fresh(tiny_protocol):
    models.save_fit(models.fit_arm("rf", "A", "log-rmse"))
    stored = joblib.load(config.fit_path("rf", "A", "log-rmse"))
    assert stored["data"] == {"seed": data.sha256_file(config.final_path("seed"))}
    assert models.load_fit("rf", "A", "log-rmse").arm == "A"


def test_load_fit_fails_on_a_changed_grid(tiny_protocol, monkeypatch):
    models.save_fit(models.fit_arm("rf", "A", "log-rmse"))
    monkeypatch.setitem(config.GRIDS, "rf", {"min_samples_leaf": [1, 2]})
    with pytest.raises(models.StaleFit, match="protocol configuration changed"):
        models.load_fit("rf", "A", "log-rmse")


def test_load_fit_fails_on_changed_training_data(tiny_protocol):
    models.save_fit(models.fit_arm("rf", "A", "log-rmse"))
    write_final("seed", synthetic_final(60, 99))
    with pytest.raises(models.StaleFit, match="training data 'seed' changed"):
        models.load_fit("rf", "A", "log-rmse")


def test_load_fit_names_a_missing_file(tiny_protocol):
    with pytest.raises(FileNotFoundError, match="sgs fit"):
        models.load_fit("rf", "A", "log-rmse")


def test_fit_all_skips_fresh_units_and_refuses_stale_ones(tiny_protocol, monkeypatch):
    unit = [("rf", "C", "log-rmse")]
    models.fit_all(units=unit)
    assert data.read_metadata()["fits"]["rf__C__log-rmse"]["n_train"] == 120
    calls = []
    real = models.fit_arm
    monkeypatch.setattr(models, "fit_arm", lambda *a: calls.append(a) or real(*a))
    models.fit_all(units=unit)
    assert calls == []
    write_final("guided", synthetic_final(60, 77))
    with pytest.raises(models.StaleFit, match="--force"):
        models.fit_all(units=unit)
    models.fit_all(units=unit, force=True)
    assert calls == [("rf", "C", "log-rmse")]


def test_fit_all_validates_the_training_sets(tiny_protocol):
    write_final("guided", synthetic_final(60, 10))       # identical to the seed set
    with pytest.raises(data.DataError, match="not disjoint"):
        models.fit_all(units=[("rf", "C", "log-rmse")])


def test_all_units_cover_the_declared_axes():
    assert len(models.all_units()) == len(config.FAMILIES) * len(config.ARMS) * len(config.ACTIVE_SCORINGS)


# --- learning curve ---------------------------------------------------------------------

import pandas as pd

from tests.helpers import shrink_roles


@pytest.fixture
def tiny_curve(root, monkeypatch):
    shrink_roles(monkeypatch, {"seed": 40, "lhs-extension": 60, "guided": 60,
                               "test-natural": 30, "test-stratified": 30})
    for i, role in enumerate(("seed", "lhs-extension", "guided", "test-natural", "test-stratified")):
        write_final(role, synthetic_final(config.ROLES[role].n, 300 + i))
    monkeypatch.setattr(config, "ESTIMATOR_FIXED", {"rf": {"n_estimators": 20}, "gbm": {}})
    monkeypatch.setitem(config.LC_PARAMS, "gbm", {**config.LC_PARAMS["gbm"], "max_iter": 20})
    monkeypatch.setattr(config, "LC_BUDGETS", (20, 40, 60))
    monkeypatch.setattr(config, "LC_REPEATS", 2)


def test_lc_plan_shares_budgets_up_to_the_seed_size():
    assert models.lc_plan(300, {"lhs": 300, "guided": 300}) == [
        ("seed-only", 100, 10), ("seed-only", 200, 10), ("seed-only", 300, 1),
        ("lhs", 400, 10), ("guided", 400, 10), ("lhs", 500, 10), ("guided", 500, 10),
        ("lhs", 600, 1), ("guided", 600, 1)]


def test_lc_plan_gives_one_repeat_when_a_budget_takes_the_whole_pool(monkeypatch):
    """A budget that exhausts a strategy's own pool (seed + all of its extra
    rows) would draw the identical row set on every repeat, same as the seed
    size; only that strategy drops to one repeat at that budget."""
    monkeypatch.setattr(config, "LC_BUDGETS", (50, 80, 100))
    monkeypatch.setattr(config, "LC_REPEATS", 5)
    assert models.lc_plan(50, {"lhs": 30, "guided": 50}) == [
        ("seed-only", 50, 1),
        ("lhs", 80, 1), ("guided", 80, 5),
        ("lhs", 100, 5), ("guided", 100, 1)]


def test_lc_sizes_refuses_a_budget_beyond_the_usable_pool(tiny_curve, monkeypatch):
    monkeypatch.setattr(config, "LC_BUDGETS", (20, 40, 60, 200))
    with pytest.raises(ValueError, match="lhs-extension"):
        models.lc_sizes()


def test_subsample_semantics():
    seed_df = pd.DataFrame({"row_id": [f"s{i}" for i in range(10)]})
    extra = pd.DataFrame({"row_id": [f"e{i}" for i in range(10)]})
    small = models.subsample(seed_df, extra, 4, models.unit_rng(4, 0))
    assert len(small) == 4 and small["row_id"].str.startswith("s").all()
    assert models.subsample(seed_df, extra, 10, models.unit_rng(10, 0)) is seed_df
    large = models.subsample(seed_df, extra, 13, models.unit_rng(13, 0))
    assert len(large) == 13 and set(seed_df["row_id"]) <= set(large["row_id"])


def test_unit_rng_is_keyed_on_budget_and_repeat():
    draw = lambda b, r: models.unit_rng(b, r).integers(0, 10**9)
    assert draw(100, 0) == draw(100, 0)
    assert len({draw(100, 0), draw(100, 1), draw(200, 0)}) == 3


def test_lc_estimator_is_fixed_and_seeded():
    rf = models.lc_estimator("rf")
    assert rf.random_state == config.SEEDS["protocol_estimator"]
    assert rf.max_features == 1.0 and rf.min_samples_leaf == 1
    gbm = models.lc_estimator("gbm")
    assert gbm.random_state == config.SEEDS["protocol_estimator"]
    assert {k: gbm.get_params()[k] for k in config.LC_PARAMS["gbm"]} == config.LC_PARAMS["gbm"]
    assert gbm.early_stopping is False


def test_learning_curve_writes_one_file_per_budget_and_refuses_stale(tiny_curve, monkeypatch):
    models.learning_curve()
    names = sorted(p.name for p in (config.ROOT / "artifacts" / "learning_curve").glob("*.joblib"))
    assert names == sorted(f"{family}__{key}.joblib" for family in ("rf", "gbm")
                           for key in ("guided__n60", "lhs__n60", "seed-only__n20", "seed-only__n40"))
    units = models.load_lc_units()
    for family in ("rf", "gbm"):
        assert units[(family, "lhs", 20)] is units[(family, "guided", 20)]
        assert len(units[(family, "lhs", 40)]) == 1 and len(units[(family, "lhs", 60)]) == 2
    rf_pred = units[("rf", "lhs", 60)][0]["predictions"]["test-natural"]["y_pred"]
    gbm_pred = units[("gbm", "lhs", 60)][0]["predictions"]["test-natural"]["y_pred"]
    assert not np.allclose(rf_pred, gbm_pred)
    monkeypatch.setattr(config, "LC_REPEATS", 3)
    with pytest.raises(models.StaleFit, match="--force"):
        models.learning_curve()
    with pytest.raises(models.StaleFit):
        models.load_lc_units()
    models.learning_curve(force=True)
    assert len(models.load_lc_units()[("gbm", "lhs", 60)]) == 3
