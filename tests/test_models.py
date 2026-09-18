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
