import json
import math

import numpy as np
import pytest

from sgs import config, evaluate, models
from tests.helpers import EVAL_KWARGS, stub_fit, stub_fits, stub_test, ARM_SCALES, write_all_finals


def _pred(y_true, y_pred, arm="C", family="rf", test="test-natural", ids=None):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ids = np.array([f"{test}:{i}" for i in range(len(y_true))]) if ids is None else np.asarray(ids)
    return evaluate.Prediction(family=family, arm=arm, scoring="log-rmse", test=test,
                               ids=ids, y_true=y_true, y_pred=y_pred)


# --- primitives -------------------------------------------------------------------------

def test_global_metrics():
    m = evaluate.global_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert m["rmse"] == pytest.approx(math.sqrt(4 / 3))
    assert m["mae"] == pytest.approx(2 / 3) and m["bias"] == pytest.approx(2 / 3)
    assert m["max_abs_error"] == 2.0 and m["r2"] == pytest.approx(1 - 4 / 2) and m["n"] == 3


def test_zone_labels_and_half_open_bounds():
    labels = [z[0] for z in evaluate.zones(config.ZONE_EDGES)]
    assert labels == ["FOS < 1", "1–1.3", "1.3–3", "3–6", "FOS ≥ 6"]
    rows = evaluate.zone_metrics(np.array([1.0, 1.3, 6.0]), np.array([1.0, 1.3, 6.0]), config.ZONE_EDGES)
    assert [r["n"] for r in rows] == [0, 1, 1, 0, 1]
    assert rows[0]["rmse"] is None and rows[0]["lo"] is None and rows[-1]["hi"] is None


def test_minimax_is_the_worst_zone_rmse():
    y = np.array([0.8, 0.9, 1.1, 1.2, 2.0, 2.5, 4.0, 7.0])
    p = y + np.array([0.1, -0.1, 0.0, 0.0, 0.5, -0.5, 0.2, 0.0])
    worst = evaluate.minimax(evaluate.zone_metrics(y, p, config.ZONE_EDGES))
    assert worst == {"zone": "1.3–3", "rmse": pytest.approx(0.5)}


def test_conformal_quantile_is_an_order_statistic():
    scores = np.arange(1, 21, dtype=float)                 # n = 20
    # k = min(ceil(21 * 0.9), 20) = 19 -> the 19th smallest value
    assert evaluate.conformal_quantile(scores, 0.10) == 19.0
    assert np.quantile(scores, 19 / 20) != 19.0            # interpolation would differ
    assert evaluate.conformal_quantile(np.array([3.0, 1.0, 2.0]), 0.10) == 3.0   # k capped at n


def test_conformal_intervals_are_multiplicative():
    y_pred = np.array([1.0, 2.0, 2.5, 8.0])
    y_true = y_pred * np.exp([0.1, -0.2, 0.6, -0.05])   # zones 1–1.3, 1.3–3, 3–6, ≥6
    out = evaluate.conformal(_pred(y_true, y_pred), np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5]),
                             alpha=0.10, edges=config.ZONE_EDGES)
    assert out["q_log"] == 0.5 and out["factor"] == pytest.approx(math.exp(0.5))
    assert out["coverage"] == 0.75                          # |0.6| > 0.5 is the only miss
    assert "no formal guarantee" in out["procedure"]
    assert [z["n"] for z in out["zones"]] == [0, 1, 1, 1, 1]


def test_subset_mask():
    y = np.array([0.4, 0.5, 3.0, 6.0, 6.1])
    assert evaluate.subset_mask(y, "all", (0.5, 6.0)).all()
    assert evaluate.subset_mask(y, "selection-range", (0.5, 6.0)).tolist() == [False, True, True, True, False]
    with pytest.raises(ValueError):
        evaluate.subset_mask(y, "band", (0.5, 6.0))


def test_predict_exponentiates_log_models():
    test = stub_test(10)
    pred = evaluate.predict(stub_fit("rf", "C", 0.0), test)
    np.testing.assert_allclose(pred.y_pred, test.y)
    assert pred.test == "test-natural" and (pred.ids == test.ids).all()


# --- paired comparisons -------------------------------------------------------------------

REG = config.COMPARISONS


def _pair(n=80, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.5, 6.0, n)
    better = _pred(y, y + rng.normal(0, 0.05, n), arm="C")
    worse = _pred(y, y + rng.normal(0, 0.30, n), arm="B")
    return better, worse


def test_registry_lookup_is_directional():
    assert evaluate.registered("C", "B", REG).alternative == "less"
    with pytest.raises(evaluate.UnregisteredComparison):
        evaluate.registered("B", "C", REG)
    with pytest.raises(evaluate.UnregisteredComparison):
        evaluate.registered("A", "B", REG)


def test_wilcoxon_on_differences_of_absolute_errors():
    c, b = _pair()
    out = evaluate.wilcoxon_abs(c, b, registry=REG)
    d = np.abs(c.y_pred - c.y_true) - np.abs(b.y_pred - b.y_true)
    assert out["median_d"] == pytest.approx(np.median(d)) and out["median_d"] < 0
    assert out["p_d_lt_0"] == pytest.approx(np.mean(d < 0))
    assert out["alternative"] == "less" and out["hypothesised"] and out["pvalue"] < 0.01
    assert "smaller absolute errors" in out["reading"]


def test_wilcoxon_is_invariant_to_row_order():
    c, b = _pair()
    perm = np.random.default_rng(1).permutation(len(c.ids))
    shuffle = lambda p: _pred(p.y_true[perm], p.y_pred[perm], arm=p.arm, ids=p.ids[perm])
    a1 = evaluate.wilcoxon_abs(c, b, registry=REG)
    a2 = evaluate.wilcoxon_abs(shuffle(c), shuffle(b), registry=REG)
    assert a1["statistic"] == a2["statistic"] and a1["pvalue"] == a2["pvalue"]


@pytest.mark.parametrize("break_it, message", [
    (lambda b: _pred(b.y_true[:-1], b.y_pred[:-1], arm="B"), "identifiers"),
    (lambda b: _pred(b.y_true, b.y_pred, arm="B", ids=b.ids[::-1]), "identifiers"),
    (lambda b: _pred(b.y_true, b.y_pred, arm="B", test="test-stratified"), "test set"),
    (lambda b: _pred(b.y_true, b.y_pred, arm="B", family="gbm"), "vertical"),
    (lambda b: _pred(b.y_true + 1, b.y_pred, arm="B"), "observed FOS"),
])
def test_misaligned_pairs_are_refused(break_it, message):
    c, b = _pair()
    with pytest.raises(evaluate.MisalignedPair, match=message):
        evaluate.wilcoxon_abs(c, break_it(b), registry=REG)
    with pytest.raises(evaluate.MisalignedPair, match=message):
        evaluate.paired_bootstrap(c, break_it(b), registry=REG, n_boot=50, seed=1, ci=0.95)


def test_bootstrap_is_reproducible_and_states_its_limitation():
    c, b = _pair()
    one = evaluate.paired_bootstrap(c, b, registry=REG, n_boot=500, seed=7, ci=0.95)
    two = evaluate.paired_bootstrap(c, b, registry=REG, n_boot=500, seed=7, ci=0.95)
    other = evaluate.paired_bootstrap(c, b, registry=REG, n_boot=500, seed=8, ci=0.95)
    assert one == two and one["ci"] != other["ci"]
    assert one["delta_rmse"] == pytest.approx(evaluate.rmse(c.y_true, c.y_pred) - evaluate.rmse(b.y_true, b.y_pred))
    assert one["ci"][0] <= one["delta_rmse"] <= one["ci"][1] and one["ci"][1] < 0
    assert "test-sampling variability only" in one["limitation"]


def test_mask_restricts_the_test():
    c, b = _pair()
    mask = c.y_true <= 3.0
    assert evaluate.wilcoxon_abs(c, b, registry=REG, mask=mask)["n"] == int(mask.sum())


# --- evaluate() and run() -------------------------------------------------------------------

def test_evaluate_covers_every_arm_and_registered_comparison():
    fits = {**stub_fits("rf"), **stub_fits("gbm")}
    result = evaluate.evaluate(fits, stub_test(), **EVAL_KWARGS)
    assert set(result.arms) == {"rf", "gbm"}
    assert set(result.arms["rf"]) == {"A", "B", "C", "C0"}
    assert len(result.comparisons) == 2 * len(config.COMPARISONS) * len(evaluate.SUBSETS)
    payload = json.loads(json.dumps(result.to_json()))
    assert payload["predictions"]["row_id"][0] == "test-natural:0"
    assert "rf__C0" in payload["predictions"]
    assert "test-sampling variability only" in payload["notes"]["bootstrap"]


def test_evaluate_refuses_a_missing_arm():
    fits = stub_fits("rf")
    fits.pop("rf__C0__log-rmse")
    with pytest.raises(ValueError, match="C0"):
        evaluate.evaluate(fits, stub_test(), **EVAL_KWARGS)


def test_evaluate_refuses_mixed_scorings():
    fits = {**stub_fits("rf"), **stub_fits("gbm", scoring="fos-rmse")}
    with pytest.raises(ValueError, match="one scoring"):
        evaluate.evaluate(fits, stub_test(), **EVAL_KWARGS)


def test_run_writes_metrics_with_provenance(root, monkeypatch):
    write_all_finals(monkeypatch)
    for family in config.FAMILIES:
        for arm, scale in ARM_SCALES.items():
            models.save_fit(stub_fit(family, arm, scale))
    monkeypatch.setattr(config, "N_BOOTSTRAP", 200)
    evaluate.run()
    for test in config.TEST_ROLES:
        metrics = json.loads(config.metrics_path(test, "log-rmse").read_text())
        assert metrics["test"] == test and set(metrics["arms"]["gbm"]) == {"A", "B", "C", "C0"}
        assert set(metrics["provenance"]["final_sha256"]) == set(config.ROLE_ORDER)
        assert set(metrics["provenance"]["fit_fingerprints"]) == {"rf__log-rmse", "gbm__log-rmse"}


# --- learning-curve metrics -------------------------------------------------------------

def test_learning_curve_metrics_average_over_repeats():
    test = stub_test(20)
    good = {"repeat": 0, "predictions": {"test-natural": {"ids": test.ids, "y_pred": test.y}}}
    off = {"repeat": 1, "predictions": {"test-natural": {"ids": test.ids, "y_pred": test.y + 1.0}}}
    rows = evaluate.learning_curve_metrics({("gbm", "lhs", 100): [good, off]}, {"test-natural": test},
                                           selection_range=config.SELECTION_RANGE,
                                           edges=config.ZONE_EDGES)["rows"]
    assert len(rows) == 1 and rows[0]["n_repeats"] == 2 and rows[0]["family"] == "gbm"
    assert rows[0]["rmse_mean"] == pytest.approx(0.5) and rows[0]["rmse_std"] == pytest.approx(0.5)
    log_off = evaluate.rmse(np.log(test.y), np.log(test.y + 1.0))
    assert rows[0]["log_rmse_mean"] == pytest.approx(log_off / 2)
    zones = rows[0]["zones"]
    assert [z["zone"] for z in zones] == [label for label, _, _ in evaluate.zones(config.ZONE_EDGES)]
    assert sum(z["n"] for z in zones) == len(test.y)
    for z in zones:
        assert (z["rmse_mean"] is None) == (z["n"] == 0)
        if z["n"]:
            assert z["rmse_mean"] == pytest.approx(0.5)


def test_learning_curve_metrics_refuse_misaligned_predictions():
    test = stub_test(20)
    unit = {"repeat": 0, "predictions": {"test-natural": {"ids": test.ids[::-1], "y_pred": test.y}}}
    with pytest.raises(evaluate.MisalignedPair):
        evaluate.learning_curve_metrics({("rf", "lhs", 100): [unit]}, {"test-natural": test},
                                        selection_range=config.SELECTION_RANGE, edges=config.ZONE_EDGES)
