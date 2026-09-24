import json
import math

import numpy as np
import pytest
from scipy import stats as _stats

from sgs import config, evaluate, models
from tests.helpers import EVAL_KWARGS, OffsetModel, stub_fit, stub_fits, stub_test, ARM_SCALES, write_all_finals


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
    assert labels == ["FOS < 0.5", "0.5–1", "1–2", "2–6", "FOS ≥ 6"]
    y = np.array([0.5, 1.0, 2.0, 6.0])
    rows = evaluate.zone_metrics(y, y, config.ZONE_EDGES)
    assert [r["n"] for r in rows] == [0, 1, 1, 1, 1]
    assert rows[0]["rmse"] is None and rows[0]["log_rmse"] is None
    assert rows[0]["lo"] is None and rows[-1]["hi"] is None


def test_zone_metrics_report_the_log_scale():
    y = np.array([1.0, 1.5])
    p = y * np.exp([0.1, -0.1])
    (row,) = [r for r in evaluate.zone_metrics(y, p, config.ZONE_EDGES) if r["n"]]
    assert row["zone"] == "1–2" and row["log_rmse"] == pytest.approx(0.1)


def test_minimax_is_the_worst_zone_rmse():
    y = np.array([0.8, 0.9, 1.1, 1.2, 2.0, 2.5, 4.0, 7.0])
    p = y + np.array([0.1, -0.1, 0.0, 0.0, 0.5, -0.5, 0.2, 0.0])
    rows = evaluate.zone_metrics(y, p, config.ZONE_EDGES)
    assert evaluate.minimax(rows) == {"zone": "2–6", "rmse": pytest.approx(math.sqrt(0.18))}
    assert evaluate.minimax(rows, key="log_rmse")["zone"] == "2–6"


def test_conformal_quantile_is_an_order_statistic():
    scores = np.arange(1, 21, dtype=float)                 # n = 20
    # k = min(ceil(21 * 0.9), 20) = 19 -> the 19th smallest value
    assert evaluate.conformal_quantile(scores, 0.10) == 19.0
    assert np.quantile(scores, 19 / 20) != 19.0            # interpolation would differ
    assert evaluate.conformal_quantile(np.array([3.0, 1.0, 2.0]), 0.10) == 3.0   # k capped at n


def test_conformal_intervals_are_multiplicative():
    y_pred = np.array([1.0, 2.0, 2.5, 8.0])
    y_true = y_pred * np.exp([0.1, -0.2, 0.6, -0.05])   # zones 1–2, 1–2, 2–6, ≥6
    out = evaluate.conformal(_pred(y_true, y_pred), np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5]),
                             alpha=0.10, edges=config.ZONE_EDGES)
    assert out["q_log"] == 0.5 and out["factor"] == pytest.approx(math.exp(0.5))
    assert out["coverage"] == 0.75                          # |0.6| > 0.5 is the only miss
    assert "no formal guarantee" in out["procedure"]
    assert [z["n"] for z in out["zones"]] == [0, 0, 2, 1, 1]


def test_subset_mask():
    y = np.array([0.4, 0.5, 1.99, 2.0, 3.0, 6.0, 6.1])
    assert evaluate.subset_mask(y, "all", (0.5, 6.0)).all()
    assert evaluate.subset_mask(y, "selection-range", (0.5, 6.0)).tolist() == [
        False, True, True, True, True, True, False]
    assert evaluate.subset_mask(y, "decision-range", (0.5, 6.0), (0.5, 2.0)).tolist() == [
        False, True, True, False, False, False, False]
    with pytest.raises(ValueError):
        evaluate.subset_mask(y, "band", (0.5, 6.0))
    with pytest.raises(ValueError, match="decision_range"):
        evaluate.subset_mask(y, "decision-range", (0.5, 6.0))


def test_wilcoxon_and_bootstrap_on_the_log_scale():
    c, b = _pair()
    out = evaluate.wilcoxon_abs(c, b, registry=REG, scale="log")
    d = (np.abs(np.log(c.y_pred) - np.log(c.y_true)) - np.abs(np.log(b.y_pred) - np.log(b.y_true)))
    assert out["scale"] == "log" and out["median_d"] == pytest.approx(np.median(d))
    boot = evaluate.paired_bootstrap(c, b, registry=REG, n_boot=200, seed=7, ci=0.95, scale="log")
    expected = (evaluate.rmse(np.log(c.y_true), np.log(c.y_pred))
               - evaluate.rmse(np.log(b.y_true), np.log(b.y_pred)))
    assert boot["scale"] == "log" and boot["delta_rmse"] == pytest.approx(expected)
    with pytest.raises(ValueError):
        evaluate.on_scale(c.y_true, "sqrt")


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
    assert set(result.arms["rf"]) == set(config.ARMS)
    assert len(result.comparisons) == (2 * len(config.COMPARISONS) * len(evaluate.SUBSETS)
                                       * len(evaluate.SCALES))
    assert {r["scale"] for r in result.comparisons} == set(evaluate.SCALES)
    payload = json.loads(json.dumps(result.to_json()))
    assert payload["predictions"]["row_id"][0] == "test-natural:0"
    assert "rf__C0" in payload["predictions"]
    assert "test-sampling variability only" in payload["notes"]["bootstrap"]


def test_evaluate_reports_the_decision_range_on_the_log_scale():
    test = stub_test()
    result = evaluate.evaluate(stub_fits("rf"), test, **EVAL_KWARGS)
    arm = result.arms["rf"]["B"]
    lo, hi = config.DECISION_RANGE
    mask = (test.y >= lo) & (test.y < hi)
    p = np.exp(OffsetModel(ARM_SCALES["B"]).predict(test.X))
    assert arm["subsets"]["decision-range"]["n"] == int(mask.sum())
    assert arm["subsets"]["decision-range"]["log_rmse"] == pytest.approx(
        evaluate.rmse(np.log(test.y[mask]), np.log(p[mask])))
    assert set(arm["subsets"]) == set(evaluate.SUBSETS)
    assert "log_rmse" in arm["global"] and arm["minimax_log"]["zone"]


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
    for scoring in config.ACTIVE_SCORINGS:
        for family in config.FAMILIES:
            for arm, scale in ARM_SCALES.items():
                models.save_fit(stub_fit(family, arm, scale, scoring=scoring))
    monkeypatch.setattr(config, "N_BOOTSTRAP", 200)
    evaluate.run()
    for scoring in config.ACTIVE_SCORINGS:
        for test in config.TEST_ROLES:
            metrics = json.loads(config.metrics_path(test, scoring).read_text())
            assert metrics["test"] == test and metrics["scoring"] == scoring
            assert set(metrics["arms"]["gbm"]) == set(config.ARMS)
            assert set(metrics["provenance"]["final_sha256"]) == set(config.ROLE_ORDER)
            assert set(metrics["provenance"]["fit_fingerprints"]) == {f"rf__{scoring}", f"gbm__{scoring}"}


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


# --- replication statistics -----------------------------------------------------------------

A5 = np.array([0.10, 0.12, 0.11, 0.13, 0.09])
B5 = np.array([0.10, 0.10, 0.10, 0.11, 0.12])


def test_welch_interval_matches_scipy():
    out = evaluate.welch_interval(A5, B5, 0.95)
    ref = _stats.ttest_ind(A5, B5, equal_var=False).confidence_interval(0.95)
    assert out["diff"] == pytest.approx(A5.mean() - B5.mean())
    assert out["ci"] == pytest.approx([ref.low, ref.high])
    assert out["level"] == 0.95 and out["df"] < 8


def test_welch_interval_with_no_spread_is_a_point():
    out = evaluate.welch_interval(np.full(5, 0.1), np.full(5, 0.2), 0.95)
    assert out["ci"] == pytest.approx([-0.1, -0.1]) and out["se"] == 0.0 and out["df"] is None


def test_paired_interval_matches_scipy():
    out = evaluate.paired_interval(A5, B5, 0.95)
    ref = _stats.ttest_rel(A5, B5).confidence_interval(0.95)
    assert out["ci"] == pytest.approx([ref.low, ref.high]) and out["df"] == 4


def test_prob_less_counts_all_pairs_and_halves_ties():
    assert evaluate.prob_less([1.0, 2.0], [1.5, 2.5]) == pytest.approx(3 / 4)
    assert evaluate.prob_less([1.0], [1.0]) == pytest.approx(0.5)


def test_prob_less_paired_counts_within_realization_pairs_and_halves_ties():
    assert evaluate.prob_less_paired([1.0, 2.0], [1.5, 2.5]) == pytest.approx(1.0)
    assert evaluate.prob_less_paired([1.0, 3.0], [2.0, 3.0]) == pytest.approx(3 / 4)
    assert evaluate.prob_less_paired([1.0], [1.0]) == pytest.approx(0.5)


def test_paired_interaction_is_a_difference_in_differences(monkeypatch):
    calls = []
    real_welch = evaluate.welch_interval
    monkeypatch.setattr(evaluate, "welch_interval",
                        lambda a, b, level: calls.append("welch") or real_welch(a, b, level))
    rf = {"X": [1.0, 2.0, 3.0, 4.0, 5.0], "Y": [1.0, 1.0, 1.0, 1.0, 1.0]}
    gbm = {"X": [2.0, 2.0, 2.0, 2.0, 2.0], "Y": [1.0, 1.0, 1.0, 1.0, 1.0]}
    rows = evaluate.interaction_rows(
        series=lambda fam, test, arm, measure: (rf if fam == "rf" else gbm)[arm],
        tests=("test-natural",), measures=("decision",),
        interactions=(config.Interaction("X", "Y", method="paired"),), level=0.95)
    expected = [np.log(a) - np.log(b) for a, b in zip(rf["X"], gbm["X"])]   # D_rf - D_gbm, Y cancels
    assert rows[0]["method"] == "paired"
    assert rows[0]["diff"] == pytest.approx(float(np.mean(expected)))
    assert rows[0]["df"] == 4                     # n - 1 over the five realizations
    assert calls == []                            # never falls back to Welch


# --- relocation reading -----------------------------------------------------------------

LN11 = float(np.log(1.10))


def cells(L, H, dR, n=4):
    """n identical cells; each value repeated over five realizations."""
    return {("rf", f"t{i}"): {"L_g": [L] * 5, "H_g": [H] * 5, "dR": [dR] * 5} for i in range(n)}


def test_gate_requires_a_gain_below_one_and_a_loss_above():
    # R > 0 but both zones worse than B: a uniform loss, never relocation.
    out = evaluate.relocation_reading(cells(L=0.05, H=0.30, dR=2 * LN11))
    assert out["reading"] == "mixed/inconclusive"
    assert out["gate_cells"] == 0


def test_larger_needs_the_gate_and_the_margin():
    out = evaluate.relocation_reading(cells(L=-0.20, H=0.20, dR=LN11))
    assert out["reading"] == "larger"


def test_similar_inside_the_margin():
    out = evaluate.relocation_reading(cells(L=-0.20, H=0.20, dR=0.9 * LN11))
    assert out["reading"] == "similar"


def test_the_margin_boundary_is_inclusive_for_larger():
    at = evaluate.relocation_reading(cells(L=-0.2, H=0.2, dR=LN11))
    just_below = evaluate.relocation_reading(cells(L=-0.2, H=0.2, dR=LN11 - 1e-12))
    assert at["reading"] == "larger" and just_below["reading"] == "similar"


def test_vanishes_when_the_gate_fails_and_the_level_is_small():
    out = evaluate.relocation_reading(cells(L=0.01, H=0.01, dR=0.0))   # R = 0
    assert out["reading"] == "vanishes"


def test_reverses_when_the_level_is_negative():
    out = evaluate.relocation_reading(cells(L=0.20, H=-0.20, dR=0.0))  # R = -0.40
    assert out["reading"] == "reverses"


def test_three_of_four_cells_is_enough():
    good = cells(L=-0.2, H=0.2, dR=LN11, n=3)
    good[("gbm", "t9")] = {"L_g": [0.1] * 5, "H_g": [-0.1] * 5, "dR": [0.0] * 5}
    assert evaluate.relocation_reading(good)["reading"] == "larger"


def _fake_units(tests, realizations=(0, 1, 2, 3), scale=None):
    """units shaped like replicate.load_units(): arm error grows with its scale."""
    scale = scale or {"A": 0.30, "B": 0.10, "C-log": 0.12, "C0-log": 0.25, "C-seq": 0.11, "C0-seq": 0.24}
    units = {}
    for family in ("rf", "gbm"):
        for r in realizations:
            arms = {}
            for arm, s in scale.items():
                if arm == "A" and r != 0:
                    continue
                noise = s * (1 + 0.1 * r) * (1.5 if family == "rf" else 1.0)
                arms[arm] = {"n_train": 10, "train_ids": np.array([f"x:{i}" for i in range(10)]),
                             "train_fos": np.linspace(0.4, 7.0, 10),
                             "predictions": {n: t.y * np.exp(noise * np.sin(np.arange(len(t.y)) + r))
                                             for n, t in tests.items()}}
            units[(family, r)] = {"test_ids": {n: t.ids for n, t in tests.items()}, "arms": arms}
    return units


REP_KW = dict(zones=config.ZONE_EDGES, bins=(0.5, 1.0, 2.0, 6.0), selection_range=config.SELECTION_RANGE,
              decision_range=config.DECISION_RANGE, level=0.95, welch=(("C-seq", "B"), ("C-log", "B")),
              paired=(("C-seq", "C0-seq"),),
              interactions=(config.Interaction("C-seq", "B", method="welch"),))


def test_replication_metrics_separate_r0_and_compare_confirmatory_realizations():
    tests = {"test-natural": stub_test(80, 3), "test-stratified": stub_test(50, 4, name="test-stratified")}
    out = evaluate.replication_metrics(_fake_units(tests), tests, **REP_KW)
    assert out["realizations"] == [0, 1, 2, 3] and out["confirmatory"] == [1, 2, 3]
    s = next(x for x in out["summaries"] if x["family"] == "gbm" and x["test"] == "test-natural"
             and x["arm"] == "B" and x["measure"] == "decision-range")
    assert len(s["values"]) == 3 and s["r0"] is not None and s["mean"] == pytest.approx(np.mean(s["values"]))
    c = next(x for x in out["comparisons"] if x["first"] == "C-seq" and x["second"] == "B"
             and x["family"] == "gbm" and x["test"] == "test-natural" and x["measure"] == "all"
             and x["scale"] == "log-rmse")
    assert c["method"] == "welch" and c["first_values"] and len(c["second_values"]) == 3
    ref = evaluate.welch_interval(c["first_values"], c["second_values"], 0.95)
    assert c["ci"] == pytest.approx(ref["ci"]) and 0.0 <= c["p_first_less"] <= 1.0
    rel = next(x for x in out["comparisons"] if x["first"] == "C-seq" and x["second"] == "B"
               and x["family"] == "gbm" and x["test"] == "test-natural" and x["measure"] == "all"
               and x["scale"] == "relative")
    assert rel["diff"] == pytest.approx(np.mean(np.log(c["first_values"])) - np.mean(np.log(c["second_values"])))
    p = next(x for x in out["comparisons"] if x["method"] == "paired")
    assert p["p_first_less"] == pytest.approx(
        evaluate.prob_less_paired(p["first_values"], p["second_values"]))
    i = next(x for x in out["interactions"] if x["test"] == "test-natural" and x["measure"] == "all")
    assert len(i["first_values"]) == 3 and "ci" in i
    v = out["values"][0]
    assert {"subsets", "zones", "bins", "train_counts"} <= set(v) and len(v["train_counts"]) == len(v["bins"])
    json.dumps(out)   # serializable


def _zoned_units(tests, noise, realizations=(0, 1, 2, 3, 4, 5)):
    """units whose log error per arm is noise[arm][zone] * sin(...): zone log-RMSE ratios
    between arms are then exact and identical in every realization and cell."""
    units = {}
    for family in ("rf", "gbm"):
        for r in realizations:
            arms = {}
            for arm, (low, high) in noise.items():
                preds = {}
                for n, t in tests.items():
                    s = np.where(t.y < 1.0, low, high) * (1 + 0.1 * r)
                    preds[n] = t.y * np.exp(s * np.sin(np.arange(len(t.y)) + r))
                arms[arm] = {"n_train": 10, "train_ids": np.array([f"x:{i}" for i in range(10)]),
                             "train_fos": np.linspace(0.4, 7.0, 10), "predictions": preds}
            units[(family, r)] = {"test_ids": {n: t.ids for n, t in tests.items()}, "arms": arms}
    return units


def _relocation(noise):
    tests = {"test-natural": stub_test(400, 3), "test-stratified": stub_test(400, 4, name="test-stratified")}
    return evaluate.replication_metrics(_zoned_units(tests, noise), tests, **REP_KW)["relocation"]


def test_relocation_dR_is_the_paired_difference_against_c_log():
    # C-logG relocates strongly against B (L = ln 0.5, H = ln 2, R = ln 4), and C-log
    # relocates exactly as much: dR = R(C-logG) - R(C-log) = 0, so "similar", never
    # "larger". Reading C-logG's own R as dR (the old wiring) called this "larger".
    out = _relocation({"B": (0.10, 0.10), "C-logG": (0.05, 0.20), "C-log": (0.05, 0.20)})
    assert out["arm"] == "C-logG" and out["reference"] == "C-log" and out["base"] == "B"
    assert out["gate_cells"] == 4 and len(out["cells"]) == 4
    for cell in out["cells"].values():
        assert cell["L"] == pytest.approx(np.log(0.5))
        assert cell["H"] == pytest.approx(np.log(2.0))
        assert cell["R"] == pytest.approx(np.log(4.0))
        assert cell["dR"] == pytest.approx(0.0, abs=1e-12)
    assert out["reading"] == "similar"


def test_relocation_dR_cancels_b_and_detects_a_larger_relocation():
    # R(C-logG) = ln 4, R(C-log) = ln(0.15/0.075) = ln 2: dR = ln 2 >= ln 1.10.
    noise = {"C-logG": (0.05, 0.20), "C-log": (0.075, 0.15)}
    for b in ((0.10, 0.10), (0.30, 0.12)):          # dR must not depend on B
        out = _relocation({"B": b, **noise})
        for cell in out["cells"].values():
            assert cell["dR"] == pytest.approx(np.log(2.0))
    assert _relocation({"B": (0.10, 0.10), **noise})["reading"] == "larger"


def test_replication_metrics_refuse_misaligned_predictions():
    tests = {"test-natural": stub_test(80, 3)}
    units = _fake_units(tests, realizations=(0, 1))
    units[("rf", 1)]["test_ids"]["test-natural"] = units[("rf", 1)]["test_ids"]["test-natural"][::-1]
    with pytest.raises(evaluate.MisalignedPair):
        evaluate.replication_metrics(units, tests, **REP_KW)


# --- the oracle reading ------------------------------------------------------------------

M = float(np.log(1.10))
CELLS4 = [("rf", "test-natural"), ("rf", "test-stratified"), ("gbm", "test-natural"), ("gbm", "test-stratified")]


def _cell(L_o, H_o, L_p, H_p, *, dec_o=(0.10,) * 5, dec_p=(0.10,) * 5, dec_b=(0.10,) * 5,
          dL=(0.0,) * 5, dH=(0.0,) * 5):
    """Lists over five realizations; scalars are broadcast. E_* are chosen so that
    ln(E_o/E_p) equals dL and dH exactly."""
    five = lambda x: [float(v) for v in (x if np.ndim(x) else [x] * 5)]
    E_p = [0.2] * 5
    return {"L_o": five(L_o), "H_o": five(H_o), "L_p": five(L_p), "H_p": five(H_p),
            "E_oL": [e * np.exp(d) for e, d in zip(E_p, five(dL))], "E_pL": E_p,
            "E_oH": [e * np.exp(d) for e, d in zip(E_p, five(dH))], "E_pH": E_p,
            "dec_o": five(dec_o), "dec_p": five(dec_p), "dec_b": five(dec_b)}


def _read(cells, rho=None):
    return evaluate.oracle_reading(dict(zip(CELLS4, cells)), rho or {"L": 1.0, "H": 1.0}, 0.95)


RELOC_P = dict(L_p=-0.1, H_p=0.1)                  # C-logP relocates: R_p = 0.2


def test_A1_with_and_without_attribution():
    tight = _read([_cell(-0.3, 0.2, **RELOC_P)] * 4)              # dR = 0.3, constant
    assert tight["row"] == "A1" and tight["attribution"] and tight["step_p"]["logp_gate"]
    wide = [_cell(-0.3, [0.2, 1.4, -0.9, 0.25, 0.2], **RELOC_P)] * 4   # median dR 0.3, interval spans 0
    out = _read(wide)
    assert out["row"] == "A1" and not out["attribution"]


def test_A2_is_the_opposite_direction():
    assert _read([_cell(-0.05, 0.02, L_p=-0.2, H_p=0.2)] * 4)["row"] == "A2"   # dR = 0.07 - 0.4


def test_A3_needs_the_interval_and_A4_takes_a_wide_one():
    assert _read([_cell(-0.1, 0.11, **RELOC_P)] * 4)["row"] == "A3"            # dR = 0.01, zero spread
    noisy = [_cell(-0.1, [0.4, -0.2, 0.12, -0.15, 0.38], **RELOC_P)] * 4       # median dR 0.02
    out = _read(noisy)
    assert out["row"] == "A4" and not out["attribution"]


def test_A5_is_the_c_logg_pattern():
    # gate passes in every cell; dR medians +0.12, -0.51, +0.14, -0.01
    # R_p = 0.6 and R_o = 0.6 + d, so dR = d; H_o = 0.55 + d stays > 0 even at d = -0.51
    cells = [_cell(-0.05, 0.55 + d, L_p=-0.3, H_p=0.3) for d in (0.12, -0.51, 0.14, -0.01)]
    out = _read(cells)
    assert out["oracle_gate"] and out["row"] == "A5"
    assert [round(c["dR"], 2) for c in out["cells"].values()] == [0.12, -0.51, 0.14, -0.01]


def test_B1_wins_whenever_c_logp_does_not_relocate():
    reversing = _read([_cell(0.2, 0.0, L_p=0.1, H_p=0.1)] * 4)                   # R_o = -0.2
    assert reversing["row"] == "B1" and not reversing["step_p"]["logp_gate"]


def test_B2_needs_the_interval_of_R_oracle():
    assert _read([_cell(0.1, 0.11, **RELOC_P)] * 4)["row"] == "B2"
    wide = _read([_cell(0.1, [0.4, -0.2, 0.12, -0.15, 0.38], **RELOC_P)] * 4)
    assert wide["row"] == "B4"


def test_attribution_needs_the_interval_on_the_side_of_the_reading():
    # B2 with C-oracle relocating MORE than C-logP: R_o = 0.05, R_p = 0.01, dR = +0.04 in
    # every realization, so the interval excludes 0, but above it. Not attributable.
    wrong_side = _read([_cell(0.0, 0.05, L_p=-0.005, H_p=0.005)] * 4)
    assert wrong_side["row"] == "B2" and not wrong_side["attribution"]
    # The same B2 with C-logP relocating: dR = 0.05 - 0.2 < 0 throughout. Attributable.
    right_side = _read([_cell(0.0, 0.05, **RELOC_P)] * 4)
    assert right_side["row"] == "B2" and right_side["attribution"]
    # A3, A4, A5, B1, B4, B5 never carry an attribution, whatever the intervals.
    assert not _read([_cell(-0.1, 0.11, **RELOC_P)] * 4)["attribution"]      # A3, dR = 0.01 exactly


def test_B3_reverses_and_B5_is_a_uniform_loss():
    assert _read([_cell(0.2, 0.0, **RELOC_P)] * 4)["row"] == "B3"
    assert _read([_cell(0.1, 0.5, **RELOC_P)] * 4)["row"] == "B5"                # R_o = 0.4 > 0: mixed


def test_group_c_is_ordered_and_c1_needs_all_three():
    win = dict(dec_o=[0.05] * 5, dec_b=[0.10] * 5)
    c1 = _read([_cell(-0.3, 0.2, **RELOC_P, dec_p=[0.10] * 5, **win)] * 4)
    assert c1["levels"] == {"C-oracle": 3, "C-logP": 0} and c1["c_row"] == "C1"
    c2 = _read([_cell(-0.3, 0.2, **RELOC_P, dec_p=[0.05] * 5, **win)] * 4)   # C-logP also level 3
    assert c2["c_row"] == "C2"
    c3 = _read([_cell(-0.3, 0.2, **RELOC_P)] * 4)
    assert c3["c_row"] == "C3" and c3["levels"]["C-oracle"] == 0


def test_levels():
    iv = lambda d, lo, hi: {"diff": d, "ci": [lo, hi]}
    assert evaluate.oracle_level([iv(-1, -2, -0.5)] * 4, 4) == 3
    assert evaluate.oracle_level([iv(-1, -2, -0.5)] + [iv(-1, -2, 1)] * 3, 4) == 2
    assert evaluate.oracle_level([iv(-1, -2, 1)] + [iv(1, -2, 3)] * 3, 4) == 1
    assert evaluate.oracle_level([iv(1, -2, 3)] * 4, 4) == 0
    assert evaluate.oracle_level([iv(-1, -2, -0.5)] * 3, 4) == 2        # an incomplete set never reaches 3


def test_density_boundaries_and_levels():
    at = evaluate.oracle_density(1.10, [-0.2, -0.15, -0.12, 0.01])
    assert at["predicted"] == "negative" and at["direction_met"] and at["magnitude_met"]
    below = evaluate.oracle_density(1 / 1.10, [0.05, 0.03, 0.02, -0.01])
    assert below["predicted"] == "positive" and below["direction_met"] and not below["magnitude_met"]
    inside = evaluate.oracle_density(1.05, [-0.3] * 4)
    assert inside["predicted"] is None and inside["direction_met"] is None


def test_oracle_block_in_replication_metrics():
    tests = {"test-natural": stub_test(400, 3), "test-stratified": stub_test(400, 4, name="test-stratified")}
    noise = {"B": (0.10, 0.10), "C-oracle": (0.05, 0.20), "C-logP": (0.075, 0.15),
             "C0-oracle": (0.2, 0.2), "C0-logP": (0.2, 0.2)}
    units = _zoned_units(tests, noise)
    for unit in units.values():   # L = [0.5, 1): 4 against 3; H = [1, 2): 2 against 2
        unit["arms"]["C-oracle"]["train_fos"] = np.array([0.6, 0.7, 0.8, 1.5, 3.0, 0.4, 7.0, 0.9, 1.2, 2.0])
        unit["arms"]["C-logP"]["train_fos"] = np.array([0.6, 0.7, 0.8, 1.5, 3.0, 0.4, 7.0, 2.5, 1.2, 2.0])
    out = evaluate.replication_metrics(units, tests, **REP_KW)
    oracle = out["oracle"]
    assert oracle is not None and oracle["row"] == "A1"          # dR = ln 4 - ln 2 = ln 2
    assert oracle["rho"]["L"] == pytest.approx(4 / 3)            # 4 points in [0.5, 1) against 3
    assert oracle["rho"]["H"] == pytest.approx(1.0)
    assert oracle["density"]["L"]["predicted"] == "negative" and oracle["density"]["H"]["predicted"] is None
    v = out["values"][0]
    assert len(v["train_zone_counts"]) == len(evaluate.zones(config.ZONE_EDGES))
    json.dumps(out)


def test_oracle_block_is_absent_without_the_arms():
    tests = {"test-natural": stub_test(80, 3), "test-stratified": stub_test(50, 4, name="test-stratified")}
    assert evaluate.replication_metrics(_fake_units(tests), tests, **REP_KW)["oracle"] is None


def test_oracle_decision_contrast_never_uses_welch(monkeypatch):
    sentinel = [7.0, 7.1, 7.2, 7.3, 7.4]
    real = evaluate.welch_interval

    def guard(a, b, level):
        # C-logP against B is Welch by design (first argument); only C-oracle against
        # C-logP, where C-logP is the second argument, must never reach it.
        assert list(b) != sentinel, "C-oracle vs C-logP went through Welch"
        return real(a, b, level)

    monkeypatch.setattr(evaluate, "welch_interval", guard)
    out = _read([_cell(-0.3, 0.2, **RELOC_P, dec_p=sentinel)] * 4)
    assert out["cells"]["rf/test-natural"]["dec_o_vs_p"]["df"] == 4
