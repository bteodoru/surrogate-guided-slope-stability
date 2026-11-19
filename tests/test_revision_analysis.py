"""The revision's analysis modules (docs/revision-declaration.md §2-§9): bootstrap, robustness
weights and readings, levels, the Gaussian process's inputs, fit paths; and, on the real
outputs when present, that the published revision tables are reproduced from
outputs/metrics/revision.json and that T-stratL is the recorded draw."""

import json

import numpy as np
import pandas as pd
import pytest

from sgs import config, data
from sgs import revision_bootstrap as rb
from sgs import revision_fit, revision_gp, revision_large, revision_report, revision_retune
from sgs import revision_robust as rr
from sgs import revision_seeds
from tests.conftest import REAL_DATA

REAL_OUTPUTS = REAL_DATA.parent / "outputs"


# --- two-level bootstrap (§7) -------------------------------------------------------------

def test_bootstrap_resamples_test_points_within_zones():
    y = np.array([0.3, 0.4, 0.7, 0.8, 0.9, 1.5, 3.0, 3.5, 4.0, 7.0])
    w = rb.test_weights(np.random.default_rng(0), y, 200)
    zone = rb._zone_index(y)
    assert w.shape == (200, len(y))
    for z in np.unique(zone):                       # each zone keeps its size in every replicate
        assert (w[:, zone == z].sum(axis=1) == (zone == z).sum()).all()
    again = rb.test_weights(np.random.default_rng(0), y, 200)
    assert (w == again).all()


def test_bootstrap_zones_are_the_reporting_zones():
    assert rb._zone_index([0.49, 0.5, 0.99, 1.0, 1.99, 2.0, 5.99, 6.0]).tolist() == [0, 1, 1, 2, 2, 3, 3, 4]


# --- robustness checks (§8) ---------------------------------------------------------------

def test_regions_bin_as_the_generator_does():
    edges = np.array([0.5, 1.0, 2.0])
    fos = [0.4, 0.5, 0.7, 1.0, 1.2, 2.0, 2.1]
    # below | (0.5, 1.0] with 0.5 included | (1.0, 2.0] | above
    assert rr._regions(fos, edges).tolist() == [0, 1, 1, 1, 2, 2, 3]


def test_weights_bring_the_counts_to_the_target():
    edges = np.array([0.5, 1.0, 2.0])
    fos = np.array([0.4, 0.4, 0.7, 1.5, 1.6, 1.7, 3.0])
    target = np.array([1, 4, 2, 0])
    w, unweighted = rr.weights(fos, edges, target)
    reg = rr._regions(fos, edges)
    assert np.allclose(np.bincount(reg, weights=w, minlength=4)[:3], target[:3])
    assert unweighted == []
    _, unweighted = rr.weights(np.array([0.4, 1.5]), edges, target)
    assert unweighted == [1]                        # a target with no point to weight


def test_crest_entry_is_the_circle_centre_within_one_percent_of_h():
    frame = pd.DataFrame({"H": [10.0, 10.0, 10.0], "Center_Y": [10.05, 10.2, 9.95]})
    assert rr.crest_entry(frame).tolist() == [True, False, True]


def test_log_rmse_is_on_a_half_open_range():
    y = np.array([0.5, 1.0, 2.0])
    pred = np.array([0.5, np.e, 100.0])
    assert rr._log_rmse(pred, y, 0.5, 2.0) == pytest.approx(np.sqrt((0 + (1 - np.log(1.0)) ** 2) / 2))


def test_verdicts_agree_between_modules():
    for ci, v in (((-0.2, -0.1), "below 0"), ((0.1, 0.2), "above 0"), ((-0.1, 0.1), "contains 0")):
        assert rr._verdict(ci) == revision_large._verdict(ci) == v


def test_compare_reports_changed_verdicts_and_gates_only_for_gate_arms():
    def arm(verdict, gate, level):
        return {"cells": {"rf/test-natural": {"verdict": verdict, "gate": gate}}, "level": level, "gate_passes": int(gate)}
    base = {"C-log": arm("contains 0", True, 1), "C0-log": arm("above 0", False, 0)}
    other = {"C-log": arm("below 0", True, 2), "C0-log": arm("above 0", True, 0)}
    out = rr.compare(base, other)
    assert out["C-log"]["cells"]["rf/test-natural"] == {"verdict": "differs", "gate": "holds",
                                                        "base_verdict": "contains 0", "verdict_now": "below 0"}
    assert out["C-log"]["level"] == [1, 2]
    assert out["C0-log"]["cells"]["rf/test-natural"]["gate"] is None and out["C0-log"]["gate_passes"] is None


# --- levels (Methods, Table 4) --------------------------------------------------------------

@pytest.mark.parametrize("cells, level", [
    ([(-0.02, (-0.03, -0.01))] * 4, 3),
    ([(-0.02, (-0.03, -0.01))] + [(0.01, (-0.01, 0.03))] * 3, 2),
    ([(-0.01, (-0.03, 0.01))] + [(0.01, (-0.01, 0.03))] * 3, 1),
    ([(0.01, (-0.01, 0.03))] * 4, 0),
])
def test_report_levels(cells, level):
    assert revision_report._level(cells) == level


# --- the Gaussian process (§9) -------------------------------------------------------------

def test_gp_inputs_are_the_five_parameters_and_log_gamma_h_over_c():
    X = pd.DataFrame({**{p: [1.0, 2.0] for p in config.PARAMS}, "gamma_H_c": [np.e, np.e ** 2], "extra": [9, 9]})
    out = revision_gp._inputs(X)
    assert out.shape == (2, len(config.PARAMS) + 1)
    assert out[:, -1].tolist() == pytest.approx([1.0, 2.0])


def test_gp_estimator_is_the_declared_one():
    gp = revision_gp.estimator()[-1]
    assert gp.n_restarts_optimizer == 5 and gp.normalize_y
    assert gp.random_state == config.SEEDS["rev_gp_restarts"]
    assert gp.kernel.k1.k2.nu == 2.5 and len(gp.kernel.k1.k2.length_scale) == len(revision_gp.INPUTS)


# --- fit artifacts --------------------------------------------------------------------------

def test_fit_paths_are_distinct_and_under_artifacts():
    paths = [revision_fit.fit_path("rf", 1), revision_fit.fit_path("gbm", 1), revision_fit.fit_path("rf", 2),
             revision_seeds.fit_path("rf", 1), revision_large.fit_path("rf", 1), revision_gp.fit_path(1),
             revision_retune.fit_path("rf", "B", 1), revision_retune.fit_path("rf", "C-log", 1)]
    assert len(set(paths)) == len(paths)
    assert all("artifacts" in p.parts for p in paths)


# --- the real outputs, when present ---------------------------------------------------------

def _real_metrics():
    rev = REAL_OUTPUTS / "metrics" / "revision.json"
    rep = REAL_OUTPUTS / "metrics" / "replications.json"
    if not (rev.exists() and rep.exists()):
        pytest.skip("no real revision outputs")
    return json.loads(rep.read_text()), json.loads(rev.read_text())


@pytest.mark.parametrize("name, build", [
    ("revision__levels", revision_report.levels_table),
    ("revision__whole_selection", revision_report.whole_selection_table),
    ("revision__whole_selection_intervals", revision_report.whole_selection_intervals_table),
    ("revision__decision_intervals", revision_report.decision_intervals_table),
    ("revision__robustness", revision_report.robustness_table),
    ("revision__bias", revision_report.bias_table),
])
def test_published_revision_tables_are_reproduced_from_the_metrics(name, build):
    rep, rev = _real_metrics()
    assert build(rep, rev) == (REAL_OUTPUTS / "tables" / f"{name}.tex").read_text()


def test_large_stratified_test_set_is_the_recorded_draw():
    _, rev = _real_metrics()
    path = REAL_DATA / "revision" / "derived" / "test_stratified_large.csv"
    if not path.exists():
        pytest.skip("no real T-stratL")
    draw = rev["large_tests"]["stratified_draw"]
    assert data.sha256_file(path) == draw["sha256"]
    frame = pd.read_csv(path)
    assert len(frame) == draw["n"] == sum(draw["allocation"])
    assert frame["bin"].value_counts().max() <= config.REV_TEST_STRAT_PER_BIN
