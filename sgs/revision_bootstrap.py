"""The revision's two-level bootstrap (docs/revision-declaration.md §7).

For every arm against B, on the decision range and on its two halves, per learner and
test set: each replicate
  - resamples the five realizations with replacement, separately for each arm;
  - resamples the test points with replacement within each reporting zone, one draw
    per test set and replicate, common to every arm (so to both arms of a contrast).
The statistic is the arm's mean log-RMSE over its resampled realizations minus B's, the
quantity the Welch interval is about; the interval is the percentile one at REP_CI over
REV_BOOTSTRAP_N replicates. One generator (SEEDS["rev_bootstrap_two_level"]): the
realization draws first, then the test-point draws of each test set in a fixed order.

It is computed on the fixed test sets (the primary ones) and on the large ones, reported
beside the Welch interval, and replaces no declared reading. With five realizations the
realization level is coarse (at most 126 distinct multisets per arm), which is stated with
the interval. Results: the ``bootstrap`` entry of ``outputs/metrics/revision.json``.
"""

from __future__ import annotations

import numpy as np

from sgs import config, data, density, evaluate
from sgs import revision_fit as rf
from sgs import revision_large as rl

MEASURES = {"decision-range": (0.5, 2.0), "0.5–1": (0.5, 1.0), "1–2": (1.0, 2.0)}
CHUNK = 1_000
NOTE = ("Two-level percentile bootstrap: realizations resampled per arm (five, so the "
        "realization level is coarse) and test points resampled within reporting zones, "
        "common to both arms. Reported beside the Welch interval; no declared reading rests on it.")


def _zone_index(y) -> np.ndarray:
    return np.searchsorted(config.ZONE_EDGES, np.asarray(y, float), side="right")


def test_weights(rng, y, n_boot) -> np.ndarray:
    """(n_boot, n) counts: each point's multiplicity, resampled within its zone."""
    zone = _zone_index(y)
    w = np.zeros((n_boot, len(y)), dtype=np.float32)
    for z in np.unique(zone):
        idx = np.flatnonzero(zone == z)
        w[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=n_boot)
    return w


def _sets():
    fixed_units = rf.merged_units()
    fixed_tests = {role: data.load_test(role) for role in config.TEST_ROLES}
    return {"fixed": (fixed_units, fixed_tests), "large": (rl.units(), rl.large_tests())}


def run() -> dict:
    n_boot, reals = config.REV_BOOTSTRAP_N, list(range(1, config.REP_REALIZATIONS + 1))
    sets = _sets()
    arms = sorted(sets["fixed"][0][(config.REP_FAMILIES[0], 1)]["arms"])
    rng = np.random.default_rng(config.SEEDS["rev_bootstrap_two_level"])
    draws = {arm: rng.integers(0, len(reals), size=(n_boot, len(reals))) for arm in arms}
    weights = {(kind, name): test_weights(rng, t.y, n_boot)
               for kind, (_, tests) in sets.items() for name, t in tests.items()}
    rows = []
    for kind, (units, tests) in sets.items():
        for family in config.REP_FAMILIES:
            for name, test in tests.items():
                y = np.asarray(test.y, float)
                sq = {arm: np.stack([(np.log(np.asarray(units[(family, r)]["arms"][arm]["predictions"][name], float))
                                      - np.log(y)) ** 2 for r in reals]) for arm in arms}
                w = weights[(kind, name)]
                for measure, (lo, hi) in MEASURES.items():
                    m = (y >= lo) & (y < hi)
                    n_sub = float(m.sum())
                    # per arm: (n_boot, realizations) log-RMSE on the resampled points
                    rm = {}
                    for arm in arms:
                        out = np.empty((n_boot, len(reals)))
                        for s in range(0, n_boot, CHUNK):
                            out[s:s + CHUNK] = (w[s:s + CHUNK][:, m] @ sq[arm][:, m].T.astype(np.float32)) / n_sub
                        rm[arm] = np.sqrt(out)
                    observed = {arm: np.sqrt(sq[arm][:, m].mean(axis=1)) for arm in arms}
                    b_boot = np.take_along_axis(rm["B"], draws["B"], axis=1).mean(axis=1)
                    for arm in arms:
                        if arm == "B":
                            continue
                        a_boot = np.take_along_axis(rm[arm], draws[arm], axis=1).mean(axis=1)
                        d = a_boot - b_boot
                        alpha = (1 - config.REP_CI) / 2
                        welch = evaluate.welch_interval(observed[arm], observed["B"], config.REP_CI)
                        rows.append({"tests": kind, "family": family, "test": name, "measure": measure,
                                     "first": arm, "second": "B",
                                     "diff": float(observed[arm].mean() - observed["B"].mean()),
                                     "bootstrap_ci": [float(np.quantile(d, alpha)), float(np.quantile(d, 1 - alpha))],
                                     "welch_ci": [float(x) for x in welch["ci"]],
                                     "n_points": int(n_sub)})
    entry = {"replicates": n_boot, "level": config.REP_CI, "note": NOTE, "comparisons": rows}
    density.write("bootstrap", entry)
    return entry
