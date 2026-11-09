import numpy as np
import pandas as pd

from sgs import benchmark, cli, config, curate


def _frame(n, rng, split):
    lo = np.array([config.DOMAIN[c][0] for c in curate.ID_COLUMNS])
    hi = np.array([config.DOMAIN[c][1] for c in curate.ID_COLUMNS])
    x = lo + rng.random((n, len(lo))) * (hi - lo)
    frame = pd.DataFrame(x, columns=list(curate.ID_COLUMNS))
    frame["FOS"] = np.exp(0.03 * frame["phi"] + 0.02 * frame["c"] - 0.05 * frame["H"] - 0.4)
    frame["simulation_id"] = [f"sim_{split}_{i}" for i in range(n)]
    return frame.set_index("simulation_id"), pd.Series(split, index=frame["simulation_id"])


def test_baseline_reports_both_test_sets_by_zone():
    rng = np.random.default_rng(0)
    parts = [_frame(400, rng, "train"), _frame(60, rng, "test-natural"), _frame(40, rng, "test-stratified")]
    sims = pd.concat([f for f, _ in parts])
    splits = pd.concat([s for _, s in parts])
    out = benchmark.run_baseline(sims, splits)
    assert set(out["results"]) == set(config.TEST_ROLES)
    for r in out["results"].values():
        assert r["overall"]["log_rmse"] > 0
        assert len(r["zones"]) == len(config.ZONE_EDGES) + 1
        assert sum(z["n"] for z in r["zones"]) == r["overall"]["n"]
    assert out["model"]["n_train"] == 400


def test_the_model_uses_its_declared_seed_and_one_job():
    params = benchmark.model().get_params()
    assert params["random_state"] == config.SEEDS["benchmark_baseline"]
    assert params["n_jobs"] == 1


def test_benchmark_command_parses():
    assert callable(cli.build_parser().parse_args(["benchmark"]).handler)
