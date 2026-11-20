import numpy as np
import pandas as pd

from sgs import benchmark, cli, config, curate


def _frame(n, rng, split, grouped):
    lo = np.array([config.DOMAIN[c][0] for c in curate.ID_COLUMNS])
    hi = np.array([config.DOMAIN[c][1] for c in curate.ID_COLUMNS])
    x = lo + rng.random((n, len(lo))) * (hi - lo)
    frame = pd.DataFrame(x, columns=list(curate.ID_COLUMNS))
    frame["FOS"] = np.exp(0.03 * frame["phi"] + 0.02 * frame["c"] - 0.05 * frame["H"] - 0.4)
    frame["simulation_id"] = [f"sim_{split}_{grouped}_{i}" for i in range(n)]
    splits = pd.DataFrame({"split": split, "split_grouped": grouped}, index=frame["simulation_id"])
    return frame.set_index("simulation_id"), splits


def _data():
    rng = np.random.default_rng(0)
    parts = [_frame(300, rng, "train", "train"), _frame(100, rng, "train", "test"),
             _frame(60, rng, "test-natural", "unassigned"), _frame(40, rng, "test-stratified", "unassigned")]
    return pd.concat([f for f, _ in parts]), pd.concat([s for _, s in parts])


def test_baseline_reports_every_test_split_by_zone():
    sims, splits = _data()
    out = benchmark.run_baseline(sims, splits)
    assert set(out["results"]["fixed"]["tests"]) == set(config.TEST_ROLES)   # absent splits skipped
    assert set(out["results"]["grouped"]["tests"]) == {"test"}
    assert out["results"]["fixed"]["n_train"] == 400 and out["results"]["grouped"]["n_train"] == 300
    for task in out["results"].values():
        for r in task["tests"].values():
            assert r["overall"]["log_rmse"] > 0
            assert len(r["zones"]) == len(config.ZONE_EDGES) + 1
            assert sum(z["n"] for z in r["zones"]) == r["overall"]["n"]


def test_nearest_neighbour_distances_and_one_nn():
    sims, splits = _data()
    out = benchmark.run_nearest_neighbour(sims, splits)
    for space in ("3d", "5d"):
        fixed = out["results"][space]["fixed"]
        assert fixed["n_train"] == 400
        q = fixed["train_to_train"]
        assert 0 < q["q05"] <= q["q50"] <= q["q95"]
        assert q["min"] <= q["q05"] and fixed["near_quantile_distance"] == q["q05"]
        for r in fixed["tests"].values():
            assert 0 <= r["share_within_near_quantile"] <= 1
            assert 0 <= r["share_within_min_distance"] <= r["share_within_near_quantile"]
            assert r["median_abs_log_fos_gap"]["all"] >= 0
            assert r["one_nn"]["overall"]["log_rmse"] > 0
    # a training point is its own nearest neighbour only when queried as a test point
    sims2, splits2 = sims.copy(), splits.copy()
    copy = sims2.loc[splits2["split"] == "train"].iloc[:5].copy()
    copy.index = [f"dup_{i}" for i in range(5)]
    sims2 = pd.concat([sims2, copy])
    splits2 = pd.concat([splits2, pd.DataFrame({"split": "test-natural", "split_grouped": "unassigned"},
                                               index=copy.index)])
    r = benchmark.run_nearest_neighbour(sims2, splits2)["results"]["5d"]["fixed"]["tests"]["test-natural"]
    assert r["test_to_train"]["q05"] == 0.0


def test_the_model_uses_its_declared_seed_and_one_job():
    params = benchmark.model().get_params()
    assert params["random_state"] == config.SEEDS["benchmark_baseline"]
    assert params["n_jobs"] == 1


def test_the_version_one_record_is_not_the_version_one_one_output():
    assert benchmark.output_path().parent != benchmark.output_dir()


def test_benchmark_command_parses():
    assert callable(cli.build_parser().parse_args(["benchmark"]).handler)
