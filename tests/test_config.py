import dataclasses

import pytest

from sgs import config


def test_domain_order_matches_params():
    # Pools are drawn one column at a time in this order.
    assert tuple(config.DOMAIN) == config.PARAMS


def test_features_end_with_the_derived_feature():
    assert config.FEATURES[:-1] == ("H", "angle", "phi", "c", "gamma")
    assert config.FEATURES[-1] == "gamma_H_c"


def test_every_role_seed_is_declared():
    for role in config.ROLES.values():
        for name in role.seeds:
            assert name in config.SEEDS


def test_every_declared_seed_has_a_user():
    used = {s for role in config.ROLES.values() for s in role.seeds}
    used |= {"screener_fit", "screener_cv", "protocol_cv_split", "protocol_estimator",
             "bootstrap_resampling", "learning_curve_subsampling", "replication_design",
             "pool_design"}
    assert used == set(config.SEEDS)


def test_role_order_covers_roles_and_respects_dependencies():
    assert set(config.ROLE_ORDER) == set(config.ROLES)
    position = {role: i for i, role in enumerate(config.ROLE_ORDER)}
    for role, spec in config.ROLES.items():
        for earlier in spec.disjoint_from:
            assert position[earlier] < position[role]
        if spec.screener:
            for earlier in config.screener_roles(role):
                assert position[earlier] < position[role]


def test_design_sizes_match_role_sizes():
    assert config.ROLES["guided"].n == config.N_BINS * config.GUIDED_PER_BIN
    assert config.ROLES["test-stratified"].n == config.N_BINS * config.TEST_PER_BIN
    assert config.GUIDED_DRAWN_PER_BIN >= config.GUIDED_PER_BIN


def test_arms_are_compositions_of_training_roles():
    for arm in config.ARMS.values():
        assert set(arm.roles) <= set(config.ROLES)
        assert not set(arm.roles) & set(config.TEST_ROLES)


def test_ablation_arms_carry_no_budget():
    for arm in config.ARMS.values():
        assert (arm.budget is None) == arm.ablation
    assert config.ARMS["C0"].ablation


def test_comparisons_are_registered_consistently():
    for comp in config.COMPARISONS:
        assert comp.first in config.ARMS and comp.second in config.ARMS
        assert comp.alternative in ("less", "greater", "two-sided")
        assert comp.hypothesised == (comp.alternative != "two-sided")
    assert any("C0" in (c.first, c.second) for c in config.COMPARISONS)


def test_scorings_and_test_roles_are_declared():
    assert set(config.ACTIVE_SCORINGS) <= set(config.SCORINGS)
    assert set(config.TEST_ROLES) <= set(config.ROLES)


def test_zone_edges_are_increasing():
    edges = config.ZONE_EDGES
    assert list(edges) == sorted(edges) and len(set(edges)) == len(edges)


def test_learning_curve_strategies_extend_the_base_role():
    assert config.LC_BASE_ROLE in config.ROLES
    assert set(config.LC_STRATEGIES.values()) <= set(config.ROLES)


def test_fit_fingerprint_is_stable_and_per_family(monkeypatch):
    rf, gbm = config.fit_fingerprint("rf", "log-rmse"), config.fit_fingerprint("gbm", "log-rmse")
    assert len(rf) == 64 and rf == config.fit_fingerprint("rf", "log-rmse")
    assert rf != config.fit_fingerprint("rf", "fos-rmse")
    monkeypatch.setitem(config.GRIDS["gbm"], "max_iter", [300, 600, 900])
    assert config.fit_fingerprint("gbm", "log-rmse") != gbm
    assert config.fit_fingerprint("rf", "log-rmse") == rf


def test_evaluation_settings_do_not_enter_the_fit_fingerprint(monkeypatch):
    before = config.fit_fingerprint("rf", "log-rmse")
    monkeypatch.setattr(config, "ALPHA", 0.2)
    monkeypatch.setattr(config, "ZONE_EDGES", (1.0, 2.0))
    assert config.fit_fingerprint("rf", "log-rmse") == before


def test_learning_curve_fingerprint_follows_its_settings(monkeypatch):
    before = config.learning_curve_fingerprint("rf")
    monkeypatch.setattr(config, "LC_REPEATS", 3)
    assert config.learning_curve_fingerprint("rf") != before


def test_learning_curve_fingerprint_is_per_family(monkeypatch):
    rf, gbm = config.learning_curve_fingerprint("rf"), config.learning_curve_fingerprint("gbm")
    assert rf != gbm
    monkeypatch.setitem(config.LC_PARAMS, "gbm", {**config.LC_PARAMS["gbm"], "learning_rate": 0.1})
    assert config.learning_curve_fingerprint("gbm") != gbm
    assert config.learning_curve_fingerprint("rf") == rf


def test_learning_curve_params_are_grid_points():
    """Fixed LC hyper-parameters must be a point the tuning grid could select."""
    assert set(config.LC_FAMILIES) <= set(config.FAMILIES)
    for family in config.LC_FAMILIES:
        for name, value in config.LC_PARAMS[family].items():
            assert value in config.GRIDS[family][name], (family, name)


def test_paths_follow_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    assert config.final_path("guided") == tmp_path / "data" / "final" / "surrogate_guided_300.csv"
    assert config.fit_path("rf", "C0", "log-rmse") == tmp_path / "artifacts" / "fits" / "rf__C0__log-rmse.joblib"
    assert config.metadata_path() == tmp_path / "data" / "run_metadata.json"


def test_roles_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.ROLES["seed"].n = 1


def test_guided_log_role_avoids_every_set_declared_before_it():
    spec = config.ROLES["guided-log"]
    assert spec.design == "guided-log-deficit" and spec.screener
    before = config.ROLE_ORDER[:config.ROLE_ORDER.index("guided-log")]
    assert set(spec.disjoint_from) == set(before)


def test_log_arms_mirror_the_linear_ones():
    assert config.ARMS["C-log"] == config.Arm(roles=("seed", "guided-log"), budget=600, ablation=False)
    assert config.ARMS["C0-log"] == config.Arm(roles=("guided-log",), budget=None, ablation=True)
    pairs = {(c.first, c.second): c.alternative for c in config.COMPARISONS}
    assert pairs[("C-log", "B")] == "less" and pairs[("C-log", "A")] == "less"
    assert pairs[("C-log", "C")] == "two-sided"
    assert pairs[("C-log", "C0-log")] == "two-sided" and pairs[("C0-log", "B")] == "two-sided"


SEQ = ("guided-seq-1", "guided-seq-2", "guided-seq-3")


def test_sequential_batches_learn_from_every_earlier_set():
    for k, role in enumerate(SEQ):
        spec = config.ROLES[role]
        assert spec.n == 100 and spec.design == "guided-log-deficit" and spec.screener
        assert spec.trained_on == ("seed", *SEQ[:k])
        assert set(spec.disjoint_from) == set(config.ROLE_ORDER[:config.ROLE_ORDER.index(role)])
    assert sum(config.ROLES[r].n for r in SEQ) == config.ROLES["guided-log"].n


def test_sequential_arms_and_comparisons():
    assert config.ARMS["C-seq"] == config.Arm(roles=("seed", *SEQ), budget=600, ablation=False)
    assert config.ARMS["C0-seq"] == config.Arm(roles=SEQ, budget=None, ablation=True)
    new = {("C-seq", "B"), ("C-seq", "A"), ("C-seq", "C-log"), ("C-seq", "C0-seq"), ("C0-seq", "B")}
    pairs = {(c.first, c.second): c.alternative for c in config.COMPARISONS}
    assert new <= set(pairs) and all(pairs[p] == "two-sided" for p in new)


def test_the_gbm_comparisons_are_all_two_sided():
    entries = {(c.first, c.second): c for c in config.COMPARISONS}
    for pair in (("C-logG", "B"), ("C-logG", "A"), ("C-logG", "C-log"),
                 ("C-logG", "C0-logG"), ("C0-logG", "B")):
        assert entries[pair].alternative == "two-sided"   # post hoc choice of screener
        assert entries[pair].hypothesised is False


def test_the_gbm_arm_intervals():
    assert ("C-logG", "C-log") in config.REP_PAIRED       # shared pool and allocation
    assert ("C-logG", "C0-logG") in config.REP_PAIRED     # shared points
    assert ("C-logG", "B") in config.REP_WELCH
    assert ("C0-logG", "B") in config.REP_WELCH


def test_interaction_methods_are_declared():
    methods = {(i.first, i.second): i.method for i in config.REP_INTERACTIONS}
    assert methods[("C-log", "B")] == "welch"
    assert methods[("C-logG", "B")] == "welch"
    assert methods[("C-logG", "C-log")] == "paired"


def test_replication_declaration():
    assert config.REP_REALIZATIONS == 5 and config.REP_EXTENSION == 300
    assert config.REP_SEQ_BATCHES == (100, 100, 100) and sum(config.REP_SEQ_BATCHES) == config.REP_EXTENSION
    assert config.REP_STRATEGIES == ("lhs", "guided-log", "guided-seq", "guided-log-g",
                                     "guided-oracle", "guided-log-p")
    assert set(config.REP_FAMILIES) <= set(config.LC_FAMILIES)
    arms = {"A", "B", "C-log", "C0-log", "C-seq", "C0-seq", "C-logG", "C0-logG",
            "C-oracle", "C0-oracle", "C-logP", "C0-logP"}
    for pairs in (config.REP_WELCH, config.REP_PAIRED):
        assert all(first in arms and second in arms for first, second in pairs)
    assert all(i.first in arms and i.second in arms for i in config.REP_INTERACTIONS)


def test_replication_fingerprint_follows_its_settings(monkeypatch):
    before = config.replication_fingerprint("rf")
    assert before != config.replication_fingerprint("gbm")
    monkeypatch.setattr(config, "REP_EXTENSION", 20)
    assert config.replication_fingerprint("rf") != before


def test_replication_fingerprint_covers_the_stream_alias(monkeypatch):
    before = config.replication_fingerprint("rf")
    monkeypatch.setattr(config, "REP_SEED_STREAM", {})
    assert config.replication_fingerprint("rf") != before


def test_replication_fingerprint_covers_the_screener_registry(monkeypatch):
    before = config.replication_fingerprint("rf")
    monkeypatch.setitem(config.SCREENERS, "gbm", {**config.SCREENERS["gbm"], "max_iter": 4000})
    assert config.replication_fingerprint("rf") != before


def test_replication_paths_follow_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    assert config.replication_path("final", "lhs_r1") == tmp_path / "data" / "replications" / "final" / "lhs_r1.csv"
    assert config.replication_fit_path("gbm", "r2") == tmp_path / "artifacts" / "replications" / "gbm__r2.joblib"
    assert config.replication_metrics_path() == tmp_path / "outputs" / "metrics" / "replications.json"


def test_structural_constants():
    assert config.STRUCTURAL_MIN_TEST_N == 10
    assert config.structural_metrics_path().name == "structural.json"


def test_screener_registry_declares_both_families():
    assert set(config.SCREENERS) == {"rf", "gbm"}
    assert config.SCREENER["family"] == "rf"          # the study's primary configuration
    assert config.SCREENERS["gbm"]["early_stopping"] is False
    # copied from LC_PARAMS, never imported: the two must be free to diverge
    assert config.SCREENERS["gbm"]["max_iter"] == config.LC_PARAMS["gbm"]["max_iter"]


def test_existing_roles_do_not_declare_a_family():
    # guided-log-g intentionally declares gbm as its screener_family; all others use the default.
    assert all(spec.screener_family is None for name, spec in config.ROLES.items() if name != "guided-log-g")


def test_guided_log_g_reuses_the_c_log_stream_and_adds_no_seed():
    spec = config.ROLES["guided-log-g"]
    assert spec.seeds == ("guided_log_stream",)        # the same pool as C-log
    assert spec.screener_family == "gbm" and spec.screener is True
    assert spec.n == 300 and spec.design == "guided-log-deficit"
    assert spec.stem == "surrogate_guided_log_g_300"
    assert config.ROLE_ORDER[-1] == "guided-log-g"


def test_guided_log_g_is_disjoint_from_everything_except_c_log():
    spec = config.ROLES["guided-log-g"]
    earlier = [r for r in config.ROLE_ORDER if r != "guided-log-g"]
    assert set(spec.disjoint_from) == set(earlier) - {"guided-log"}


def test_the_gbm_screener_arms():
    assert config.ARMS["C-logG"].roles == ("seed", "guided-log-g")
    assert config.ARMS["C-logG"].budget == 600 and config.ARMS["C-logG"].ablation is False
    assert config.ARMS["C0-logG"].roles == ("guided-log-g",)
    assert config.ARMS["C0-logG"].budget is None and config.ARMS["C0-logG"].ablation is True


def test_the_gbm_arm_is_not_in_the_learning_curve():
    # Neither C-log nor C-logG is; the curve compares lhs-extension and guided.
    assert set(config.LC_STRATEGIES.values()) == {"lhs-extension", "guided"}


def test_pool_configuration():
    assert config.SEEDS["pool_design"] == 20260930
    assert config.POOL_CAP == 50_000 and config.POOL_CHUNK == 1_000
    assert config.POOL_CHUNK % config.REP_REALIZATIONS == 0
    assert config.REP_POOL_STRATEGIES == {"guided-oracle": "FOS", "guided-log-p": "FOS_pred"}
    assert config.ORACLE_DENSITY_RATIO == 1.10


def test_pool_strategies_are_appended_last():
    assert config.REP_STRATEGIES == ("lhs", "guided-log", "guided-seq", "guided-log-g",
                                     "guided-oracle", "guided-log-p")


def test_pool_path(root):
    assert config.pool_path("final", 7) == root / "data" / "pool" / "final" / "pool_c007.csv"


def test_oracle_intervals_are_declared():
    for pair in (("C-oracle", "B"), ("C-logP", "B"), ("C0-oracle", "B"), ("C0-logP", "B"),
                 ("C-logP", "C-log")):
        assert pair in config.REP_WELCH
    for pair in (("C-oracle", "C-logP"), ("C-oracle", "C0-oracle"), ("C-logP", "C0-logP")):
        assert pair in config.REP_PAIRED and pair not in config.REP_WELCH
    assert config.Interaction("C-oracle", "C-logP", method="paired") in config.REP_INTERACTIONS


def test_replication_fingerprint_covers_the_pool(monkeypatch):
    before = config.replication_fingerprint("rf")
    monkeypatch.setattr(config, "POOL_CHUNK", 500)
    assert config.replication_fingerprint("rf") != before
