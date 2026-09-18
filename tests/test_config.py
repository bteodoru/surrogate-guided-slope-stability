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
             "bootstrap_resampling", "learning_curve_subsampling"}
    assert used == set(config.SEEDS)


def test_role_order_covers_roles_and_respects_dependencies():
    assert set(config.ROLE_ORDER) == set(config.ROLES)
    position = {role: i for i, role in enumerate(config.ROLE_ORDER)}
    for role, spec in config.ROLES.items():
        for earlier in spec.disjoint_from:
            assert position[earlier] < position[role]
        if spec.screener:
            assert position[config.SCREENER["trained_on"]] < position[role]


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
