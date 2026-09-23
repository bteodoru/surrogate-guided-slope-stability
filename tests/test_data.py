import json

import numpy as np
import pandas as pd
import pytest

from sgs import config, data
from tests.helpers import shrink_roles, synthetic_final, write_final


def test_schema_order():
    assert data.RESULT_COLUMNS[:6] == ["angle", "phi", "c", "gamma", "H", "FOS"]
    assert data.FINAL_COLUMNS == data.RESULT_COLUMNS + ["repair_variant", "excluded"]


def test_derived_feature():
    frame = synthetic_final(3, 0)
    out = data.add_features(frame)
    np.testing.assert_allclose(out["gamma_H_c"], frame["gamma"] * frame["H"] / frame["c"])
    assert "gamma_H_c" not in frame


def test_load_role_drops_excluded_and_keeps_file_row_ids(root):
    frame = synthetic_final(5, 1)
    frame.loc[1, "excluded"] = "unresolved"
    frame.loc[3, "FOS"] = np.nan
    frame.loc[3, "excluded"] = "solver_error"
    write_final("seed", frame)
    loaded = data.load_role("seed")
    assert list(loaded["row_id"]) == ["seed:0", "seed:2", "seed:4"]
    assert loaded.columns[-1] == "gamma_H_c"


def test_load_role_names_the_missing_step(root):
    with pytest.raises(FileNotFoundError, match="sgs generate"):
        data.load_role("seed")


def test_load_arm_concatenates_its_roles(root):
    write_final("seed", synthetic_final(4, 1))
    write_final("guided", synthetic_final(3, 2))
    arm = data.load_arm("C")
    assert len(arm) == 7
    assert arm["row_id"].iloc[-1] == "guided:2"


def test_load_test(root):
    write_final("test-natural", synthetic_final(4, 3))
    test = data.load_test("test-natural")
    assert list(test.X.columns) == list(config.FEATURES)
    assert test.ids.tolist() == [f"test-natural:{i}" for i in range(4)]
    assert test.y.shape == (4,)


def test_load_test_refuses_training_roles(root):
    with pytest.raises(ValueError):
        data.load_test("seed")


def test_is_true_reads_csv_booleans():
    assert data.is_true(pd.Series(["True", "false", "", " TRUE "])).tolist() == [True, False, False, True]


def test_normalize_maps_domain_to_unit_cube():
    corners = pd.DataFrame([{p: lo for p, (lo, _) in config.DOMAIN.items()},
                            {p: hi for p, (_, hi) in config.DOMAIN.items()}])
    np.testing.assert_allclose(data.normalize(corners), [[0] * 5, [1] * 5])


def _problem_names(frame, role="seed"):
    return {p.split(":")[0] for p in data.set_problems(frame, role)}


def test_clean_set_has_no_problems():
    assert data.set_problems(synthetic_final(300, 1), "seed") == []


@pytest.mark.parametrize("name, corrupt", [
    ("schema", lambda f: f.drop(columns="Error")),
    ("row_count", lambda f: f.iloc[:-1]),
    ("domain", lambda f: f.assign(angle=np.r_[70.0, f["angle"].to_numpy()[1:]])),
    ("exclusion_reasons", lambda f: f.assign(excluded=["odd"] + [""] * (len(f) - 1))),
    ("fos_present", lambda f: f.assign(FOS=np.r_[np.nan, f["FOS"].to_numpy()[1:]])),
    ("no_untreated_boundary", lambda f: f.assign(At_Boundary=["True"] + ["False"] * (len(f) - 1))),
    ("no_internal_duplicates", lambda f: f.assign(**{p: np.r_[f[p].iloc[1], f[p].to_numpy()[1:]]
                                                     for p in config.PARAMS})),
])
def test_each_violation_is_named(name, corrupt):
    assert name in _problem_names(corrupt(synthetic_final(300, 1)))


def test_excluded_rows_are_not_held_to_fos_or_boundary_rules():
    frame = synthetic_final(300, 1)
    frame.loc[0, ["FOS", "excluded"]] = [np.nan, "solver_error"]
    frame.loc[1, ["At_Boundary", "excluded"]] = ["True", "unresolved"]
    assert data.set_problems(frame, "seed") == []


def test_validate_sets_passes_and_catches_overlap(root, monkeypatch):
    shrink_roles(monkeypatch, {"seed": 50, "lhs-extension": 50})
    write_final("seed", synthetic_final(50, 1))
    write_final("lhs-extension", synthetic_final(50, 2))
    data.validate_sets(["seed", "lhs-extension"])
    write_final("lhs-extension", synthetic_final(50, 1))
    with pytest.raises(data.DataError, match="lhs-extension/seed: not disjoint"):
        data.validate_sets(["seed", "lhs-extension"])


def test_validate_sets_names_the_failing_role(root, monkeypatch):
    shrink_roles(monkeypatch, {"seed": 50})
    frame = synthetic_final(50, 1)
    frame.loc[0, "At_Boundary"] = "True"
    write_final("seed", frame)
    with pytest.raises(data.DataError, match="seed: no_untreated_boundary"):
        data.validate_sets(["seed"])


def test_record_merges_entries_and_serializes_numpy(root):
    data.record("datasets", "seed", {"n": np.int64(3), "x": np.array([1.0, 2.0])})
    data.record("fits", "rf__A__log-rmse", {"best_params": {"min_samples_leaf": 1}})
    meta = json.loads(config.metadata_path().read_text())
    assert meta["datasets"]["seed"] == {"n": 3, "x": [1.0, 2.0]}
    assert data.read_metadata()["fits"]["rf__A__log-rmse"]["best_params"] == {"min_samples_leaf": 1}


def _backup(root):
    path = config.metadata_path()
    return path.with_name(path.name + ".bak")


def test_record_writes_the_documented_bytes(root):
    data.record("datasets", "seed", {"n": 3})
    data.record("fits", "rf__A__log-rmse", {"ok": True})
    expected = json.dumps({"datasets": {"seed": {"n": 3}},
                           "fits": {"rf__A__log-rmse": {"ok": True}}},
                          indent=2, sort_keys=True) + "\n"
    assert config.metadata_path().read_bytes() == expected.encode()


def test_record_keeps_a_backup_of_the_previous_state(root):
    assert not _backup(root).exists()
    data.record("datasets", "seed", {"n": 3})
    assert not _backup(root).exists()          # nothing good to back up yet
    first = config.metadata_path().read_bytes()
    data.record("datasets", "lhs-extension", {"n": 5})
    assert _backup(root).read_bytes() == first
    assert json.loads(_backup(root).read_text()) == {"datasets": {"seed": {"n": 3}}}


@pytest.mark.parametrize("broken", ["", "{\"datasets\": {\"seed\": {\"n\": 3", "not json"])
def test_read_metadata_falls_back_to_the_backup(root, capsys, broken):
    data.record("datasets", "seed", {"n": 3})
    data.record("datasets", "lhs-extension", {"n": 5})
    good = json.loads(config.metadata_path().read_text())
    good.pop("datasets")["lhs-extension"]                      # the backup lacks the last entry
    config.metadata_path().write_text(broken)

    recovered = data.read_metadata()
    assert recovered == {"datasets": {"seed": {"n": 3}}}
    warning = capsys.readouterr().err
    assert "run_metadata.json" in warning and "run_metadata.json.bak" in warning

    data.record("datasets", "guided", {"n": 7})                # the next write repairs the file
    assert json.loads(config.metadata_path().read_text()) == {
        "datasets": {"seed": {"n": 3}, "guided": {"n": 7}}}
    assert json.loads(_backup(root).read_text()) == {"datasets": {"seed": {"n": 3}}}


def test_read_metadata_raises_a_recovery_hint_when_both_are_unreadable(root):
    data.record("datasets", "seed", {"n": 3})
    data.record("datasets", "lhs-extension", {"n": 5})
    config.metadata_path().write_text("")
    _backup(root).write_text("{oops")
    with pytest.raises(data.MetadataError, match="run_metadata.json.bak"):
        data.read_metadata()
    with pytest.raises(data.MetadataError, match="git checkout"):
        data.read_metadata()
    with pytest.raises(data.MetadataError, match="re-solve|replicate"):
        data.read_metadata()


def test_read_metadata_never_silently_starts_over(root):
    data.record("datasets", "seed", {"n": 3})
    config.metadata_path().write_text("")
    _backup(root).unlink(missing_ok=True)
    with pytest.raises(data.MetadataError):
        data.read_metadata()


def test_environment_lists_the_pinned_packages():
    env = data.environment()
    assert env["python"] and set(env["packages"]) >= {"numpy", "scikit-learn", "xslope"}


def test_frame_problems_checks_an_expected_size(tmp_path):
    frame = synthetic_final(30, 4)
    assert data.frame_problems(frame, 30) == []
    assert any(p.startswith("row_count") for p in data.frame_problems(frame, 31))


def test_load_final_prefixes_row_ids_and_adds_features(tmp_path):
    path = tmp_path / "x.csv"
    synthetic_final(12, 5).to_csv(path, index=False)
    frame = data.load_final(path, "rep:x")
    assert frame["row_id"].iloc[0] == "rep:x:0" and "gamma_H_c" in frame and len(frame) == 12
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        data.load_final(tmp_path / "missing.csv", "rep:y")


def test_validate_sets_skips_declared_alternatives(root, monkeypatch):
    shrink_roles(monkeypatch, {"seed": 40, "guided-log": 40, "lhs-extension": 40})
    frame = synthetic_final(40, 5)
    write_final("seed", synthetic_final(40, 6))
    write_final("guided-log", frame)
    write_final("lhs-extension", frame)          # a deliberate duplicate of guided-log
    monkeypatch.setattr(config, "ALTERNATIVES",
                        frozenset({frozenset({"guided-log", "lhs-extension"})}))
    data.validate_sets(("seed", "guided-log", "lhs-extension"))   # must not raise

    monkeypatch.setattr(config, "ALTERNATIVES", frozenset())
    with pytest.raises(data.DataError, match="not disjoint"):
        data.validate_sets(("seed", "guided-log", "lhs-extension"))


def test_alternatives_declares_the_two_guided_log_roles():
    assert frozenset({"guided-log", "guided-log-g"}) in config.ALTERNATIVES
