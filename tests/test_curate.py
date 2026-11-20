import csv
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from sgs import cli, config, curate, data
from tests.conftest import REAL_DATA


def _row(values, *, fos="1.5", runtime="4.2", **membership):
    H, angle, phi, c, gamma = (repr(float(v)) for v in values)
    row = {column: "" for column in data.FINAL_COLUMNS}
    row.update({column: "1.0" for column in curate._FLOAT})
    row.update(H=H, angle=angle, phi=phi, c=c, gamma=gamma, FOS=fos, Runtime_s=runtime,
               At_Boundary="False", Search_Converged="True", N_Circles_Tested="10",
               N_Valid_Surfaces="9", N_Unsolved="0")
    row.update({"source_file": "f.csv", "source_row": 0, "origin": "pool", "set": "pool",
                "realization": None, "batch": None, "seed_index": None, "design": "uniform-iid",
                "is_copy": False})
    row.update(membership)
    row["simulation_id"] = curate.simulation_id(values)
    return row


def _rows(*rows):
    return pd.DataFrame(list(rows))


POOL = [(5.0, 30.0, 20.0, 10.0, 18.0), (20.0, 50.0, 35.0, 60.0, 15.0)]
NATURAL = (10.0, 40.0, 10.0, 40.0, 20.0)
STRATIFIED = (2.0, 15.0, 8.0, 3.0, 21.0)


def _tests():
    return [_row(NATURAL, set="test-natural", origin="fixed", design="lhs"),
            _row(STRATIFIED, set="test-stratified", origin="fixed", design="stratified-random")]


def test_simulation_id_follows_the_stated_recipe():
    values = (12.5, 33.0, 21.25, 7.0, 19.0)
    text = "|".join(f"{float(v):.7g}" for v in values)
    assert curate.simulation_id(values) == "sim_" + hashlib.sha256(text.encode()).hexdigest()[:16]
    # strings or floats, and a last-digit difference, give the same id; a real change gives another
    assert curate.simulation_id([repr(v) for v in values]) == curate.simulation_id(values)
    assert curate.simulation_id((12.5, 33.0, 21.25, 7.0, 19.000000000000004)) == curate.simulation_id(values)
    assert curate.simulation_id((12.5, 33.0, 21.25, 7.0, 19.0001)) != curate.simulation_id(values)


def test_an_input_at_a_rounding_boundary_is_refused():
    frame = pd.DataFrame({c: [1.0] for c in curate.INPUTS})
    frame["gamma"] = [19.000005]          # halfway between 19.00000 and 19.00001 at 7 digits
    with pytest.raises(curate.CurateError, match="rounding boundary"):
        curate.stable_ids(frame)


def test_copies_and_repeated_runs_collapse_to_one_simulation():
    rows = _rows(_row(POOL[0]), _row(POOL[1]),
                 _row(POOL[0], set="guided-oracle", origin="replication", is_copy=True, runtime="9.9"),
                 _row(POOL[1], set="guided-log", origin="replication", runtime="7.0"),
                 *_tests())
    tables = curate.build(rows, reruns={})
    assert len(tables["simulations"]) == 4
    assert len(tables["membership"]) == 6
    checks = tables["checks"]
    assert checks["copied_rows"] == 1
    assert [p["simulation_id"] for p in checks["repeated_solver_runs"]] == [curate.simulation_id(POOL[1])]
    kept = tables["simulations"].set_index("simulation_id").loc[curate.simulation_id(POOL[1])]
    assert kept["Runtime_s"] == "4.2"                    # the first row in canonical order
    splits = tables["splits"].set_index("simulation_id")["split"]
    assert splits[curate.simulation_id(POOL[0])] == "train"
    assert splits[curate.simulation_id(NATURAL)] == "test-natural"
    assert set(splits) <= set(curate.SPLITS) and splits.index.is_unique


def test_rows_of_one_simulation_must_agree_on_outputs():
    rows = _rows(_row(POOL[0]), _row(POOL[0], fos="1.6", set="guided-log", origin="replication"),
                 *_tests())
    with pytest.raises(curate.CurateError, match="disagree on"):
        curate.build(rows, reruns={})


def test_a_copy_without_a_solver_run_is_refused():
    rows = _rows(_row(POOL[0]), _row(POOL[1], set="guided-oracle", origin="replication", is_copy=True),
                 *_tests())
    with pytest.raises(curate.CurateError, match="no solver run"):
        curate.build(rows, reruns={})


def test_a_non_test_point_near_a_test_set_is_refused():
    near = tuple(np.array(NATURAL) + 1e-4)       # a distinct id, well inside MIN_DISTANCE
    rows = _rows(_row(POOL[0]), _row(near, set="guided-log", origin="replication"), *_tests())
    with pytest.raises(curate.CurateError, match="not disjoint"):
        curate.build(rows, reruns={})


def test_a_file_that_differs_from_the_recorded_hash_is_refused(root):
    path = root / "data" / "final" / "x.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame([_row(POOL[0])])[data.FINAL_COLUMNS].to_csv(path, index=False)
    entry = {"path": path, "origin": "pool", "set": "pool", "realization": None, "batch": None,
             "seed_index": None, "design": "uniform-iid", "is_copy": False, "sha256": "0" * 64}
    with pytest.raises(curate.CurateError, match="SHA-256"):
        curate.read_sources([entry])
    entry["sha256"] = data.sha256_file(path)
    assert len(curate.read_sources([entry])) == 1


def test_every_dictionary_entry_names_a_real_column():
    dictionary = curate.data_dictionary()
    rows = _rows(_row(POOL[0]), _row(POOL[1]), *_tests())
    tables = curate.build(rows, reruns={})
    for name in ("simulations", "membership", "splits"):
        assert dictionary.loc[dictionary["table"] == name, "column"].tolist() == list(tables[name].columns)


def test_written_deposit_round_trips(root):
    pytest.importorskip("pyarrow")
    rows = _rows(_row(POOL[0]), _row(POOL[1]), *_tests())
    tables = curate.build(rows, reruns={})
    entries = [{"path": root / "data" / "final" / "x.csv", "sha256": "0" * 64}]
    (curate.out_dir() / "data").mkdir(parents=True, exist_ok=True)
    (curate.out_dir() / ".DS_Store").write_bytes(b"x")            # a hidden file is not recorded
    (curate.out_dir() / "data" / ".DS_Store").write_bytes(b"x")
    provenance = curate.write(tables, entries, curate.out_dir())
    base = curate.out_dir()
    csv = pd.read_csv(base / "data" / "simulations.csv", float_precision="round_trip")
    ids = [curate.simulation_id(v) for v in csv[list(curate.ID_COLUMNS)].itertuples(index=False)]
    assert ids == csv["simulation_id"].tolist()          # full precision survives the CSV
    parquet = pd.read_parquet(base / "data" / "simulations.parquet")
    assert parquet["FOS"].dtype == float and parquet["At_Boundary"].dtype == bool
    assert parquet["simulation_id"].tolist() == csv["simulation_id"].tolist()
    recorded = {f["path"]: f["sha256"] for f in provenance["files"]}
    assert not any(path.split("/")[-1].startswith(".") for path in recorded)
    assert recorded["data/simulations.parquet"] == data.sha256_file(base / "data" / "simulations.parquet")
    assert json.loads((base / "metadata" / "provenance.json").read_text())["relation"] == "IsDerivedFrom"


def test_curate_command_parses():
    assert callable(cli.build_parser().parse_args(["curate"]).handler)


# --- on the published data, when present ---------------------------------------------------

@pytest.fixture(scope="module")
def published():
    if not (REAL_DATA / "run_metadata.json").exists() or not (REAL_DATA / "pool" / "final").exists():
        pytest.skip("the published data are not unpacked under data/")
    entries = curate.sources()
    return curate.build(curate.read_sources(entries))


def test_published_counts(published):
    assert len(published["membership"]) == 55_500
    assert len(published["simulations"]) == 43_637
    assert published["checks"]["copied_rows"] == 11_850
    assert published["checks"]["solver_runs"] == 43_650
    assert len(published["checks"]["repeated_solver_runs"]) == 13
    # version 1.0 is a part of version 1.1, unchanged in its counts
    old = published["membership"]["origin"] != "revision"
    assert old.sum() == 33_250
    assert published["membership"].loc[old, "simulation_id"].nunique() == 30_237
    assert (published["membership"].loc[old & published["membership"]["is_copy"]].shape[0]) == 3_000
    revision = published["membership"].loc[~old]
    assert (~revision["is_copy"]).sum() == 13_400       # the revision's solved analyses
    assert not set(revision.loc[~revision["is_copy"], "simulation_id"]) & set(
        published["membership"].loc[old, "simulation_id"])
    sets = published["membership"].groupby("simulation_id")["set"].agg(frozenset)
    repeated = [p["simulation_id"] for p in published["checks"]["repeated_solver_runs"]]
    # all within C-log / C-logG, which share one candidate pool by design
    assert all(sets[s] <= {"guided-log", "guided-log-g"} for s in repeated)


def test_published_coverage(published):
    sims = published["simulations"]["simulation_id"]
    assert sims.is_unique
    assert set(published["membership"]["simulation_id"]) == set(sims)
    assert published["splits"]["simulation_id"].tolist() == sims.tolist()
    assert published["splits"]["split"].value_counts().to_dict() == {
        "train": 22_000, "unassigned": 16_187, "test-natural-large": 5_000, "test-natural": 300,
        "test-stratified": 150}


def test_published_test_sets_are_disjoint_from_every_other_simulation(published):
    dis = published["checks"]["disjointness"]
    for test in curate.TEST_SPLITS:
        assert dis[f"non_test__{test}"]["n"] == 38_187
        assert dis[f"non_test__{test}"]["min_normalized_distance"] > config.MIN_DISTANCE
        assert dis[f"train__{test}"]["min_normalized_distance"] > config.MIN_DISTANCE
    for i, first in enumerate(curate.TEST_SPLITS):
        for second in curate.TEST_SPLITS[i + 1:]:
            assert dis[f"{first}__{second}"]["min_normalized_distance"] > config.MIN_DISTANCE


def test_published_ids_are_reproducible_from_the_written_csv(published, tmp_path):
    """The README's recipe, applied to the CSV as written and read with exact parsing."""
    published["simulations"].head(2000).to_csv(tmp_path / "s.csv", index=False)
    csv = pd.read_csv(tmp_path / "s.csv", float_precision="round_trip")
    ids = [curate.simulation_id(v) for v in csv[list(curate.ID_COLUMNS)].itertuples(index=False)]
    assert ids == csv["simulation_id"].tolist()


def test_published_ids_survive_every_way_of_reading_the_csv(published, tmp_path):
    pytest.importorskip("pyarrow")
    sims = published["simulations"]
    sims.to_csv(tmp_path / "s.csv", index=False)
    curate.typed(sims).to_parquet(tmp_path / "s.parquet", index=False)
    columns = list(curate.ID_COLUMNS)
    readers = {
        "pandas default": pd.read_csv(tmp_path / "s.csv"),
        "pandas round_trip": pd.read_csv(tmp_path / "s.csv", float_precision="round_trip"),
        "csv module": pd.DataFrame(list(csv.DictReader((tmp_path / "s.csv").open()))),
        "parquet": pd.read_parquet(tmp_path / "s.parquet"),
    }
    for name, frame in readers.items():
        ids = [curate.simulation_id(v) for v in frame[columns].itertuples(index=False)]
        assert ids == sims["simulation_id"].tolist(), name
    assert published["checks"]["id_stability"]["inputs_moved"] == 0


def test_pool_backed_designs_give_the_same_ids_as_their_final_rows():
    if not (REAL_DATA / "replications" / "designs").exists():
        pytest.skip("the published data are not unpacked under data/")
    columns = list(curate.ID_COLUMNS)
    for r in range(1, config.REP_REALIZATIONS + 1):
        for key in (f"guided_oracle_r{r}", f"guided_log_p_r{r}"):
            design = pd.read_csv(config.replication_path("designs", key), dtype=str)
            final = pd.read_csv(config.replication_path("final", key), dtype=str)
            assert ([curate.simulation_id(v) for v in design[columns].itertuples(index=False)]
                    == [curate.simulation_id(v) for v in final[columns].itertuples(index=False)]), key


def test_published_arm_table_follows_the_arm_registry(published):
    rows = {(origin, name): arms for origin, name, arms in curate.arm_rows(published["membership"])}
    assert rows[("fixed", "lhs-extension")] == "B"
    assert rows[("fixed", "guided-log")] == "C-log, C0-log"
    assert rows[("replication", "guided-oracle")] == "C-oracle, C0-oracle"
    assert "C0-oracle" not in rows[("fixed", "seed")]
    assert len(rows) == published["membership"][["origin", "set"]].drop_duplicates().shape[0]


# --- version 1.1: derived columns, the grouped split, the revision's sets -----------------

def test_tension_fields_parse_the_one_warning_xslope_writes():
    warnings = pd.Series(["", "base tension on 2 of 30 slices (worst N' = -3.3 at slice 30)",
                          "base tension on 1 of 31 slices (worst N' = -0.0 at slice 1)"])
    out = curate.tension_fields(warnings)
    assert out["Tension_Slices"].tolist() == ["0", "2", "1"]
    assert out["N_Slices"].tolist() == ["", "30", "31"]
    assert out["Min_Normal_Force"].tolist() == ["", "-3.3", "-0.0"]
    assert out["Min_Normal_Force_Slice"].tolist() == ["", "30", "1"]
    with pytest.raises(curate.CurateError, match="other than"):
        curate.tension_fields(pd.Series(["base tension on 2 of 30 slices; something else"]))


def _derived(rows, reruns):
    tables = curate.build(rows, reruns=reruns)
    return tables["simulations"].set_index("simulation_id"), tables["checks"]["derived_columns"]


def test_crest_entry_follows_the_centre_height():
    crest = _row(POOL[0], H="10.0", Center_Y="10.09")       # |10.09 - 10| / 10 < 0.01
    clear = _row(POOL[1], H="10.0", Center_Y="10.2")
    sims, summary = _derived(_rows(crest, clear, *_tests()), {})
    assert sims.loc[crest["simulation_id"], "Crest_Entry"] == "True"
    assert sims.loc[clear["simulation_id"], "Crest_Entry"] == "False"
    assert summary["crest_entry"]["n"] == 1


def test_a_rerun_fills_the_converged_value_and_leaves_fos_alone():
    flagged = _row(POOL[0], fos="2.5", Search_Converged="False", source_file="data/x.csv", source_row=3)
    other = _row(POOL[1], Search_Converged="False", source_file="data/x.csv", source_row=4)
    reruns = {("data/x.csv", 3): ("2.5", "2.49", 200)}
    sims, summary = _derived(_rows(flagged, other, *_tests()), reruns)
    row = sims.loc[flagged["simulation_id"]]
    assert (row["FOS"], row["FOS_converged"], row["Converged_Max_Iter"]) == ("2.5", "2.49", "200")
    assert sims.loc[other["simulation_id"], "FOS_converged"] == ""
    assert summary["non_converged"]["without_converged_value"] == ["data/x.csv:4"]
    with pytest.raises(curate.CurateError, match="not the recorded"):
        _derived(_rows(flagged, other, *_tests()), {("data/x.csv", 3): ("2.6", "2.49", 200)})
    converged = _row(POOL[0], fos="2.5", source_file="data/x.csv", source_row=3)
    with pytest.raises(curate.CurateError, match="not the recorded"):
        _derived(_rows(converged, *_tests()), reruns)


def test_dimensionless_bounds_follow_the_domain():
    d = config.DOMAIN
    corners = pd.DataFrame({"H": [d["H"][1], d["H"][0]], "angle": [d["angle"][0], d["angle"][1]],
                            "phi": [d["phi"][0], d["phi"][1]], "c": [d["c"][0], d["c"][1]],
                            "gamma": [d["gamma"][1], d["gamma"][0]]})
    assert np.allclose(curate.dimensionless_unit(corners), [[0, 0, 0], [1, 1, 1]])
    k = config.CURATED_GROUPED_CELLS
    assert curate.cell_ids(np.array([[0, 0, 0], [1, 1, 1]], dtype=float)).tolist() == [0, k ** 3 - 1]
    with pytest.raises(curate.CurateError, match="outside"):
        curate.cell_ids(np.array([[0, 0, 1.01]]))


def test_held_out_cells_are_deterministic_and_follow_their_seed_key(monkeypatch):
    first, second = curate.held_out_cells(), curate.held_out_cells()
    assert first.tolist() == second.tolist()
    n_cells = config.CURATED_GROUPED_CELLS ** 3
    assert len(first) == round(config.CURATED_GROUPED_SHARE * n_cells) == len(set(first.tolist()))
    assert first.min() >= 0 and first.max() < n_cells
    monkeypatch.setitem(config.SEEDS, "curated_grouped_cells", config.SEEDS["curated_grouped_cells"] + 1)
    assert curate.held_out_cells().tolist() != first.tolist()


def test_the_grouped_split_covers_the_pool_only_and_never_shares_a_cell():
    rng = np.random.default_rng(0)
    lo = np.array([config.DOMAIN[c][0] for c in curate.INPUTS])
    hi = np.array([config.DOMAIN[c][1] for c in curate.INPUTS])
    points = lo + rng.random((400, len(lo))) * (hi - lo)
    rows = _rows(*[_row(tuple(p)) for p in points],
                 _row(tuple(lo + 0.37 * (hi - lo)), set="guided-log", origin="replication"), *_tests())
    splits = curate.build(rows, reruns={})["splits"]
    pool = splits["split"] == "train"
    assert set(splits.loc[pool, "split_grouped"]) == {"train", "test"}
    assert (splits.loc[~pool, "split_grouped"] == "unassigned").all()
    held = set(curate.held_out_cells().tolist())
    test_cells = set(splits.loc[splits["split_grouped"] == "test", "cell_id"])
    train_cells = set(splits.loc[splits["split_grouped"] == "train", "cell_id"])
    assert test_cells <= held and not (train_cells & held) and not (test_cells & train_cells)


def test_a_subset_file_is_read_from_the_rows_it_names(root):
    source = [_row(POOL[0], set="src"), _row(POOL[1], set="src")]
    frame = pd.DataFrame(source)
    derived = frame.iloc[[1]][data.FINAL_COLUMNS].copy()
    derived.insert(0, "row_id", ["rev:src:1"])
    derived["H"] = repr(float(derived["H"].iat[0]) * (1 + 2e-16))     # a last-digit difference
    path = root / "derived.csv"
    derived.to_csv(path, index=False)
    entry = {"path": path, "subset_of": "src", "row_prefix": "rev:src:", "origin": "revision",
             "set": "sub", "realization": None, "batch": None, "seed_index": None,
             "design": "stratified-solved", "is_copy": True}
    picked = curate._subset_rows(entry, [frame])
    assert picked["H"].tolist() == [source[1]["H"]]                     # the source's own strings
    assert picked["source_row"].tolist() == [0] and picked["set"].tolist() == ["sub"]
    derived["FOS"] = "9.9"
    derived.to_csv(path, index=False)
    with pytest.raises(curate.CurateError, match="outputs differ"):
        curate._subset_rows(entry, [frame])


def test_published_derived_columns(published):
    summary = published["checks"]["derived_columns"]
    assert summary["crest_entry"]["n"] == 3_645
    assert summary["tension"]["n_with_tension"] == 33_560
    conv = summary["non_converged"]
    # 22 in docs/non-converged-check.md, 8 in docs/non-converged-check-revision/
    assert (conv["n"], conv["with_converged_value"]) == (30, 30)
    assert conv["max_relative_change"] < 7e-4
    assert conv["without_converged_value"] == []
    assert len(curate.revision_reruns()) == 8
    sims = published["simulations"]
    flagged = sims["Search_Converged"] == "False"
    assert ((sims["FOS_converged"] != "") <= flagged).all()


def test_published_grouped_split(published):
    grouped = published["checks"]["grouped_split"]
    assert grouped["n"] == {"train": 17_227, "test": 4_773, "unassigned": 21_637}
    assert grouped["pool_per_cell"]["empty"] == 0
    splits = published["splits"]
    assert ((splits["split_grouped"] != "unassigned") == (splits["split"] == "train")).all()


def test_published_revision_sets_follow_their_records(published):
    mem = published["membership"]
    revision = mem.loc[mem["origin"] == "revision"]
    counts = revision.groupby("set").size().to_dict()
    assert counts["seed-study-seed"] == counts["seed-study-lhs-extension"] == 1_200
    assert counts["test-natural-large"] == 5_000 and counts["test-stratified-large-source"] == 6_000
    assert counts["test-stratified-large"] == 450
    strat = revision.loc[revision["set"] == "test-stratified-large", "simulation_id"]
    source = revision.loc[revision["set"] == "test-stratified-large-source", "simulation_id"]
    assert set(strat) <= set(source)
    pool = set(mem.loc[mem["set"] == "pool", "simulation_id"])
    assert set(revision.loc[revision["design"] == "pool-selection", "simulation_id"]) <= pool
    seeds = revision.loc[revision["set"].str.startswith("seed-study"), "seed_index"]
    assert sorted(seeds.unique().tolist()) == [1, 2, 3, 4]


def _rerun_csv(path, rows):
    fields = ["file", "row", "max_iter", "fos", "converged", "stored_fos"]
    pd.DataFrame(rows, columns=fields).to_csv(path, index=False)
    return path


def test_revision_reruns_take_the_first_converged_limit(tmp_path):
    rows = [("f.csv", 3, 50, "1.000500", "False", "1.000500"), ("f.csv", 3, 100, "1.000400", "False", "1.000500"),
            ("f.csv", 3, 200, "1.000100", "True", "1.000500"), ("f.csv", 3, 400, "1.000100", "True", "1.000500"),
            ("g.csv", 1, 50, "2.000000", "False", "2.000000"), ("g.csv", 1, 400, "1.999000", "False", "2.000000")]
    out = curate.revision_reruns(_rerun_csv(tmp_path / "r.csv", rows))
    assert out == {("f.csv", 3): ("1.000500", "1.000100", 200)}       # g.csv never converged: left out


def test_revision_reruns_refuse_a_moving_or_unreproduced_value(tmp_path):
    moving = [("f.csv", 3, 50, "1.0005", "False", "1.0005"), ("f.csv", 3, 100, "1.0001", "True", "1.0005"),
              ("f.csv", 3, 200, "1.0000", "True", "1.0005")]
    with pytest.raises(curate.CurateError, match="changes after"):
        curate.revision_reruns(_rerun_csv(tmp_path / "a.csv", moving))
    unreproduced = [("f.csv", 3, 50, "1.0004", "False", "1.0005"), ("f.csv", 3, 100, "1.0001", "True", "1.0005")]
    with pytest.raises(curate.CurateError, match="does not reproduce"):
        curate.revision_reruns(_rerun_csv(tmp_path / "b.csv", unreproduced))


def test_tension_by_quartile_counts_shares_and_medians():
    n = 8
    frame = pd.DataFrame({"angle": np.arange(n, dtype=float), "c": np.arange(n, dtype=float)[::-1],
                          "H": np.ones(n) + np.arange(n),
                          "Tension_Slices": ["0", "1", "0", "2", "1", "1", "3", "2"],
                          "Min_Normal_Force": ["", "-1.0", "", "-3.0", "-2.0", "-4.0", "-6.0", "-8.0"]})
    out = curate.tension_by_quartile(frame)
    angle = out["angle"]
    assert [r["n"] for r in angle] == [2, 2, 2, 2]
    assert [r["share_with_tension"] for r in angle] == [0.5, 0.5, 1.0, 1.0]
    assert [r["median_min_normal_force_kN_per_m"] for r in angle] == [-1.0, -3.0, -3.0, -7.0]
    assert out["c"][0]["median_min_normal_force_kN_per_m"] == -7.0      # c runs the other way
