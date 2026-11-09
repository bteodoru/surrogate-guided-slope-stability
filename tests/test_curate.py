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
                "realization": None, "batch": None, "design": "uniform-iid", "is_copy": False})
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
    text = "|".join(repr(float(v)) for v in values)
    assert curate.simulation_id(values) == "sim_" + hashlib.sha256(text.encode()).hexdigest()[:16]
    # the same inputs as strings or floats give the same id; any change gives another
    assert curate.simulation_id([repr(v) for v in values]) == curate.simulation_id(values)
    assert curate.simulation_id((12.5, 33.0, 21.25, 7.0, 19.000000000000004)) != curate.simulation_id(values)


def test_copies_and_repeated_runs_collapse_to_one_simulation():
    rows = _rows(_row(POOL[0]), _row(POOL[1]),
                 _row(POOL[0], set="guided-oracle", origin="replication", is_copy=True, runtime="9.9"),
                 _row(POOL[1], set="guided-log", origin="replication", runtime="7.0"),
                 *_tests())
    tables = curate.build(rows)
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
        curate.build(rows)


def test_a_copy_without_a_solver_run_is_refused():
    rows = _rows(_row(POOL[0]), _row(POOL[1], set="guided-oracle", origin="replication", is_copy=True),
                 *_tests())
    with pytest.raises(curate.CurateError, match="no solver run"):
        curate.build(rows)


def test_a_non_test_point_near_a_test_set_is_refused():
    near = tuple(np.array(NATURAL) + 1e-6)
    rows = _rows(_row(POOL[0]), _row(near, set="guided-log", origin="replication"), *_tests())
    with pytest.raises(curate.CurateError, match="not disjoint"):
        curate.build(rows)


def test_a_file_that_differs_from_the_recorded_hash_is_refused(root):
    path = root / "data" / "final" / "x.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame([_row(POOL[0])])[data.FINAL_COLUMNS].to_csv(path, index=False)
    entry = {"path": path, "origin": "pool", "set": "pool", "realization": None, "batch": None,
             "design": "uniform-iid", "is_copy": False, "sha256": "0" * 64}
    with pytest.raises(curate.CurateError, match="SHA-256"):
        curate.read_sources([entry])
    entry["sha256"] = data.sha256_file(path)
    assert len(curate.read_sources([entry])) == 1


def test_every_dictionary_entry_names_a_real_column():
    dictionary = curate.data_dictionary()
    rows = _rows(_row(POOL[0]), _row(POOL[1]), *_tests())
    tables = curate.build(rows)
    for name in ("simulations", "membership", "splits"):
        assert dictionary.loc[dictionary["table"] == name, "column"].tolist() == list(tables[name].columns)


def test_written_deposit_round_trips(root):
    pytest.importorskip("pyarrow")
    rows = _rows(_row(POOL[0]), _row(POOL[1]), *_tests())
    tables = curate.build(rows)
    entries = [{"path": root / "data" / "final" / "x.csv", "sha256": "0" * 64}]
    provenance = curate.write(tables, entries, curate.out_dir())
    base = curate.out_dir()
    csv = pd.read_csv(base / "data" / "simulations.csv", float_precision="round_trip")
    ids = [curate.simulation_id(v) for v in csv[list(curate.ID_COLUMNS)].itertuples(index=False)]
    assert ids == csv["simulation_id"].tolist()          # full precision survives the CSV
    parquet = pd.read_parquet(base / "data" / "simulations.parquet")
    assert parquet["FOS"].dtype == float and parquet["At_Boundary"].dtype == bool
    assert parquet["simulation_id"].tolist() == csv["simulation_id"].tolist()
    recorded = {f["path"]: f["sha256"] for f in provenance["files"]}
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
    assert len(published["membership"]) == 33_250
    assert len(published["simulations"]) == 30_237
    assert published["checks"]["copied_rows"] == 3_000
    assert published["checks"]["solver_runs"] == 30_250
    assert len(published["checks"]["repeated_solver_runs"]) == 13
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
        "train": 22_000, "unassigned": 7_787, "test-natural": 300, "test-stratified": 150}


def test_published_test_sets_are_disjoint_from_every_other_simulation(published):
    dis = published["checks"]["disjointness"]
    for test in config.TEST_ROLES:
        assert dis[f"non_test__{test}"]["n"] == 29_787
        assert dis[f"non_test__{test}"]["min_normalized_distance"] > config.MIN_DISTANCE
        assert dis[f"train__{test}"]["min_normalized_distance"] > config.MIN_DISTANCE


def test_published_ids_are_reproducible_from_the_written_csv(published, tmp_path):
    """The README's recipe, applied to the CSV as written and read with exact parsing."""
    published["simulations"].head(2000).to_csv(tmp_path / "s.csv", index=False)
    csv = pd.read_csv(tmp_path / "s.csv", float_precision="round_trip")
    ids = [curate.simulation_id(v) for v in csv[list(curate.ID_COLUMNS)].itertuples(index=False)]
    assert ids == csv["simulation_id"].tolist()
