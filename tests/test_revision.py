"""The revision's solved sets: design, disjointness, parts, resume, separate metadata."""

import pandas as pd
import pytest

from sgs import config, data, design, revision
from tests.helpers import fake_analyse, tamper


@pytest.fixture
def small(root, monkeypatch):
    monkeypatch.setattr(config, "REV_SEEDS", 1)
    monkeypatch.setattr(config, "REV_SEED_N", 20)
    monkeypatch.setattr(config, "REV_TEST_NATURAL_N", 25)
    monkeypatch.setattr(config, "REV_TEST_STRAT_SOURCE_N", 15)
    monkeypatch.setattr(config, "REV_SOLVE_CHUNK", 10)
    _, other = design.lhs(30, 1)
    return {"other": other}


def test_sets_are_disjoint_and_split_in_parts(small):
    designs = revision.build_designs(small)
    assert list(designs) == ["seed_s1", "extension_s1", "test_natural_large", "test_stratified_large_source"]
    frames = [f for f, _ in designs.values()]
    design.assert_disjoint(pd.concat(frames, ignore_index=True), small, config.MIN_DISTANCE)
    keys = [k for key, (f, _) in designs.items() for k, _ in revision.parts(key, f)]
    assert "seed_s1_p01" in keys and "test_natural_large_p03" in keys and "test_natural_large_p04" not in keys


def test_near_points_are_dropped_and_recorded(small):
    seed_frame, _ = revision.build_designs(small)["seed_s1"]
    refs = {"other": pd.concat([small["other"], seed_frame.iloc[[3]]], ignore_index=True)}
    frame, info = revision.build_designs(refs)["seed_s1"]
    assert len(frame) == config.REV_SEED_N - 1
    assert info["n_dropped"] == 1 and info["dropped"] == {"other": [3]}


def test_generate_writes_only_revision_metadata_and_resumes(small, capsys):
    revision.generate_all(workers=1, fn=fake_analyse, references=small)
    assert not config.metadata_path().exists()
    meta = data.read_metadata(config.revision_metadata_path())["sets"]
    assert len(meta) == 2 + 2 + 3 + 2
    assert len(revision.load_set("test_natural_large")) == config.REV_TEST_NATURAL_N
    capsys.readouterr()
    revision.generate_all(workers=1, fn=fake_analyse, references=small)
    assert capsys.readouterr().out.count("up to date") == len(meta)
    tamper(config.revision_path("final", "seed_s1_p01"), "broken")
    revision.generate_all(workers=1, fn=fake_analyse, references=small)
    assert "seed_s1_p01] incomplete" in capsys.readouterr().out


def test_status_reports_parts(small):
    rows = revision.status(small)
    assert len(rows) == 9 and not any(r["solved"] for r in rows)
