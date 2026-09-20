import json

import pytest

from sgs import cli, config, data, solver
from tests.helpers import fake_analyse


def test_every_command_parses():
    parser = cli.build_parser()
    for argv in (["generate"], ["generate", "--role", "guided", "--workers", "4", "--force"],
                 ["fit", "--force"], ["evaluate"], ["report"], ["learning-curve", "--force"],
                 ["all", "--workers", "2"], ["inspect", "--role", "seed", "--row", "3"],
                 ["inspect", "--role", "guided", "--row", "1", "--stage", "results"]):
        assert callable(parser.parse_args(argv).handler)


def test_unknown_role_is_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["generate", "--role", "nope"])


def test_no_scientific_flags_exist():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["generate", "--seed", "1"])


def test_workers_must_be_positive():
    with pytest.raises(SystemExit):
        cli.main(["generate", "--role", "seed", "--workers", "0"])


@pytest.fixture
def fake_solver(root, monkeypatch):
    calls = []

    def counting(combo):
        calls.append(combo)
        return fake_analyse(combo)

    monkeypatch.setattr(solver, "analyse", counting)
    return calls


def test_generate_writes_three_files_and_records_them(fake_solver):
    entry = cli.generate("seed")
    for path in (config.design_path("seed"), config.results_path("seed"), config.final_path("seed")):
        assert path.exists()
    assert len(fake_solver) == 300
    assert list(data.read_table(config.final_path("seed")).columns) == data.FINAL_COLUMNS
    recorded = data.read_metadata()["datasets"]["seed"]
    assert recorded == entry
    assert recorded["design_sha256"] == data.sha256_file(config.design_path("seed"))
    assert recorded["seeds"] == {"seed_lhs_design": 42} and recorded["repair"]["n_boundary"] == 0


def test_generate_reuses_existing_outputs(fake_solver):
    cli.generate("seed")
    results = data.sha256_file(config.results_path("seed"))
    cli.generate("seed")
    assert len(fake_solver) == 300
    assert data.sha256_file(config.results_path("seed")) == results


def test_repair_is_redone_from_untouched_results(fake_solver):
    cli.generate("seed")
    results = data.sha256_file(config.results_path("seed"))
    config.final_path("seed").unlink()
    cli.generate("seed")
    assert len(fake_solver) == 300 and config.final_path("seed").exists()
    assert data.sha256_file(config.results_path("seed")) == results


def test_changed_design_is_refused_without_force(fake_solver, monkeypatch):
    cli.generate("seed")
    monkeypatch.setitem(config.SEEDS, "seed_lhs_design", 43)
    with pytest.raises(cli.GenerateError, match="differs"):
        cli.generate("seed")
    cli.generate("seed", force=True)
    assert len(fake_solver) == 600


def test_results_from_another_design_are_refused(fake_solver):
    cli.generate("seed")
    entry = data.read_metadata()["datasets"]["seed"]
    data.record("datasets", "seed", {**entry, "design_sha256": "0" * 64})
    with pytest.raises(cli.GenerateError, match="different design"):
        cli.generate("seed")


def test_learning_curve_fits_then_evaluates(monkeypatch):
    from sgs import evaluate, models

    calls = []
    monkeypatch.setattr(models, "learning_curve", lambda *, force: calls.append(("learning_curve", force)))
    monkeypatch.setattr(evaluate, "run_learning_curve", lambda: calls.append(("run_learning_curve",)))
    cli.main(["learning-curve", "--force"])
    assert calls == [("learning_curve", True), ("run_learning_curve",)]


def test_generate_is_idempotent_in_run_metadata(fake_solver, monkeypatch):
    cli.generate("seed")
    path = config.metadata_path()
    before = path.read_bytes()

    class _Later:
        @staticmethod
        def now(tz):
            import datetime as real_dt
            return real_dt.datetime.now(tz) + real_dt.timedelta(hours=1)

    monkeypatch.setattr(cli, "_dt", type("_dt", (), {"datetime": _Later, "timezone": cli._dt.timezone}))
    cli.generate("seed")
    after = path.read_bytes()
    assert before == after


def test_missing_metadata_entry_is_reported_as_no_record(fake_solver):
    cli.generate("seed")
    meta = data.read_metadata()
    del meta["datasets"]["seed"]
    path = config.metadata_path()
    path.write_text(json.dumps(meta))
    with pytest.raises(cli.GenerateError, match="no record"):
        cli.generate("seed")


def test_replicate_command_takes_only_execution_controls():
    from sgs import cli
    args = cli.build_parser().parse_args(["replicate", "--workers", "2", "--force"])
    assert args.workers == 2 and args.force and args.handler is cli._replicate
