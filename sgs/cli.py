"""sgs — the surrogate-guided sampling pipeline.

  sgs generate        design → solve → repair, for every role or one
  sgs fit             tuned searches with nested OOF residuals, cached
  sgs evaluate        metrics on the test sets (seconds)
  sgs report          booktabs tables and figures
  sgs learning-curve  optional; RF and GBM with fixed hyper-parameters
  sgs replicate       optional; replicated designs, fixed-HP fits, interval metrics
  sgs pool            optional; the reference pool, solved in chunks until its declared rule is met
  sgs structural      optional; descriptive density-and-error artifacts from the replications
  sgs curate          optional; the curated deposit (unique analyses, membership, splits), from data/
  sgs benchmark       optional; the curated deposit's FOS baseline, from data/curated/
  sgs all             generate, fit, evaluate, report
  sgs inspect         rerun one stored row and draw it with xslope's plots

Scientific parameters live in sgs/config.py; the command line only selects a
role and controls execution.
"""

from __future__ import annotations

import os

# One OpenMP thread per process: the grid searches already use every core
# (n_jobs=-1), and HistGradientBoosting's own threads on top of them
# oversubscribed the machine. Measured identical results, see CLAUDE.md.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import datetime as _dt
import platform
import sys

from sgs import config, data, design, solver

REFERENCE_PYTHON = "3.14.4"


class GenerateError(RuntimeError):
    """Existing outputs of a role do not match what would be produced now."""


def generate(role: str, *, workers: int = 1, force: bool = False) -> dict:
    """Design, solve and repair one role; reuse outputs that still match."""
    spec = config.ROLES[role]
    frame, info = design.build(role)
    design_path, results_path, final_path = (config.design_path(role), config.results_path(role),
                                             config.final_path(role))

    text = frame.to_csv(index=False)
    if design_path.exists() and design_path.read_text() != text and not force:
        raise GenerateError(f"{design_path.name} differs from the design the current configuration "
                            f"and data produce; rerun with --force to regenerate {role!r}")
    if not design_path.exists() or design_path.read_text() != text:
        design_path.parent.mkdir(parents=True, exist_ok=True)
        design_path.write_text(text)
    design_sha = data.sha256_file(design_path)
    fingerprint = config.solver_fingerprint()

    previous = data.read_metadata().get("datasets", {}).get(role, {})
    solved = False
    if results_path.exists() and not force:
        if not previous:
            raise GenerateError(f"there is no record of {results_path.name} for {role!r}; "
                                f"rerun with --force to solve it again")
        if previous.get("design_sha256") != design_sha:
            raise GenerateError(f"{results_path.name} was solved from a different design; "
                                f"rerun with --force to solve {role!r} again")
        if previous.get("solver_fingerprint") != fingerprint:
            raise GenerateError(f"{results_path.name} was solved under other solver settings, or "
                                f"before they were recorded; rerun with --force to solve {role!r} again")
        if previous.get("results_sha256") != data.sha256_file(results_path):
            raise GenerateError(f"{results_path.name} is not the file that was recorded; "
                                f"rerun with --force to solve {role!r} again")
        print(f"[{role}] results up to date")
    else:
        rows = solver.run_set(frame[list(config.PARAMS)].to_numpy(float), workers=workers, label=role)
        data.quarantine(results_path)
        solver.write_rows(results_path, rows, data.RESULT_COLUMNS)
        solved = True

    final_stale = (not final_path.exists()
                   or previous.get("final_sha256") != data.sha256_file(final_path))
    if solved or force or final_stale:
        rows, columns = solver.read_rows(results_path)
        if columns != data.RESULT_COLUMNS:
            raise GenerateError(f"{results_path.name} does not match the result schema")
        repair = solver.repair_rows(rows)
        data.quarantine(final_path)
        solver.write_rows(final_path, rows, data.FINAL_COLUMNS)
        print(f"[{role}] repair: {repair}")
    else:
        repair = previous["repair"]
        print(f"[{role}] final up to date")

    entry = {"n": spec.n, **info, "design_sha256": design_sha,
             "solver_fingerprint": fingerprint,
             "results_sha256": data.sha256_file(results_path),
             "final_sha256": data.sha256_file(final_path), "repair": repair,
             "non_converged": data.non_converged(data.read_table(final_path)),
             "environment": data.environment()}
    unchanged = previous and {k: v for k, v in previous.items() if k != "updated"} == entry
    entry["updated"] = (previous["updated"] if unchanged
                        else _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    data.record("datasets", role, entry)
    return entry


def _generate(args):
    roles = [args.role] if getattr(args, "role", None) else list(config.ROLE_ORDER)
    for role in roles:
        generate(role, workers=args.workers, force=args.force)


def _fit(args):
    from sgs import models
    models.fit_all(force=args.force)


def _evaluate(args):
    from sgs import evaluate
    evaluate.run()


def _report(args):
    from sgs import report
    report.run()


def _learning_curve(args):
    from sgs import evaluate, models
    models.learning_curve(force=args.force)
    evaluate.run_learning_curve()


def _replicate(args):
    from sgs import evaluate, replicate
    replicate.generate_all(workers=args.workers, force=args.force)
    replicate.fit_all(force=args.force)
    evaluate.run_replications()


def _pool(args):
    from sgs import pool
    if args.status:
        print(pool.format_status(pool.status()))
        return
    if args.workers > 2:
        # execution control only: warned, never refused
        print(f"warning: --workers {args.workers}; the standing preference for xslope is at most 2",
              file=sys.stderr)
    pool.run(chunks=args.chunks, workers=args.workers)


def _structural(args):
    from sgs import structural
    structural.run()


def _curate(args):
    from sgs import curate
    curate.run()


def _benchmark(args):
    from sgs import benchmark
    benchmark.run()


def _inspect(args):
    from sgs import inspection
    found = inspection.inspect_row(args.role, args.row, args.stage)
    print(f"[{found['role']} row {found['row']}, {found['stage']}] "
          + ", ".join(f"{k}={v:g}" for k, v in found["params"].items()))
    if found["repair_variant"]:
        print(f"  repair variant: {found['repair_variant']}")
    print(f"  FOS stored={found['fos_stored']}  recomputed={found['fos']:.6f}")
    print(f"  at boundary: {found['at_boundary']} {found['boundary_flags'] or ''}")
    for path in found["figures"].values():
        print(f"  {path}")


def _all(args):
    _generate(args)
    _fit(args)
    _evaluate(args)
    _report(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sgs", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, handler, help, *, role=False, workers=False, force=False, force_help=None):
        p = sub.add_parser(name, help=help)
        if role:
            p.add_argument("--role", choices=tuple(config.ROLES), default=None,
                           help="one role (default: all, in dependency order)")
        if workers:
            p.add_argument("--workers", type=int, default=1, help="solver processes")
        if force:
            p.add_argument("--force", action="store_true",
                           help=force_help or "regenerate outputs instead of reusing or refusing them")
        p.set_defaults(handler=handler)

    command("generate", _generate, "design, solve and repair the data sets", role=True, workers=True, force=True)
    command("fit", _fit, "tuned searches with nested OOF residuals", force=True)
    command("evaluate", _evaluate, "metrics on the test sets")
    command("report", _report, "booktabs tables and figures from outputs/metrics")
    command("learning-curve", _learning_curve, "optional: RF and GBM learning curves with fixed hyper-parameters",
            force=True)
    command("replicate", _replicate, "optional: replicated designs of the added points, fixed-HP fits, "
            "interval metrics", workers=True, force=True,
            force_help="regenerate outputs instead of reusing or refusing them; WARNING: this flag is "
                       "shared with generation, so it re-solves all replicated sets (hours of solver "
                       "time). An interrupted run resumes without it; use it only for a real "
                       "configuration mismatch")
    command("structural", _structural, "optional: descriptive density-and-error artifacts "
            "from the replications; reads committed data only")
    command("curate", _curate, "optional: the curated deposit in data/curated/, read from data/ "
            "and checked against run_metadata.json")
    command("benchmark", _benchmark, "optional: the curated deposit's FOS baseline (random forest, "
            "library defaults), written to docs/curated-benchmark/")
    p = sub.add_parser("pool", help="optional: solve the reference pool in chunks, until the declared "
                                    "stopping rule is met (then it solves nothing more)")
    p.add_argument("--chunks", type=int, default=1,
                   help=f"solve at most this many more chunks of {config.POOL_CHUNK} analyses")
    p.add_argument("--workers", type=int, default=1, help="solver processes")
    p.add_argument("--status", action="store_true",
                   help="read-only: counts per sub-pool and bin, platforms, and the rule's state")
    p.set_defaults(handler=_pool)
    p = sub.add_parser("inspect", help="rerun one stored row and draw it with xslope's plots")
    p.add_argument("--role", choices=tuple(config.ROLES), required=True)
    p.add_argument("--row", type=int, required=True, help="1-based row, in design order")
    p.add_argument("--stage", choices=("final", "results"), default="final",
                   help="final: on the row's repair variant; results: on the original domain")
    p.set_defaults(handler=_inspect)
    command("all", _all, "generate, fit, evaluate, report", workers=True, force=True)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "workers", 1) < 1:
        parser.error("--workers must be at least 1")
    if getattr(args, "chunks", 1) < 1:
        parser.error("--chunks must be at least 1")
    if platform.python_version() != REFERENCE_PYTHON:
        print(f"warning: running on Python {platform.python_version()}; "
              f"the reference environment is {REFERENCE_PYTHON}", file=sys.stderr)
    args.handler(args)


if __name__ == "__main__":
    main()
