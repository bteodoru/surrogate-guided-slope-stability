"""sgs — the surrogate-guided sampling pipeline.

  sgs generate        design → solve → repair, for every role or one
  sgs fit             tuned searches with nested OOF residuals, cached
  sgs evaluate        metrics on the test sets (seconds)
  sgs report          booktabs tables and figures
  sgs learning-curve  optional; RF and GBM with fixed hyper-parameters
  sgs all             generate, fit, evaluate, report

Scientific parameters live in sgs/config.py; the command line only selects a
role and controls execution.
"""

from __future__ import annotations

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

    previous = data.read_metadata().get("datasets", {}).get(role, {})
    solved = False
    if results_path.exists() and not force:
        if not previous:
            raise GenerateError(f"there is no record of {results_path.name} for {role!r}; "
                                f"rerun with --force to solve it again")
        if previous.get("design_sha256") != design_sha:
            raise GenerateError(f"{results_path.name} was solved from a different design; "
                                f"rerun with --force to solve {role!r} again")
        print(f"[{role}] results up to date")
    else:
        rows = solver.run_set(frame[list(config.PARAMS)].to_numpy(float), workers=workers, label=role)
        solver.write_rows(results_path, rows, data.RESULT_COLUMNS)
        solved = True

    if solved or force or not final_path.exists():
        rows, columns = solver.read_rows(results_path)
        if columns != data.RESULT_COLUMNS:
            raise GenerateError(f"{results_path.name} does not match the result schema")
        repair = solver.repair_rows(rows)
        solver.write_rows(final_path, rows, data.FINAL_COLUMNS)
        print(f"[{role}] repair: {repair}")
    else:
        repair = previous["repair"]
        print(f"[{role}] final up to date")

    entry = {"n": spec.n, **info, "design_sha256": design_sha,
             "results_sha256": data.sha256_file(results_path),
             "final_sha256": data.sha256_file(final_path), "repair": repair,
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


def _all(args):
    _generate(args)
    _fit(args)
    _evaluate(args)
    _report(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sgs", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, handler, help, *, role=False, workers=False, force=False):
        p = sub.add_parser(name, help=help)
        if role:
            p.add_argument("--role", choices=tuple(config.ROLES), default=None,
                           help="one role (default: all, in dependency order)")
        if workers:
            p.add_argument("--workers", type=int, default=1, help="solver processes")
        if force:
            p.add_argument("--force", action="store_true",
                           help="regenerate outputs instead of reusing or refusing them")
        p.set_defaults(handler=handler)

    command("generate", _generate, "design, solve and repair the data sets", role=True, workers=True, force=True)
    command("fit", _fit, "tuned searches with nested OOF residuals", force=True)
    command("evaluate", _evaluate, "metrics on the test sets")
    command("report", _report, "booktabs tables and figures from outputs/metrics")
    command("learning-curve", _learning_curve, "optional: RF and GBM learning curves with fixed hyper-parameters",
            force=True)
    command("all", _all, "generate, fit, evaluate, report", workers=True, force=True)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "workers", 1) < 1:
        parser.error("--workers must be at least 1")
    if platform.python_version() != REFERENCE_PYTHON:
        print(f"warning: running on Python {platform.python_version()}; "
              f"the reference environment is {REFERENCE_PYTHON}", file=sys.stderr)
    args.handler(args)


if __name__ == "__main__":
    main()
