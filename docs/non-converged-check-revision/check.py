"""The revision's non-converged searches, solved again with larger iteration limits.

  PYTHONPATH=. .venv/bin/python docs/non-converged-check-revision/check.py

The same check as docs/non-converged-check.md, for the eight cases of the revision's
sets (`non_converged` in data/revision/run_metadata.json), which that note predates.
Each case is solved through sgs.solver.calculate_fos with only SOLVER.search_max_iter
replaced (50, 100, 200, 400), on the original model extent (none of the eight was
repaired). 50 must reproduce the stored row. Nothing in data/ is written; the result
goes to results.csv next to this script, and sgs curate reads the converged values from
there (curate.NON_CONVERGED_RERUNS_REVISION).
"""

from __future__ import annotations

import csv
import dataclasses
import json
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from sgs import config, solver

MAX_ITERS = (50, 100, 200, 400)
WORKERS = 2          # xslope: at most two solver processes
HERE = Path(__file__).parent


def cases() -> list[dict]:
    meta = json.loads(config.revision_metadata_path().read_text())["sets"]
    out = []
    for part, entry in meta.items():
        frame = pd.read_csv(config.revision_path("final", part), dtype=str, keep_default_na=False)
        for row in entry["non_converged"]:
            stored = frame.iloc[row]
            if stored["repair_variant"] or stored["Search_Converged"] != "False":
                raise SystemExit(f"{part}:{row}: not an unrepaired non-converged row")
            out.append({"file": f"data/revision/final/{part}.csv", "row": row,
                        **{p: stored[p] for p in config.PARAMS}, "stored_fos": stored["FOS"]})
    return out


def solve(task: tuple[dict, int]) -> dict:
    case, max_iter = task
    config.SOLVER = dataclasses.replace(config.SOLVER, search_max_iter=max_iter)
    r = solver.calculate_fos(*(float(case[p]) for p in ("angle", "phi", "c", "gamma", "H")))
    if r["error"]:
        raise RuntimeError(f"{case['file']}:{case['row']} at {max_iter}: {r['error']}")
    return {"file": case["file"], "row": case["row"], "max_iter": max_iter,
            "fos": f"{round(r['fos'], 6):.6f}", "converged": r["search_converged"],
            "n_circles_tested": r["n_circles_tested"], "warnings": r["warnings"],
            "runtime_s": round(r["runtime_s"], 3)}


def main() -> None:
    todo = cases()
    tasks = [(case, m) for case in todo for m in MAX_ITERS]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(solve, tasks))
    stored = {(c["file"], c["row"]): c["stored_fos"] for c in todo}
    for r in rows:
        r["stored_fos"] = stored[(r["file"], r["row"])]
        r["platform"] = f"{platform.system()} {platform.machine()}"
    with (HERE / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    bad = [r for r in rows if r["max_iter"] == 50 and float(r["fos"]) != float(r["stored_fos"])]
    for r in rows:
        print(f"{r['file']}:{r['row']} max_iter={r['max_iter']}: FOS {r['fos']} "
              f"(stored {r['stored_fos']}), converged={r['converged']}, {r['runtime_s']} s")
    if bad:
        sys.exit(f"{len(bad)} cases do not reproduce the stored FOS at max_iter = 50")


if __name__ == "__main__":
    main()
