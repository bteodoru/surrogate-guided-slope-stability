"""The reference pool: uniform i.i.d. analyses over the domain, solved in chunks
the user launches, from which C-oracle and C-logP select.

Spec: docs/superpowers/specs/2026-09-23-oracle-reference-pool-design.md.

  sgs pool --chunks N --workers W   solve up to N more chunks; stop at the rule
  sgs pool --status                 read-only: counts per sub-pool and bin

The design is drawn once, in full, at POOL_CAP rows. Rows within MIN_DISTANCE
of the seed or of either test set are removed, and row i of what remains
belongs to sub-pool i % REP_REALIZATIONS + 1, independently of FOS. Rows beyond
the stopping point are never solved.
"""

from __future__ import annotations

import datetime as _dt
import platform

import numpy as np
import pandas as pd

from sgs import config, data, design, solver


class PoolError(RuntimeError):
    """The pool, its files or its stopping rule are not in the declared state."""


REFERENCE_ROLES = ("seed", *config.TEST_ROLES)


# --- the design -------------------------------------------------------------------------

def full_design() -> tuple[pd.DataFrame, int]:
    """The capped design after the disjointness filter, with pool_row and subpool,
    and the number of rows the filter removed. Deterministic; cheap to recompute."""
    n_sub = config.REP_REALIZATIONS
    if config.POOL_CHUNK % n_sub:
        raise PoolError(f"POOL_CHUNK={config.POOL_CHUNK} is not a multiple of the {n_sub} sub-pools")
    raw = design.uniform_pool(np.random.default_rng(config.SEEDS["pool_design"]), config.POOL_CAP)
    references = pd.concat([design.read_design(role)[list(config.PARAMS)] for role in REFERENCE_ROLES],
                           ignore_index=True)
    keep = design.nearest_distance(raw, references) > config.MIN_DISTANCE
    frame = raw.loc[keep].reset_index(drop=True)
    frame["pool_row"] = np.arange(len(frame))
    frame["subpool"] = frame["pool_row"] % n_sub + 1
    return frame, int((~keep).sum())


def n_chunks(frame: pd.DataFrame) -> int:
    return len(frame) // config.POOL_CHUNK


def chunk_design(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    """The rows of chunk k (1-based), in design order."""
    if not 1 <= k <= n_chunks(frame):
        raise PoolError(f"chunk {k} is outside 1..{n_chunks(frame)}")
    lo = (k - 1) * config.POOL_CHUNK
    return frame.iloc[lo:lo + config.POOL_CHUNK].reset_index(drop=True)


# --- solving -------------------------------------------------------------------------------

STAGES = ("designs", "results", "final")


def chunk_key(k: int) -> str:
    return f"c{k:03d}"


def chunk_paths(k: int) -> dict:
    return {stage: config.pool_path(stage, k) for stage in STAGES}


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _platform() -> dict:
    return {"system": platform.system(), "machine": platform.machine()}


def chunk_problem(frame: pd.DataFrame, k: int, previous: dict) -> str | None:
    """Why chunk k is not complete, or None. A recorded chunk whose design or solver
    settings differ from the declared ones is refused, not re-solved: the pool's
    design is declared and must not change once solving has started."""
    text = chunk_design(frame, k).to_csv(index=False)
    if previous and (previous.get("design_sha256") != data.text_sha256(text)
                     or previous.get("solver_fingerprint") != config.solver_fingerprint()):
        raise PoolError(f"{chunk_key(k)}: the stored chunk was solved from another design or under "
                        "other solver settings; the pool's design is declared and must not change")
    return data.stored_set_problem(chunk_paths(k), previous, config.POOL_CHUNK)


def _intact_results(paths: dict, chunk: pd.DataFrame, previous: dict) -> list[dict] | None:
    """The chunk's solved results, when they can be kept without solving again.

    Kept when the results file is the one recorded for the chunk. With no record, it is
    kept when it matches the design row for row: write_rows is atomic, so a results file
    on disk is whole, and one without a record is a run killed after solving and before
    recording, which leaves 48 min of analyses to keep. Anything else returns None."""
    path = paths["results"]
    if not path.exists() or not paths["designs"].exists():
        return None
    if paths["designs"].read_text() != chunk.to_csv(index=False):
        return None
    if previous:
        return solver.read_rows(path)[0] if previous.get("results_sha256") == data.sha256_file(path) else None
    rows, columns = solver.read_rows(path)
    if columns != data.RESULT_COLUMNS or len(rows) != len(chunk):
        return None
    try:
        solved = np.array([[float(row[p]) for p in config.PARAMS] for row in rows])
    except (KeyError, ValueError):
        return None
    return rows if np.allclose(solved, chunk[list(config.PARAMS)].to_numpy(float)) else None


def solve_chunk(frame: pd.DataFrame, k: int, *, workers: int, fn=None) -> dict:
    """Design, solve and repair chunk k; reuse it when complete.

    Solver output is protected (spec §4.2, and the plan's protection rules):
    - intact results are never solved again: a damaged or missing final is rebuilt
      from them by repair alone;
    - results are solved again only when they are missing, damaged or do not match
      the design; the old files are quarantined, never deleted, and only once the new
      rows exist, so an interrupted solve leaves the old files where they were;
    - a recorded chunk's three files are read-only.
    Nothing is recorded until the chunk is whole."""
    key, paths = chunk_key(k), chunk_paths(k)
    previous = data.read_metadata().get("pool", {}).get(key, {})
    reason = chunk_problem(frame, k, previous)
    if reason is None:
        print(f"[pool {key}] up to date")
        return previous
    chunk = chunk_design(frame, k)
    rows = _intact_results(paths, chunk, previous)
    if rows is not None:
        print(f"[pool {key}] {reason}; the solved results are intact, rebuilding the final from them "
              "(no new solve)", flush=True)
    else:
        if any(path.exists() for path in paths.values()):
            print(f"[pool {key}] incomplete ({reason}); solving it again", flush=True)
        text = chunk.to_csv(index=False)
        if not paths["designs"].exists() or paths["designs"].read_text() != text:
            data.quarantine(paths["designs"])
            data.write_text_atomic(paths["designs"], text)
        solved = solver.run_set(chunk[list(config.PARAMS)].to_numpy(float), workers=workers,
                                label=f"pool {key}", fn=fn)
        data.quarantine(paths["results"])              # only now: the new rows exist
        solver.write_rows(paths["results"], solved, data.RESULT_COLUMNS)
        rows, _ = solver.read_rows(paths["results"])
    repair = solver.repair_rows(rows)
    data.quarantine(paths["final"])
    solver.write_rows(paths["final"], rows, data.FINAL_COLUMNS)
    final = data.read_table(paths["final"])
    problems = data.frame_problems(final, config.POOL_CHUNK)
    if problems:
        raise PoolError(f"{key}: {'; '.join(problems)}")
    entry = {"chunk": k, "n": config.POOL_CHUNK, "repair": repair,
             "n_excluded": int((final["excluded"] != "").sum()),
             "non_converged": data.non_converged(final), "environment": data.environment(),
             "platform": _platform(), "solver_fingerprint": config.solver_fingerprint(),
             **{f"{name}_sha256": data.sha256_file(paths[stage]) for stage, name in
                (("designs", "design"), ("results", "results"), ("final", "final"))},
             "updated": _now()}
    data.record("pool", key, entry)
    for path in paths.values():
        data.protect(path)
    print(f"[pool {key}] repair: {repair}")
    return entry
