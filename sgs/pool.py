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

    Kept when the results file is the one recorded for the chunk: the recorded hash
    proves it, whatever became of the design file, which is rewritten from the chunk.
    With no record, it is kept when the design file is the chunk's and the results
    match it row for row: write_rows is atomic, so a results file on disk is whole, and
    one without a record is a run killed after solving and before recording, which
    leaves 48 min of analyses to keep. Anything else returns None."""
    path = paths["results"]
    if not path.exists():
        return None
    if previous:
        return solver.read_rows(path)[0] if previous.get("results_sha256") == data.sha256_file(path) else None
    if not paths["designs"].exists() or paths["designs"].read_text() != chunk.to_csv(index=False):
        return None
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
      from them by repair alone, and a damaged or missing design from the chunk;
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
    text = chunk.to_csv(index=False)
    rows = _intact_results(paths, chunk, previous)
    if rows is not None:
        print(f"[pool {key}] {reason}; the solved results are intact, rebuilding the final from them "
              "(no new solve)", flush=True)
    elif any(path.exists() for path in paths.values()):
        print(f"[pool {key}] incomplete ({reason}); solving it again", flush=True)
    if not paths["designs"].exists() or paths["designs"].read_text() != text:
        data.quarantine(paths["designs"])              # a damaged design is kept, never deleted
        data.write_text_atomic(paths["designs"], text)
    if rows is None:
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


# --- the allocation and the screener, as C-log's ------------------------------------------

def _edges() -> np.ndarray:
    return design.bin_edges_log(config.N_BINS, config.SELECTION_RANGE)


def seed_counts() -> np.ndarray:
    """The seed's solved FOS per selection bin, as the generator bins it."""
    return design.bin_counts(data.load_role("seed")["FOS"].to_numpy(float), _edges())


def allocation() -> np.ndarray:
    """C-log's allocation: the seed water-filled to REP_EXTENSION points."""
    return design.deficit_allocation(seed_counts(), config.REP_EXTENSION)


def screener():
    """The screener C-log r1..r5 used: the primary family, trained on the seed alone."""
    return design.fit_screener(data.load_role("seed"), config.SCREENER["family"])


# --- eligibility and the rule --------------------------------------------------------------

def solved_prefix(frame: pd.DataFrame) -> int:
    """How many chunks, from the first, are complete."""
    recorded = data.read_metadata().get("pool", {})
    k = 0
    while k < n_chunks(frame) and chunk_problem(frame, k + 1, recorded.get(chunk_key(k + 1), {})) is None:
        k += 1
    return k


def eligible(frame: pd.DataFrame, k_max: int, model) -> pd.DataFrame:
    """Eligible rows of chunks 1..k_max (final present, not excluded), with their
    pool identity and the screener's FOS_pred. Both arms and the rule see only these."""
    parts = []
    for k in range(1, k_max + 1):
        chunk = chunk_design(frame, k)
        final = data.read_table(config.pool_path("final", k))
        if len(final) != len(chunk) or not np.allclose(final[list(config.PARAMS)].to_numpy(float),
                                                       chunk[list(config.PARAMS)].to_numpy(float)):
            raise PoolError(f"{chunk_key(k)}: the final is not aligned with the design row for row")
        final = final.assign(pool_row=chunk["pool_row"].to_numpy(), subpool=chunk["subpool"].to_numpy(),
                             chunk=k, chunk_row=np.arange(len(final)))
        parts.append(data.usable(final))
    rows = pd.concat(parts, ignore_index=True)
    rows["FOS"] = rows["FOS"].astype(float)
    rows["FOS_pred"] = design.predict_fos(model, rows)
    return rows


def rule_table(rows: pd.DataFrame, alloc) -> dict:
    """Solved and predicted counts per sub-pool and selection bin, (lo, hi] bins."""
    edges = _edges()
    solved, predicted = [], []
    for r in range(1, config.REP_REALIZATIONS + 1):
        sub = rows[rows["subpool"] == r]
        solved.append(design.bin_counts(sub["FOS"].to_numpy(float), edges).tolist())
        predicted.append(design.bin_counts(sub["FOS_pred"].to_numpy(float), edges).tolist())
    return {"allocation": np.asarray(alloc, dtype=int).tolist(), "solved": solved, "predicted": predicted}


def rule_met(table: dict) -> bool:
    a = np.asarray(table["allocation"], dtype=int)
    return bool((np.asarray(table["solved"]) >= a).all() and (np.asarray(table["predicted"]) >= a).all())


def stop_record() -> dict | None:
    return data.read_metadata().get("pool", {}).get("stop")


def verified_stop(frame: pd.DataFrame) -> dict:
    """The stop record, once the pool on disk is still the pool the stop describes.

    Chunks 1..k must be complete under their own metadata, and each final must be the
    one the stop recorded: a chunk re-solved or edited after the stop would otherwise
    change both selections without a trace. The screener that bins C-logP, and that
    counted the rule's predicted side, is platform-dependent, so the platform must be
    the one that evaluated the rule."""
    stop = stop_record()
    if stop is None:
        raise PoolError("the reference pool has not met its stopping rule; run `sgs pool` until it has")
    recorded = data.read_metadata().get("pool", {})
    for j in range(1, stop["k"] + 1):
        key = chunk_key(j)
        problem = chunk_problem(frame, j, recorded.get(key, {}))
        if problem is not None:
            raise PoolError(f"{key}: {problem}; the pool changed after the stop")
        if data.sha256_file(config.pool_path("final", j)) != stop["chunks_sha256"][key]:
            raise PoolError(f"{key}: the final is not the one the stop recorded")
    if stop["platform"] != _platform():
        raise PoolError(f"the stop was evaluated on {stop['platform']['system']}/{stop['platform']['machine']} "
                        f"and this is {_platform()['system']}/{_platform()['machine']}; the screener is "
                        "platform-dependent, so the pool-backed sets are generated only on the stop's platform")
    return stop


def _platforms(k: int) -> list[str]:
    recorded = data.read_metadata().get("pool", {})
    return sorted({f"{e['platform']['system']}/{e['platform']['machine']}"
                   for j in range(1, k + 1) if (e := recorded.get(chunk_key(j)))})


# --- status and the run loop ---------------------------------------------------------------

def status() -> dict:
    frame, n_filtered = full_design()
    k = solved_prefix(frame)
    table = None
    if k:
        model, _ = screener()
        table = rule_table(eligible(frame, k, model), allocation())
    return {"solved_chunks": k, "n_chunks": n_chunks(frame), "n_filtered": n_filtered,
            "table": table, "met": bool(table and rule_met(table)), "platforms": _platforms(k),
            "stop": stop_record()}


def format_status(st: dict) -> str:
    lines = [f"[pool] chunks solved: {st['solved_chunks']} of {st['n_chunks']} "
             f"({config.POOL_CHUNK} analyses each; {st['n_filtered']} design rows filtered)",
             f"[pool] platforms: {', '.join(st['platforms']) or 'none yet'}"]
    table = st["table"]
    if table is not None:
        a = np.asarray(table["allocation"])
        for r, (s, p) in enumerate(zip(table["solved"], table["predicted"]), start=1):
            s, p = np.asarray(s), np.asarray(p)
            short = [f"bin {b}: solved {s[b]}/{a[b]}, predicted {p[b]}/{a[b]}"
                     for b in range(len(a)) if s[b] < a[b] or p[b] < a[b]]
            lines.append(f"  sub-pool r{r}: bin 0 solved {s[0]}/{a[0]} predicted {p[0]}/{a[0]}; "
                         f"bin 1 solved {s[1]}/{a[1]} predicted {p[1]}/{a[1]}; "
                         + ("complete" if not short else "short: " + "; ".join(short)))
    if st["stop"]:
        lines.append(f"[pool] the stopping rule was met at chunk {st['stop']['k']}; nothing more to solve")
    else:
        lines.append("[pool] stopping rule " + ("met" if st["met"] else "not met"))
    return "\n".join(lines)


def run(*, chunks: int, workers: int, fn=None) -> dict | None:
    """Solve up to `chunks` more chunks, checking the rule after each; record the stop
    at the first chunk that meets it. Returns the stop record, or None if not met yet."""
    stop = stop_record()
    if stop:
        print(f"[pool] the stopping rule was met at chunk {stop['k']}; nothing to solve")
        return stop
    frame, n_filtered = full_design()
    model, _ = screener()
    alloc = allocation()
    k, done = solved_prefix(frame), 0
    while True:
        table = rule_table(eligible(frame, k, model), alloc) if k else None
        if table is not None and rule_met(table):
            recorded = data.read_metadata()["pool"]
            stop = {"k": k, "table": table, "n_filtered": n_filtered,
                    "chunks_sha256": {chunk_key(j): recorded[chunk_key(j)]["final_sha256"]
                                      for j in range(1, k + 1)},
                    "platforms": _platforms(k), "platform": _platform(),
                    "environment": data.environment(), "updated": _now()}
            data.record("pool", "stop", stop)
            print(f"[pool] the stopping rule is met at chunk {k}; solving stops for good")
            return stop
        if done == chunks:
            print(format_status({"solved_chunks": k, "n_chunks": n_chunks(frame), "n_filtered": n_filtered,
                                 "table": table, "met": False, "platforms": _platforms(k), "stop": None}))
            return None
        if k == n_chunks(frame):
            raise PoolError(f"the cap of {config.POOL_CAP} rows ({k} chunks) is reached and the stopping "
                            "rule is not met; it is not relaxed, a new decision is needed")
        solve_chunk(frame, k + 1, workers=workers, fn=fn)
        k, done = k + 1, done + 1
