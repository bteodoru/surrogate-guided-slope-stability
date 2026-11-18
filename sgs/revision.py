"""The revision's analyses (docs/revision-declaration.md, tag revision-declared).

  sgs revision            design and solve the revision's sets, reusing what still matches
  sgs revision --status   read-only: what is solved

Additive only. Nothing here reads or writes a file of the published study except to
read it: the revision's sets live under ``data/revision/``, with their provenance in
``data/revision/run_metadata.json``; ``data/run_metadata.json`` is never written.

The solved sets (declaration §5 and §6) are LHS designs, one SEEDS key each, built in
the order of ``config.rev_solved_sets()``. Each is made disjoint from every existing
design (the roles, the replications and the solved reference pool) and from every
earlier revision set: a point within ``MIN_DISTANCE`` is dropped, not replaced, and
the drop is recorded. A set larger than ``REV_SOLVE_CHUNK`` is solved in parts, each
recorded on its own. The resume contract is replicate's: a part is reused only when
its record matches the design produced now, the solver fingerprint and the hashes of
its files on disk; otherwise it is solved again.
"""

from __future__ import annotations

import datetime as _dt
import math
import platform

import numpy as np
import pandas as pd

from sgs import config, data, design, solver

SECTION = "sets"


class RevisionError(RuntimeError):
    """A stored revision set does not match what the configuration produces."""


def _meta() -> dict:
    return data.read_metadata(config.revision_metadata_path())


def _record(key: str, entry: dict) -> None:
    data.record(SECTION, key, entry, path=config.revision_metadata_path())


# --- designs --------------------------------------------------------------------------------

def existing_references() -> dict[str, pd.DataFrame]:
    """Every design of the published study: the roles, the replicated sets and the
    solved chunks of the reference pool."""
    from sgs import pool
    params = list(config.PARAMS)
    refs = {f"role:{role}": design.read_design(role)[params] for role in config.ROLE_ORDER}
    rep_dir = config.replication_path("designs", "x").parent
    for path in sorted(rep_dir.glob("*.csv")):
        refs[f"rep:{path.stem}"] = pd.read_csv(path)[params]
    stop = pool.stop_record()
    if stop is None:
        raise RevisionError("the reference pool has no recorded stop; its rows cannot be referenced")
    frame, _ = pool.full_design()
    refs["pool"] = frame.iloc[:stop["k"] * config.POOL_CHUNK][params].reset_index(drop=True)
    return refs


def _drop_near(frame: pd.DataFrame, references: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    keep = np.ones(len(frame), dtype=bool)
    dropped = {}
    for name, reference in references.items():
        near = design.nearest_distance(frame, reference) <= config.MIN_DISTANCE
        if near.any():
            dropped[name] = np.flatnonzero(near).tolist()
        keep &= ~near
    return frame.loc[keep].reset_index(drop=True), dropped


def build_designs(references: dict[str, pd.DataFrame] | None = None) -> dict[str, tuple[pd.DataFrame, dict]]:
    """Every solved revision set's design, in generation order, with what produced it."""
    references = dict(existing_references() if references is None else references)
    out = {}
    for key, n, seed_key in config.rev_solved_sets():
        unit, frame = design.lhs(n, config.SEEDS[seed_key])
        if not design.lhs_strata_complete(unit):
            raise design.DesignError(f"{key}: LHS stratification incomplete")
        frame, dropped = _drop_near(frame, references)
        design.assert_disjoint(frame, references, config.MIN_DISTANCE)
        info = {"set": key, "n_design": n, "seed_key": seed_key, "seed": config.SEEDS[seed_key],
                "n_dropped": sum(len(v) for v in dropped.values()), "dropped": dropped}
        out[key] = (frame, info)
        references[f"rev:{key}"] = frame
    return out


def parts(key: str, frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """(part key, rows) of one set; a set no larger than REV_SOLVE_CHUNK is one part."""
    size = config.REV_SOLVE_CHUNK
    if len(frame) <= size:
        return [(key, frame)]
    count = math.ceil(len(frame) / size)
    return [(f"{key}_p{i + 1:02d}", frame.iloc[i * size:(i + 1) * size].reset_index(drop=True))
            for i in range(count)]


# --- solving --------------------------------------------------------------------------------

def _paths(key: str) -> dict:
    return {stage: config.revision_path(stage, key) for stage in ("designs", "results", "final")}


def solve_part(key: str, frame: pd.DataFrame, info: dict, *, workers: int, fn=None) -> dict:
    """Solve and repair one part, or reuse it when its record and files match."""
    paths = _paths(key)
    text = frame.to_csv(index=False)
    n = len(frame)
    fingerprint = config.solver_fingerprint()
    previous = _meta().get(SECTION, {}).get(key, {})
    if previous:
        if previous.get("design_sha256") != data.text_sha256(text):
            raise RevisionError(f"{key}: the stored part was solved from another design than the one "
                                "the configuration produces now")
        if previous.get("solver_fingerprint") != fingerprint:
            raise RevisionError(f"{key}: the stored part was solved under other solver settings")
        reason = data.stored_set_problem(paths, previous, n)
        if reason is None:
            print(f"[{key}] up to date")
            return previous
        print(f"[{key}] incomplete ({reason}); solving it again", flush=True)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        data.quarantine(path)
    data.write_text_atomic(paths["designs"], text)
    rows = solver.run_set(frame[list(config.PARAMS)].to_numpy(float), workers=workers, label=key, fn=fn)
    solver.write_rows(paths["results"], rows, data.RESULT_COLUMNS)
    rows, _ = solver.read_rows(paths["results"])
    repair = solver.repair_rows(rows)
    solver.write_rows(paths["final"], rows, data.FINAL_COLUMNS)
    problems = data.frame_problems(data.read_table(paths["final"]), n)
    if problems:
        raise RevisionError(f"{key}: {'; '.join(problems)}")
    entry = {**info, "part": key, "n": n, "repair": repair, "environment": data.environment(),
             "platform": {"system": platform.system(), "machine": platform.machine()},
             "solver_fingerprint": fingerprint,
             "non_converged": data.non_converged(data.read_table(paths["final"])),
             **{f"{name}_sha256": data.sha256_file(paths[stage]) for stage, name in
                (("designs", "design"), ("results", "results"), ("final", "final"))},
             "updated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    _record(key, entry)
    for path in paths.values():
        data.protect(path)
    print(f"[{key}] repair: {repair}")
    return entry


def generate_all(*, workers: int, fn=None, references=None) -> None:
    for key, (frame, info) in build_designs(references).items():
        for part_key, rows in parts(key, frame):
            solve_part(part_key, rows, info, workers=workers, fn=fn)


def load_set(key: str) -> pd.DataFrame:
    """A solved revision set, its parts concatenated in order, with row ids."""
    frames = []
    meta = _meta().get(SECTION, {})
    names = sorted(k for k in meta if k == key or k.startswith(f"{key}_p"))
    if not names:
        raise FileNotFoundError(f"revision set {key!r} is not solved; run `sgs revision`")
    for name in names:
        frames.append(data.read_table(config.revision_path("final", name)))
    frame = pd.concat(frames, ignore_index=True)
    return data.usable(frame).reset_index(drop=True)


def status(references=None) -> list[dict]:
    meta = _meta().get(SECTION, {})
    rows = []
    for key, (frame, info) in build_designs(references).items():
        for part_key, part in parts(key, frame):
            entry = meta.get(part_key)
            rows.append({"part": part_key, "n": len(part), "dropped": info["n_dropped"],
                         "solved": bool(entry) and entry.get("design_sha256") == data.text_sha256(
                             part.to_csv(index=False))})
    return rows


def format_status(rows: list[dict]) -> str:
    done = sum(r["solved"] for r in rows)
    lines = [f"revision sets: {done}/{len(rows)} parts solved"]
    lines += [f"  {r['part']:<36} n={r['n']:<5} dropped={r['dropped']:<3} "
              f"{'solved' if r['solved'] else 'pending'}" for r in rows]
    return "\n".join(lines)
