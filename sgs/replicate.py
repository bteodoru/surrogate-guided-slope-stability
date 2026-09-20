"""Replications (optional): new realizations of the points each strategy adds
to the fixed seed, fitted with common fixed hyper-parameters.

  sgs replicate   generate (solver) -> fit -> metrics, reusing what still matches

Realization r0 is the existing main-pipeline sets refitted the same way
(exploratory); r1..rN are new (confirmatory). The seed, both test sets, the
model seed and the hyper-parameters are fixed; every conclusion is conditional
on them.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

from sgs import config, data, design, solver


class ReplicationError(RuntimeError):
    """A stored replicated set does not match what the configuration produces."""


# --- generation ------------------------------------------------------------------------------

def rep_seed(strategy: str, realization: int, batch: int = 0) -> int:
    sequence = np.random.SeedSequence(config.SEEDS["replication_design"],
                                      spawn_key=(config.REP_STRATEGIES.index(strategy), realization, batch))
    return int(sequence.generate_state(1)[0])


def set_keys(realization: int) -> list[tuple[str, str, int]]:
    """(key, strategy, batch) of one realization, in generation order."""
    r = realization
    keys = [(f"lhs_r{r}", "lhs", 0), (f"guided_log_r{r}", "guided-log", 0)]
    keys += [(f"guided_seq_r{r}_b{b}", "guided-seq", b) for b in range(1, len(config.REP_SEQ_BATCHES) + 1)]
    return keys


def load_set(key: str) -> pd.DataFrame:
    return data.load_final(config.replication_path("final", key), f"rep:{key}")


def build_design(key: str, strategy: str, realization: int, batch: int) -> tuple[pd.DataFrame, dict]:
    """The design of one replicated set, and what produced it. Guided sets are
    disjoint from the seed, both test sets and, for a sequential batch, the
    realization's earlier batches."""
    seed = rep_seed(strategy, realization, batch)
    info = {"strategy": strategy, "realization": realization, "batch": batch, "seed": seed}
    references = {role: design.read_design(role) for role in ("seed", *config.TEST_ROLES)}
    if strategy == "lhs":
        unit, frame = design.lhs(config.REP_EXTENSION, seed)
        if not design.lhs_strata_complete(unit):
            raise design.DesignError(f"{key}: LHS stratification incomplete")
    else:
        earlier = [f"guided_seq_r{realization}_b{b}" for b in range(1, batch)] if strategy == "guided-seq" else []
        n = config.REP_SEQ_BATCHES[batch - 1] if strategy == "guided-seq" else config.REP_EXTENSION
        for k in earlier:
            references[k] = pd.read_csv(config.replication_path("designs", k))
        training = pd.concat([data.load_role("seed"), *(load_set(k) for k in earlier)], ignore_index=True)
        model, diagnostics = design.fit_screener(training)
        edges = design.bin_edges_log(config.N_BINS, config.SELECTION_RANGE)
        have = design.bin_counts(training["FOS"].to_numpy(float), edges)
        allocation = design.deficit_allocation(have, n)
        info["screener"] = {**diagnostics, "trained_on": ["seed", *earlier]}
        info["allocation"] = {"edges": edges.tolist(), "trained_counts": have.tolist(),
                              "added": allocation.tolist()}
        frame = design.guided_log_deficit(model, rng=np.random.default_rng(seed), allocation=allocation,
                                          n_pool=config.N_GUIDED_POOL, selection_range=config.SELECTION_RANGE)
    design.assert_disjoint(frame, references, config.MIN_DISTANCE)
    return frame, info


def generate_set(key: str, strategy: str, realization: int, batch: int, *, workers: int,
                 force: bool = False, fn=None) -> dict:
    """Design, solve and repair one replicated set; reuse it when its stored
    design is the one the configuration produces now."""
    frame, info = build_design(key, strategy, realization, batch)
    paths = {stage: config.replication_path(stage, key) for stage in ("designs", "results", "final")}
    text = frame.to_csv(index=False)
    previous = data.read_metadata().get("replications", {}).get(key, {})
    if paths["final"].exists() and not force:
        if (paths["designs"].exists() and paths["designs"].read_text() == text
                and previous.get("design_sha256") == data.sha256_file(paths["designs"])):
            print(f"[{key}] up to date")
            return previous
        raise ReplicationError(f"{key}: the stored set does not match the design the configuration "
                               "produces now; rerun with --force")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["designs"].write_text(text)
    rows = solver.run_set(frame[list(config.PARAMS)].to_numpy(float), workers=workers, label=key, fn=fn)
    solver.write_rows(paths["results"], rows, data.RESULT_COLUMNS)
    rows, _ = solver.read_rows(paths["results"])
    repair = solver.repair_rows(rows)
    solver.write_rows(paths["final"], rows, data.FINAL_COLUMNS)
    problems = data.frame_problems(data.read_table(paths["final"]), len(frame))
    if problems:
        raise ReplicationError(f"{key}: {'; '.join(problems)}")
    entry = {**info, "n": int(len(frame)), "repair": repair, "environment": data.environment(),
             **{f"{stage}_sha256": data.sha256_file(path) for stage, path in
                (("design", paths["designs"]), ("results", paths["results"]), ("final", paths["final"]))},
             "updated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    data.record("replications", key, entry)
    print(f"[{key}] repair: {repair}")
    return entry


def generate_all(*, workers: int, force: bool = False, fn=None) -> None:
    """Every set of every new realization, realization by realization, so an
    interrupted run leaves whole realizations behind."""
    for r in range(1, config.REP_REALIZATIONS + 1):
        for key, strategy, batch in set_keys(r):
            generate_set(key, strategy, r, batch, workers=workers, force=force, fn=fn)
