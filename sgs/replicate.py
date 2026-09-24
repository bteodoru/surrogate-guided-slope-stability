"""Replications (optional): new realizations of the points each strategy adds
to the fixed seed, fitted with common fixed hyper-parameters.

  sgs replicate   generate (solver) -> fit -> metrics, reusing what still matches

Realization r0 is the existing main-pipeline sets refitted the same way
(exploratory); r1..rN are new (confirmatory). The seed, both test sets, the
model seed and the hyper-parameters are fixed; every conclusion is conditional
on them.

Resume contract (generation takes hours and may be killed at any moment):
a set is reused only when its metadata entry exists, records the design the
configuration produces now and the same solver fingerprint, its recorded
hashes match the design, results and final files on disk, and the final still
satisfies the data contract for its expected size. Any other state -- no
metadata entry, a missing, torn or stale file, a final that fails the contract
-- is an interrupted run: that one set is solved again, with a printed reason,
and its files are overwritten (the design is written atomically, so a kill
cannot tear it). Nothing else is touched and no ``--force`` is needed.
``--force`` is reserved for a real configuration mismatch, which is read from
the metadata, never from a file that may be damaged: a recorded design hash or
solver fingerprint that differs from what the configuration produces now.

A set that is complete but whose screener learned from files that have since
been solved again (the same design and the same FOS, other runtimes) keeps its
data and has only its provenance re-recorded.
"""

from __future__ import annotations

import datetime as _dt

import joblib

import numpy as np
import pandas as pd

from sgs import config, data, design, models, solver


class ReplicationError(RuntimeError):
    """A stored replicated set does not match what the configuration produces."""


# --- generation ------------------------------------------------------------------------------

def rep_seed(strategy: str, realization: int, batch: int = 0) -> int:
    stream = config.REP_SEED_STREAM.get(strategy, strategy)
    sequence = np.random.SeedSequence(config.SEEDS["replication_design"],
                                      spawn_key=(config.REP_STRATEGIES.index(stream), realization, batch))
    return int(sequence.generate_state(1)[0])


def strategy_family(strategy: str) -> str:
    return config.REP_SCREENER_FAMILY.get(strategy, config.SCREENER["family"])


def set_keys(realization: int) -> list[tuple[str, str, int]]:
    """(key, strategy, batch) of one realization, in generation order."""
    r = realization
    keys = [(f"lhs_r{r}", "lhs", 0), (f"guided_log_r{r}", "guided-log", 0),
            (f"guided_log_g_r{r}", "guided-log-g", 0)]
    keys += [(f"guided_seq_r{r}_b{b}", "guided-seq", b) for b in range(1, len(config.REP_SEQ_BATCHES) + 1)]
    keys += [(f"guided_oracle_r{r}", "guided-oracle", 0), (f"guided_log_p_r{r}", "guided-log-p", 0)]
    return keys


def keys_by_strategy(realization: int) -> dict[str, list[str]]:
    """Set keys of one realization grouped by strategy, in generation order. Arms are
    composed from this, never from positions in set_keys."""
    out: dict[str, list[str]] = {}
    for key, strategy, _ in set_keys(realization):
        out.setdefault(strategy, []).append(key)
    return out


def load_set(key: str) -> pd.DataFrame:
    return data.load_final(config.replication_path("final", key), f"rep:{key}")


def expected_n(strategy: str, batch: int) -> int:
    """How many points the set of one strategy adds."""
    return config.REP_SEQ_BATCHES[batch - 1] if strategy == "guided-seq" else config.REP_EXTENSION


def read_set_design(key: str) -> pd.DataFrame:
    path = config.replication_path("designs", key)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing; run `sgs replicate`")
    return pd.read_csv(path)


def build_design(key: str, strategy: str, realization: int, batch: int) -> tuple[pd.DataFrame, dict]:
    """The design of one replicated set, and what produced it. Guided sets are
    disjoint from the seed, both test sets and, for a sequential batch, the
    realization's earlier batches."""
    seed = rep_seed(strategy, realization, batch)
    n = expected_n(strategy, batch)
    info = {"strategy": strategy, "realization": realization, "batch": batch, "seed": seed}
    references = {role: design.read_design(role) for role in ("seed", *config.TEST_ROLES)}
    if strategy in config.REP_POOL_STRATEGIES:
        return _pool_design(key, strategy, realization, info, references)
    if strategy == "lhs":
        unit, frame = design.lhs(n, seed)
        if not design.lhs_strata_complete(unit):
            raise design.DesignError(f"{key}: LHS stratification incomplete")
    else:
        earlier = [f"guided_seq_r{realization}_b{b}" for b in range(1, batch)] if strategy == "guided-seq" else []
        for k in earlier:
            references[k] = read_set_design(k)
        training = pd.concat([data.load_role("seed"), *(load_set(k) for k in earlier)], ignore_index=True)
        family = strategy_family(strategy)
        model, diagnostics = design.fit_screener(training, family)
        edges = design.bin_edges_log(config.N_BINS, config.SELECTION_RANGE)
        have = design.bin_counts(training["FOS"].to_numpy(float), edges)
        allocation = design.deficit_allocation(have, n)
        # The solved files the screener learned from, as design._training_record
        # writes them for a role trained on more than one set.
        trained_paths = {"seed": config.final_path("seed"),
                         **{k: config.replication_path("final", k) for k in earlier}}
        info["screener"] = {**diagnostics, "family": family,
                            "params": dict(config.SCREENERS[family]),
                            "fit_seed": config.SEEDS["screener_fit"],
                            "trained_on": ["seed", *earlier],
                            "trained_on_sha256": {k: data.sha256_file(p)
                                                  for k, p in trained_paths.items()}}
        info["allocation"] = {"edges": edges.tolist(), "trained_counts": have.tolist(),
                              "added": allocation.tolist()}
        frame = design.guided_log_deficit(model, rng=np.random.default_rng(seed), allocation=allocation,
                                          n_pool=config.N_GUIDED_POOL, selection_range=config.SELECTION_RANGE)
    if len(frame) != n:
        raise design.DesignError(f"{key}: {len(frame)} points, expected {n}")
    design.assert_disjoint(frame, references, config.MIN_DISTANCE)
    return frame, info


def _pool_design(key: str, strategy: str, realization: int, info: dict,
                 references: dict) -> tuple[pd.DataFrame, dict]:
    """C-oracle or C-logP of one realization: C-log's allocation and within-bin draw,
    on the eligible rows of sub-pool `realization`, binned on the strategy's column
    (spec 2026-09-23 oracle, §5). No solver: the rows are already solved."""
    from sgs import pool
    frame, _ = pool.full_design()
    try:
        # the stop, the chunk files it recorded, and the platform, all checked here
        stop = pool.verified_stop(frame)
    except pool.PoolError as error:
        raise ReplicationError(f"{key}: {error}") from error
    model, diagnostics = pool.screener()
    rows = pool.eligible(frame, stop["k"], model)
    rows = rows[rows["subpool"] == realization].reset_index(drop=True)
    column = config.REP_POOL_STRATEGIES[strategy]
    edges = design.bin_edges_log(config.N_BINS, config.SELECTION_RANGE)
    added = pool.allocation()
    index, bins = design.select_in_bins(rows[column].to_numpy(float), edges, added,
                                        np.random.default_rng(info["seed"]))
    chosen = rows.iloc[index].reset_index(drop=True)
    out = chosen[[*config.PARAMS, "FOS_pred"]].copy()
    out["bin"] = bins
    out["pool_chunk"] = chosen["chunk"].to_numpy()
    out["pool_chunk_row"] = chosen["chunk_row"].to_numpy()
    out["pool_row"] = chosen["pool_row"].to_numpy()
    family = config.SCREENER["family"]
    info["screener"] = {**diagnostics, "family": family, "params": dict(config.SCREENERS[family]),
                        "fit_seed": config.SEEDS["screener_fit"], "trained_on": ["seed"],
                        "trained_on_sha256": {"seed": data.sha256_file(config.final_path("seed"))}}
    info["allocation"] = {"edges": edges.tolist(), "trained_counts": pool.seed_counts().tolist(),
                          "added": added.tolist()}
    info["pool"] = {"stop_k": stop["k"], "subpool": realization, "column": column,
                    "chunks_sha256": stop["chunks_sha256"], "platform": stop["platform"]}
    if len(out) != expected_n(strategy, 0):
        raise design.DesignError(f"{key}: {len(out)} points, expected {expected_n(strategy, 0)}")
    design.assert_disjoint(out, references, config.MIN_DISTANCE)   # the pool filter already ensures it
    return out, info


def _reuse(key: str, previous: dict, info: dict) -> dict:
    """A complete, matching set. When the files its screener learned from were
    solved again (same design, same FOS, other runtimes), only its provenance is
    stale: re-record it, and keep the data. When its screener provenance lacks keys
    the current code records (an entry written before they existed), backfill them
    the same way, without solving anything."""
    old_screener, new_screener = previous.get("screener", {}), info.get("screener", {})
    stored = old_screener.get("trained_on_sha256")
    fresh = new_screener.get("trained_on_sha256")
    if fresh is None or stored == fresh:
        missing = sorted(set(new_screener) - set(old_screener)) if fresh is not None else []
        if not missing:
            print(f"[{key}] up to date")
            return previous
        entry = {**previous, "screener": {**old_screener, **{k: new_screener[k] for k in missing}},
                 "updated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
        data.record("replications", key, entry)
        print(f"[{key}] up to date; screener provenance backfilled ({', '.join(missing)})")
        return entry
    entry = {**previous, **info,
             "updated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    data.record("replications", key, entry)
    print(f"[{key}] up to date; its training files were solved again, screener provenance re-recorded")
    return entry


def generate_set(key: str, strategy: str, realization: int, batch: int, *, workers: int,
                 force: bool = False, fn=None) -> dict:
    """Design, solve and repair one replicated set; reuse it when it is complete
    and matches the configuration, and solve it again when an interrupted run
    left it half-written (see the resume contract at the top of the module)."""
    if strategy in config.REP_POOL_STRATEGIES:
        return _copy_from_pool(key, strategy, realization, batch, force=force)
    frame, info = build_design(key, strategy, realization, batch)
    paths = {stage: config.replication_path(stage, key) for stage in ("designs", "results", "final")}
    text = frame.to_csv(index=False)
    n = expected_n(strategy, batch)
    fingerprint = config.solver_fingerprint()
    previous = data.read_metadata().get("replications", {}).get(key, {})
    if not force:
        if previous and previous.get("design_sha256") != data.text_sha256(text):
            raise ReplicationError(f"{key}: the stored set was solved from another design than the "
                                   "one the configuration produces now; rerun with --force")
        if previous and previous.get("solver_fingerprint") != fingerprint:
            raise ReplicationError(f"{key}: the stored set was solved under other solver settings; "
                                   "rerun with --force")
        reason = data.stored_set_problem(paths, previous, n)
        if reason is None:
            return _reuse(key, previous, info)
        if any(path.exists() for path in paths.values()):
            print(f"[{key}] incomplete ({reason}); solving it again", flush=True)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    data.write_text_atomic(paths["designs"], text)
    rows = solver.run_set(frame[list(config.PARAMS)].to_numpy(float), workers=workers, label=key, fn=fn)
    data.quarantine(paths["results"])
    solver.write_rows(paths["results"], rows, data.RESULT_COLUMNS)
    rows, _ = solver.read_rows(paths["results"])
    repair = solver.repair_rows(rows)
    data.quarantine(paths["final"])
    solver.write_rows(paths["final"], rows, data.FINAL_COLUMNS)
    problems = data.frame_problems(data.read_table(paths["final"]), n)
    if problems:
        raise ReplicationError(f"{key}: {'; '.join(problems)}")
    entry = {**info, "n": n, "repair": repair, "environment": data.environment(),
             "solver_fingerprint": fingerprint,
             "non_converged": data.non_converged(data.read_table(paths["final"])),
             **{f"{stage}_sha256": data.sha256_file(path) for stage, path in
                (("design", paths["designs"]), ("results", paths["results"]), ("final", paths["final"]))},
             "updated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    data.record("replications", key, entry)
    print(f"[{key}] repair: {repair}")
    return entry


def _copy_from_pool(key: str, strategy: str, realization: int, batch: int, *, force: bool = False) -> dict:
    """Write a pool-backed set: its design, and the results and final rows copied from
    the pool's chunk files, row for row as the solver wrote them.

    The resume contract of the module holds here too. A damaged or missing set, or one
    with no metadata entry, is simply copied again: that is cheap. A set whose record
    says it came from another design or another pool state is a configuration mismatch.
    It is refused, never rewritten silently, because fits and stage 1 may already rest on it."""
    frame, info = build_design(key, strategy, realization, batch)
    paths = {stage: config.replication_path(stage, key) for stage in ("designs", "results", "final")}
    text = frame.to_csv(index=False)
    n = expected_n(strategy, batch)
    previous = data.read_metadata().get("replications", {}).get(key, {})
    if previous and not force:
        if previous.get("design_sha256") != data.text_sha256(text) or previous.get("pool") != info["pool"]:
            raise ReplicationError(f"{key}: the stored set was copied from another design or another state "
                                   "of the reference pool than the configuration produces now; rerun "
                                   "with --force only after deciding that it must change")
        reason = data.stored_set_problem(paths, previous, n)
        if reason is None:
            print(f"[{key}] up to date")
            return previous
        print(f"[{key}] incomplete ({reason}); copying it again", flush=True)
    for path in paths.values():
        data.quarantine(path)                 # the pool is the source; old copies are kept aside
    data.write_text_atomic(paths["designs"], text)
    results, finals = [], []
    for k in sorted(set(frame["pool_chunk"])):
        result_rows, result_columns = solver.read_rows(config.pool_path("results", k))
        final_rows, final_columns = solver.read_rows(config.pool_path("final", k))
        if result_columns != data.RESULT_COLUMNS or final_columns != data.FINAL_COLUMNS:
            raise ReplicationError(f"{key}: pool chunk {k} does not match the result or final schema")
        for position in frame.index[frame["pool_chunk"] == k]:
            chunk_row = int(frame.at[position, "pool_chunk_row"])
            results.append((position, result_rows[chunk_row]))
            finals.append((position, final_rows[chunk_row]))
    # The rows are written as read (strings), so the copy is byte-faithful to the pool.
    solver.write_rows(paths["results"], [row for _, row in sorted(results, key=lambda x: x[0])],
                      data.RESULT_COLUMNS)
    solver.write_rows(paths["final"], [row for _, row in sorted(finals, key=lambda x: x[0])],
                      data.FINAL_COLUMNS)
    final = data.read_table(paths["final"])
    problems = data.frame_problems(final, n)
    if problems:
        raise ReplicationError(f"{key}: {'; '.join(problems)}")
    entry = {**info, "n": n, "source": "pool",
             "repair": {"n_rows": n, "n_repaired": int((final["repair_variant"] != "").sum()),
                        "source": "pool"},
             "environment": data.environment(), "solver_fingerprint": config.solver_fingerprint(),
             "non_converged": data.non_converged(final),
             **{f"{name}_sha256": data.sha256_file(paths[stage]) for stage, name in
                (("designs", "design"), ("results", "results"), ("final", "final"))},
             "updated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    data.record("replications", key, entry)
    for path in paths.values():
        data.protect(path)
    print(f"[{key}] copied {n} rows from the reference pool (no solver)")
    return entry


def generate_all(*, workers: int, force: bool = False, fn=None) -> None:
    """Every set of every new realization, realization by realization, so an
    interrupted run leaves whole realizations behind."""
    for r in range(1, config.REP_REALIZATIONS + 1):
        for key, strategy, batch in set_keys(r):
            generate_set(key, strategy, r, batch, workers=workers, force=force, fn=fn)


# --- fitting (fixed hyper-parameters) --------------------------------------------------------

REP_ARMS = ("A", "B", "C-log", "C0-log", "C-seq", "C0-seq", "C-logG", "C0-logG",
            "C-oracle", "C0-oracle", "C-logP", "C0-logP")
# Arms of the pool-backed strategies: confirmatory realizations only, no r0.
POOL_ARMS = ("C-oracle", "C0-oracle", "C-logP", "C0-logP")
SEQ_ROLES_R0 = tuple(f"guided-seq-{b}" for b in range(1, 4))
# The non-seed, non-test r0 roles arm_frames draws on: the LHS extension, both
# guided-log screeners' designs, and the sequential batches. Named once so
# data_hashes and fit_all's disjointness check cannot drift from arm_frames.
R0_GUIDED_ROLES = ("lhs-extension", "guided-log", "guided-log-g", *SEQ_ROLES_R0)


def arm_frames(realization: int) -> dict[str, pd.DataFrame]:
    """Training frames of one realization. r0 is the existing main-pipeline sets
    (A, the seed alone, is fitted only there)."""
    seed = data.load_role("seed")
    if realization == 0:
        extension, guided = data.load_role("lhs-extension"), data.load_role("guided-log")
        guided_g = data.load_role("guided-log-g")
        batches = [data.load_role(role) for role in SEQ_ROLES_R0]
        out = {"A": seed}
    else:
        by = keys_by_strategy(realization)
        extension, guided = load_set(by["lhs"][0]), load_set(by["guided-log"][0])
        guided_g = load_set(by["guided-log-g"][0])
        batches = [load_set(key) for key in by["guided-seq"]]
        oracle, logp = load_set(by["guided-oracle"][0]), load_set(by["guided-log-p"][0])
        out = {"C-oracle": pd.concat([seed, oracle], ignore_index=True), "C0-oracle": oracle,
               "C-logP": pd.concat([seed, logp], ignore_index=True), "C0-logP": logp}
    sequential = pd.concat(batches, ignore_index=True)
    out.update({"B": pd.concat([seed, extension], ignore_index=True),
                "C-log": pd.concat([seed, guided], ignore_index=True), "C0-log": guided,
                "C-seq": pd.concat([seed, sequential], ignore_index=True), "C0-seq": sequential,
                "C-logG": pd.concat([seed, guided_g], ignore_index=True), "C0-logG": guided_g})
    return out


def data_hashes(realization: int) -> dict:
    paths = [config.final_path(role) for role in ("seed", *config.TEST_ROLES)]
    if realization == 0:
        paths += [config.final_path(role) for role in R0_GUIDED_ROLES]
    else:
        paths += [config.replication_path("final", key) for key, _, _ in set_keys(realization)]
    return {path.name: data.sha256_file(path) for path in paths}


def _state(family: str, realization: int) -> dict:
    return {"fingerprint": config.replication_fingerprint(family), "data": data_hashes(realization)}


def fit_realization(family: str, realization: int, tests) -> dict:
    arms = {}
    for arm, frame in arm_frames(realization).items():
        model = models.lc_estimator(family).fit(frame[list(config.FEATURES)],
                                                np.log(frame["FOS"].to_numpy(float)))
        arms[arm] = {"n_train": int(len(frame)), "train_ids": frame["row_id"].to_numpy(str),
                     "train_fos": frame["FOS"].to_numpy(float),
                     "predictions": {name: np.exp(model.predict(test.X)) for name, test in tests.items()}}
    return arms


def fit_all(*, force: bool = False) -> None:
    data.validate_sets(("seed", *config.TEST_ROLES, *R0_GUIDED_ROLES))
    tests = {role: data.load_test(role) for role in config.TEST_ROLES}
    for family in config.REP_FAMILIES:
        for r in range(0, config.REP_REALIZATIONS + 1):
            path = config.replication_fit_path(family, f"r{r}")
            state = _state(family, r)
            if path.exists():
                if joblib.load(path)["state"] == state:
                    print(f"[replicate] {path.name}: up to date")
                    continue
                if not force:
                    raise models.StaleFit(f"{path.name} was fitted under another configuration or other "
                                          "data; rerun with --force")
            print(f"[replicate] fit {family} r{r}", flush=True)
            models._dump(path, {"state": state, "family": family, "realization": r,
                                "test_ids": {name: test.ids for name, test in tests.items()},
                                "arms": fit_realization(family, r, tests)})


def load_units() -> dict:
    out = {}
    for family in config.REP_FAMILIES:
        for r in range(0, config.REP_REALIZATIONS + 1):
            path = config.replication_fit_path(family, f"r{r}")
            if not path.exists():
                raise FileNotFoundError(f"{path.name} is missing; run `sgs replicate`")
            stored = joblib.load(path)
            if stored["state"] != _state(family, r):
                raise models.StaleFit(f"{path.name} is stale; run `sgs replicate --force`")
            out[(family, r)] = stored
    return out
