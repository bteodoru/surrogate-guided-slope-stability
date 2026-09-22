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
import hashlib
import os

import joblib

import numpy as np
import pandas as pd

from sgs import config, data, design, models, solver


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
    if strategy == "lhs":
        unit, frame = design.lhs(n, seed)
        if not design.lhs_strata_complete(unit):
            raise design.DesignError(f"{key}: LHS stratification incomplete")
    else:
        earlier = [f"guided_seq_r{realization}_b{b}" for b in range(1, batch)] if strategy == "guided-seq" else []
        for k in earlier:
            references[k] = read_set_design(k)
        training = pd.concat([data.load_role("seed"), *(load_set(k) for k in earlier)], ignore_index=True)
        model, diagnostics = design.fit_screener(training)
        edges = design.bin_edges_log(config.N_BINS, config.SELECTION_RANGE)
        have = design.bin_counts(training["FOS"].to_numpy(float), edges)
        allocation = design.deficit_allocation(have, n)
        # The solved files the screener learned from, as design._training_record
        # writes them for a role trained on more than one set.
        trained_paths = {"seed": config.final_path("seed"),
                         **{k: config.replication_path("final", k) for k in earlier}}
        info["screener"] = {**diagnostics, "trained_on": ["seed", *earlier],
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


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _write_design(path, text: str) -> None:
    """Write the design atomically, so a kill cannot leave it torn."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _incomplete(paths: dict, previous: dict, n: int) -> str | None:
    """Why the stored set cannot be reused as it stands, or None when it can.
    Its metadata entry is already known to record the design the configuration
    produces now, so a file that differs from it is a damaged one."""
    if not previous:
        return "it has no metadata entry"
    for stage, recorded in (("designs", "design"), ("results", "results"), ("final", "final")):
        path = paths[stage]
        if not path.exists():
            return f"the {stage} file is missing"
        if previous.get(f"{recorded}_sha256") != data.sha256_file(path):
            return f"the {stage} file is not the one that was recorded"
    problems = data.frame_problems(data.read_table(paths["final"]), n)
    return "; ".join(problems) if problems else None


def _reuse(key: str, previous: dict, info: dict) -> dict:
    """A complete, matching set. When the files its screener learned from were
    solved again (same design, same FOS, other runtimes), only its provenance is
    stale: re-record it, and keep the data."""
    stored = previous.get("screener", {}).get("trained_on_sha256")
    fresh = info.get("screener", {}).get("trained_on_sha256")
    if fresh is None or stored == fresh:
        print(f"[{key}] up to date")
        return previous
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
    frame, info = build_design(key, strategy, realization, batch)
    paths = {stage: config.replication_path(stage, key) for stage in ("designs", "results", "final")}
    text = frame.to_csv(index=False)
    n = expected_n(strategy, batch)
    fingerprint = config.solver_fingerprint()
    previous = data.read_metadata().get("replications", {}).get(key, {})
    if not force:
        if previous and previous.get("design_sha256") != _text_sha256(text):
            raise ReplicationError(f"{key}: the stored set was solved from another design than the "
                                   "one the configuration produces now; rerun with --force")
        if previous and previous.get("solver_fingerprint") != fingerprint:
            raise ReplicationError(f"{key}: the stored set was solved under other solver settings; "
                                   "rerun with --force")
        reason = _incomplete(paths, previous, n)
        if reason is None:
            return _reuse(key, previous, info)
        if any(path.exists() for path in paths.values()):
            print(f"[{key}] incomplete ({reason}); solving it again", flush=True)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_design(paths["designs"], text)
    rows = solver.run_set(frame[list(config.PARAMS)].to_numpy(float), workers=workers, label=key, fn=fn)
    solver.write_rows(paths["results"], rows, data.RESULT_COLUMNS)
    rows, _ = solver.read_rows(paths["results"])
    repair = solver.repair_rows(rows)
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


def generate_all(*, workers: int, force: bool = False, fn=None) -> None:
    """Every set of every new realization, realization by realization, so an
    interrupted run leaves whole realizations behind."""
    for r in range(1, config.REP_REALIZATIONS + 1):
        for key, strategy, batch in set_keys(r):
            generate_set(key, strategy, r, batch, workers=workers, force=force, fn=fn)


# --- fitting (fixed hyper-parameters) --------------------------------------------------------

REP_ARMS = ("A", "B", "C-log", "C0-log", "C-seq", "C0-seq")
SEQ_ROLES_R0 = tuple(f"guided-seq-{b}" for b in range(1, 4))


def arm_frames(realization: int) -> dict[str, pd.DataFrame]:
    """Training frames of one realization. r0 is the existing main-pipeline sets
    (A, the seed alone, is fitted only there)."""
    seed = data.load_role("seed")
    if realization == 0:
        extension, guided = data.load_role("lhs-extension"), data.load_role("guided-log")
        batches = [data.load_role(role) for role in SEQ_ROLES_R0]
        out = {"A": seed}
    else:
        keys = [key for key, _, _ in set_keys(realization)]
        extension, guided = load_set(keys[0]), load_set(keys[1])
        batches = [load_set(key) for key in keys[2:]]
        out = {}
    sequential = pd.concat(batches, ignore_index=True)
    out.update({"B": pd.concat([seed, extension], ignore_index=True),
                "C-log": pd.concat([seed, guided], ignore_index=True), "C0-log": guided,
                "C-seq": pd.concat([seed, sequential], ignore_index=True), "C0-seq": sequential})
    return out


def data_hashes(realization: int) -> dict:
    paths = [config.final_path(role) for role in ("seed", *config.TEST_ROLES)]
    if realization == 0:
        paths += [config.final_path(role) for role in ("lhs-extension", "guided-log", *SEQ_ROLES_R0)]
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
    data.validate_sets(("seed", *config.TEST_ROLES, "lhs-extension", "guided-log", *SEQ_ROLES_R0))
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
