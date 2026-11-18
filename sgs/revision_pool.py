"""The revision's pool-backed sets (docs/revision-declaration.md §2 and §4).

  D-oracle, D-P   C-log's procedure with the decision-range target: the seed water-filled
                  to 300 points over REV_D_BINS geometric bins on DECISION_RANGE, a random
                  draw within each bin of sub-pool r, binned on solved FOS (oracle) or on
                  the seed-trained RF screener's prediction (P).
  C-seqF, C-seqP  three batches of 100 from sub-pool r against C-log's 15-bin deficit,
                  recomputed after each batch on the solved FOS of the seed and the earlier
                  batches; the screener stays the seed's (F) or is retrained on the seed and
                  the earlier batches before each batch (P).

No solver: the rows are already solved; results and final rows are copied from the pool's
chunk files as written. Each draw has its own stream, a SeedSequence on
SEEDS["rev_pool_design"] keyed on (strategy index, realization, batch). Provenance goes to
the ``pool_sets`` section of data/revision/run_metadata.json.
"""

from __future__ import annotations

import datetime as _dt
import json

import numpy as np
import pandas as pd

from sgs import config, data, design, pool, solver
from sgs.revision import RevisionError

SECTION = "pool_sets"
STRATEGIES = ("d-oracle", "d-p", "seq-f", "seq-p")
# C-seqF and C-seqP share their draw stream, so their first batches (both screened by the
# seed's RF, against the same deficit) are the same rows, and the two differ only from
# the retraining on.
STREAM = {"seq-p": "seq-f"}
COLUMN = {"d-oracle": "FOS", "d-p": "FOS_pred"}


def stream(strategy: str, realization: int, batch: int = 0) -> np.random.Generator:
    key = STREAM.get(strategy, strategy)
    sequence = np.random.SeedSequence(config.SEEDS["rev_pool_design"],
                                      spawn_key=(STRATEGIES.index(key), realization, batch))
    return np.random.default_rng(sequence)


def d_edges() -> np.ndarray:
    return design.bin_edges_log(config.REV_D_BINS, config.DECISION_RANGE)


def d_allocation(seed_fos) -> np.ndarray:
    return design.deficit_allocation(design.bin_counts(seed_fos, d_edges()), config.REP_EXTENSION)


def capped_allocation(have, alloc, avail) -> tuple[np.ndarray, int]:
    """Deviation recorded in docs/predictions-and-outcomes.md (2026-11-18), before any
    draw: a bin asked for more points than its sub-pool has left keeps all of them, and
    the shortfall is water-filled, as the deficit itself is, over the bins that still
    have points, so every batch keeps its size. Returns the allocation and the shortfall."""
    alloc, avail = np.asarray(alloc, dtype=int), np.asarray(avail, dtype=int)
    capped = np.minimum(alloc, avail)
    short = int(alloc.sum() - capped.sum())
    total = np.asarray(have, dtype=int) + capped
    for _ in range(short):
        open_ = np.flatnonzero(capped < avail)
        if not len(open_):
            raise RevisionError("the sub-pool has fewer unused points than the batch")
        b = open_[np.argmin(total[open_])]
        capped[b] += 1
        total[b] += 1
    return capped, short


def sel_edges() -> np.ndarray:
    return design.bin_edges_log(config.N_BINS, config.SELECTION_RANGE)


def verified_rows() -> tuple[pd.DataFrame, dict, dict]:
    """The pool's eligible rows with the seed screener's FOS_pred, after the same checks
    the oracle's sets passed (replicate._pool_design)."""
    frame, _ = pool.full_design()
    stop = pool.verified_stop(frame)
    model, diagnostics = pool.screener()
    rows = pool.eligible(frame, stop["k"], model)
    if pool.rule_table(rows, pool.allocation()) != stop["table"]:
        raise RevisionError("the screener, the seed or the allocation differs from the one that "
                            "evaluated the pool's stop")
    return rows, stop, diagnostics


def _as_design(chosen: pd.DataFrame, bins: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    out = chosen[list(config.PARAMS)].copy().reset_index(drop=True)
    out["FOS_pred"] = np.asarray(pred, dtype=float)
    out["bin"] = np.asarray(bins, dtype=int)
    out["pool_chunk"] = chosen["chunk"].to_numpy()
    out["pool_chunk_row"] = chosen["chunk_row"].to_numpy()
    out["pool_row"] = chosen["pool_row"].to_numpy()
    return out


def build_designs() -> dict[str, tuple[pd.DataFrame, dict]]:
    """Every pool-backed revision set, keyed, with its provenance."""
    rows, stop, diagnostics = verified_rows()
    seed = data.load_role("seed")
    a_d = d_allocation(seed["FOS"].to_numpy(float))
    if tuple(a_d.tolist()) != config.REV_D_ALLOCATION:
        raise RevisionError(f"the D allocation {a_d.tolist()} is not the declared {config.REV_D_ALLOCATION}")
    base = {"stop_k": stop["k"], "chunks_sha256": stop["chunks_sha256"], "platform": stop["platform"],
            "seed_sha256": data.sha256_file(config.final_path("seed"))}
    out = {}
    for r in range(1, config.REP_REALIZATIONS + 1):
        sub = rows[rows["subpool"] == r].reset_index(drop=True)
        for strategy in ("d-oracle", "d-p"):
            index, bins = design.select_in_bins(sub[COLUMN[strategy]].to_numpy(float), d_edges(), a_d,
                                                stream(strategy, r))
            chosen = sub.iloc[index]
            out[f"{strategy.replace('-', '_')}_r{r}"] = (
                _as_design(chosen, bins, chosen["FOS_pred"]),
                {**base, "strategy": strategy, "realization": r, "batch": 0, "column": COLUMN[strategy],
                 "edges": d_edges().tolist(), "added": a_d.tolist(), "screener": diagnostics})
        for strategy in ("seq-f", "seq-p"):
            taken = np.zeros(len(sub), dtype=bool)
            earlier: list[pd.DataFrame] = []
            for b, size in enumerate(config.REP_SEQ_BATCHES, 1):
                training = pd.concat([seed[[*config.PARAMS, "FOS"]],
                                      *(e[[*config.PARAMS, "FOS"]] for e in earlier)], ignore_index=True)
                have = design.bin_counts(training["FOS"].to_numpy(float), sel_edges())
                alloc = design.deficit_allocation(have, size)
                if strategy == "seq-p" and earlier:
                    model, diag = design.fit_screener(data.add_features(training), config.SCREENER["family"])
                    pred = design.predict_fos(model, sub)
                else:
                    pred, diag = sub["FOS_pred"].to_numpy(float), diagnostics
                values = np.where(taken, np.nan, pred)       # NaN is in no bin
                avail = design.bin_counts(values[~np.isnan(values)], sel_edges())
                requested = alloc
                alloc, short = capped_allocation(have, requested, avail)
                index, bins = design.select_in_bins(values, sel_edges(), alloc, stream(strategy, r, b))
                taken[index] = True
                chosen = sub.iloc[index]
                earlier.append(chosen)
                out[f"{strategy.replace('-', '_')}_r{r}_b{b}"] = (
                    _as_design(chosen, bins, pred[index]),
                    {**base, "strategy": strategy, "realization": r, "batch": b, "column": "FOS_pred",
                     "edges": sel_edges().tolist(), "trained_counts": have.tolist(),
                     "added": alloc.tolist(), "requested": requested.tolist(),
                     "shortfall": short, "screener": diag,
                     "screener_trained_on": "seed" if strategy == "seq-f" or b == 1
                     else f"seed + batches 1..{b - 1}"})
    return out


def _paths(key: str) -> dict:
    return {stage: config.revision_path(stage, key) for stage in ("designs", "results", "final")}


def copy_set(key: str, frame: pd.DataFrame, info: dict) -> dict:
    paths = _paths(key)
    text = frame.to_csv(index=False)
    meta_path = config.revision_metadata_path()
    previous = data.read_metadata(meta_path).get(SECTION, {}).get(key, {})
    if previous:
        if previous.get("design_sha256") != data.text_sha256(text):
            raise RevisionError(f"{key}: the stored set was drawn from another design than the one "
                                "the configuration produces now")
        if data.stored_set_problem(paths, previous, len(frame)) is None:
            print(f"[{key}] up to date")
            return previous
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        data.quarantine(path)
    data.write_text_atomic(paths["designs"], text)
    results, finals = [], []
    for k in sorted(set(frame["pool_chunk"])):
        result_rows, rc = solver.read_rows(config.pool_path("results", k))
        final_rows, fc = solver.read_rows(config.pool_path("final", k))
        if rc != data.RESULT_COLUMNS or fc != data.FINAL_COLUMNS:
            raise RevisionError(f"{key}: pool chunk {k} does not match the result or final schema")
        for position in frame.index[frame["pool_chunk"] == k]:
            row = int(frame.at[position, "pool_chunk_row"])
            results.append((position, result_rows[row]))
            finals.append((position, final_rows[row]))
    solver.write_rows(paths["results"], [r for _, r in sorted(results, key=lambda x: x[0])], data.RESULT_COLUMNS)
    solver.write_rows(paths["final"], [r for _, r in sorted(finals, key=lambda x: x[0])], data.FINAL_COLUMNS)
    final = data.read_table(paths["final"])
    problems = data.frame_problems(final, len(frame))
    if problems:
        raise RevisionError(f"{key}: {'; '.join(problems)}")
    if not np.allclose(final[list(config.PARAMS)].to_numpy(float), frame[list(config.PARAMS)].to_numpy(float)):
        raise RevisionError(f"{key}: the copied rows are not the drawn rows")
    entry = {**json.loads(json.dumps(info, default=float)), "n": len(frame), "source": "pool",
             "environment": data.environment(),
             **{f"{name}_sha256": data.sha256_file(paths[stage]) for stage, name in
                (("designs", "design"), ("results", "results"), ("final", "final"))},
             "updated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    data.record(SECTION, key, entry, path=meta_path)
    for path in paths.values():
        data.protect(path)
    print(f"[{key}] copied {len(frame)} rows from the reference pool (no solver)")
    return entry


def generate_all() -> None:
    for key, (frame, info) in build_designs().items():
        copy_set(key, frame, info)


def load_set(key: str) -> pd.DataFrame:
    return data.load_final(config.revision_path("final", key), f"rev:{key}")
