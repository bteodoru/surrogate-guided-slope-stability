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


# --- stage 1 (declaration §4.1 and the prediction of §3): training data only ---------------------

ZONES = ((None, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 6.0), (6.0, None))


def zone_counts(fos) -> list[int]:
    fos = np.asarray(fos, dtype=float)
    return [int((((fos >= lo) if lo is not None else True) & ((fos < hi) if hi is not None else True)).sum())
            for lo, hi in ZONES]


def arm_sets(r: int) -> dict[str, pd.DataFrame]:
    """The added points of each new arm in realization r (the ablations' training sets)."""
    out = {"D0-oracle": load_set(f"d_oracle_r{r}"), "D0-P": load_set(f"d_p_r{r}")}
    for name, s in (("C0-seqF", "seq_f"), ("C0-seqP", "seq_p")):
        out[name] = pd.concat([load_set(f"{s}_r{r}_b{b}") for b in range(1, len(config.REP_SEQ_BATCHES) + 1)],
                              ignore_index=True)
    return out


def stage1() -> dict:
    from sgs import density, replicate
    seed = data.load_role("seed")
    seed_fos = seed["FOS"].to_numpy(float)
    a_d = np.asarray(config.REV_D_ALLOCATION)
    intended_d = design.bin_counts(seed_fos, d_edges()) + a_d
    intended_seq = design.bin_counts(seed_fos, sel_edges()) + pool.allocation()
    meta = data.read_metadata(config.revision_metadata_path())[SECTION]
    per_r, b_zone = [], []
    for r in range(1, config.REP_REALIZATIONS + 1):
        b_zone.append(zone_counts(np.concatenate([seed_fos, replicate.load_set(f"lhs_r{r}")["FOS"]])))
        oracle = set(pd.read_csv(config.replication_path("designs", f"guided_oracle_r{r}"))["pool_row"])
        logp = set(pd.read_csv(config.replication_path("designs", f"guided_log_p_r{r}"))["pool_row"])
        entry = {"realization": r, "arms": {}}
        for ablation, added in arm_sets(r).items():
            arm = ablation.replace("0", "", 1)
            full = np.concatenate([seed_fos, added["FOS"].to_numpy(float)])
            edges, intended = (d_edges(), intended_d) if arm.startswith("D") else (sel_edges(), intended_seq)
            achieved = design.bin_counts(full, edges)
            rows = set(pd.read_csv(config.revision_path("designs", f"{'d_oracle' if arm == 'D-oracle' else 'd_p'}_r{r}"))
                       ["pool_row"]) if arm.startswith("D") else set().union(*(
                           set(pd.read_csv(config.revision_path("designs", f"{'seq_f' if arm == 'C-seqF' else 'seq_p'}_r{r}_b{b}"))["pool_row"])
                           for b in (1, 2, 3)))
            entry["arms"][arm] = {
                "achieved": achieved.tolist(), "intended": intended.tolist(),
                "sum_abs_dev": int(np.abs(achieved - intended).sum()),
                "zones_train": zone_counts(full), "zones_added": zone_counts(added["FOS"]),
                "below_0_5": int((added["FOS"] < 0.5).sum()), "above_6": int((added["FOS"] > 6).sum()),
                "overlap_C_oracle": len(rows & oracle), "overlap_C_logP": len(rows & logp)}
            if arm.startswith("C-seq"):
                entry["arms"][arm]["shortfall"] = [meta[f"{'seq_f' if arm == 'C-seqF' else 'seq_p'}_r{r}_b{b}"]
                                                   ["shortfall"] for b in (1, 2, 3)]
        per_r.append(entry)
    b_mean = np.mean(b_zone, axis=0)
    a = json.loads(density.revision_metrics_path().read_text())["density"]["estimates"]["pooled"]["a"]
    prediction = {}
    for arm in ("D-oracle", "D-P"):
        ratios = [[np.log(e["arms"][arm]["zones_train"][z] / b_mean[z]) if e["arms"][arm]["zones_train"][z] else None
                   for z in range(len(ZONES))] for e in per_r]
        prediction[arm] = {"predicted_log_ratio_vs_B": [[None if x is None else float(-a * x) for x in row]
                                                         for row in ratios]}
    # The decision-range prediction per cell: the predicted squared zone errors on [0.5, 1)
    # and [1, 2), each B's mean zone error times exp(predicted log ratio, median over r),
    # weighted by the test set's share of decision-range points in the zone.
    values = json.loads(config.replication_metrics_path().read_text())["values"]
    for arm in prediction:
        med = np.median(np.array(prediction[arm]["predicted_log_ratio_vs_B"], dtype=float), axis=0)
        cells = {}
        for family in config.REP_FAMILIES:
            for test in config.TEST_ROLES:
                bs = [v for v in values if v["arm"] == "B" and v["family"] == family and v["test"] == test
                      and v["realization"] >= 1]
                e = {z: np.mean([[x for x in v["zones"] if x["zone"] == z][0]["log_rmse"] for v in bs])
                     for z in ("0.5–1", "1–2")}
                n = {z: [x for x in bs[0]["zones"] if x["zone"] == z][0]["n"] for z in ("0.5–1", "1–2")}
                e_dr = np.mean([v["subsets"]["decision-range"]["log_rmse"] for v in bs])
                w = {z: n[z] / (n["0.5–1"] + n["1–2"]) for z in n}
                pred = np.sqrt(w["0.5–1"] * (e["0.5–1"] * np.exp(med[1])) ** 2
                               + w["1–2"] * (e["1–2"] * np.exp(med[2])) ** 2)
                cells[f"{family}/{test}"] = {"B_mean_decision_range": float(e_dr),
                                             "predicted_decision_range": float(pred),
                                             "predicted_log_ratio": float(np.log(pred / e_dr))}
        prediction[arm]["decision_range"] = cells
    return {"realizations": per_r, "B_zone_train_mean": b_mean.tolist(), "a_pooled": a,
            "zones": [list(z) for z in ZONES], "density_prediction": prediction,
            "note": "training data only, apart from B's existing zone errors used for the "
                    "decision-range combination; committed before any fit of a new arm"}
