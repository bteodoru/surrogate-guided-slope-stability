"""Sampling designs: parameter sets before the solver.

Three procedures, one per ``SetRole.design``:
  lhs                Latin hypercube over the input domain;
  guided-maxmin      screener-predicted FOS bins, greedy max-min inside a bin
                     (the guided training selection);
  stratified-random  screener-predicted FOS bins, random selection inside a
                     bin (the target-stratified test set).
The screener is refitted in memory whenever a screener role is built; the fit
takes seconds and is deterministic.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import qmc, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict

from sgs import config, data


class DesignError(RuntimeError):
    """A design violates one of its own invariants."""


# --- LHS ------------------------------------------------------------------------

def _bounds():
    lo = np.array([config.DOMAIN[p][0] for p in config.PARAMS], dtype=float)
    hi = np.array([config.DOMAIN[p][1] for p in config.PARAMS], dtype=float)
    return lo, hi


def lhs(n: int, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    # ``seed=``, not ``rng=``: in SciPy 1.18 the two keywords seed different
    # generators, and the published designs were drawn with ``seed=``.
    sampler = qmc.LatinHypercube(d=len(config.PARAMS), seed=seed)
    unit = sampler.random(n)
    lo, hi = _bounds()
    return unit, pd.DataFrame(qmc.scale(unit, lo, hi), columns=list(config.PARAMS))


def lhs_strata_complete(unit: np.ndarray) -> bool:
    """One point in each of the n strata along every dimension."""
    n = len(unit)
    strata = np.floor(np.clip(unit, 0.0, np.nextafter(1.0, 0.0)) * n).astype(int)
    return all(len(np.unique(strata[:, j])) == n for j in range(unit.shape[1]))


# --- disjointness -----------------------------------------------------------------

def nearest_distance(points: pd.DataFrame, reference: pd.DataFrame) -> np.ndarray:
    return cKDTree(data.normalize(reference)).query(data.normalize(points), k=1)[0]


def assert_disjoint(frame: pd.DataFrame, references: Mapping[str, pd.DataFrame],
                    min_distance: float) -> None:
    for name, reference in references.items():
        d = nearest_distance(frame, reference)
        if d.min() <= min_distance:
            raise DesignError(f"{int((d <= min_distance).sum())} point(s) within {min_distance} "
                              f"of {name!r} (closest {d.min():.2e})")
    unit = data.normalize(frame)
    self_d = cKDTree(unit).query(unit, k=2)[0][:, 1]
    if self_d.min() <= min_distance:
        raise DesignError(f"duplicate points inside the set (closest pair {self_d.min():.2e})")


# --- screener -----------------------------------------------------------------------

def screener_model(*, oob: bool) -> RandomForestRegressor:
    # n_jobs=1: parallel prediction sums tree outputs in thread-completion order, which changes FOS_pred in the last bit and with it the design hash.
    return RandomForestRegressor(n_estimators=config.SCREENER["n_estimators"], oob_score=oob,
                                 random_state=config.SEEDS["screener_fit"], n_jobs=1)


def fit_screener(frame: pd.DataFrame):
    """Fit on log(FOS) of ``frame`` (features included); return model and diagnostics."""
    X = frame[list(config.FEATURES)]
    fos = frame["FOS"].to_numpy(float)
    y = np.log(fos)
    model = screener_model(oob=True).fit(X, y)
    cv = KFold(config.SCREENER["cv_folds"], shuffle=True, random_state=config.SEEDS["screener_cv"])
    oof = np.exp(cross_val_predict(screener_model(oob=False), X, y, cv=cv))
    lo, hi = config.SELECTION_RANGE
    inside = (fos >= lo) & (fos <= hi)
    diagnostics = {
        "n_train": int(len(frame)),
        "oob_r2_log": float(model.oob_score_),
        "oof_rmse": float(np.sqrt(np.mean((oof - fos) ** 2))),
        "oof_rmse_selection_range": float(np.sqrt(np.mean((oof[inside] - fos[inside]) ** 2))),
        "oof_spearman": float(spearmanr(oof, fos).statistic),
    }
    return model, diagnostics


# --- screener-driven designs --------------------------------------------------------------

def uniform_pool(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Uniform candidates over the domain, one column at a time in PARAMS order."""
    return pd.DataFrame({p: rng.uniform(config.DOMAIN[p][0], config.DOMAIN[p][1], n)
                         for p in config.PARAMS})


def predict_fos(model, frame: pd.DataFrame) -> np.ndarray:
    return np.exp(model.predict(data.add_features(frame)[list(config.FEATURES)]))


def greedy_maxmin(Xs: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy max-min selection of k rows from a random start."""
    n = len(Xs)
    if k >= n:
        return np.arange(n)
    chosen = [int(rng.integers(n))]
    dmin = np.linalg.norm(Xs - Xs[chosen[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(dmin.argmax())
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(Xs - Xs[nxt], axis=1))
    return np.array(chosen)


def bin_edges(n_bins: int, selection_range) -> np.ndarray:
    lo, hi = selection_range
    return np.linspace(lo, hi, n_bins + 1)


def guided_maxmin(model, *, rng, n_pool, n_bins, per_bin, drawn_per_bin, selection_range) -> pd.DataFrame:
    """The guided training selection.

    One generator, in this order: the pool, then per bin a max-min draw of
    ``drawn_per_bin`` candidates and a permutation of which the first
    ``per_bin`` are kept. Bins are (lo, hi] on screener-predicted FOS.
    """
    params = list(config.PARAMS)
    pool = uniform_pool(rng, n_pool)
    pool["FOS_pred"] = predict_fos(model, pool)
    edges = bin_edges(n_bins, selection_range)
    lo, hi = selection_range
    inside = pool[(pool["FOS_pred"] >= lo) & (pool["FOS_pred"] <= hi)]
    counts = pd.cut(inside["FOS_pred"], edges).value_counts().sort_index()
    thin = counts[counts < drawn_per_bin]
    if len(thin):
        raise DesignError(f"bins with fewer than {drawn_per_bin} candidates:\n{thin.to_string()}")

    mu, sd = pool[params].mean(), pool[params].std()
    chosen, bins = [], []
    for b, (a, z) in enumerate(zip(edges[:-1], edges[1:])):
        candidates = pool[(pool["FOS_pred"] > a) & (pool["FOS_pred"] <= z)]
        Xs = ((candidates[params] - mu) / sd).to_numpy()
        drawn = candidates.index.to_numpy()[greedy_maxmin(Xs, drawn_per_bin, rng)]
        kept = drawn[rng.permutation(len(drawn))[:per_bin]]
        chosen.extend(kept.tolist())
        bins.extend([b] * len(kept))
    out = pool.loc[chosen, params + ["FOS_pred"]].reset_index(drop=True)
    out["bin"] = bins
    return out


def stratified_random(model, *, pool_rng, selection_rng, n_pool, n_bins, per_bin,
                      selection_range, references, min_distance) -> pd.DataFrame:
    """The target-stratified test set: random selection within each bin.

    An independent pool (its own generator) is screened; candidates outside
    the selection range or within ``min_distance`` of any reference point are
    dropped; ``per_bin`` candidates are then drawn at random from each bin on
    a separate generator. The first bin is closed [lo, hi], the others (lo, hi].
    """
    params = list(config.PARAMS)
    lo, hi = selection_range
    pool = uniform_pool(pool_rng, n_pool)
    pool["FOS_pred"] = predict_fos(model, pool)
    pool = pool[pool["FOS_pred"].between(lo, hi, inclusive="both")].reset_index(drop=True)
    if references:
        reference = pd.concat([r[params] for r in references.values()], ignore_index=True)
        pool = pool.loc[nearest_distance(pool, reference) > min_distance].reset_index(drop=True)

    edges = bin_edges(n_bins, selection_range)
    chosen, bins = [], []
    for b, (a, z) in enumerate(zip(edges[:-1], edges[1:])):
        lower = pool["FOS_pred"] >= a if b == 0 else pool["FOS_pred"] > a
        candidates = pool.loc[lower & (pool["FOS_pred"] <= z)]
        if len(candidates) < per_bin:
            raise DesignError(f"bin {b} [{a:.3f}, {z:.3f}] has only {len(candidates)} candidates")
        picked = selection_rng.choice(len(candidates), per_bin, replace=False)
        chosen.extend(candidates.index.to_numpy()[picked].tolist())
        bins.extend([b] * per_bin)
    out = pool.loc[chosen, params + ["FOS_pred"]].reset_index(drop=True)
    out["bin"] = bins
    return out


# --- roles ------------------------------------------------------------------------

def read_design(role: str) -> pd.DataFrame:
    path = config.design_path(role)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing; generate role {role!r} first")
    return pd.read_csv(path)


def build(role: str) -> tuple[pd.DataFrame, dict]:
    """The design of a role, and what produced it (seeds, screener diagnostics)."""
    spec = config.ROLES[role]
    references = {r: read_design(r) for r in spec.disjoint_from}
    seeds = [config.SEEDS[name] for name in spec.seeds]
    info = {"seeds": dict(zip(spec.seeds, seeds))}

    if spec.design == "lhs":
        unit, frame = lhs(spec.n, seeds[0])
        if not lhs_strata_complete(unit):
            raise DesignError(f"{role}: LHS stratification incomplete")
    else:
        trained_on = config.SCREENER["trained_on"]
        model, diagnostics = fit_screener(data.load_role(trained_on))
        info["screener"] = {**diagnostics, "trained_on": trained_on,
                            "trained_on_sha256": data.sha256_file(config.final_path(trained_on))}
        if spec.design == "guided-maxmin":
            frame = guided_maxmin(model, rng=np.random.default_rng(seeds[0]),
                                  n_pool=config.N_GUIDED_POOL, n_bins=config.N_BINS,
                                  per_bin=config.GUIDED_PER_BIN,
                                  drawn_per_bin=config.GUIDED_DRAWN_PER_BIN,
                                  selection_range=config.SELECTION_RANGE)
        elif spec.design == "stratified-random":
            frame = stratified_random(model, pool_rng=np.random.default_rng(seeds[0]),
                                      selection_rng=np.random.default_rng(seeds[1]),
                                      n_pool=config.N_TEST_POOL, n_bins=config.N_BINS,
                                      per_bin=config.TEST_PER_BIN,
                                      selection_range=config.SELECTION_RANGE,
                                      references=references, min_distance=config.MIN_DISTANCE)
        else:
            raise ValueError(f"unknown design procedure {spec.design!r} for role {role!r}")

    if len(frame) != spec.n:
        raise DesignError(f"{role}: {len(frame)} points, expected {spec.n}")
    assert_disjoint(frame, references, config.MIN_DISTANCE)
    return frame, info
