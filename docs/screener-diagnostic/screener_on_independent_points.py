# Read-only, no solver. Handoff 2026-09-22 section 2: score both screeners on solved LHS
# points that NEITHER screener selected, so the comparison is not conditional on the RF's
# own selection. Evaluation sets: test-natural (300) and the replication lhs_r1..r5 (1500).
# Both screeners are trained on the seed alone, as the strategy fixes them.
import numpy as np, pandas as pd
from sgs import config, data, design, models, replicate

EDGES = design.bin_edges_log(config.N_BINS, config.SELECTION_RANGE)   # the 15 selection bins
ZONES = np.array([-np.inf, *config.ZONE_EDGES, np.inf])
LO, HI = config.SELECTION_RANGE


def sel_bin(fos):
    """Index of the selection bin, as the generator bins: (lo, hi], first closed. -1 below, 99 above."""
    fos = np.asarray(fos, float)
    idx = np.searchsorted(EDGES, fos, side="left") - 1
    idx = np.where(fos <= EDGES[0], 0, idx)
    return np.where(fos < LO, -1, np.where(fos > HI, 99, np.clip(idx, 0, config.N_BINS - 1)))


def row(name, model, X, y):
    p = np.exp(model.predict(X))
    r = np.log(y / p)
    bt, bp = sel_bin(y), sel_bin(p)
    zt = np.asarray(pd.cut(y, ZONES, right=False).codes)
    zp = np.asarray(pd.cut(p, ZONES, right=False).codes)
    pred_in = (p >= LO) & (p <= HI)          # what the gate would let through
    true_in = (y >= LO) & (y <= HI)
    low, high = y < 0.6, y > 6
    return {
        "screener": name, "n": len(y),
        "log-RMSE": np.sqrt(np.mean(r**2)), "log-bias": np.mean(r),
        "same sel-bin": np.mean(bt == bp), "same zone": np.mean(zt == zp),
        "|bin displacement|": np.mean(np.abs(np.clip(bt, -1, 15) - np.clip(bp, -1, 15))),
        "min pred": p.min(),
        "pred in range": np.mean(pred_in),
        "true in | pred in": np.mean(true_in[pred_in]) if pred_in.any() else np.nan,
        "p/y true<0.6": np.median(p[low] / y[low]) if low.any() else np.nan,
        "p/y true>6": np.median(p[high] / y[high]) if high.any() else np.nan,
    }


def report(title, frame, screeners):
    X, y = frame[list(config.FEATURES)], frame["FOS"].to_numpy(float)
    print(f"\n== {title}  (n={len(y)}, none selected by either screener)")
    out = pd.DataFrame([row(n, m, X, y) for n, m in screeners]).set_index("screener")
    print(out.to_string(float_format=lambda v: f"{v:.3f}"))


def main():
    seed = data.load_role("seed")
    Xs, ys = seed[list(config.FEATURES)], np.log(seed["FOS"].to_numpy(float))
    rf, _ = design.fit_screener(seed, "rf")   # family argument added after 2026-09-23
    gbm = models.lc_estimator("gbm").fit(Xs, ys)
    screeners = (("RF (used)", rf), ("GBM (LC_PARAMS)", gbm))

    natural = data.load_role("test-natural")
    reps = {f"lhs_r{r}": replicate.load_set(f"lhs_r{r}") for r in range(1, config.REP_REALIZATIONS + 1)}

    report("test-natural", natural, screeners)
    report("lhs_r1..r5 pooled", pd.concat(reps.values(), ignore_index=True), screeners)
    report("all independent LHS points", pd.concat([natural, *reps.values()], ignore_index=True), screeners)

    # Per realization, so the pooled numbers are not carried by one draw.
    print("\n== log-RMSE per set")
    per = {k: {n: np.sqrt(np.mean(np.log(f["FOS"].to_numpy(float)
                                         / np.exp(m.predict(f[list(config.FEATURES)]))) ** 2))
               for n, m in screeners}
           for k, f in (("test-natural", natural), *reps.items())}
    print(pd.DataFrame(per).T.to_string(float_format=lambda v: f"{v:.3f}"))

    # The floor: what the RF does to candidates it predicts into the lowest selection bins.
    print("\n== points PREDICTED into the lowest three selection bins (what the gate lets through)")
    allp = pd.concat([natural, *reps.values()], ignore_index=True)
    X, y = allp[list(config.FEATURES)], allp["FOS"].to_numpy(float)
    for n, m in screeners:
        p = np.exp(m.predict(X))
        sel = sel_bin(p)
        mask = np.isin(sel, (0, 1, 2)) & (p >= LO)
        if not mask.any():
            print(f"  {n:16s} no candidate predicted there")
            continue
        print(f"  {n:16s} n={mask.sum():4d}  true<0.5 {np.mean(y[mask] < LO):.0%}  "
              f"true in same bin {np.mean(sel_bin(y[mask]) == sel[mask]):.0%}  "
              f"median true {np.median(y[mask]):.2f}")


if __name__ == "__main__":
    main()
