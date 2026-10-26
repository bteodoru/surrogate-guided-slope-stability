# Read-only, no solver, no test data. docs/oracle-declaration.md §9.1:
# stage 1 of the oracle, recorded before any C-oracle or C-logP fit exists. Inputs are the
# designs, the solved finals and the allocation metadata only. Allocation quantities bin as
# the generator does, (lo, hi]; rho uses the evaluation zones [lo, hi).
from pathlib import Path

import numpy as np

from sgs import config, data, design, evaluate, replicate, structural

R = list(range(1, config.REP_REALIZATIONS + 1))
EDGES = structural.EDGES()
META = data.read_metadata()["replications"]


def achieved(key):
    return design.bin_counts(replicate.load_set(key)["FOS"].to_numpy(float), EDGES)


def outside(key):
    fos = replicate.load_set(key)["FOS"].to_numpy(float)
    return int((fos < EDGES[0]).sum()), int((fos > EDGES[-1]).sum())


def stayed(key):
    d, f = replicate.read_set_design(key), replicate.load_set(key)
    return float(np.mean(d["bin"].to_numpy(int) == structural._true_bin(f["FOS"].to_numpy(float))))


def zone_counts(key):
    fos = np.concatenate([data.load_role("seed")["FOS"].to_numpy(float),
                          replicate.load_set(key)["FOS"].to_numpy(float)])
    return int(((fos >= 0.5) & (fos < 1.0)).sum()), int(((fos >= 1.0) & (fos < 2.0)).sum())


rows, dev = [], {"o": [], "p": [], "c": []}
for r in R:
    o, p, c = f"guided_oracle_r{r}", f"guided_log_p_r{r}", f"guided_log_r{r}"
    intended = np.asarray(META[o]["allocation"]["added"])
    assert np.array_equal(intended, META[p]["allocation"]["added"])
    for tag, key in (("o", o), ("p", p), ("c", c)):
        dev[tag].append(int(np.abs(achieved(key) - intended).sum()))
    (oL, oH), (pL, pH) = zone_counts(o), zone_counts(p)
    shared = len(set(replicate.read_set_design(o)["pool_row"]) & set(replicate.read_set_design(p)["pool_row"]))
    rows.append({"r": r, "o_dev": dev["o"][-1], "o_out": outside(o), "o_stayed": stayed(o),
                 "p_dev": dev["p"][-1], "c_dev": dev["c"][-1], "p_out": outside(p), "p_stayed": stayed(p),
                 "rho_L": oL / pL, "rho_H": oH / pH, "shared": shared})
    # construction check (declaration §9.1, first row): asserted, not predicted
    assert dev["o"][-1] == 0 and outside(o) == (0, 0) and stayed(o) == 1.0, f"r{r}: C-oracle construction"

welch = evaluate.welch_interval(dev["p"], dev["c"], config.REP_CI)
sanity = welch["ci"][0] <= 0 <= welch["ci"][1] and abs(welch["diff"]) <= 15
rho_L = float(np.median([x["rho_L"] for x in rows]))
rho_H = float(np.median([x["rho_H"] for x in rows]))
ratio = config.ORACLE_DENSITY_RATIO
# Rows the solver excluded: in the C-log sets (lost after selection) and in the pool
# chunks up to the stop (never eligible). Counts only; no test data.
c_log_lost = [config.REP_EXTENSION - len(replicate.load_set(f"guided_log_r{r}")) for r in R]
pool_meta = data.read_metadata()["pool"]
pool_excluded = sum(pool_meta[f"c{j:03d}"]["n_excluded"] for j in range(1, pool_meta["stop"]["k"] + 1))
excluded_note = (f"C-log lost {sum(c_log_lost)} selected points to exclusion over r1-r5 "
                 f"({', '.join(map(str, c_log_lost))}); {pool_excluded} of "
                 f"{pool_meta['stop']['k'] * config.POOL_CHUNK} pool rows were excluded before selection")
direction = lambda rho: "dZ_O < 0" if rho >= ratio else "dZ_O > 0" if rho <= 1 / ratio else "none"

lines = ["| r | C-oracle Σ\\|dev\\| | < 0.5, > 6 | stayed | C-logP Σ\\|dev\\| | C-log Σ\\|dev\\| | C-logP < 0.5, > 6 "
         "| C-logP stayed | ρ_L | ρ_H | shared rows |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for x in rows:
    lines.append(f"| r{x['r']} | {x['o_dev']} | {x['o_out'][0]}, {x['o_out'][1]} | {100 * x['o_stayed']:.0f}% "
                 f"| {x['p_dev']} | {x['c_dev']} | {x['p_out'][0]}, {x['p_out'][1]} | {100 * x['p_stayed']:.1f}% "
                 f"| {x['rho_L']:.3f} | {x['rho_H']:.3f} | {x['shared']} |")
text = f"""# Stage 1 of the oracle — training data only

Declared text: `docs/oracle-declaration.md` §9.1 (committed with the tag `oracle-declared`).
Script: `stage1.py` beside this note, run as `PYTHONPATH=. .venv/bin/python docs/oracle-stage1/stage1.py`.

**Recorded before any fit of C-oracle or C-logP.** The replication fits under `artifacts/replications/`
predate the new sets and are stale under the fingerprint declared in `config.py`, so
`evaluate.run_replications` cannot have run on them. Inputs: the designs, the solved finals and the
allocation metadata. No test set is read.

Allocation quantities bin as the generator does, `(lo, hi]`, the first bin closed. ρ counts the whole
training set (seed + selection) in the evaluation zones `[lo, hi)`: L = [0.5, 1), H = [1, 2).
These are net distortion of the requested distribution, not prediction error.

## Per realization

{chr(10).join(lines)}

## Verdicts (`docs/oracle-declaration.md` §9.1)

- **Construction check, C-oracle:** Σ|achieved − intended| = 0, no point outside [0.5, 6], 100% in its
  bin, in every realization (asserted by the script).
- **Sanity check, C-logP − C-log on Σ|achieved − intended|:** mean {welch['diff']:+.1f} points, Welch 95%
  interval [{welch['ci'][0]:.1f}, {welch['ci'][1]:.1f}] → {'met' if sanity else 'NOT met'}
  (needs the interval to contain 0 and |mean| ≤ 15).{'' if sanity else ' The paired contrast C-oracle vs C-logP is unaffected; statements comparing with C-log carry "differs also in the candidate pool".'}
- **ρ (binding for stage 2):** ρ_L = {rho_L:.3f}, ρ_H = {rho_H:.3f} (declared expectation 1.27 and 1.06).
  Density prediction: L → {direction(rho_L)}; H → {direction(rho_H)}.

**Candidate pools differ in one more respect than size.** C-log and C-logG select from 200k unsolved
candidates, so a selected point that the solver then excludes is lost from their training set. C-oracle
and C-logP select only among pool rows that were already solved and kept (eligible rows). For C-logP this
is a small head start over C-log that owes nothing to the screener: it cannot draw a point that would fail
to solve. It is negligible here ({excluded_note}), but it belongs to "differs also in the candidate pool",
never to the screener. It does not touch the paired contrast C-oracle vs C-logP, since both
draw from the same eligible rows.

What this note does not say: anything about test accuracy or relocation, which are stage 2.
"""
Path(__file__).with_name("RESULTS.md").write_text(text)
print(text)
