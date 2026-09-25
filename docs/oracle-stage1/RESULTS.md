# Stage 1 of the oracle — training data only

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

| r | C-oracle Σ\|dev\| | < 0.5, > 6 | stayed | C-logP Σ\|dev\| | C-log Σ\|dev\| | C-logP < 0.5, > 6 | C-logP stayed | ρ_L | ρ_H | shared rows |
|---|---|---|---|---|---|---|---|---|---|---|
| r1 | 0 | 0, 0 | 100% | 90 | 94 | 37, 25 | 29.7% | 1.217 | 1.096 | 48 |
| r2 | 0 | 0, 0 | 100% | 99 | 76 | 54, 25 | 28.3% | 1.367 | 1.075 | 49 |
| r3 | 0 | 0, 0 | 100% | 85 | 100 | 43, 26 | 31.7% | 1.339 | 1.013 | 47 |
| r4 | 0 | 0, 0 | 100% | 88 | 88 | 40, 28 | 31.3% | 1.207 | 1.090 | 40 |
| r5 | 0 | 0, 0 | 100% | 74 | 93 | 40, 26 | 35.7% | 1.227 | 1.088 | 42 |

## Verdicts (`docs/oracle-declaration.md` §9.1)

- **Construction check, C-oracle:** Σ|achieved − intended| = 0, no point outside [0.5, 6], 100% in its
  bin, in every realization (asserted by the script).
- **Sanity check, C-logP − C-log on Σ|achieved − intended|:** mean -3.0 points, Welch 95%
  interval [-16.2, 10.2] → met
  (needs the interval to contain 0 and |mean| ≤ 15).
- **ρ (binding for stage 2):** ρ_L = 1.227, ρ_H = 1.088 (declared expectation 1.27 and 1.06).
  Density prediction: L → dZ_O < 0; H → none.

**Candidate pools differ in one more respect than size.** C-log and C-logG select from 200k unsolved
candidates, so a selected point that the solver then excludes is lost from their training set. C-oracle
and C-logP select only among pool rows that were already solved and kept (eligible rows). For C-logP this
is a small head start over C-log that owes nothing to the screener: it cannot draw a point that would fail
to solve. It is negligible here (C-log lost 0 selected points to exclusion over r1-r5 (0, 0, 0, 0, 0); 0 of 22000 pool rows were excluded before selection), but it belongs to "differs also in the candidate pool",
never to the screener. It does not touch the paired contrast C-oracle vs C-logP, since both
draw from the same eligible rows.

What this note does not say: anything about test accuracy or relocation, which are stage 2.
