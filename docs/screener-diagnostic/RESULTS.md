# Screener diagnostic on independent points — 2026-09-23

Run on `main` at `469737a`, i.e. on the re-solved data of the final rerun.
Read-only, no solver. Script: `screener_on_independent_points.py`.

**Moved here on 2026-09-28** from `docs/superpowers/notes/2026-09-23-diagnostics/`, which is not tracked, because the manuscript cites this diagnostic (Methods, C-logG).
- **Script change:** one line only, the `family` argument that `design.fit_screener` has taken since the GBM-screener arm.
- **Check:** re-run on 2026-09-28 from the repository root with `PYTHONPATH=. .venv/bin/python docs/screener-diagnostic/screener_on_independent_points.py`. It reproduces every number below.
- **References to untracked notes:** the GBM-screener handoff and the 2026-09-19 diagnostics are internal notes, not part of the repository. The rest of this file stands on its own.
- **The natural test set** is read here to compare screeners, never to fit, tune or select a surrogate.

This answers the limit recorded in the GBM-screener handoff §2: the earlier
comparison scored both screeners on the points the **RF had selected**, so it
was conditional on the RF's own selection. Here both screeners, trained on the
seed alone, are scored on solved LHS points **neither** selected: `test-natural`
(300) and the replication sets `lhs_r1..r5` (1500).

## Reconfirmation of the old script, after the rerun

`2026-09-19-diagnostics/screener_rf_vs_gbm.py`, on `guided-log`:
RF log-RMSE **0.215**, GBM **0.110** — unchanged from the note.
Zone agreement reads 74% / 87%, not the 85% / 91% quoted in §2: that figure
predates the 2026-09-19 change of `ZONE_EDGES` and is not comparable.

## On 1800 independent LHS points

| | RF (used) | GBM (`LC_PARAMS`) |
|---|---|---|
| log-RMSE | 0.153 | **0.079** |
| log-bias | −0.007 | +0.003 |
| same selection bin (15 bins) | 48.9% | **69.6%** |
| mean \|bin displacement\| | 0.60 | **0.31** |
| same reporting zone | 87.0% | 93.8% |
| median pred/true, true < 0.6 | 1.29 | 1.10 |
| median pred/true, true > 6 | 0.89 | 0.94 |

Per set, with no overlap between the two screeners' ranges:

| set | RF | GBM |
|---|---|---|
| test-natural | 0.169 | 0.088 |
| lhs_r1 | 0.143 | 0.072 |
| lhs_r2 | 0.154 | 0.076 |
| lhs_r3 | 0.141 | 0.078 |
| lhs_r4 | 0.146 | 0.076 |
| lhs_r5 | 0.161 | 0.085 |

## What the selection gate would let through

Candidates **predicted** into the lowest three selection bins — the ones the
guided rule is most eager to buy:

| | n | truly below 0.5 | truly in the predicted bin | median true FOS |
|---|---|---|---|---|
| RF | 77 | **31%** | 22% | 0.59 |
| GBM | 98 | **14%** | 51% | 0.65 |

This is the mechanism behind the 45 guided-log points that realized below
FOS 0.5, measured without conditioning on the RF's own selection.

## A correction the handoff needs

§3b states that *the RF never predicts below about 0.50*. On these independent
points its minimum prediction is **0.343** (GBM: 0.240). There is no hard
floor; what there is, is **end compression** — the RF over-predicts low points
by about 29% in the median against the GBM's 10%. The conclusion of §3b is
unaffected, and if anything better supported by the bin-crossing rows above,
but the sentence itself is false as written and must not reach the manuscript.

## Scope of this diagnostic

The evaluation points are LHS draws over the parameter space, so they are
unconditional on either screener's selection but still carry the LHS
distribution of FOS, not the guided pool's. It estimates screener behaviour
over the sampled domain, not over the candidates the rule ends up preferring.
It says nothing about which candidates either screener picks inside a bin.
