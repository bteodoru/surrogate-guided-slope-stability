# Screener diagnostic on independent points

Run on `main` at `469737a`, on the re-solved data of the final rerun. Read-only, no solver. The manuscript cites it (Methods, C-logG).

- **Reproduce:** from the repository root, `PYTHONPATH=. .venv/bin/python docs/screener-diagnostic/screener_on_independent_points.py`. It reproduces every number below.
- **The natural test set** is read here to compare screeners, never to fit, tune or select a surrogate.

An earlier comparison scored both screeners on the points the **RF had
selected**, so it was conditional on the RF's own selection. Here both
screeners, trained on the seed alone, are scored on solved LHS points
**neither** selected: `test-natural` (300) and the replication sets
`lhs_r1..r5` (1500).

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

## No hard floor in the RF's predictions

On these independent points the RF's minimum prediction is **0.343** (GBM:
0.240), so the RF has no hard floor near 0.5. What it has is **end
compression**: it over-predicts low points by about 29% in the median, against
the GBM's 10%.

## Scope of this diagnostic

The evaluation points are LHS draws over the parameter space, so they are
unconditional on either screener's selection but still carry the LHS
distribution of FOS, not the guided pool's. It estimates screener behaviour
over the sampled domain, not over the candidates the rule ends up preferring.
It says nothing about which candidates either screener picks inside a bin.
