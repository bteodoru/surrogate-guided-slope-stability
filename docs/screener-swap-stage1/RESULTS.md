# Stage 1 of the screener swap — training data only

Spec: `2026-09-23-gbm-screener-arm-design.md` §10.1 (local design notes, not tracked).
Script: `stage1.py` beside this note, run as
`PYTHONPATH=. .venv/bin/python docs/screener-swap-stage1/stage1.py`
at commit caf7a04 (guided_log_g_r1..r5 solved), on 2026-09-23.

**Recorded before any test evaluation of C-logG.** At the time of the run:
- the replication fits in `artifacts/replications/` all predate the C-logG solves
  (11:15–11:20; the widened fingerprint makes them stale, and `fit_all` refused them
  with `StaleFit`, so `evaluate.run_replications` never ran);
- no file under `artifacts/` or `outputs/` was newer than `data/run_metadata.json`,
  and none mentioned C-logG;
- `replications.json` and the test sets are not read by the script. Its inputs are the
  designs (`FOS_pred`, predicted bin), the solved finals (true FOS) and the allocation
  metadata, all through `sgs.structural`.

Δ = GBM − RF, paired per realization: C-logG minus C-log, which share the seed, the
candidate pool (`REP_SEED_STREAM` aliases guided-log-g to guided-log) and the intended
allocation, and differ only in the screener. Binning is the generator's, `(lo, hi]`, the
first bin closed at both ends; points outside [0.5, 6] are excluded from the 15 bin
counts and reported in the last two rows. "Stayed" is the share of all 300 selected
points whose true FOS falls in the bin their prediction implied (a point that left the
range counts as not stayed); mean |displacement| is over the points that stayed inside
[0.5, 6]. r0 is exploratory and enters no threshold.

These are descriptive quantities of what the allocation rule requested and what it
received. They are net distortion of the requested distribution, not prediction error,
and they say nothing yet about test accuracy (stage 2, §10.2).

## Per realization (RF = C-log, GBM = C-logG, Δ = GBM − RF)

| quantity | r1 RF / GBM / Δ | r2 RF / GBM / Δ | r3 RF / GBM / Δ | r4 RF / GBM / Δ | r5 RF / GBM / Δ |
|---|---|---|---|---|---|
| Σ|achieved − intended| (points) | 94 / 49 / -45 | 76 / 55 / -21 | 100 / 50 / -50 | 88 / 43 / -45 | 93 / 50 / -43 |
| stayed in predicted bin (pp) | 36.67 / 64.67 / +28.00 | 35.33 / 58.33 / +23.00 | 31.33 / 59.67 / +28.33 | 34.00 / 62.33 / +28.33 | 38.00 / 62.67 / +24.67 |
| mean |displacement| (bins) | 0.64 / 0.30 / -0.34 | 0.71 / 0.33 / -0.38 | 0.72 / 0.35 / -0.37 | 0.70 / 0.33 / -0.37 | 0.64 / 0.34 / -0.30 |
| realized FOS < 0.5 (points) | 49 / 23 / -26 | 36 / 30 / -6 | 48 / 24 / -24 | 45 / 19 / -26 | 40 / 15 / -25 |
| realized FOS > 6 (points) | 25 / 4 / -21 | 22 / 13 / -9 | 22 / 8 / -14 | 23 / 6 / -17 | 23 / 7 / -16 |

## Verdicts over r1..r5 (spec §10.1)

| quantity | mean Δ | Δ per realization | anticipated sign | threshold | met |
|---|---|---|---|---|---|
| Σ|achieved − intended| (points) | -40.80 | -45, -21, -50, -45, -43 | < 0 in 5/5 | mean ≤ -10, sign in ≥ 4/5 | yes |
| stayed in predicted bin (pp) | +26.47 | +28.00, +23.00, +28.33, +28.33, +24.67 | > 0 in 5/5 | mean ≥ +10, sign in ≥ 4/5 | yes |
| mean |displacement| (bins) | -0.35 | -0.34, -0.38, -0.37, -0.37, -0.30 | < 0 in 5/5 | mean ≤ -0.15, sign in ≥ 4/5 | yes |
| realized FOS < 0.5 (points) | -21.40 | -26, -6, -24, -26, -25 | — | directional only | — |
| realized FOS > 6 (points) | -15.40 | -21, -9, -14, -17, -16 | — | directional only | — |

## r0 (exploratory, enters no threshold)

| quantity | RF | GBM | Δ |
|---|---|---|---|
| Σ|achieved − intended| (points) | 92 | 44 | -48 |
| stayed in predicted bin (pp) | 36.00 | 63.33 | +27.33 |
| mean |displacement| (bins) | 0.66 | 0.32 | -0.34 |
| realized FOS < 0.5 (points) | 45 | 20 | -25 |
| realized FOS > 6 (points) | 19 | 8 | -11 |

## Shared candidates (points of C-logG within MIN_DISTANCE of a C-log point)

- r0: 4/300
- r1: 2/300
- r2: 2/300
- r3: 2/300
- r4: 0/300
- r5: 3/300

## Reading

All three thresholded stage-1 quantities meet their declared thresholds, each with the
anticipated sign in 5 of 5 realizations:

- **Primary.** Σ|achieved − intended| over the 15 selection bins: mean Δ = −40.8 points
  (threshold ≤ −10). With the GBM screener the achieved allocation departs from the
  intended one by about half as much as with the RF (43–55 against 76–100 points).
- **Secondary.** Share staying in the predicted bin: mean Δ = +26.5 pp (threshold ≥ +10).
  Mean |displacement|: mean Δ = −0.35 bins (threshold ≤ −0.15).
- **Directional.** Fewer selected points realize outside the selection range with the
  GBM screener, both below 0.5 (mean Δ −21.4) and above 6 (mean Δ −15.4), in every
  realization.

r0 falls within the r1..r5 spread on every row.

The two screeners pick almost disjoint points from the shared pool (0–4 of 300 within
`MIN_DISTANCE`), so C-logG is a different training set, not a perturbation of C-log.

What this note does not say: whether the closer allocation changes test accuracy or the
relocation pattern. That is stage 2, which reads the test sets and is run after this
note is committed. Nothing here isolates the screener's error from the allocation rule
in the sense the oracle would; it compares two screeners under one rule.
