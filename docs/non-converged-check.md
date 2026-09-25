# Non-converged searches: sensitivity check

Some kept rows have `Search_Converged = False`. In those rows the circular search stopped at `config.SOLVER.search_max_iter` (50) before it met its tolerances (`search_tol` 0.01, `search_fs_tol` 0.0005). These rows are kept, flagged, and listed under `non_converged` in `run_metadata.json` (see `data/README.md`). This note records the check that justifies keeping them unchanged.

## Method

Each case was solved again, outside the pipeline, with `scripts/xslope_colab.py` (not versioned). The script is a self-contained copy of `sgs/solver.py`: same xslope 0.5.2, same model, same search window and same `config.SOLVER` settings.

- The runs used `max_iter` = 50, 100, 200 and 400. Nothing else changed.
- They ran on Colab (x86), while the data were solved on darwin arm64.
- At `max_iter = 50`, all 19 cases reproduced the stored FOS to every printed digit.

Nothing in `data/` was modified. The pipeline reads nothing this check produced.

## Results (2026-09-25, complete)

All five cases in the study sets and all 14 in the pool, chunks c001–c022. The pool stopped at c022. The first 14 cases were checked on 2026-09-25 as c001–c015 ran, and the last five after the stop. Row numbers are 0-based positions in `data/final/<set>.csv` and `data/pool/final/<chunk>.csv`.

| Set | Row | Stored FOS (50 it.) | Converged FOS | Converged at | ΔFOS | Relative Δ |
|---|---:|---:|---:|---:|---:|---:|
| lhs_continued_300 | 85 | 3.628919 | 3.628214 | 200 | −7.05e−4 | −1.9e−4 |
| test_stratified_150 | 65 | 2.770601 | 2.770490 | 100 | −1.10e−4 | −4.0e−5 |
| replications/guided_log_r1 | 75 | 0.566515 | 0.566506 | 100 | −9.0e−6 | −1.6e−5 |
| replications/guided_log_r5 | 115 | 0.830120 | 0.829919 | 100 | −2.02e−4 | −2.4e−4 |
| replications/lhs_r1 | 186 | 1.100943 | 1.100866 | 100 | −7.7e−5 | −7.0e−5 |
| pool/pool_c001 | 131 | 2.086449 | 2.086426 | 100 | −2.3e−5 | −1.1e−5 |
| pool/pool_c001 | 937 | 2.178500 | 2.178282 | 100 | −2.18e−4 | −1.0e−4 |
| pool/pool_c003 | 376 | 3.154162 | 3.153745 | 200 | −4.17e−4 | −1.3e−4 |
| pool/pool_c004 | 266 | 1.703435 | 1.703366 | 100 | −6.8e−5 | −4.0e−5 |
| pool/pool_c005 | 910 | 2.051945 | 2.051924 | 100 | −2.1e−5 | −1.0e−5 |
| pool/pool_c006 | 544 | 2.163868 | 2.163847 | 100 | −2.1e−5 | −9.5e−6 |
| pool/pool_c011 | 947 | 2.762275 | 2.762080 | 100 | −1.94e−4 | −7.0e−5 |
| pool/pool_c012 | 658 | 0.707920 | 0.707519 | 100 | −4.02e−4 | −5.7e−4 |
| pool/pool_c015 | 965 | 2.927884 | 2.927800 | 100 | −8.4e−5 | −2.9e−5 |
| pool/pool_c016 | 572 | 4.251066 | 4.251066 | 100 | 0 | 0 |
| pool/pool_c017 | 471 | 2.598604 | 2.598597 | 100 | −6.7e−6 | −2.6e−6 |
| pool/pool_c020 | 115 | 0.638141 | 0.638141 | 100 | 0 | 0 |
| pool/pool_c020 | 621 | 1.453592 | 1.452597 | 200 | −9.95e−4 | −6.8e−4 |
| pool/pool_c022 | 716 | 4.359769 | 4.359493 | 100 | −2.76e−4 | −6.3e−5 |

"Converged at" is the smallest tested `max_iter` whose search converged. Every larger `max_iter` tested returned the same FOS. The c015 run printed up to 200 only, and the c022 run up to 100 only.

## Reading

- **All 19 converge within 200 iterations.** Sixteen converge within 100.
- **FOS never increases.** It decreases in 17 cases and is unchanged in 2 (pool_c016/572 and pool_c020/115), where the search met its tolerance a few circles later at the same minimum. The stored value lies at or slightly above the converged minimum, as expected when a minimum search is stopped early.
- **The largest change is small.**
  - It is at pool_c020/621 (FOS 1.45, in the decision range [0.5, 2)): 9.95 × 10⁻⁴ absolute, 6.8 × 10⁻⁴ relative.
  - Both are of the order of the search's own FOS tolerance (5 × 10⁻⁴).
  - Seven of the 19 cases lie in the decision range, and none of them changed by more than this.
  - On the log scale used for training and evaluation, |Δ log FOS| ≤ 6.8 × 10⁻⁴. That is at least two orders of magnitude below the surrogates' errors.
- **No case changes FOS bin or reporting zone.** This covers the 15 geometric oracle bins on [0.5, 6] and the zones < 0.5 | 0.5–1 | 1–2 | 2–6 | ≥ 6. The pool's stopping rule and the oracle binning would come out the same with the converged values.

## Why the rows are not replaced

- **One solver configuration.** Every row comes from the same solver configuration. Replacing only the flagged rows would mix two protocols, and the choice of which rows to replace would depend on a property of the result.
- **Reproducibility.** Regenerating from `sgs/config.py` reproduces the data exactly, and the recorded pool chunk hashes must keep verifying.
- **No effect on the study.** The changes above alter no bin, no zone and no conclusion.

## Status

Complete. The pool stopped at c022 and every non-converged row in `data/` is in the table above. The manuscript's Methods sentence quotes these counts and the maximum change (see `paper/HANDOFF.md`).
