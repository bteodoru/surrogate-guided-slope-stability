# Non-converged searches: sensitivity check

Some kept rows have `Search_Converged = False`. In those rows the circular search stopped at `config.SOLVER.search_max_iter` (50) before it met its tolerances (`search_tol` 0.01, `search_fs_tol` 0.0005). These rows are kept, flagged, and listed under `non_converged` in `run_metadata.json` (see `data/README.md`). This note records the check that justifies keeping them unchanged.

## Method

Each case was solved again, outside the pipeline, with `scripts/xslope_colab.py` (not versioned). The script is a self-contained copy of `sgs/solver.py`: same xslope 0.5.2, same model, same search window and same `config.SOLVER` settings.

- The runs used `max_iter` = 50, 100, 200 and 400. Nothing else changed.
- They ran on Colab (x86), while the data were solved on darwin arm64.
- At `max_iter = 50`, all 14 cases reproduced the stored FOS to every printed digit.

Nothing in `data/` was modified. The pipeline reads nothing this check produced.

## Results (2026-09-25)

All five cases in the study sets and all nine in pool chunks c001–c015. Row numbers are 0-based positions in `data/final/<set>.csv` and `data/pool/final/<chunk>.csv`.

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

"Converged at" is the smallest tested `max_iter` whose search converged. Every larger `max_iter` tested returned the same FOS. The c015 run printed up to 200 only.

## Reading

- **All 14 converge within 200 iterations.** Twelve converge within 100.
- **FOS decreases in every case.** The stored value lies slightly above the converged minimum, as expected when a minimum search is stopped early.
- **The largest change is small.**
  - Relative: 5.7 × 10⁻⁴, at pool_c012/658, the only case in the decision range [0.5, 2).
  - Absolute: 7.05 × 10⁻⁴, at lhs_continued_300/85.
  - Both are of the order of the search's own FOS tolerance (5 × 10⁻⁴).
  - On the log scale used for training and evaluation, |Δ log FOS| ≤ 5.7 × 10⁻⁴. That is at least two orders of magnitude below the surrogates' errors.
- **No case changes FOS bin or reporting zone.** This covers the 15 geometric oracle bins on [0.5, 6] and the zones < 0.5 | 0.5–1 | 1–2 | 2–6 | ≥ 6. The pool's stopping rule and the oracle binning would come out the same with the converged values.

## Why the rows are not replaced

- **One solver configuration.** Every row comes from the same solver configuration. Replacing only the flagged rows would mix two protocols, and the choice of which rows to replace would depend on a property of the result.
- **Reproducibility.** Regenerating from `sgs/config.py` reproduces the data exactly, and the recorded pool chunk hashes must keep verifying.
- **No effect on the study.** The changes above alter no bin, no zone and no conclusion.

## To update

When the pool stops, extract the non-converged rows of the remaining chunks, check them the same way, and add them to the table. The manuscript's Methods sentence quotes the final counts and the maximum change (see `paper/HANDOFF.md`).
