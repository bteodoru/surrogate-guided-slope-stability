# Non-converged searches of the revision's sets: sensitivity check

`docs/non-converged-check.md` covers the 22 non-converged rows of the published study (data version 1.0). It predates the revision's sets (`data/revision/`, data version 1.1). Eight of their rows also have `Search_Converged = False`: the `non_converged` lists of `data/revision/run_metadata.json`. This note records the same check for those eight.

## Method

- **Script.** `check.py` solves each case through `sgs.solver.calculate_fos`. Only `config.SOLVER.search_max_iter` is replaced, with the values 50, 100, 200 and 400.
- **Platform and model.** The runs used two worker processes on darwin arm64, the platform on which the sets were solved. None of the eight was repaired, so each is solved on its original model extent.
- **Reproduction.** At `max_iter = 50`, all eight reproduced the stored FOS to every printed digit.
- **Nothing in `data/` was modified.** The output is `results.csv`. `sgs curate` reads it for the `FOS_converged` and `Converged_Max_Iter` columns of the curated deposit, version 1.1 (`curate.revision_reruns`).

Run on 2026-11-20, at the user's request.

## Results

Row numbers are 0-based positions in `data/revision/final/<part>.csv`.

| Part | Row | Stored FOS (50 it.) | Converged FOS | Converged at | Relative Δ |
|---|---:|---:|---:|---:|---:|
| test_natural_large_p02 | 271 | 2.693436 | 2.693435 | 100 | −3.7e−7 |
| test_natural_large_p03 | 37 | 1.477627 | 1.477617 | 100 | −6.8e−6 |
| test_natural_large_p03 | 580 | 4.799390 | 4.799356 | 100 | −7.1e−6 |
| test_natural_large_p04 | 245 | 2.574162 | 2.574151 | 100 | −4.3e−6 |
| test_stratified_large_source_p01 | 478 | 3.050992 | 3.050956 | 100 | −1.2e−5 |
| test_stratified_large_source_p01 | 593 | 0.542444 | 0.542315 | 100 | −2.4e−4 |
| test_stratified_large_source_p01 | 682 | 2.347521 | 2.347411 | 100 | −4.7e−5 |
| test_stratified_large_source_p03 | 112 | 0.619243 | 0.618968 | 100 | −4.4e−4 |

"Converged at" is the smallest tested `max_iter` whose search converged. Every larger `max_iter` tested returned the same FOS.

## Reading

- **All eight converge within 100 iterations.**
- **FOS never increases.** It decreases in all eight cases.
- **The largest change is 4.4 × 10⁻⁴ (relative).** It is at test_stratified_large_source_p03/112, FOS 0.62. This is below the 6.8 × 10⁻⁴ of the earlier note, so over all 30 non-converged rows in data version 1.1 the maximum stays 6.8 × 10⁻⁴.
- **No case changes reporting zone** (< 0.5 | 0.5–1 | 1–2 | 2–6 | ≥ 6).
- **No case changes geometric selection bin** on [0.5, 6]. Those bins are the ones from which T-stratL was drawn, on solved FOS. None of the four source cases is among the 450 drawn points, and the draw would come out the same with the converged values.

The stored rows are kept unchanged, for the reasons given in `docs/non-converged-check.md`.
