# data/

Everything in this directory is produced by `sgs generate`. CSV files are not
tracked in git; `run_metadata.json` is. For every set it records the seeds,
the SHA-256 of the design, results and final files, the repair summary, the
screener diagnostics where a screener was used, and the Python and library
versions.

| directory   | content                                              |
|-------------|------------------------------------------------------|
| `designs/`  | parameter sets, before the solver                    |
| `results/`  | raw xslope/Bishop output, never modified by repair   |
| `final/`    | boundary rows repaired; consumed by the analysis     |

## File names and the Zenodo deposit

The first Zenodo versions (concept DOI 10.5281/zenodo.20706202) used the old
names below. Two of them are misleading: `results_uniform_bishop.csv` holds
the *guided* set, and `results_test_bishop.csv` holds the *natural* test.

| old name (Zenodo ≤ previous version) | new name                   | role                        |
|--------------------------------------|----------------------------|-----------------------------|
| `initial_lhs_300_bishop.csv`         | `seed_lhs_300.csv`         | seed, arm A                 |
| `results_lhs_bishop.csv`             | `lhs_continued_300.csv`    | LHS extension, arm B        |
| `results_uniform_bishop.csv`         | `surrogate_guided_300.csv` | guided selection, C and C₀  |
| `results_test_bishop.csv`            | `test_natural_300.csv`     | natural-300 test            |
| `results_test_150_bishop.csv`        | `test_stratified_150.csv`  | target-stratified-150 test  |

Rows with a non-empty `excluded` column (`solver_error`, `unresolved`) are kept
in `final/` for transparency and dropped at load time.
