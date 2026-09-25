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

## Files

| file | role |
|---|---|
| `seed_lhs_300.csv` | seed, arm A |
| `lhs_continued_300.csv` | LHS extension, arm B |
| `surrogate_guided_300.csv` | guided selection, arms C and C₀ |
| `surrogate_guided_log_300.csv` | guided-log selection, arms C-log and C₀-log |
| `surrogate_guided_seq1_100.csv`, `surrogate_guided_seq2_100.csv`, `surrogate_guided_seq3_100.csv` | sequential guided-log batches, arms C-seq and C₀-seq |
| `test_natural_300.csv` | natural-300 test |
| `test_stratified_150.csv` | target-stratified-150 test |

## Pool and pool-backed replication files

- `data/pool/{designs,results,final}/pool_cNNN.csv`: the reference pool, uniform i.i.d. over the
  domain (seed `pool_design`), solved in chunks of 1 000 by `sgs pool` until the stopping rule in
  `config.py` was met. Chunks after the stop are never solved.
- `data/replications/*/guided_oracle_rN.csv`, `guided_log_p_rN.csv`: C-oracle and C-logP, rows
  copied from sub-pool N of the reference pool (no solver).

## Surface geometry columns

`Surface_L` and `Surface_R` bound the clipped critical arc. `Surface_Min_Y` is
its lowest point, and the domain-bottom check uses it. `Center_X`,
`Center_Y` and `Radius` define the full circle. `Circle_Nadir_Y` is that circle's lowest
point (`Center_Y − Radius`, xslope's `Depth`). It lies on the arc only when the centre is
above it; for a toe circle it is below the failure surface. Before the final
rerun this column was named `Surface_Depth` and fed the bottom check. That was
conservative and never produced a false flag.

## Non-converged searches

Rows whose circular search stopped at the iteration limit
(`Search_Converged = False`) are kept. Each set's entry in `run_metadata.json`
lists them under `non_converged`. The five present before the final rerun,
solved again to convergence, moved in the fourth decimal of FOS. The same
check on the pool (c001–c015, 2026-09-25) is in
`docs/non-converged-check.md`. The pool cases also converged within 200
iterations, and no case changed FOS bin or zone. That file is updated when the
pool stops.

Rows with a non-empty `excluded` column (`solver_error`, `unresolved`) are kept
in `final/` for transparency and dropped at load time.
