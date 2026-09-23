# Surrogate-guided sampling for slope-stability surrogates

Research code for a study of how a fixed budget of physics-based slope-stability
analyses should be allocated across the response space. A preliminary random
forest (the screener), trained on a Latin hypercube seed, screens a large
candidate pool. The remaining budget goes to candidates chosen to even out the
factor-of-safety (FOS) distribution of the training set over the selection range
[0.5, 6.0]. The paper calls this strategy *response-balanced sampling* and
compares it, at the same budget, against continued Latin hypercube sampling.

The finding: at a fixed budget, response-balanced sampling does not
systematically improve accuracy over uniform sampling of the parameter space.
Instead it relocates accuracy along the FOS axis, following the training density
actually achieved. The relocation holds across five replicated designs (lower
error in FOS 0.5–1, higher in 1–2). On the primary criterion, no guided arm
shows an advantage.

Data are generated with xslope 0.5.2 (Bishop's simplified method), homogeneous
drained slopes without a water table.

| Arm   | Training data                                    | Solver budget       |
|-------|--------------------------------------------------|---------------------|
| A     | seed LHS-300                                     | 300                 |
| B     | seed LHS-300 + LHS-300                           | 600                 |
| C     | seed LHS-300 + guided-300 (linear bins, max-min) | 600                 |
| C₀    | guided-300 only                                  | ablation, no budget |
| C-log | seed LHS-300 + guided-log-300                    | 600                 |
| C₀-log| guided-log-300 only                              | ablation, no budget |
| C-seq | seed LHS-300 + 3 × guided-log-100, screener retrained before each batch | 600 |
| C₀-seq| the three sequential batches only                | ablation, no budget |
| C-logG | seed LHS-300 + guided-log-g-300: C-log's rule and candidate pool, GBM screener | 600 |
| C₀-logG| guided-log-g-300 only                           | ablation, no budget |

> guided-log fills the seed's deficit in 15 geometric log(FOS) bins over the selection range, with random selection inside each bin. C-seq runs the same procedure in three batches of 100. Before each batch the screener is refitted, and the deficit recomputed, on the solved FOS of the seed and every earlier batch. C-logG repeats C-log with a gradient-boosting screener in place of the random forest, drawing from the same candidate pool against the same intended allocation, so only the screener differs.
>
> Results are reported in the zones FOS < 0.5 | 0.5–1 | 1–2 | 2–6 | ≥ 6. The primary criterion is log-RMSE on the *decision range* [0.5, 2) (defined after the C-log results, before C-seq); the FOS scale is secondary.

Two test sets represent two deployment scenarios: `test-natural` (input-space
LHS, 300) and `test-stratified` (random selection within predicted-FOS bins,
150), each disjoint from every other set.

## Running the pipeline

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install --no-deps -e .
.venv/bin/sgs generate --workers 2   # ten data sets, about three hours of solver time with 2 workers
.venv/bin/sgs fit                    # 40 fits (10 arms × 2 families × 2 scorings), each with a nested CV; hours
.venv/bin/sgs evaluate               # seconds
.venv/bin/sgs report                 # tables and figures
```

On macOS the files under `.venv` can carry the `hidden` flag, and Python 3.14
skips hidden `.pth` files, so the editable install is ignored and `sgs` fails
with `No module named 'sgs'`. Clearing it with `chflags nohidden` need not last:
the flag can come back within minutes. Run from the repository root with
`PYTHONPATH=.` (e.g. `PYTHONPATH=. .venv/bin/sgs report`).

`sgs all --workers 2` runs the four in order. `sgs learning-curve` runs the
optional RF and GBM learning curves (fixed hyper-parameters, not part of `sgs all`).
`sgs replicate --workers 2` is optional as well and is not part of `sgs all`: it
generates and solves five further realizations of the points each strategy adds
to the fixed seed (B, C-log, C-seq, C-logG), fits them with common fixed
hyper-parameters and writes the interval metrics. Summed over the per-row
`Runtime_s` of `data/replications/final/`, the 6000 replicated rows cost about
nine hours of solver time, so roughly four and a half hours of wall clock with 2 workers.
A command reuses outputs that already exist and stops, naming the cause, if a
design, a results file or a fit no longer matches the current configuration or
data; `--force` regenerates; under `sgs replicate` it also re-solves every replicated
set, so to refit the replications alone call `replicate.fit_all(force=True)`. All scientific parameters live in
`sgs/config.py`; none are accepted on the command line.

`sgs structural` is optional as well and is not part of `sgs all`: it reads
`outputs/metrics/replications.json`, `data/run_metadata.json` and the stored
designs, and writes `outputs/metrics/structural.json` in seconds. It touches no
generation path and refits nothing. Its procedure is declared in
`docs/predictions-and-outcomes.md`, and what it may and may not claim is in
`CLAUDE.md`.

## Data

The xslope data sets are not yet published; `data/README.md` describes the
files. The earlier HYRCAN data are on Zenodo, concept DOI
[10.5281/zenodo.20706202](https://doi.org/10.5281/zenodo.20706202).

The initial approach of this study (HYRCAN/Spencer generator, single Colab
notebook) is preserved at the git tag `v0-hyrcan-colab`, and the HYRCAN
generator's last state at `hyrcan-final`.

Declared predictions and their observed outcomes: docs/predictions-and-outcomes.md.

## Notebook

`notebooks/surrogate_guided_sampling.ipynb` holds the narrative and figures. It imports
`sgs.report` and reads `outputs/metrics/`; it fits nothing and computes no metric. On
Colab its first cell clones the repository, which already contains the metrics.
