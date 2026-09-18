# Surrogate-guided sampling for slope-stability surrogates

Research code for a study of how a fixed budget of physics-based slope-stability
analyses should be allocated. A preliminary random forest, trained on a Latin
hypercube seed, screens a large candidate pool; the remaining budget goes to
candidates chosen for uniform coverage of the factor-of-safety (FOS) selection
range [0.5, 6.0]. The claim is tested in a budget-fair comparison against
continued Latin hypercube sampling.

Data are generated with xslope 0.5.2 (Bishop's simplified method), homogeneous
drained slopes without a water table.

| Arm | Training data                  | Solver budget       |
|-----|--------------------------------|---------------------|
| A   | seed LHS-300                   | 300                 |
| B   | seed LHS-300 + LHS-300         | 600                 |
| C   | seed LHS-300 + guided-300      | 600                 |
| C₀  | guided-300 only                | ablation, no budget |

Two test sets represent two deployment scenarios: `test-natural` (input-space
LHS, 300) and `test-stratified` (random selection within predicted-FOS bins,
150), each disjoint from every other set.

## Running the pipeline

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install --no-deps -e .
.venv/bin/sgs generate --workers 2   # five data sets, about an hour of solver time
.venv/bin/sgs fit                    # 8 fits, 48 grid searches with the nested CV; hours
.venv/bin/sgs evaluate               # seconds
.venv/bin/sgs report                 # tables and figures
```

`sgs all --workers 2` runs the four in order. `sgs learning-curve` runs the
optional RF and GBM learning curves (fixed hyper-parameters, not part of `sgs all`).
A command reuses outputs that already exist and stops, naming the cause, if a
design, a results file or a fit no longer matches the current configuration or
data; `--force` regenerates. All scientific parameters live in
`sgs/config.py`; none are accepted on the command line.

## Data

Dataset: [Zenodo, concept DOI 10.5281/zenodo.20706202](https://doi.org/10.5281/zenodo.20706202).
See `data/README.md` for the mapping between file names in the deposit and in
the pipeline.

The HYRCAN generator used in an earlier version of this study is preserved at
the git tag `hyrcan-final`.

## Notebook

`notebooks/surrogate_guided_sampling.ipynb` holds the narrative and figures. It imports
`sgs.report` and reads `outputs/metrics/`; it fits nothing and computes no metric. On
Colab its first cell clones the repository, which already contains the metrics.
