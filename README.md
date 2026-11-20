# Response-balanced sampling for slope-stability surrogates

Research code for a study of how a fixed budget of physics-based slope-stability
analyses should be allocated across the response space. A preliminary random
forest (the screener), trained on a Latin hypercube seed, screens a large
candidate pool. The remaining budget goes to candidates chosen to even out the
factor-of-safety (FOS) distribution of the training set over the selection range
[0.5, 6.0]. The study calls this strategy *response-balanced sampling* and
compares it, at the same budget, against continued Latin hypercube sampling.

The solved analyses are on Zenodo,
[10.5281/zenodo.23618253](https://doi.org/10.5281/zenodo.23618253); how to cite
the code and the data is in `CITATION.cff` and in the section [Citing](#citing).

The finding: at a fixed budget, response-balanced sampling does not
systematically improve accuracy over uniform sampling of the parameter space.
Instead it relocates accuracy along the FOS axis, following the training density
actually achieved. The relocation holds across five replicated designs (lower
error in FOS 0.5–1, higher in 1–2). On the primary criterion, no guided arm
shows an advantage. C-oracle, which is not a budget arm (it costs the solved
reference pool), reaches level 2 against B on the decision range (sustained
improvement: the Welch interval entirely below 0 in one cell) but not level 3
(all four cells), so "no demonstrated gain" stands (row C3,
`docs/oracle-declaration.md` §9.3).

The further designs and checks (see [Further designs and checks](#further-designs-and-checks)) place the target on the
decision range itself: D-P, the arm that could be used in practice, reaches level 2,
not level 3, so "no demonstrated gain" extends to that target, in this design. The
direction of relocation recurs over five LHS seeds and on two larger test sets. Under
a Gaussian process fitted on the same sets, an order of magnitude more accurate,
relocation is not observed: it is stated as a behaviour of the two tree ensembles.

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
| C-oracle | seed LHS-300 + guided-oracle-300: C-log's allocation, drawn from the solved reference pool by solved FOS; replication-only, no r0 | 600, plus the pool |
| C₀-oracle | guided-oracle-300 only; replication-only, no r0 | ablation, no budget |
| C-logP | seed LHS-300 + guided-log-p-300: C-log's allocation, drawn from the same pool by the RF screener's predicted FOS; replication-only, no r0 | 600, plus the pool |
| C₀-logP | guided-log-p-300 only; replication-only, no r0 | ablation, no budget |

> guided-log fills the seed's deficit in 15 geometric log(FOS) bins over the selection range, with random selection inside each bin. C-seq runs the same procedure in three batches of 100. Before each batch the screener is refitted, and the deficit recomputed, on the solved FOS of the seed and every earlier batch. C-logG repeats C-log with a gradient-boosting screener in place of the random forest, drawing from the same candidate pool against the same intended allocation, so only the screener differs.
>
> Results are reported in the zones FOS < 0.5 | 0.5–1 | 1–2 | 2–6 | ≥ 6. The primary criterion is log-RMSE on the *decision range* [0.5, 2) (defined after the C-log results, before C-seq); the FOS scale is secondary.

Two test sets represent two deployment scenarios: `test-natural` (input-space
LHS, 300) and `test-stratified` (random selection within predicted-FOS bins,
150), each disjoint from every other set.

## Running the pipeline

With the deposited data unpacked (see [Data](#data)), the tables and figures
are rebuilt from the tracked metrics in seconds, and the fits from the data in
hours; nothing needs to be solved:

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install --no-deps -e .
.venv/bin/sgs report                 # tables and figures, from outputs/metrics/
.venv/bin/sgs fit                    # 40 fits (10 arms × 2 families × 2 scorings), each with a nested CV; hours
.venv/bin/sgs evaluate               # seconds
```

Skip `sgs generate` and `sgs all` in this case. They find every set up to date
and solve nothing, but they rewrite the records in `data/run_metadata.json`, after
which `sgs report` asks for `sgs structural` to be rerun. To regenerate the data
from scratch:

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install --no-deps -e .
.venv/bin/sgs generate --workers 2   # ten data sets, about three hours of solver time with 2 workers
.venv/bin/sgs fit                    # 40 fits (10 arms × 2 families × 2 scorings), each with a nested CV; hours
.venv/bin/sgs evaluate               # seconds
.venv/bin/sgs report                 # tables and figures
.venv/bin/sgs pool --chunks 5 --workers 2  # optional: solve the reference pool in chunks until its rule is met (--status: read-only)
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
to the fixed seed (B, C-log, C-seq, C-logG, and the pool-backed C-oracle,
C₀-oracle, C-logP, C₀-logP, whose rows are copied from the solved reference
pool rather than solved), fits them with common fixed
hyper-parameters and writes the interval metrics. Summed over the per-row
`Runtime_s` of `data/replications/final/`, the 6000 replicated rows cost about
nine hours of solver time, so roughly four and a half hours of wall clock with 2 workers.
A command reuses outputs that already exist and stops, naming the cause, if a
design, a results file or a fit no longer matches the current configuration or
data; `--force` regenerates; under `sgs replicate` it also re-solves every replicated
set, so to refit the replications alone call `replicate.fit_all(force=True)`. All scientific parameters live in
`sgs/config.py`; none are accepted on the command line.

`sgs pool` is optional as well and is not part of `sgs all`: it solves the
*reference pool*, uniform i.i.d. over the parameter domain and split into five
sub-pools, one per replicated realization, from which the pool arms draw:
C-oracle (bins on solved FOS), C-logP (bins on the seed-trained screener's
prediction) and their ablations C₀-oracle and C₀-logP, all against C-log's
allocation. It solves chunks of 1 000 analyses, at most `--chunks` more per
call, and after each chunk checks the stopping rule declared before the first
chunk (`docs/oracle-declaration.md` §4.4): every sub-pool must hold, in every
one of the 15 log bins, at least as many solved points, and as many points
predicted by the screener, as C-log's allocation asks for. At the first chunk
that meets it, solving stops for good; the rule is never relaxed. It was met at
chunk 22 (22 000 rows). `--status` prints the counts and the rule's state
without solving. The files are described in `data/README.md`; the declared
readings and their outcome are in `docs/predictions-and-outcomes.md`.

`sgs structural` is optional as well and is not part of `sgs all`: it reads
`outputs/metrics/replications.json`, `data/run_metadata.json` and the stored
designs, and writes `outputs/metrics/structural.json` in seconds. It touches no
generation path and refits nothing. Its procedure is declared in
`docs/predictions-and-outcomes.md`, and what it may and may not claim is in
`CLAUDE.md`.

### Further designs and checks

The further designs and checks were declared in
`docs/revision-declaration.md`, committed and tagged `revision-declared` before any
of their sets was drawn or solved; the outcomes are appended to
`docs/predictions-and-outcomes.md`. They add, without changing any earlier set, fit
or result:

- D-oracle and D-P, C-log's procedure with the target on the decision range, and
  C-seqF and C-seqP, which decompose C-seq, all drawn from the solved reference
  pool, with their ablations;
- a seed study: four new LHS seeds, each with its own continued LHS, and C-logP and
  D-P recomputed on each;
- two larger test sets, T-natL (an LHS of 5 000) and T-stratL (30 points per
  selection bin, on the solved FOS of a further LHS of 6 000);
- a two-level bootstrap, the robustness checks (independent retuning, three
  features, no crest entries in training, density reweighting of B, size-matched
  ablations, two alternative decision ranges), a Gaussian process, and the signed
  bias per zone (post hoc, descriptive).

`sgs revision` designs and solves the new LHS sets into `data/revision/`
(`--status` is read-only); the pool-backed sets are copied from the reference pool,
not solved. The analyses have no command of their own: each module's `run()` writes
its entry of `outputs/metrics/revision.json`, in the order of the declaration —
`density`, `revision_pool` (`stage1`, `generate_all`, `generate_seed_sets`),
`revision_fit`, `revision_seeds`, `revision_large`, `revision_bootstrap`,
`revision_robust`, `revision_retune`, `revision_gp`, `revision_bias` — and
`revision_report.run()` writes the tables `outputs/tables/revision__*.tex` from the
metrics alone. Fits go to `artifacts/revision/`.

## Data

The solved analyses are deposited on Zenodo,
[10.5281/zenodo.23618253](https://doi.org/10.5281/zenodo.23618253): every set of
the study, the replications and the reference pool, as designs, raw solver
output and repaired final files. Version 1.1 of the deposit,
[10.5281/zenodo.24374247](https://doi.org/10.5281/zenodo.24374247), adds the
further sets under `data/revision/` (13 400 analyses, and the pool-backed sets
copied from the reference pool), with their provenance in
`data/revision/run_metadata.json`. The CSV files are not tracked here. Download
`data.zip` and unzip it in the repository root. It holds only the CSV files and
unpacks into `data/`, next to the tracked `data/README.md`, which describes the
files, and `data/run_metadata.json`, whose SHA-256 hashes the commands check
before they use a file.

The guided designs depend on the computing platform: the screener's predictions,
and with them the points it selects, differ between x86 and arm64 processors. A
rerun of `sgs generate` on another platform therefore gives different guided
sets. The deposited sets, generated on macOS (arm64), are the reference; the
fits, evaluation and report run from them on any platform.

The earlier HYRCAN data are on Zenodo, concept DOI
[10.5281/zenodo.20706202](https://doi.org/10.5281/zenodo.20706202), which holds
nothing from the xslope runs.

The initial approach of this study (HYRCAN/Spencer generator, single Colab
notebook) is preserved at the git tag `v0-hyrcan-colab`, and the HYRCAN
generator's last state at `hyrcan-final`.

Declared predictions and their observed outcomes: docs/predictions-and-outcomes.md.

## Notebook

`notebooks/surrogate_guided_sampling.ipynb` holds the narrative and figures. It imports
`sgs.report` and reads `outputs/metrics/`; it fits nothing and computes no metric. On
Colab its first cell clones the repository, which already contains the metrics.

## Repository history

The commit history records the order in which the study's declarations,
designs and results were committed, and it is never rewritten. The tag `oracle-declared`
marks the frozen oracle declaration, and `revision-declared` that of the further designs.

`CLAUDE.md` holds the working rules given to the AI coding assistant used in
the study (Claude Code): terminology, the
disciplines of the analysis and the wording the results allow.

## Citing

Cite the code by its Zenodo record and, when you use the data, the data
deposit: version 1.1, [10.5281/zenodo.24374247](https://doi.org/10.5281/zenodo.24374247),
for the further sets, or version 1.0,
[10.5281/zenodo.23618253](https://doi.org/10.5281/zenodo.23618253), for the original sets;
`CITATION.cff` holds both. Each release of the code is archived on Zenodo, and
the repository with its full history on Software Heritage.

## License

The code is released under the MIT License (`LICENSE`). The data deposit is
licensed separately, under CC BY 4.0.
