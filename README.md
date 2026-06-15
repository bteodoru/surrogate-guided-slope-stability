# Surrogate-Guided Sampling for Slope Stability Surrogates

## Overview

This repository contains the complete, deterministic pipeline used to
generate, select, train, and evaluate surrogate models for slope stability
prediction. The study proposes a surrogate-guided sampling methodology that
controls the distribution of the target variable (factor of safety, FOS)
directly, rather than sampling uniformly in the input parameter space.

The pipeline compares four experimental arms in a budget-fair design:

| Arm | Training data                          | Solver budget |
| --- | -------------------------------------- | ------------- |
| A   | LHS seed (300)                         | 300           |
| B   | LHS seed + continued LHS (300)         | 600           |
| C₀  | Surrogate-guided only (300) — ablation | —             |
| C   | LHS seed + surrogate-guided (300)      | 600           |

All arms are evaluated on a common independent test set of 150
configurations, uniformly stratified over the FOS band [0.5, 6.0].

## Usage

The analysis pipeline is a single Google Colab notebook:
**`surrogate_guided_sampling.ipynb`**

1. Download the four CSV files from the dataset archive: [\[Zenodo\]](https://doi.org/10.5281/zenodo.20690924)
2. Open the notebook in Google Colab (free, no installation required)
3. Upload the CSV files to the Colab session (`/content/`)
4. Run all cells in order: Runtime → Run all

All results are deterministic (random seed 42).

HYRCAN scripts (`hyrcan/`) are provided for transparency and
reproducibility of the data generation step. They require HYRCAN
and are not needed to reproduce the analysis results —
the complete dataset is available on Zenodo.
