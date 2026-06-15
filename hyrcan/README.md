## HYRCAN scripts

These scripts automate Spencer-method slope stability analyses in [HYRCAN](http://www.geowizard.org/download_hyrcan.html)
and must be executed within the HYRCAN Python scripting environment.
They are not standalone Python scripts.

HYRCAN is freely available.

Reference: [Geraili Mikola (2023)](https://doi.org/10.22541/au.167533016.65709246/v1)

These scripts are provided for transparency of the data generation step.
To reproduce the analysis results, the complete dataset is available
on Zenodo and the HYRCAN scripts are not needed.

### run_analyses.py

Reads a CSV of parameter combinations (angle, phi, c, gamma, H), sets
adaptive domain geometry (extension = max(1.5H, 0.5·slope_length),
depth = max(H, 0.5·slope_length)), runs Spencer analysis for each
configuration, and appends results including critical slip circle
geometry and a boundary QC flag.

### run_repair.py

Idempotent repair routine for boundary-flagged configurations.
Progressively enlarges the analysis domain in three steps
(depth_2x → both_2x → both_3x) until the critical surface detaches
from the search boundary. Backs up each intermediate state.
The reference system origin is at the slope toe.
