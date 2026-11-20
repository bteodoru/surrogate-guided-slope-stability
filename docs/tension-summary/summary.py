"""Base tension by quarter of the slope angle, the cohesion and the height, on the curated deposit.

  PYTHONPATH=. .venv/bin/python docs/tension-summary/summary.py

Reads data/curated/ (written by `sgs curate`) as a user would and writes summary.json next to this
script, with the commit and the hash of the table it read. The numbers are those of
curate.tension_by_quartile, which `sgs curate` also records in the deposit's provenance; the Data
Descriptor (Methods, Base Tension and Crest Entries) quotes them.
"""

import json

import pandas as pd

from sgs import config, curate, data

base = curate.out_dir()
sims = pd.read_csv(base / "data" / "simulations.csv", dtype=str, keep_default_na=False)
provenance = json.loads((base / "metadata" / "provenance.json").read_text())
out = {"definition": "quarters of each input over all simulations (pd.qcut, 4 bins, right-closed); "
                     "share of analyses with Tension_Slices > 0; median Min_Normal_Force among them",
       "n_simulations": len(sims),
       "n_with_tension": int((sims["Tension_Slices"].astype(int) > 0).sum()),
       "by_quartile": curate.tension_by_quartile(sims),
       "deposit": {"simulations_sha256": data.sha256_file(base / "data" / "simulations.csv"),
                   "curated_version": provenance.get("version"),
                   "curated_at_commit": provenance["code"]["commit"]},
       "code": curate._git_state()}
path = config.ROOT / "docs" / "tension-summary" / "summary.json"
data.write_text_atomic(path, json.dumps(out, indent=2) + "\n")
for column, rows in out["by_quartile"].items():
    print(column, [(round(r["share_with_tension"], 3), r["median_min_normal_force_kN_per_m"]) for r in rows])
