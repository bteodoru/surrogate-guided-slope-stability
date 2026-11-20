"""Signed bias per reporting zone, the regression analogue of a false-safe count.

Post hoc and descriptive, added in the research article's revision and declared as such:
the mean of ln(predicted / true FOS) over the test points of each zone, per arm, learner,
test set and realization r1..r5, on the fixed test sets, from the stored replication and
revision fits. A positive value is an over-prediction of the FOS (unconservative), a
negative one an under-prediction. It needs no threshold, so it does not count a 0.05 error
across FOS = 1 like a 0.5 one (CLAUDE.md, discipline 2). The ``signed_bias`` entry of
``outputs/metrics/revision.json`` holds the per-realization values and their means; it
introduces no test and no reading.
"""

from __future__ import annotations

import numpy as np

from sgs import config, data, density
from sgs import revision_fit as rf

ZONES = ("FOS < 0.5", "0.5–1", "1–2", "2–6", "FOS ≥ 6")


def run() -> dict:
    tests = {role: data.load_test(role) for role in config.TEST_ROLES}
    units = rf.merged_units()
    zone = {n: np.searchsorted(config.ZONE_EDGES, t.y, side="right") for n, t in tests.items()}
    per = {}
    for (family, r), u in units.items():
        for arm, a in u["arms"].items():
            for name, t in tests.items():
                e = np.log(np.asarray(a["predictions"][name], float) / t.y)
                for zi, label in enumerate(ZONES):
                    m = zone[name] == zi
                    per.setdefault(f"{family}/{name}/{arm}/{label}", {})[f"r{r}"] = float(e[m].mean())
    means = {k: float(np.mean(list(v.values()))) for k, v in per.items()}
    entry = {"note": "post hoc, descriptive (R1.10); mean ln(pred/true) per zone; no test, no reading",
             "per_realization": per, "means": means,
             "n_points": {n: {label: int((zone[n] == zi).sum()) for zi, label in enumerate(ZONES)}
                          for n in tests}}
    density.write("signed_bias", entry)
    return entry
