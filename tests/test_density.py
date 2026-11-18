"""The density model's slope (revision declaration §3)."""

import numpy as np

from sgs import density


def test_fit_slope_recovers_a_common_slope_with_group_intercepts():
    rng = np.random.default_rng(0)
    rows = []
    for zone, alpha in (("0.5–1", 1.0), ("1–2", 0.2), ("2–6", -0.5)):
        for family in ("rf", "gbm"):
            for n in rng.integers(20, 400, size=30):
                e = np.exp(alpha - 0.55 * np.log(n) + rng.normal(0, 0.01))
                rows.append({"zone": zone, "family": family, "test": "t", "n_train": int(n),
                             "log_rmse": float(e), "realization": 1})
    assert abs(density.fit_slope(rows) - 0.55) < 0.01
