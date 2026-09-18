"""Synthetic fixtures. The formula is not a slope-stability model; it only
gives a smooth, positive, FOS-like response over the real domain."""

import dataclasses

import numpy as np
import pandas as pd

from sgs import config, data


def synthetic_fos(frame):
    return (0.3
            + np.tan(np.radians(frame["phi"])) / np.tan(np.radians(frame["angle"]))
            + 4.0 * frame["c"] / (frame["gamma"] * frame["H"])).to_numpy(float)


class FormulaModel:
    """Stands in for a fitted regressor: predicts log(FOS) of the formula."""

    def predict(self, X):
        return np.log(synthetic_fos(X))


_DEFAULTS = {"At_Boundary": "False", "Search_Converged": "True", "Warnings": "",
             "Error": "", "repair_variant": "", "excluded": ""}


def synthetic_final(n, seed):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({p: rng.uniform(lo, hi, n) for p, (lo, hi) in config.DOMAIN.items()})
    frame["FOS"] = synthetic_fos(frame)
    for column in data.FINAL_COLUMNS:
        if column not in frame:
            frame[column] = _DEFAULTS.get(column, 0.0)
    return frame[data.FINAL_COLUMNS]


def write_final(role, frame):
    path = config.final_path(role)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def shrink_roles(monkeypatch, sizes):
    roles = dict(config.ROLES)
    for role, n in sizes.items():
        roles[role] = dataclasses.replace(roles[role], n=n)
    monkeypatch.setattr(config, "ROLES", roles)


def fake_analyse(combo):
    """Stands in for solver.analyse: a solved, boundary-free row. Module level
    so that a spawned worker can import it."""
    angle, phi, c, gamma, H = combo
    row = {column: "" for column in data.RESULT_COLUMNS}
    row.update(angle=angle, phi=phi, c=c, gamma=gamma, H=H, FOS=round(c / H, 6),
               At_Boundary=False, Search_Converged=True, Runtime_s=0.0)
    return row


def boundary_row(fos="1.5", boundary="False", c=10.0):
    """A results row as read back from CSV (all strings)."""
    row = {column: "" for column in data.RESULT_COLUMNS}
    row.update(angle="45.0", phi="30.0", c=str(c), gamma="18.0", H="10.0", FOS=fos, At_Boundary=boundary)
    return row
