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


class OffsetModel:
    """log(FOS) of the formula, perturbed row-wise by a fixed pattern."""

    def __init__(self, scale):
        self.scale = scale

    def predict(self, X):
        return np.log(synthetic_fos(X)) + self.scale * np.sin(np.arange(len(X)) * 1.7)


ARM_SCALES = {"A": 0.30, "B": 0.20, "C": 0.10, "C0": 0.25}


def stub_fit(family, arm, scale, scoring="log-rmse", n_train=40):
    from sgs.models import FitArtifact
    rng = np.random.default_rng(int(scale * 1000) + len(arm))
    return FitArtifact(family=family, arm=arm, scoring=scoring, target="log", model=OffsetModel(scale),
                       best_params={}, cv_results={},
                       train_ids=np.array([f"seed:{i}" for i in range(n_train)]),
                       oof_pred=np.zeros(n_train), oof_residual=rng.normal(0.0, 0.1, n_train),
                       fold=np.arange(n_train) % 5)


def stub_fits(family="rf", scoring="log-rmse"):
    return {f"{family}__{arm}__{scoring}": stub_fit(family, arm, scale, scoring)
            for arm, scale in ARM_SCALES.items()}


def stub_test(n=60, seed=3, name="test-natural"):
    frame = data.add_features(synthetic_final(n, seed))
    return data.TestSet(name=name, X=frame[list(config.FEATURES)].reset_index(drop=True),
                        y=frame["FOS"].to_numpy(float), ids=np.array([f"{name}:{i}" for i in range(n)]))


EVAL_KWARGS = dict(alpha=0.10, zones=config.ZONE_EDGES, comparisons=config.COMPARISONS,
                   selection_range=config.SELECTION_RANGE, n_boot=300, bootstrap_seed=1,
                   bootstrap_ci=0.95)


def write_all_finals(monkeypatch, n=60):
    """Small, valid, pairwise-disjoint final sets for every role."""
    shrink_roles(monkeypatch, {role: n for role in config.ROLE_ORDER})
    for i, role in enumerate(config.ROLE_ORDER):
        write_final(role, synthetic_final(n, 200 + i))
