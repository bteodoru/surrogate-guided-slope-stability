"""The tuned learners: one fixed grid per family, identical for every arm.

Hyper-parameters are selected by K-fold CV on the arm's training data only;
no test set is touched. Out-of-fold residuals come from a nested CV:
``cross_val_predict`` receives the search object, so each OOF prediction is
made with hyper-parameters selected without its fold.

Each (family, arm, scoring) fit is cached in artifacts/fits/ together with a
fingerprint of its protocol configuration and the hashes of its training
data; a fit whose configuration or data changed is refused.
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

from sgs import config, data


@dataclass
class FitArtifact:
    family: str
    arm: str
    scoring: str
    target: str                  # "log": predictions are exponentiated at evaluation
    model: object                # refit on the whole arm with best_params
    best_params: dict
    cv_results: dict             # every grid point: params, mean and std CV score
    train_ids: np.ndarray
    oof_pred: np.ndarray         # nested out-of-fold predictions, target scale
    oof_residual: np.ndarray     # y_target - oof_pred, signed, target scale
    fold: np.ndarray             # outer fold of each training row

    @property
    def scores(self) -> np.ndarray:
        """Raw conformity scores |y - ŷ| on the target scale."""
        return np.abs(self.oof_residual)


# --- the protocol -----------------------------------------------------------------

def estimator(family: str):
    fixed = config.ESTIMATOR_FIXED[family]
    random_state = config.SEEDS["protocol_estimator"]
    if family == "rf":
        return RandomForestRegressor(**fixed, random_state=random_state, n_jobs=1)
    if family == "gbm":
        return HistGradientBoostingRegressor(**fixed, random_state=random_state)
    raise ValueError(f"unknown family {family!r}")


def _fos_rmse(y_log, p_log) -> float:
    return float(np.sqrt(np.mean((np.exp(p_log) - np.exp(y_log)) ** 2)))


def scorer(scoring: str):
    if scoring == "log-rmse":
        return "neg_root_mean_squared_error"
    if scoring == "fos-rmse":
        return make_scorer(_fos_rmse, greater_is_better=False)
    raise ValueError(f"unknown scoring {scoring!r}")


def cv_splitter() -> KFold:
    return KFold(config.CV_FOLDS, shuffle=True, random_state=config.SEEDS["protocol_cv_split"])


def searcher(family: str, scoring: str) -> GridSearchCV:
    return GridSearchCV(estimator(family), config.GRIDS[family], cv=cv_splitter(),
                        scoring=scorer(scoring), n_jobs=-1)


def fit_arm(family: str, arm: str, scoring: str) -> FitArtifact:
    if config.TARGET != "log":
        raise ValueError(f"unsupported target convention {config.TARGET!r}")
    frame = data.load_arm(arm)
    X = frame[list(config.FEATURES)]
    y = np.log(frame["FOS"].to_numpy(float))

    search = searcher(family, scoring).fit(X, y)
    outer = cv_splitter()
    oof = cross_val_predict(searcher(family, scoring), X, y, cv=outer, n_jobs=1)
    fold = np.empty(len(y), dtype=int)
    for k, (_, held_out) in enumerate(outer.split(X)):
        fold[held_out] = k

    cv = search.cv_results_
    return FitArtifact(
        family=family, arm=arm, scoring=scoring, target=config.TARGET,
        model=search.best_estimator_, best_params=dict(search.best_params_),
        cv_results={"params": [dict(p) for p in cv["params"]],
                    "mean_test_score": cv["mean_test_score"].tolist(),
                    "std_test_score": cv["std_test_score"].tolist()},
        train_ids=frame["row_id"].to_numpy(str), oof_pred=oof, oof_residual=y - oof, fold=fold,
    )


# --- the fit cache ------------------------------------------------------------------

class StaleFit(RuntimeError):
    """A cached fit no longer matches its configuration or training data."""


def training_hashes(arm: str) -> dict:
    return {role: data.sha256_file(config.final_path(role)) for role in config.ARMS[arm].roles}


def _dump(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    joblib.dump(payload, tmp)
    os.replace(tmp, path)


def save_fit(artifact: FitArtifact) -> None:
    _dump(config.fit_path(artifact.family, artifact.arm, artifact.scoring), {
        "artifact": artifact,
        "fingerprint": config.fit_fingerprint(artifact.family, artifact.scoring),
        "data": training_hashes(artifact.arm),
        "environment": data.environment(),
    })


def fit_problems(stored: dict) -> list[str]:
    artifact = stored["artifact"]
    problems = []
    if stored["fingerprint"] != config.fit_fingerprint(artifact.family, artifact.scoring):
        problems.append("protocol configuration changed (grid, fixed parameters, CV, seeds, target or features)")
    now = training_hashes(artifact.arm)
    for role in sorted(set(now) | set(stored["data"])):
        if stored["data"].get(role) != now.get(role):
            problems.append(f"training data {role!r} changed")
    return problems


def load_fit(family: str, arm: str, scoring: str) -> FitArtifact:
    path = config.fit_path(family, arm, scoring)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing; run `sgs fit`")
    stored = joblib.load(path)
    problems = fit_problems(stored)
    if problems:
        raise StaleFit(f"{path.name} is stale: {'; '.join(problems)}. Run `sgs fit --force`.")
    return stored["artifact"]


def all_units() -> list[tuple[str, str, str]]:
    return [(f, a, s) for s in config.ACTIVE_SCORINGS for f in config.FAMILIES for a in config.ARMS]


def fit_all(*, force: bool = False, units=None) -> None:
    units = list(units or all_units())
    data.validate_sets(sorted({role for _, arm, _ in units for role in config.ARMS[arm].roles}))
    for index, (family, arm, scoring) in enumerate(units, 1):
        path = config.fit_path(family, arm, scoring)
        label = f"[fit {index}/{len(units)}] {family} {arm} {scoring}"
        if path.exists():
            problems = fit_problems(joblib.load(path))
            if not problems:
                print(f"{label}: up to date")
                continue
            if not force:
                raise StaleFit(f"{path.name} is stale: {'; '.join(problems)}. "
                               "Rerun with --force to refit.")
        print(label, flush=True)
        artifact = fit_arm(family, arm, scoring)
        save_fit(artifact)
        print(f"  best {artifact.best_params}")
        data.record("fits", f"{family}__{arm}__{scoring}", {
            "best_params": artifact.best_params, "n_train": int(len(artifact.train_ids)),
            "fingerprint": config.fit_fingerprint(family, scoring), "data": training_hashes(arm),
            "environment": data.environment(),
            "updated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        })


# --- learning curve (optional; fixed hyper-parameters, both families) --------------------

SHARED = "seed-only"


def lc_estimator(family: str):
    """The protocol estimator with the fixed LC_PARAMS of its family; not tuned (spec §8)."""
    return estimator(family).set_params(**config.LC_PARAMS[family])


def unit_rng(budget: int, repeat: int) -> np.random.Generator:
    sequence = np.random.SeedSequence(config.SEEDS["learning_curve_subsampling"],
                                      spawn_key=(budget, repeat))
    return np.random.default_rng(sequence)


def subsample(seed_df, extra_df, budget: int, rng):
    n_seed = len(seed_df)
    if budget < n_seed:
        return seed_df.iloc[rng.choice(n_seed, budget, replace=False)]
    if budget == n_seed:
        return seed_df
    if budget == n_seed + len(extra_df):
        # The whole pool, in canonical order: the same frame the arm's own
        # fit trains on (load_arm concatenates its roles in order).
        return pd.concat([seed_df, extra_df], ignore_index=True)
    take = extra_df.iloc[rng.choice(len(extra_df), budget - n_seed, replace=False)]
    return pd.concat([seed_df, take], ignore_index=True)


def lc_plan(n_seed: int, n_extra: dict[str, int]) -> list[tuple[str, int, int]]:
    """Budgets up to the seed size do not depend on the strategy: fitted once.
    At the seed size, or at a strategy's full pool (seed plus all of that
    strategy's extra rows), every repeat would draw the identical row set:
    one repeat, for that strategy only."""
    out = []
    for budget in config.LC_BUDGETS:
        if budget <= n_seed:
            repeats = 1 if budget == n_seed else config.LC_REPEATS
            out.append((SHARED, budget, repeats))
        else:
            for strategy in config.LC_STRATEGIES:
                repeats = 1 if budget == n_seed + n_extra[strategy] else config.LC_REPEATS
                out.append((strategy, budget, repeats))
    return out


def lc_roles() -> list[str]:
    return [config.LC_BASE_ROLE, *config.LC_STRATEGIES.values(), *config.TEST_ROLES]


def lc_state(family: str) -> dict:
    return {"fingerprint": config.learning_curve_fingerprint(family),
            "data": {role: data.sha256_file(config.final_path(role)) for role in lc_roles()}}


def _check_lc_budgets(n_seed: int, n_extra: dict[str, int]) -> None:
    """A budget beyond a strategy's usable pool (seed + its extra rows) would
    ask subsample() for more rows than exist."""
    needed = max(config.LC_BUDGETS)
    for strategy, extra in n_extra.items():
        usable = n_seed + extra
        if usable < needed:
            role = config.LC_STRATEGIES[strategy]
            raise ValueError(f"{role!r} has {usable} usable rows (seed {n_seed} + extra {extra}); "
                             f"the largest learning-curve budget ({needed}) needs more")


def lc_sizes() -> tuple[int, dict[str, int]]:
    """Row counts backing the plan, guarded against a budget the data cannot fill."""
    n_seed = len(data.load_role(config.LC_BASE_ROLE))
    n_extra = {strategy: len(data.load_role(role)) for strategy, role in config.LC_STRATEGIES.items()}
    _check_lc_budgets(n_seed, n_extra)
    return n_seed, n_extra


def run_lc_budget(family, key, budget, repeats, seed_df, extras, tests) -> dict:
    units = []
    for repeat in range(repeats):
        extra = seed_df.iloc[:0] if key == SHARED else extras[key]
        frame = subsample(seed_df, extra, budget, unit_rng(budget, repeat))
        model = lc_estimator(family).fit(frame[list(config.FEATURES)], np.log(frame["FOS"].to_numpy(float)))
        units.append({
            "repeat": repeat, "n_train": int(len(frame)), "train_ids": frame["row_id"].to_numpy(str),
            "predictions": {name: {"ids": test.ids, "y_pred": np.exp(model.predict(test.X))}
                            for name, test in tests.items()},
        })
    return {"family": family, "key": key, "budget": budget, "units": units}


def learning_curve(*, force: bool = False) -> None:
    roles = lc_roles()
    data.validate_sets(roles)
    seed_df = data.load_role(config.LC_BASE_ROLE)
    extras = {strategy: data.load_role(role) for strategy, role in config.LC_STRATEGIES.items()}
    tests = {role: data.load_test(role) for role in config.TEST_ROLES}
    n_extra = {strategy: len(extra) for strategy, extra in extras.items()}
    _check_lc_budgets(len(seed_df), n_extra)
    for family in config.LC_FAMILIES:
        state = lc_state(family)
        for key, budget, repeats in lc_plan(len(seed_df), n_extra):
            path = config.learning_curve_path(family, key, budget)
            if path.exists():
                if joblib.load(path)["state"] == state:
                    print(f"[learning-curve] {path.name}: up to date")
                    continue
                if not force:
                    raise StaleFit(f"{path.name} was fitted under another configuration or other data; "
                                   "rerun with --force")
            print(f"[learning-curve] {family} {key} n={budget} x{repeats}", flush=True)
            _dump(path, {**run_lc_budget(family, key, budget, repeats, seed_df, extras, tests),
                         "state": state})


def load_lc_units() -> dict:
    """{(family, curve, budget): units}; budgets up to the seed size are shared by both curves."""
    n_seed, n_extra = lc_sizes()
    out = {}
    for family in config.LC_FAMILIES:
        state = lc_state(family)
        for key, budget, _ in lc_plan(n_seed, n_extra):
            path = config.learning_curve_path(family, key, budget)
            if not path.exists():
                raise FileNotFoundError(f"{path.name} is missing; run `sgs learning-curve`")
            stored = joblib.load(path)
            if stored["state"] != state:
                raise StaleFit(f"{path.name} is stale; run `sgs learning-curve --force`")
            for curve in (tuple(config.LC_STRATEGIES) if key == SHARED else (key,)):
                out[(family, curve, budget)] = stored["units"]
    return out
