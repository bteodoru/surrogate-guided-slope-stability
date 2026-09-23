"""The reference pool: uniform i.i.d. analyses over the domain, solved in chunks
the user launches, from which C-oracle and C-logP select.

Spec: docs/superpowers/specs/2026-09-23-oracle-reference-pool-design.md.

  sgs pool --chunks N --workers W   solve up to N more chunks; stop at the rule
  sgs pool --status                 read-only: counts per sub-pool and bin

The design is drawn once, in full, at POOL_CAP rows. Rows within MIN_DISTANCE
of the seed or of either test set are removed, and row i of what remains
belongs to sub-pool i % REP_REALIZATIONS + 1, independently of FOS. Rows beyond
the stopping point are never solved.
"""

from __future__ import annotations

import datetime as _dt
import platform

import numpy as np
import pandas as pd

from sgs import config, data, design, solver


class PoolError(RuntimeError):
    """The pool, its files or its stopping rule are not in the declared state."""


REFERENCE_ROLES = ("seed", *config.TEST_ROLES)


# --- the design -------------------------------------------------------------------------

def full_design() -> tuple[pd.DataFrame, int]:
    """The capped design after the disjointness filter, with pool_row and subpool,
    and the number of rows the filter removed. Deterministic; cheap to recompute."""
    n_sub = config.REP_REALIZATIONS
    if config.POOL_CHUNK % n_sub:
        raise PoolError(f"POOL_CHUNK={config.POOL_CHUNK} is not a multiple of the {n_sub} sub-pools")
    raw = design.uniform_pool(np.random.default_rng(config.SEEDS["pool_design"]), config.POOL_CAP)
    references = pd.concat([design.read_design(role)[list(config.PARAMS)] for role in REFERENCE_ROLES],
                           ignore_index=True)
    keep = design.nearest_distance(raw, references) > config.MIN_DISTANCE
    frame = raw.loc[keep].reset_index(drop=True)
    frame["pool_row"] = np.arange(len(frame))
    frame["subpool"] = frame["pool_row"] % n_sub + 1
    return frame, int((~keep).sum())


def n_chunks(frame: pd.DataFrame) -> int:
    return len(frame) // config.POOL_CHUNK


def chunk_design(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    """The rows of chunk k (1-based), in design order."""
    if not 1 <= k <= n_chunks(frame):
        raise PoolError(f"chunk {k} is outside 1..{n_chunks(frame)}")
    lo = (k - 1) * config.POOL_CHUNK
    return frame.iloc[lo:lo + config.POOL_CHUNK].reset_index(drop=True)
