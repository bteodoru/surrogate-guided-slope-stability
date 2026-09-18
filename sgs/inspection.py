"""Rebuild the model of one stored row, rerun its search and draw it with
xslope's own plots. A diagnostic: nothing it writes is read by the pipeline.

``row`` is 1-based, in design order, as ``repair`` numbers rows. On the
``final`` stage the model is built on the row's repair variant, if any; on
``results`` it is built on the original domain.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sgs import config, data, solver

def variant_factors(label: str):
    """The (ext_factor, depth_factor) of a ``repair_variant`` label."""
    if not label:
        return None, None
    for factors in config.REPAIR_VARIANTS:
        if solver.variant_label(*factors) == label:
            return factors
    raise ValueError(f"unknown repair variant {label!r}")


def inspect_row(role: str, row: int, stage: str = "final") -> dict:
    from xslope.plot import plot_circular_search_results, plot_solution

    path = config.final_path(role) if stage == "final" else config.results_path(role)
    table = data.read_table(path)
    if not 1 <= row <= len(table):
        raise IndexError(f"{path.name} has rows 1..{len(table)}; got {row}")
    stored = table.iloc[row - 1]
    ext_factor, depth_factor = (variant_factors(stored["repair_variant"] or "") if stage == "final"
                                else (None, None))

    combo = [float(stored[p]) for p in config.PARAMS]
    found = solver.search(*combo, ext_factor, depth_factor)
    if not found["fs_cache"]:
        raise RuntimeError("xslope circular search returned no valid surface")
    best = found["fs_cache"][0]
    left, right, depth = solver.surface_limits(best)
    at_boundary, flags = solver.check_boundary(left, right, depth, found["geometry"],
                                               float(stored["H"]))

    paths = {view: config.inspect_path(role, stage, row, view) for view in ("search", "solution")}
    paths["search"].parent.mkdir(parents=True, exist_ok=True)
    fig = plot_circular_search_results(found["model"], found["fs_cache"], found["search_path"],
                                       found["circle_cache"], fig=plt.figure(figsize=(12, 7)))
    fig.savefig(paths["search"], dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig = plot_solution(found["model"], best["slices"], best["failure_surface"],
                        best["solver_result"], fig=plt.figure(figsize=(12, 7)))
    fig.savefig(paths["solution"], dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"role": role, "row": row, "stage": stage,
            "params": dict(zip(config.PARAMS, combo)),
            "repair_variant": stored["repair_variant"] if stage == "final" else "",
            "fos_stored": None if pd.isna(stored["FOS"]) else float(stored["FOS"]),
            "fos": float(best["FS"]), "at_boundary": at_boundary, "boundary_flags": flags,
            "figures": paths}
