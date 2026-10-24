"""One xslope/Bishop analysis for arbitrary parameters, outside the pipeline.

Self-contained copy of ``sgs/solver.py`` for Colab: the same model, domain,
search window and ``config.SOLVER`` settings, with nothing imported from
``sgs``. It is a diagnostic; nothing it produces is read by the pipeline.

In Colab:

    !pip install -q xslope==0.5.2 shapely
    %run xslope_colab.py            # or paste this file into a cell
    r = run(beta=10.975369851859272, phi=33.83530776453324, c=1.854369392083913,
            gamma=21.045944752536563, H=23.356046748138173)
    r = run(..., max_iter=200)      # the non-converged question (defect 2)
    r = run(..., plot=True)         # xslope's search and solution plots

Units: beta and phi in degrees, c in kPa, gamma in kN/m3, H in m. Colab runs
on x86; the stored data were solved on darwin arm64, so a last-digit
difference in FOS is not by itself a discrepancy.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import math
import time
import urllib.request
from copy import deepcopy
from pathlib import Path

# --- config.SOLVER, copied verbatim ----------------------------------------------

XSLOPE_VERSION = "0.5.2"
METHOD = "bishop"
NUM_SLICES = 30
SEARCH_START = "grid"
SEARCH_TOL = 0.01
SEARCH_FS_TOL = 0.0005
SEARCH_MAX_ITER = 50
SEARCH_MIN_GRID_FRAC = 0.03
EXTENSION_H_FACTOR = 1.5
EXTENSION_L_FACTOR = 0.5
DEPTH_L_FACTOR = 0.5
EXIT_RANGE_L_FRAC = 1 / 3
ENTRY_RANGE_L_FRAC = 2 / 3
BOUNDARY_TOL_H_FRAC = 0.01
TEMPLATE_URL = ("https://raw.githubusercontent.com/njones61/xslope/"
                "v0.5.2/docs/inputs/slope/xslope_simple1.xlsx")
TEMPLATE_SHA256 = "82b2c69293b2afc5f2f2f0026948a860c5032fe01eb029b64297bbb42d1c5b37"
TEMPLATE_PATH = Path("xslope_simple1_v0.5.2.xlsx")

# config.REPAIR_VARIANTS: (extension factor, depth factor)
REPAIR_VARIANTS = ((2.0, None), (None, 2.0), (1.5, 1.5), (2.0, 2.0), (3.0, 3.0))

# The kept rows whose search did not converge (Search_Converged == False),
# as (set, 0-based row, beta, phi, c, gamma, H, stored FOS).
NON_CONVERGED = (
    ("lhs_continued_300", 85, 10.975369851859272, 33.83530776453324, 1.854369392083913,
     21.045944752536563, 23.356046748138173, 3.628919),
    ("test_stratified_150", 65, 13.777432287195458, 31.595969473802626, 2.5859731141454763,
     21.67079062470726, 14.533013532036922, 2.770601),
    ("replications/guided_log_r1", 75, 57.81139066813939, 32.87200049695873, 2.9984453391600723,
     20.93415879316596, 23.99978360419644, 0.566515),
    ("replications/guided_log_r5", 115, 44.74711188551581, 33.67858837452263, 2.731074702531025,
     21.774836578586108, 21.6470185489641, 0.83012),
    ("replications/lhs_r1", 186, 44.634593501561405, 38.232834526571104, 4.664342556643642,
     17.595567180172, 18.37996400969391, 1.100943),
)

_BASE_MODEL = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_model():
    global _BASE_MODEL
    if _BASE_MODEL is None:
        import xslope
        from xslope.fileio import load_slope_data

        installed = getattr(xslope, "__version__", "unknown")
        if installed != XSLOPE_VERSION:
            raise RuntimeError(f"xslope=={XSLOPE_VERSION} required; installed {installed}")
        if not TEMPLATE_PATH.exists() or _sha256(TEMPLATE_PATH) != TEMPLATE_SHA256:
            urllib.request.urlretrieve(TEMPLATE_URL, TEMPLATE_PATH)
            if _sha256(TEMPLATE_PATH) != TEMPLATE_SHA256:
                raise RuntimeError("xslope template checksum mismatch")
        _BASE_MODEL = load_slope_data(TEMPLATE_PATH)
    return _BASE_MODEL


def _geometry(H, beta, ext_factor=None, depth_factor=None):
    slope = math.tan(math.radians(beta))
    slope_length = H / slope
    extension = max(EXTENSION_H_FACTOR * H, EXTENSION_L_FACTOR * slope_length)
    depth = max(H, DEPTH_L_FACTOR * slope_length)
    if ext_factor:
        extension *= ext_factor
    if depth_factor:
        depth *= depth_factor
    return {"slope": slope, "slope_length": slope_length, "extension": extension, "depth": depth,
            "exit_range": (-depth, EXIT_RANGE_L_FRAC * slope_length),
            "entry_range": (ENTRY_RANGE_L_FRAC * slope_length, slope_length + extension)}


def _model(beta, phi, c, gamma, H, geometry):
    from shapely.geometry import Polygon
    from xslope.fileio import build_ground_surface_from_polygons
    from xslope.generators import generate_starting_circles
    from xslope.mesh import build_polygons

    depth, slope_length, extension = geometry["depth"], geometry["slope_length"], geometry["extension"]
    model = deepcopy(_base_model())
    ground = [(-depth, 0.0), (0.0, 0.0), (slope_length, H), (slope_length + extension, H)]
    model["max_depth"] = -depth
    model["profile_lines"] = [{"coords": ground, "mat_id": 0, "size": None}]
    raw = build_polygons(slope_data={"profile_lines": model["profile_lines"],
                                     "max_depth": model["max_depth"]})
    model["polygons"] = [{"polygon": Polygon(p["coords"]), "mat_id": int(p["mat_id"]),
                          "size": p.get("size")} for p in raw]
    model["ground_surface"], model["domain_polygon"] = build_ground_surface_from_polygons(model["polygons"])
    model["materials"][0].update(name="Soil", option="mc", gamma=float(gamma), c=float(c),
                                 phi=float(phi), ru=0.0, u="none")
    model.update(piezo_line=[], piezo_line2=[], piezo_phreatic=False, piezo_phreatic2=False,
                 dloads=[], dloads2=[], dload_dirs=[], dload2_dirs=[], reinforce_lines=[],
                 reinforcement_lines=[], pile_lines=[], line_loads=[], k_seismic=0.0,
                 tcrack_depth=0.0, tcrack_water=0.0, circular=True, non_circ=[])
    circles = generate_starting_circles(model, report=True)
    if not circles["circles"]:
        raise RuntimeError("no starting circle: " + circles.get("reason", "unknown reason"))
    model["circles"] = circles["circles"]
    return model


def run(beta, phi, c, gamma, H, *, max_iter=SEARCH_MAX_ITER, ext_factor=None, depth_factor=None,
        plot=False, quiet=True) -> dict:
    """Solve one slope with the pipeline's settings; ``max_iter`` and the domain
    factors can be overridden for diagnosis. Returns FOS, convergence and the
    critical circle, plus xslope's raw outputs under ``"raw"``."""
    from xslope.search import circular_search

    started = time.perf_counter()
    geometry = _geometry(H, beta, ext_factor, depth_factor)
    model = _model(beta, phi, c, gamma, H, geometry)
    unsolved = {}
    stdout = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with stdout:
        fs_cache, converged, search_path, circle_cache = circular_search(
            model, METHOD, num_slices=NUM_SLICES, seed=SEARCH_START, tol=SEARCH_TOL,
            fs_tol=SEARCH_FS_TOL, max_iter=max_iter, min_grid_frac=SEARCH_MIN_GRID_FRAC,
            diagnostic=False, entry_range=geometry["entry_range"],
            exit_range=geometry["exit_range"], unsolved_out=unsolved)
    if not fs_cache:
        raise RuntimeError("xslope circular search returned no valid surface")
    best = fs_cache[0]
    left, bottom, right, _ = map(float, best["failure_surface"].bounds)  # the arc's lowest point, as in sgs/solver.py
    circle_nadir_y = float(best["Depth"])
    tol = BOUNDARY_TOL_H_FRAC * H
    flags = [name for name, hit in (
        ("left_boundary", left <= -geometry["depth"] + tol),
        ("right_boundary", right >= geometry["slope_length"] + geometry["extension"] - tol),
        ("bottom_boundary", bottom <= -geometry["depth"] + tol)) if hit]
    result = {
        "FOS": float(best["FS"]), "converged": bool(converged), "max_iter": max_iter,
        "n_search_steps": len(search_path), "n_circles_tested": len(circle_cache),
        "n_valid_surfaces": len(fs_cache), "n_unsolved": int(unsolved.get("unsolved", 0)),
        "at_boundary": bool(flags), "boundary_flags": flags,
        "Xo": float(best["Xo"]), "Yo": float(best["Yo"]),
        "radius": float(best["Yo"]) - circle_nadir_y, "circle_nadir_y": circle_nadir_y,
        "surface_left": left, "surface_right": right,
        "ext_factor": ext_factor, "depth_factor": depth_factor,
        "runtime_s": round(time.perf_counter() - started, 2),
        "raw": {"model": model, "fs_cache": fs_cache, "search_path": search_path,
                "circle_cache": circle_cache, "geometry": geometry},
    }
    if plot:
        import matplotlib.pyplot as plt
        from xslope.plot import plot_circular_search_results, plot_solution

        plot_circular_search_results(model, fs_cache, search_path, circle_cache,
                                     fig=plt.figure(figsize=(12, 7)))
        plot_solution(model, best["slices"], best["failure_surface"], best["solver_result"],
                      fig=plt.figure(figsize=(12, 7)))
        plt.show()
    return result


def summary(r: dict) -> str:
    return (f"FOS={r['FOS']:.6f} converged={r['converged']} max_iter={r['max_iter']} "
            f"steps={r['n_search_steps']} circles={r['n_circles_tested']} "
            f"boundary={r['boundary_flags'] or '-'} {r['runtime_s']} s")


def sweep_max_iter(beta, phi, c, gamma, H, iters=(50, 100, 200, 400), **kwargs) -> list[dict]:
    """The same slope at increasing ``max_iter``: where does it converge, and
    how far does FOS move from the 50-iteration value?"""
    rows = []
    for n in iters:
        r = run(beta, phi, c, gamma, H, max_iter=n, **kwargs)
        rows.append(r)
        print(f"  {summary(r)}  dFOS vs first = {r['FOS'] - rows[0]['FOS']:+.2e}")
    return rows


if __name__ == "__main__":
    for name, row, *params, stored in NON_CONVERGED:
        print(f"{name} row {row}: stored FOS {stored}")
        sweep_max_iter(*params)
