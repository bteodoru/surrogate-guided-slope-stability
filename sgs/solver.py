"""xslope/Bishop: one homogeneous, dry, unloaded slope per analysis; a set of
analyses in a process pool; boundary repair on enlarged domains.

Every model is built in memory; the version-pinned xslope workbook supplies
only a valid input schema. Solver settings come from ``config.SOLVER``.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import math
import multiprocessing as mp
import os
import time
import urllib.request
from copy import deepcopy
from pathlib import Path

from sgs import config

_BASE_MODEL = None


# --- the template and the model -------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_template() -> Path:
    """Return the verified, version-pinned xslope workbook, downloading it once."""
    settings = config.SOLVER
    path = config.solver_template_path()
    if path.exists() and _sha256(path) == settings.template_sha256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    print(f"Downloading xslope {settings.package_version} input template...")
    urllib.request.urlretrieve(settings.template_url, path)
    actual = _sha256(path)
    if actual != settings.template_sha256:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"xslope template checksum mismatch: expected "
                           f"{settings.template_sha256}, got {actual}")
    return path


def get_base_model():
    """Load the template once per process, after checking the solver version."""
    global _BASE_MODEL
    if _BASE_MODEL is None:
        import xslope
        from xslope.fileio import load_slope_data

        installed = getattr(xslope, "__version__", "unknown")
        if installed != config.SOLVER.package_version:
            raise RuntimeError(f"xslope=={config.SOLVER.package_version} required; "
                               f"installed version is {installed}")
        _BASE_MODEL = load_slope_data(ensure_template())
    return _BASE_MODEL


def model_dims(H, angle, ext_factor=None, depth_factor=None):
    settings = config.SOLVER
    slope = math.tan(math.radians(angle))
    slope_length = H / slope
    extension = max(settings.extension_h_factor * H, settings.extension_l_factor * slope_length)
    depth = max(H, settings.depth_l_factor * slope_length)
    if ext_factor:
        extension *= ext_factor
    if depth_factor:
        depth *= depth_factor
    return slope, slope_length, extension, depth


def initialize_model(H, angle, phi, c, gamma, ext_factor=None, depth_factor=None):
    """Construct one homogeneous, dry xslope model entirely in memory."""
    from shapely.geometry import Polygon
    from xslope.fileio import build_ground_surface_from_polygons
    from xslope.generators import generate_starting_circles
    from xslope.mesh import build_polygons

    slope, slope_length, extension, depth = model_dims(H, angle, ext_factor, depth_factor)
    model = deepcopy(get_base_model())

    ground = [(-depth, 0.0), (0.0, 0.0), (slope_length, H), (slope_length + extension, H)]
    model["max_depth"] = -depth
    model["profile_lines"] = [{"coords": ground, "mat_id": 0, "size": None}]

    raw_polygons = build_polygons(slope_data={"profile_lines": model["profile_lines"],
                                              "max_depth": model["max_depth"]})
    model["polygons"] = [{"polygon": Polygon(item["coords"]), "mat_id": int(item["mat_id"]),
                          "size": item.get("size")} for item in raw_polygons]
    model["ground_surface"], model["domain_polygon"] = build_ground_surface_from_polygons(model["polygons"])

    model["materials"][0].update(name="Soil", option="mc", gamma=float(gamma), c=float(c),
                                 phi=float(phi), ru=0.0, u="none")

    # The dry, unloaded problem is explicit rather than inherited from the
    # schema workbook.
    model.update(piezo_line=[], piezo_line2=[], piezo_phreatic=False, piezo_phreatic2=False,
                 dloads=[], dloads2=[], dload_dirs=[], dload2_dirs=[], reinforce_lines=[],
                 reinforcement_lines=[], pile_lines=[], line_loads=[], k_seismic=0.0,
                 tcrack_depth=0.0, tcrack_water=0.0, circular=True, non_circ=[])

    circle_report = generate_starting_circles(model, report=True)
    if not circle_report["circles"]:
        raise RuntimeError("xslope could not generate a starting circle: "
                           + circle_report.get("reason", "unknown reason"))
    model["circles"] = circle_report["circles"]

    # Toe-side exit range and crest-side entry range.
    geometry = {
        "slope": slope, "slope_length": slope_length, "extension": extension, "depth": depth,
        "exit_range": (-depth, slope_length / 3.0),
        "entry_range": (2.0 * slope_length / 3.0, slope_length + extension),
    }
    return model, geometry


def surface_limits(best):
    """Bounds of the clipped critical arc; xslope's ``Depth`` is an elevation."""
    failure_surface = best.get("failure_surface")
    if failure_surface is None or failure_surface.is_empty:
        raise RuntimeError("xslope did not return the critical failure surface")
    left, _, right, _ = map(float, failure_surface.bounds)
    return left, right, float(best["Depth"])


def check_boundary(surface_left, surface_right, surface_depth, geometry, H):
    """Flag a critical surface that reaches an analysis-domain boundary."""
    domain_left = -geometry["depth"]
    domain_right = geometry["slope_length"] + geometry["extension"]
    domain_bottom = -geometry["depth"]
    tol = config.SOLVER.boundary_tol_h_frac * H
    flags = []
    if surface_left <= domain_left + tol:
        flags.append("left_boundary")
    if surface_right >= domain_right - tol:
        flags.append("right_boundary")
    if surface_depth <= domain_bottom + tol:
        flags.append("bottom_boundary")
    return bool(flags), flags


def calculate_fos(angle, phi, c, gamma, H, ext_factor=None, depth_factor=None) -> dict:
    """One xslope/Bishop circular search. ``fos`` is None on solver error."""
    from xslope.search import circular_search

    settings = config.SOLVER
    started = time.perf_counter()
    result = {"fos": None, "surface_left": None, "surface_right": None, "surface_depth": None,
              "cx": None, "cy": None, "radius": None, "at_boundary": None, "boundary_flags": [],
              "search_converged": False, "n_circles_tested": 0, "n_valid_surfaces": 0,
              "n_unsolved": 0, "warnings": "", "error": "", "runtime_s": None}
    try:
        model, geometry = initialize_model(H, angle, phi, c, gamma, ext_factor, depth_factor)
        unsolved = {}
        with contextlib.redirect_stdout(io.StringIO()):
            fs_cache, converged, _, circle_cache = circular_search(
                model, settings.method,
                num_slices=settings.num_slices,
                seed=settings.search_start,
                tol=settings.search_tol,
                fs_tol=settings.search_fs_tol,
                max_iter=settings.search_max_iter,
                min_grid_frac=settings.search_min_grid_frac,
                diagnostic=False,
                entry_range=geometry["entry_range"],
                exit_range=geometry["exit_range"],
                unsolved_out=unsolved,
            )
        if not fs_cache:
            raise RuntimeError("xslope circular search returned no valid surface")
        best = fs_cache[0]
        cy = float(best["Yo"])
        surface_left, surface_right, surface_depth = surface_limits(best)
        at_boundary, flags = check_boundary(surface_left, surface_right, surface_depth, geometry, H)
        solver_result = best.get("solver_result") or {}
        result.update(
            fos=float(best["FS"]), surface_left=surface_left, surface_right=surface_right,
            surface_depth=surface_depth, cx=float(best["Xo"]), cy=cy,
            radius=cy - float(best["Depth"]), at_boundary=at_boundary, boundary_flags=flags,
            search_converged=bool(converged), n_circles_tested=len(circle_cache),
            n_valid_surfaces=len(fs_cache), n_unsolved=int(unsolved.get("unsolved", 0)),
            warnings=" | ".join(map(str, solver_result.get("warnings", []))),
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["runtime_s"] = time.perf_counter() - started
    return result


def analysis_row(combo, analysis: dict, ext_factor=None, depth_factor=None) -> dict:
    """One result row, keyed by ``data.RESULT_COLUMNS``."""
    angle, phi, c, gamma, H = (float(v) for v in combo)
    _, slope_length, extension, depth = model_dims(H, angle, ext_factor, depth_factor)
    solved = analysis["fos"] is not None

    def r6(value):
        return round(value, 6) if solved else ""

    return {
        "angle": angle, "phi": phi, "c": c, "gamma": gamma, "H": H,
        "FOS": r6(analysis["fos"]),
        "Surface_L": r6(analysis["surface_left"]),
        "Surface_R": r6(analysis["surface_right"]),
        "Surface_Depth": r6(analysis["surface_depth"]),
        "Center_X": r6(analysis["cx"]),
        "Center_Y": r6(analysis["cy"]),
        "Radius": r6(analysis["radius"]),
        "At_Boundary": analysis["at_boundary"] if solved else "",
        "slope_length": round(slope_length, 6),
        "extension": round(extension, 6),
        "depth": round(depth, 6),
        "Search_Converged": analysis["search_converged"] if solved else False,
        "N_Circles_Tested": analysis["n_circles_tested"],
        "N_Valid_Surfaces": analysis["n_valid_surfaces"],
        "N_Unsolved": analysis["n_unsolved"],
        "Runtime_s": round(analysis["runtime_s"], 3),
        "Warnings": analysis["warnings"],
        "Error": "" if solved else analysis["error"],
    }


def analyse(combo) -> dict:
    """Solve one parameter combination ``(angle, phi, c, gamma, H)``."""
    angle, phi, c, gamma, H = (float(v) for v in combo)
    return analysis_row(combo, calculate_fos(angle, phi, c, gamma, H))


# --- a set of analyses ----------------------------------------------------------------

def run_set(combos, *, workers: int, label: str, fn=None) -> list[dict]:
    """Solve every combination, in design order. Nothing is written here: a set
    of 300 takes minutes, so an interrupted set is simply rerun."""
    fn = fn or analyse
    combos = [tuple(float(v) for v in combo) for combo in combos]
    rows = []

    def progress(index, row):
        state = row["Error"] or f"FOS={row['FOS']}" + (" AT BOUNDARY" if row["At_Boundary"] is True else "")
        print(f"[{label} {index}/{len(combos)}] {state}", flush=True)

    if workers == 1:
        for index, combo in enumerate(combos, 1):
            rows.append(fn(combo))
            progress(index, rows[-1])
    else:
        # spawn: portable to macOS, and no solver state inherited from the parent.
        with mp.get_context("spawn").Pool(workers) as pool:
            for index, row in enumerate(pool.imap(fn, combos), 1):
                rows.append(row)
                progress(index, row)
    return rows


def read_rows(path) -> tuple[list[dict], list[str]]:
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader), list(reader.fieldnames or [])


def write_rows(path, rows, columns) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


# --- repair -------------------------------------------------------------------------

def variant_label(ext_factor, depth_factor) -> str:
    return f"ext{ext_factor or 1}x_depth{depth_factor or 1}x"


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def repair_rows(rows: list[dict], calculate=None) -> dict:
    """Recompute boundary-truncated rows on the enlarged domains of
    ``config.REPAIR_VARIANTS``, in order, until the surface sits inside the
    domain. Rows no variant frees are excluded as ``unresolved``; rows the
    solver could not analyse are excluded as ``solver_error``."""
    calculate = calculate or calculate_fos
    for row in rows:
        row["repair_variant"] = ""
        row["excluded"] = ""
    n_boundary = n_repaired = n_error = 0
    for index, row in enumerate(rows):
        if _float(row["FOS"]) is None:
            row["excluded"] = "solver_error"
            n_error += 1
            continue
        if str(row["At_Boundary"]).strip().lower() != "true":
            continue
        n_boundary += 1
        combo = [float(row[p]) for p in config.PARAMS]
        for ext_factor, depth_factor in config.REPAIR_VARIANTS:
            label = variant_label(ext_factor, depth_factor)
            analysis = calculate(*combo, ext_factor, depth_factor)
            if analysis["fos"] is None or analysis["at_boundary"]:
                print(f"  row {index + 1}: {label} did not free the surface")
                continue
            row.update(analysis_row(combo, analysis, ext_factor, depth_factor))
            row["repair_variant"] = label
            n_repaired += 1
            print(f"  row {index + 1}: repaired with {label}, FOS={analysis['fos']:.4f}")
            break
        else:
            row["excluded"] = "unresolved"
            print(f"  row {index + 1}: no variant freed the surface; excluded")
    return {"n_rows": len(rows), "n_boundary": n_boundary, "n_repaired": n_repaired,
            "n_unresolved": n_boundary - n_repaired, "n_solver_error": n_error}
