import math

import numpy as np
import pytest

from sgs import config, data, solver
from tests.helpers import boundary_row, fake_analyse


def test_model_dims_matches_the_adaptive_rule():
    slope, L, ext, depth = solver.model_dims(10.0, 45.0)
    assert math.isclose(L, 10.0)
    assert math.isclose(ext, 15.0)          # max(1.5*H, 0.5*L)
    assert math.isclose(depth, 10.0)        # max(H, 0.5*L)
    _, _, ext2, depth2 = solver.model_dims(10.0, 45.0, 2.0, 3.0)
    assert math.isclose(ext2, 30.0) and math.isclose(depth2, 30.0)


def test_boundary_flags():
    geometry = {"depth": 10.0, "slope_length": 10.0, "extension": 15.0}
    assert solver.check_boundary(-9.95, 5.0, -2.0, geometry, 10.0) == (True, ["left_boundary"])
    assert solver.check_boundary(0.0, 24.95, -9.95, geometry, 10.0) == (True, ["right_boundary", "bottom_boundary"])
    assert solver.check_boundary(0.0, 20.0, -5.0, geometry, 10.0) == (False, [])


def _analysis(fos=1.5, at_boundary=False):
    return {"fos": fos, "surface_left": 0.1, "surface_right": 9.0, "surface_min_y": 0.2,
            "circle_nadir_y": -1.0, "cx": 2.0, "cy": 8.0, "radius": 9.0, "at_boundary": at_boundary,
            "boundary_flags": [], "search_converged": True, "n_circles_tested": 10,
            "n_valid_surfaces": 5, "n_unsolved": 0, "warnings": "", "error": "",
            "runtime_s": 1.23456}


def test_analysis_row_schema_for_solved_and_failed_cases():
    combo = (45.0, 30.0, 10.0, 18.0, 10.0)
    solved = solver.analysis_row(combo, _analysis())
    assert list(solved) == data.RESULT_COLUMNS
    assert solved["FOS"] == 1.5 and solved["Error"] == "" and solved["Runtime_s"] == 1.235
    assert solved["Surface_Min_Y"] == 0.2 and solved["Circle_Nadir_Y"] == -1.0
    failed = solver.analysis_row(combo, {**_analysis(fos=None), "error": "RuntimeError: x"})
    assert failed["FOS"] == "" and failed["At_Boundary"] == "" and failed["Error"] == "RuntimeError: x"
    assert failed["slope_length"] == 10.0


def test_surface_limits_take_the_arc_minimum_not_the_circle_nadir():
    from shapely.geometry import LineString
    # A toe circle: its centre lies left of the arc, so the full circle's
    # lowest point is well below the lowest point of the failure surface.
    best = {"failure_surface": LineString([(0.0, 0.1), (5.0, 3.0), (10.0, 8.0)]), "Depth": -12.0}
    assert solver.surface_limits(best) == (0.0, 10.0, 0.1, -12.0)


def test_a_circle_nadir_below_the_bottom_is_not_a_bottom_flag():
    geometry = {"depth": 10.0, "slope_length": 10.0, "extension": 15.0}
    from shapely.geometry import LineString
    best = {"failure_surface": LineString([(0.0, 0.1), (5.0, 3.0), (10.0, 8.0)]), "Depth": -12.0}
    left, right, surface_min_y, _ = solver.surface_limits(best)
    assert solver.check_boundary(left, right, surface_min_y, geometry, 10.0) == (False, [])


def _combos(n=7):
    rng = np.random.default_rng(0)
    return np.column_stack([rng.uniform(lo, hi, n) for lo, hi in config.DOMAIN.values()])


@pytest.mark.parametrize("workers", [1, 2])
def test_run_set_keeps_design_order(workers):
    combos = _combos()
    rows = solver.run_set(combos, workers=workers, label="t", fn=fake_analyse)
    np.testing.assert_allclose([[r[p] for p in config.PARAMS] for r in rows], combos)


def test_rows_round_trip_through_csv(tmp_path):
    rows = solver.run_set(_combos(3), workers=1, label="t", fn=fake_analyse)
    path = tmp_path / "out" / "r.csv"
    solver.write_rows(path, rows, data.RESULT_COLUMNS)
    back, columns = solver.read_rows(path)
    assert columns == data.RESULT_COLUMNS and len(back) == 3
    assert not list(path.parent.glob(".*.tmp"))


# --- repair ---------------------------------------------------------------------------

def fake_calculate(frees_at):
    """Returns a boundary-free surface only on the given variant."""
    def calculate(angle, phi, c, gamma, H, ext_factor=None, depth_factor=None):
        free = (ext_factor, depth_factor) == frees_at
        return {**_analysis(fos=1.2, at_boundary=not free), "runtime_s": 0.1}
    return calculate


def test_label():
    assert solver.variant_label(2.0, None) == "ext2.0x_depth1x"


def test_rows_without_problems_are_left_alone():
    rows = [boundary_row(), boundary_row()]
    summary = solver.repair_rows(rows, calculate=fake_calculate(None))
    assert summary["n_boundary"] == 0 and summary["n_repaired"] == 0
    assert all(r["repair_variant"] == "" and r["excluded"] == "" for r in rows)


def test_boundary_row_is_repaired_by_the_first_freeing_variant():
    rows = [boundary_row(boundary="True")]
    summary = solver.repair_rows(rows, calculate=fake_calculate((None, 2.0)))
    assert summary["n_repaired"] == 1
    assert rows[0]["repair_variant"] == "ext1x_depth2.0x"
    assert rows[0]["FOS"] == 1.2 and rows[0]["At_Boundary"] is False and rows[0]["excluded"] == ""


def test_unfreeable_row_is_excluded():
    rows = [boundary_row(boundary="True")]
    summary = solver.repair_rows(rows, calculate=fake_calculate(("never", "never")))
    assert summary["n_unresolved"] == 1 and rows[0]["excluded"] == "unresolved"


def test_solver_error_is_excluded_not_repaired():
    rows = [boundary_row(fos="")]
    summary = solver.repair_rows(rows, calculate=fake_calculate(None))
    assert summary["n_solver_error"] == 1 and rows[0]["excluded"] == "solver_error"


@pytest.mark.slow
def test_one_real_analysis():
    row = solver.analyse((24.70810805776474, 33.64879751536226, 12.628212513088526,
                          17.341403519225082, 4.832465812168988))
    # First row of the seed design: FOS 3.171339 in the reference data, not at boundary.
    assert row["Error"] == ""
    assert abs(row["FOS"] - 3.171339) < 1e-4
    assert row["At_Boundary"] is False
    assert row["Surface_Min_Y"] >= row["Circle_Nadir_Y"]
