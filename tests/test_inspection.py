import pytest

from sgs import config, inspection, solver
from tests.helpers import synthetic_final, write_final


def test_variant_factors_invert_the_repair_labels():
    assert inspection.variant_factors("") == (None, None)
    for factors in config.REPAIR_VARIANTS:
        assert inspection.variant_factors(solver.variant_label(*factors)) == factors
    with pytest.raises(ValueError):
        inspection.variant_factors("ext9x_depth9x")


def test_row_out_of_range_is_refused(root):
    write_final("seed", synthetic_final(5, 1))
    with pytest.raises(IndexError):
        inspection.inspect_row("seed", 6)


@pytest.mark.slow
def test_inspect_reproduces_a_stored_row(root):
    frame = synthetic_final(1, 1)
    frame.loc[0, list(config.PARAMS)] = (24.70810805776474, 33.64879751536226, 12.628212513088526,
                                         17.341403519225082, 4.832465812168988)
    frame.loc[0, "FOS"] = 3.171339
    write_final("seed", frame)
    found = inspection.inspect_row("seed", 1)
    assert abs(found["fos"] - 3.171339) < 1e-4 and found["at_boundary"] is False
    assert all(path.exists() for path in found["figures"].values())
