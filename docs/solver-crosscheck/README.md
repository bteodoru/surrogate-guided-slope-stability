# Independent cross-check of the solver

**Purpose.** An independent check of a sample of xslope results with another limit-equilibrium program, carried out by co-authors M. Aniculăesi and I. Lungu. It supports the CRediT role *Validation* and one sentence in Methods.

## The sample

- **Source:** 15 cases from `data/final/test_natural_300.csv`, the natural test set.
- **Eligible rows:** converged search, no boundary repair, and a critical surface clear of the domain edge (296 of 300).
- **Selection:** 3 cases per reporting zone (FOS < 0.5, 0.5–1, 1–2, 2–6, ≥ 6). Within each zone the rows are sorted by FOS and taken at the 1/6, 1/2 and 5/6 positions. The selection is deterministic, with no random draw.

## Files

- **`cases.csv`: sent to the checkers.** It holds the inputs only. The xslope FOS is withheld, so the check is blind. The columns to fill are `FOS_other`, `program_method`, `circle_*` and `notes`.
- **`xslope_reference.csv`: kept here, not sent.** It holds the xslope FOS and the critical circle of each case, with its source row.

## The model to reproduce

Every case is the same model.

- **Soil:** a homogeneous slope in one Mohr–Coulomb soil (c, φ, γ), drained, with no water table and no external load.
- **Geometry:**
  - the toe is at (0, 0) and the crest at (`crest_x`, `crest_y` = H);
  - the ground is flat at y = 0 from `left_x` to the toe, and at y = H from the crest to `right_x`;
  - the soil extends down to `bottom_y`, and the same soil lies below the toe level.
- **Method:** Bishop simplified, circular slip surfaces. The xslope run uses 30 slices.
  - **Exit range** on the toe side: from `left_x` to L/3, where L = `crest_x`.
  - **Entry range** on the crest side: from 2L/3 to `right_x`.
  - The checker may use their program's standard circular search inside the domain. They record the critical circle found.

## Reading the results (to fill in when they arrive)

- **Measure:** the relative difference FOS_other / FOS_xslope − 1, for each case.
- **What it reads:** only agreement between the two programs. A difference is first traced to the search window or the slice count, before it is read as an error.
- **Methods sentence:** written only from the numbers, for example: "A sample of 15 analyses spanning the five reporting zones was re-computed independently with [program], Bishop simplified; the FOS agreed within X%."
