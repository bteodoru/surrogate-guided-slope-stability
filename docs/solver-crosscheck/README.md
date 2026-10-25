# Independent cross-check of the solver

**Purpose.** An independent check of a sample of xslope results with another limit-equilibrium program, carried out by co-authors M. Aniculăesi and I. Lungu. It supports the CRediT role *Validation* and one sentence in Methods.

## The sample

- **Source:** 15 cases from `data/final/test_natural_300.csv`, the natural test set.
- **Eligible rows:** converged search, no boundary repair, and a critical surface clear of the domain edge (296 of 300).
- **Selection:** 3 cases per reporting zone (FOS < 0.5, 0.5–1, 1–2, 2–6, ≥ 6). Within each zone the rows are sorted by FOS and taken at the 1/6, 1/2 and 5/6 positions. The selection is deterministic, with no random draw.

## Files

- **`cases.csv`: sent to the checkers.** It holds the inputs only. The xslope FOS is withheld, so the check is blind. The columns to fill are `FOS_other`, `program_method`, `circle_*` and `notes`.
- **`dxf/V01.dxf` … `dxf/V15.dxf`: may be sent with `cases.csv`.** These hold the geometry of each case as the pipeline builds it, exported with xslope's own `export_dxf`:
  - the soil polygon is on layer `SOIL`, and the ground profile on `PROFILE_SOIL`;
  - units are metres, with the toe at (0, 0);
  - no circle is written, so the check stays blind;
  - the material properties are in `cases.csv`.
  - **Relation to `cases.csv`:** its coordinates are rounded to the millimetre, and the DXF coordinates differ from them by at most 2 mm.
  - **How they were written:** `export_dxf.py`, which needs `ezdxf` (xslope's `cad` extra), not installed by the pipeline.
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

## Results (2026-10-25)

- **The check:** M. Aniculăesi re-computed the cases blind, with Slide2 9.044 (Rocscience): Bishop simplified, 50 slices, Auto Refine Search. The reports are kept by the authors; the extracted values are in `results.csv`.
- **14 cases:** the FOS agree within 1.8%. The median absolute difference is 0.26%, and 13 of the 14 cases agree within 0.6%.
  - Slide2 is the higher in 12 of the 14 cases, consistent with xslope's search finding a slightly lower minimum.
  - The largest difference is V06 (+1.77%), where the two critical circles differ.
- **V09:** the friction angle was entered as 16.858° instead of 13.937°, so this case is not comparable. With 16.858°, xslope gives 2.1477 against Slide2's 2.1494, a difference of +0.08%. A rerun with 13.937° has been requested.

### HYRCAN (2026-10-25)

- **The check:** the same 15 cases were solved with HYRCAN 2.0.15, Bishop simplified, by the authors' script (`../spencer_datasets/repo/hyrcan/run_analyses.py`, outside this repository). The script uses the same geometry and the same entry and exit windows as xslope (L/3, 2L/3), and HYRCAN's default of 25 slices. The values are in `FOS_hyrcan` in `results.csv`, to three decimals, and V06 is 0.95769. HYRCAN was given the correct inputs for all 15, V09 included.
- **A correction:** the first copy of `cases_solved.csv` carried Slide2's value for V06 (0.956) by mistake; HYRCAN's own is 0.958.
- **Agreement:** within 1.93%. The median absolute difference is 0.29%. Leaving V06 aside, every case agrees within 0.71%. V09: 1.863 against xslope's 1.860, +0.17%.
- **V06:** the largest difference in both programs. Slide2 gives 0.956 and HYRCAN 0.958, on practically the same circle: centre about (−0.4 to −0.6, 21.4 to 21.7), radius about 21.5 m. xslope gives 0.940.
  - **Not rounding:** with the inputs rounded as in `cases.csv`, xslope gives 0.93958.
  - **The slice count does not explain it.** With 50 slices, xslope gives 0.9435. HYRCAN, with 25 slices, and Slide2, with 50, agree with each other.
  - **The rest is the surface:** the case is a steep (59°), strongly cohesive slope (c ≈ 50 kPa, φ ≈ 6°). xslope's critical circle enters the crest almost vertically, with base tension on the last slices. Slide2 and HYRCAN settle on a slightly flatter circle.

### Open (2026-10-25)

- **Slide2, V09:** rerun with φ = 13.937°.
- **Slide2, V06:** the FOS on xslope's own critical circle, as a user-defined surface, with 30 and 50 slices. The circle has its centre at (0.474, 17.632) and radius 17.636 m, and runs from x = 0.001 to x = 18.109.
  - **If it gives about 0.94,** the 1.8% is only the search: xslope finds the more critical circle.
  - **If it is refused or very different,** an admissibility rule (m_α or base tension) explains it, and Methods must say so.
- **HYRCAN slice count:** confirmed, 25.
