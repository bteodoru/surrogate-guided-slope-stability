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

### HYRCAN (2026-10-25): an extra check, not cited in the paper

- **Status:** run by the first author as an additional check. The paper cites only Slide2 (decided 2026-10-25).

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

### V06 in detail (2026-10-25; not in the paper)

- **The circle:** xslope's critical circle has its centre slightly above crest height, 0.009 m higher. So the circle enters the crest almost vertically, but not vertically. Slide2's critical circle has its centre much higher, at y ≈ 21.4 against a crest at 17.6.
- **xslope, on its own circle:** the FOS depends on the slice count. It is 0.9375 with 25 slices, 0.9396 with 30, 0.9440 with 50 and 0.9480 with 200.
- **HYRCAN, on the same circle:** the run had to be set to this surface, with its slip limits fixed at x = 0.001 and x = 18.109. HYRCAN does not refuse the circle. It gives 0.966, and m_α on the last of its 30 slices is 0.2412, above 0.2. The 2.8% gap between the two programs on the same surface comes from how each **discretizes the last slice**, at the crest. Base tension is not the cause: HYRCAN keeps negative normals too, and its N' there is −880 kN.
  - **xslope:** α is the tangent at the slice's midpoint, 79.2°, and the weight comes from the true area under the arc, 39.2 kN. Then m_α = 0.297 and N' = −452 kN.
  - **HYRCAN:** the base is the chord between the slice's ends, rising 4.64 m over 0.628 m. So α = 82.3°, the base length is 4.70 m and the weight, from the area under the chord, is 27.5 kN. Then m_α = 0.241 and N' = −880 kN, as in HYRCAN's slice report.
  - **Why it matters here:** on that slice the arc turns from about 75° to 90° within 0.63 m, so the two conventions differ greatly. With cohesion dominant, this one slice moves the total by a few percent.
  - **The slice count:** xslope's FOS on its circle rises with it, from 0.940 with 30 slices to 0.948 with 200. Both programs should converge to the same limit as the slices get finer.
  - **To confirm:** run HYRCAN on this circle with 100–200 slices.
- **On HYRCAN's own circle the two agree:** xslope gives 0.9525 with 25 slices, against HYRCAN's 0.9577.
- **How common this geometry is:** about 12% of all critical circles in the data have their centre within 0.01 H of crest height. This is a cohesive slope with no tension crack modelled.
- **Decision:** the paper reports only the largest difference, in FOS units, with its FOS. The point may matter for the data descriptor.
