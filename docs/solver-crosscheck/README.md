# Independent cross-check of the solver

**Purpose.** A check of a sample of xslope results with another limit-equilibrium program, Slide2. It supports the paragraph "Independent check" in Methods.

## The sample

- **Source:** 15 cases from `data/final/test_natural_300.csv`, the natural test set.
- **Eligible rows:** converged search, no boundary repair, and a critical surface clear of the domain edge (296 of 300).
- **Selection:** 3 cases per reporting zone (FOS < 0.5, 0.5–1, 1–2, 2–6, ≥ 6). Within each zone the rows are sorted by FOS and taken at the 1/6, 1/2 and 5/6 positions. The selection is deterministic, with no random draw.

## Files

- **`cases.csv`:** the inputs of each case, as given to the checker. The xslope FOS is withheld, so the check is blind.
- **`dxf/V01.dxf` … `dxf/V15.dxf`:** the geometry of each case as the pipeline builds it, exported with xslope's own `export_dxf`:
  - the soil polygon is on layer `SOIL`, and the ground profile on `PROFILE_SOIL`;
  - units are metres, with the toe at (0, 0);
  - no circle is written, so the check stays blind;
  - the material properties are in `cases.csv`, whose coordinates are rounded to the millimetre; the DXF coordinates differ from them by at most 2 mm.
  - `export_dxf.py` writes them. It needs `ezdxf` (xslope's `cad` extra), which the pipeline does not install.
- **`xslope_reference.csv`:** the xslope FOS and critical circle of each case, with its source row.
- **`results.csv`:** the FOS from each program and the relative differences. The `note` of V09 records that it was first entered with a wrong friction angle; the values kept are from the rerun with the correct one.

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
  - The checker uses the program's own circular search inside the domain and records the critical circle found.

## Results

Slide2 9.044 (Rocscience), Bishop simplified, 50 slices, Auto Refine Search, all 15 cases. The program reports are kept by the authors; the extracted values are in `results.csv`.

- **Lower FOS:** xslope gave the lower FOS in 13 of the 15 cases. In the other two, V02 and V03, it was higher by at most 0.59%. Slide2's own search therefore finds no lower critical circle than xslope's restricted one.
- **Agreement:** the FOS agree within 0.6% in 14 of the 15 cases. The median absolute difference is 0.26%.
- **V06:** the difference is 0.017 (1.8%, at FOS 0.94). It is explained below.

## V06 in detail

- **The case:** a steep (59°), strongly cohesive slope (c ≈ 50 kPa, φ ≈ 6°).
- **The circle:** xslope's critical circle has its centre 0.009 m above crest height, so it enters the crest almost vertically. Slide2's critical circle has its centre much higher, at y ≈ 21.4 against a crest at 17.6.
- **The cause is the last slice.** On the same circle, the programs discretize the last slice, at the crest, differently:
  - **xslope:** α is the tangent at the slice's midpoint, 79.2°, and the weight comes from the true area under the arc, 39.2 kN. Then m_α = 0.297.
  - **Chord-based programs** (HYRCAN, below): the base is the chord between the slice's ends, rising 4.64 m over 0.628 m. So α = 82.3°, and the weight, from the area under the chord, is 27.5 kN. Then m_α = 0.241.
  - On that slice the arc turns from about 75° to 90° within 0.63 m, so the two conventions differ greatly. With cohesion dominant, this one slice moves the total by a few percent.
  - Base tension is not the cause: both conventions keep negative normals on the last slice.
- **On the same circle, the programs converge from opposite sides as the slices get finer:**

  | slices | xslope (arc) | HYRCAN (chord) | Slide2 |
  |---|---|---|---|
  | 30 | 0.9396 | 0.963 | 0.964 |
  | 40 | 0.9426 | 0.959 | — |
  | 50 | 0.9440 | 0.957 | 0.958 |
  | 60 | 0.9449 | 0.955 | — |
  | 70+ | 0.9467 (100), 0.9480 (200) | refused: no valid surface | — |

  - Slide2 and HYRCAN agree within 0.001 on this circle.
  - The gap narrows from 2.4% at 30 slices to 1.1% at 60. From 70 slices on, HYRCAN refuses the surface: the last slice's chord passes 85°, and m_α falls below 0.2. Computed for this circle, m_α is 0.240 at 30 slices (HYRCAN reports 0.2412), 0.202 at 60 and 0.196 at 70.
- **Reading:** the circle is at the edge of Bishop's admissibility, and its FOS depends on how the last slice is discretized.
- **How common this geometry is:** about 12% of all critical circles in the data have their centre within 0.01 H of crest height, in cohesive slopes with no tension crack modelled. The study's results do not depend on them: `../crest-entry-check/RESULTS.md`.

## HYRCAN: an additional check, not cited in the paper

- **The check:** the same 15 cases were solved with HYRCAN 2.0.15, Bishop simplified, with the same geometry and the same entry and exit windows as xslope (L/3, 2L/3), and HYRCAN's default of 25 slices. The values are in `FOS_hyrcan` in `results.csv`.
- **Agreement:** within 1.93%, with a median absolute difference of 0.29%. Leaving V06 aside, every case agrees within 0.71%.
- **On HYRCAN's own V06 circle,** xslope gives 0.9525 with 25 slices, against HYRCAN's 0.9577.

## How the programs treat base tension at the crest

- **Slide2.** The user guide (Project Settings › Advanced, https://www.rocscience.com/help/slide2/documentation/slide-model/project-settings/advanced) gives the following defaults:
  - tensile normal stress may be calculated on the base of steeply inclined slices near the crest;
  - the tensile stress check is off by default, and even when on it leaves the crest slices unchecked;
  - "zero shear strength when base in tension" is on by default, and acts only when the shear strength would be negative. That is not the case on V06's last slice.
- **xslope.** The search page (https://xslope.org/en/stable/lem/search/#tension-at-the-crest) says:
  - substantial tension at the crest draws a note ("Consider adding a tension crack") and does not change the FOS or the reported surface;
  - circles whose ground crossing lies above the centre are rejected, which is why searches stop with the centre just above the crest.
