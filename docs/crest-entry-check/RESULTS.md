# Robustness to near-vertical crest entries (2026-10-26)

## Why this check exists

- **The geometry.** About 12% of the critical circles in the data have their centre within 0.01 H of crest height. They enter the crest almost vertically, in cohesive slopes, with no tension crack modelled.
- **Why it matters.** xslope rejects circles whose ground crossing lies above the centre, so the search often stops at that limit. On such a circle, the FOS depends on how a program discretizes the steep last slice. In V06, xslope and HYRCAN differ by 2.8% on the same circle (`../solver-crosscheck/README.md`).
- **The question.** Does that alter the study's results?

## What was done

`check.py` drops the test points with such a circle. These are 18 of the 300 natural test points, 5 of them on the decision range, and 13 of the 150 stratified ones, 3 on the decision range. It then recomputes two things from the stored replication predictions, r1–r5, with nothing refitted:

- **Every arm against B:** 10 arms, ablations included, on the decision range and its two halves, for each learner and test set. That makes 120 Welch intervals.
- **The relocation gate:** median L < 0 < median H, per learner × test cell.

On the full test sets, the script reproduces `outputs/metrics/replications.json` exactly.

## Result

- **The decision range, the primary criterion:** no verdict changes.
- **The relocation gate:** the same counts for every arm, with and without these points. C-log passes in 4 of 4 cells, C-seq in 4 of 4, and C-logG, C-logP and C-oracle in 3 of 4 each.
- **Four of the 120 intervals change verdict, all on a half of the decision range:**
  - **RF, natural test set, [1, 2), C-log against B.** The interval goes from [+0.0016, +0.0224] to [−0.0001, +0.0226], with the mean going from 0.012 to 0.011. Its lower end now reaches 0 at the fourth decimal; it was already marginal.
  - **RF, stratified test set, [0.5, 1), C₀-log, C₀-oracle and C₀-logP against B.** On 10 points instead of 12, the intervals move from containing 0 to lying below 0. This is the same direction as before, only sharper.

## Conclusion

- **Test side:** the results do not depend on these points.
- **Training side:** the labels are xslope's for every arm. Its convention on these surfaces is therefore part of the function every surrogate learns, the same for all arms.

`intervals.csv` and `gates.csv` hold the full output.
