# Read-only, no solver, no test data. Stage 1 of the screener swap, recorded before any
# C-logG test prediction exists.
# Paired Δ = GBM − RF (C-logG − C-log) per realization r1..r5; r0 is reported apart and
# enters no threshold. Inputs are designs, solved finals and allocation metadata only,
# through sgs.structural, which bins as the generator does: (lo, hi], first bin closed,
# points outside [0.5, 6] excluded from the bin counts and reported in the last row.
import numpy as np
from sgs import config, structural

REALIZATIONS = range(1, config.REP_REALIZATIONS + 1)


def quantities(r):
    alloc, disp = structural.allocation_rows(r), structural.displacement_rows(r)
    out = {}
    for arm in ("C-log", "C-logG"):
        a, d = alloc[arm], disp[arm]
        out[arm] = {"abs_dev": a["abs_deviation"],
                    "stayed_pp": 100 * d["stayed"],
                    "shift": d["mean_abs_shift"],
                    "below": a["achieved_outside"]["< 0.5"],
                    "above": a["achieved_outside"]["> 6"],
                    "n": d["n"]}
    return out


def delta(q, key):
    return q["C-logG"][key] - q["C-log"][key]


# key, label, anticipated sign of Δ, threshold on mean Δ (None = directional only)
ROWS = [("abs_dev", "Σ|achieved − intended| (points)", -1, -10),
        ("stayed_pp", "stayed in predicted bin (pp)", +1, +10),
        ("shift", "mean |displacement| (bins)", -1, -0.15),
        ("below", "realized FOS < 0.5 (points)", None, None),
        ("above", "realized FOS > 6 (points)", None, None)]


def fmt(x):
    return f"{x:+.2f}" if isinstance(x, float) else f"{x:+d}"


def main():
    per_r = {r: quantities(r) for r in REALIZATIONS}
    r0 = quantities(0)
    for r, q in per_r.items():
        assert q["C-log"]["n"] == q["C-logG"]["n"] == 300, (r, q)

    print("## Per realization (RF = C-log, GBM = C-logG, Δ = GBM − RF)\n")
    print("| quantity | " + " | ".join(f"r{r} RF / GBM / Δ" for r in REALIZATIONS) + " |")
    print("|---|" + "---|" * len(REALIZATIONS))
    for key, label, _, _ in ROWS:
        cells = []
        for r in REALIZATIONS:
            q = per_r[r]
            rf, gbm = q["C-log"][key], q["C-logG"][key]
            f = (lambda v: f"{v:.2f}") if isinstance(rf, float) else str
            cells.append(f"{f(rf)} / {f(gbm)} / {fmt(delta(q, key))}")
        print(f"| {label} | " + " | ".join(cells) + " |")

    print("\n## Verdicts over r1..r5\n")
    print("| quantity | mean Δ | Δ per realization | anticipated sign | threshold | met |")
    print("|---|---|---|---|---|---|")
    for key, label, sign, threshold in ROWS:
        ds = [delta(per_r[r], key) for r in REALIZATIONS]
        mean = float(np.mean(ds))
        listed = ", ".join(fmt(d) for d in ds)
        if sign is None:
            print(f"| {label} | {mean:+.2f} | {listed} | — | directional only | — |")
            continue
        n_sign = sum(np.sign(d) == sign for d in ds)
        mean_ok = mean <= threshold if sign < 0 else mean >= threshold
        met = "yes" if mean_ok and n_sign >= 4 else "no"
        cmp = "≤" if sign < 0 else "≥"
        print(f"| {label} | {mean:+.2f} | {listed} | {'<' if sign < 0 else '>'} 0 in {n_sign}/5 "
              f"| mean {cmp} {threshold:+g}, sign in ≥ 4/5 | {met} |")

    print("\n## r0 (exploratory, enters no threshold)\n")
    print("| quantity | RF | GBM | Δ |")
    print("|---|---|---|---|")
    for key, label, _, _ in ROWS:
        rf, gbm = r0["C-log"][key], r0["C-logG"][key]
        f = (lambda v: f"{v:.2f}") if isinstance(rf, float) else str
        print(f"| {label} | {f(rf)} | {f(gbm)} | {fmt(delta(r0, key))} |")

    print("\n## Shared candidates (points of C-logG within MIN_DISTANCE of a C-log point)\n")
    for row in structural.shared_candidate_rows():
        print(f"- r{row['realization']}: {row['shared']}/{row['n']}")


if __name__ == "__main__":
    main()
