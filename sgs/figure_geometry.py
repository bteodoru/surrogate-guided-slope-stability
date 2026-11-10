"""The geometry figure of the curated deposit's data article: one real analysis, annotated with
the column names of `simulations`.

  PYTHONPATH=. python -m sgs.figure_geometry     read data/curated/, write docs/curated-figures/

The analysis is named by its simulation_id (FIGURE_ID). It is a toe circle whose centre lies left of
the toe, so the lowest point of the full circle (Circle_Nadir_Y) falls outside the slip surface and
differs from Surface_Min_Y.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Polygon

from sgs import config, curate

FIGURE_ID = "sim_045ad082175f7789"


def output_dir() -> Path:
    return config.ROOT / "docs" / "curated-figures"


def _dimension(ax, x0, y0, x1, y1, label, *, offset=(0, 0), ha="center", va="center", family="monospace"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "<->", "lw": 0.8, "color": "0.25", "shrinkA": 0, "shrinkB": 0})
    ax.text((x0 + x1) / 2 + offset[0], (y0 + y1) / 2 + offset[1], label, ha=ha, va=va, fontsize=8,
            family=family, color="0.15")


def draw(row: pd.Series) -> plt.Figure:
    H, beta = row["H"], np.radians(row["angle"])
    L, ext, depth = row["slope_length"], row["extension"], row["depth"]
    cx, cy, r = row["Center_X"], row["Center_Y"], row["Radius"]
    xl, xr = row["Surface_L"], row["Surface_R"]

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    soil = [(-depth, 0), (0, 0), (L, H), (L + ext, H), (L + ext, -depth), (-depth, -depth)]
    ax.add_patch(Polygon(soil, closed=True, facecolor="#e9e4d8", edgecolor="0.3", lw=0.9))

    # the full circle, dashed, and the slip surface on it, between Surface_L and Surface_R
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), ls=(0, (4, 3)), lw=0.7, color="0.55")
    xs = np.linspace(xl, xr, 200)
    ax.plot(xs, cy - np.sqrt(np.maximum(r ** 2 - (xs - cx) ** 2, 0)), lw=2.2, color="#b2182b")

    # centre and radius
    ax.plot(cx, cy, "o", ms=4, color="0.1")
    ax.text(cx + 0.4, cy + 0.3, "(Center_X, Center_Y)", fontsize=8, family="monospace", va="bottom")
    xm = xl + 0.6 * (xr - xl)                              # radius drawn to a point of the slip surface
    a = np.arctan2(-np.sqrt(r ** 2 - (xm - cx) ** 2), xm - cx)
    ax.plot([cx, cx + r * np.cos(a)], [cy, cy + r * np.sin(a)], lw=0.8, color="0.1")
    ax.text(cx + 0.5 * r * np.cos(a) + 0.5, cy + 0.5 * r * np.sin(a), "Radius", fontsize=8,
            family="monospace", ha="left")

    # lowest point of the full circle, and of the slip surface
    nadir = row["Circle_Nadir_Y"]
    ax.plot(cx, nadir, "v", ms=5, color="0.1")
    ax.text(cx, nadir - 0.6, "Circle_Nadir_Y", fontsize=8, family="monospace", ha="center", va="top")
    ax.plot(xl, row["Surface_Min_Y"], "s", ms=4, color="#b2182b")
    ax.annotate("Surface_Min_Y", xy=(xl, row["Surface_Min_Y"]), xytext=(xl + 2.2, -2.6), fontsize=8,
                family="monospace", arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "0.3"})

    # where the slip surface meets the ground
    for x, y, name, dx in ((xl, 0.0, "Surface_L", -0.3), (xr, H, "Surface_R", 0.3)):
        ax.plot([x, x], [y, y + 1.2], lw=0.8, color="#b2182b")
        ax.text(x + dx, y + 1.35, name, fontsize=8, family="monospace", va="bottom",
                ha="right" if dx < 0 else "left")

    # height and angle
    _dimension(ax, L + ext + 1.2, 0, L + ext + 1.2, H, "H", offset=(0.5, 0), ha="left")
    ax.plot([L, L + ext + 1.6], [0, 0], lw=0.4, color="0.6", ls=":")
    ax.add_patch(Arc((0, 0), 5, 5, theta1=0, theta2=np.degrees(beta), lw=0.8, color="0.25"))
    ax.text(3.0, 0.9, "angle", fontsize=8, family="monospace")

    # model extent, below the domain
    y = -depth - 1.2
    _dimension(ax, 0, y, L, y, "slope_length", offset=(0, -0.7), va="top")
    _dimension(ax, L, y, L + ext, y, "extension", offset=(0, -0.7), va="top")
    _dimension(ax, -depth - 1.2, 0, -depth - 1.2, -depth, "depth", offset=(-0.5, 0), ha="right")
    for x in (0, L, L + ext):
        ax.plot([x, x], [-depth, y - 0.4], lw=0.4, color="0.6", ls=":")

    ax.set_aspect("equal")
    ax.set_xlim(-depth - 5.5, L + ext + 4.5)
    ax.set_ylim(-depth - 3.5, cy + 2.5)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def run() -> Path:
    sims = pd.read_parquet(curate.out_dir() / "data" / "simulations.parquet").set_index("simulation_id")
    fig = draw(sims.loc[FIGURE_ID])
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "geometry.pdf")
    fig.savefig(out / "geometry.png", dpi=600)
    plt.close(fig)
    return out / "geometry.pdf"


if __name__ == "__main__":
    print(run())
