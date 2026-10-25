"""Write the geometry of the 15 cross-check cases as DXF files, for the checkers.

Each file holds the model exactly as the pipeline builds it (sgs.solver.initialize_model):
the soil polygon on layer SOIL and the ground profile on PROFILE_SOIL, in metres, toe at
(0, 0). No circle is written, so the check stays blind. Run from the repository root:

    PYTHONPATH=. python docs/solver-crosscheck/export_dxf.py

xslope's DXF export needs ezdxf (xslope[cad]), which the pipeline does not install.
"""
import csv
from pathlib import Path

from sgs import solver
from xslope.cad import export_dxf

HERE = Path(__file__).parent


def main():
    out = HERE / "dxf"
    out.mkdir(exist_ok=True)
    with open(HERE / "cases.csv", newline="") as f:
        for row in csv.DictReader(f):
            model, geometry = solver.initialize_model(
                float(row["H_m"]), float(row["angle_deg"]), float(row["phi_deg"]),
                float(row["c_kPa"]), float(row["gamma_kNm3"]))
            model["circles"] = []          # blind: no search or critical circle
            export_dxf(model, out / f"{row['case']}.dxf")
            print(row["case"], geometry["slope_length"], geometry["extension"], geometry["depth"])


if __name__ == "__main__":
    main()
