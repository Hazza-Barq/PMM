import pandas as pd
import numpy as np
from pathlib import Path

# Path to Excel file (no hardcoding to your PC path)
CSV_FILE = Path(__file__).resolve().parent.parent / "data" / "Assegai_Aero_Table2.xlsx"

# Load once into NumPy arrays for speed
df = pd.read_excel(CSV_FILE, engine="openpyxl")
mach_values = df["mach"].to_numpy()
cd_values = df["C_D0"].to_numpy()

def get_cd(mach: float) -> float:
    """Return Cd by linear interpolation from Mach number."""
    if mach <= mach_values[0]:
        return cd_values[0]
    if mach >= mach_values[-1]:
        return cd_values[-1]

    idx = np.searchsorted(mach_values, mach)
    m1, m2 = mach_values[idx - 1], mach_values[idx]
    c1, c2 = cd_values[idx - 1], cd_values[idx]

    t = (mach - m1) / (m2 - m1)
    return c1 + t * (c2 - c1)
