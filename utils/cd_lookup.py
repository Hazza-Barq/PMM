# utils/cd_lookup.py
import pandas as pd
import numpy as np
from pathlib import Path

CSV_FILE = Path(__file__).resolve().parent.parent / "data" / "Assegai_Aero_Table2.xlsx"

# Safe load: allow csv or xlsx
if CSV_FILE.suffix.lower() in (".xlsx", ".xls"):
    df = pd.read_excel(CSV_FILE, engine="openpyxl")
else:
    df = pd.read_csv(CSV_FILE)

# Normalize column names (lowercase, strip)
cols = {c: c.strip().lower() for c in df.columns}
df.rename(columns=cols, inplace=True)

# required: a Mach column (try many common names)
_possible_mach_names = ["mach", "m", "mach_number"]
mach_col = None
for n in _possible_mach_names:
    if n in df.columns:
        mach_col = n; break
if mach_col is None:
    raise RuntimeError("Could not find Mach column in aero table; expected one of: " + ", ".join(_possible_mach_names))

mach_values = df[mach_col].to_numpy(dtype=float)

# pick a set of coefficient columns common in your tables (fallback if missing)
# keys: normalized lower-case names we will return
_expected_coeffs = {
    "cd0": ["cd0", "c_d0", "c_d", "cd_0", "c_d_0", "C_D0"],
    "cd_alpha2": ["cd_alpha2", "c_d_alpha2", "cd_a2", "C_D2"],
    "cl_alpha": ["cl_alpha", "c_l_alpha", "C_La"],
    "cl_alpha3": ["cl_alpha3", "c_l_alpha3", "C_L3"],
    "cm_alpha": ["cm_alpha","c_m_alpha"],
    "cm_alpha3": ["cm_alpha3"],
    # add any other names you expect (Cmag_f, Cmag_m, CM_q, etc.)
    "cm_q": ["cm_q", "C_Mq"],
    "cm_alphadot": ["cm_alphadot", "c_m_alphadot"],
    "cmag_f": ["cmag_f", "c_m_f", "C_ma-f"],
    "cmag_m": ["cmag_m", "c_mag_m", "C_mag-m"],
    "cspin": ["cspin", "C_spin"]
}

# build arrays for available coefficients (if column missing, fill with zeros or use sensible fallback)
coeff_arrays = {}
for key, name_list in _expected_coeffs.items():
    found = None
    for n in name_list:
        if n in df.columns:
            found = n; break
    if found is not None:
        coeff_arrays[key] = df[found].to_numpy(dtype=float)
    else:
        # fallback: zeros
        coeff_arrays[key] = np.zeros_like(mach_values)

def get_coeffs(mach: float):
    """Return a dict of interpolated coefficients at requested mach (scalar)."""
    m = float(mach)
    # clamp outside-range to endpoints
    def interp(arr):
        return float(np.interp(m, mach_values, arr, left=arr[0], right=arr[-1]))
    out = {}
    out["mach"] = m
    for k, arr in coeff_arrays.items():
        out[k] = interp(arr)
    # also include the simple Cd single value if present in df
    # attempt to find a 'cd' or 'c_d0' fallback
    if "cd" in df.columns:
        out.setdefault("cd0", float(np.interp(m, mach_values, df["cd"].to_numpy(), left=df["cd"].iloc[0], right=df["cd"].iloc[-1])))
    return out
