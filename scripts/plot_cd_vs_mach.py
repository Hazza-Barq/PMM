# scripts/plot_cd_vs_mach.py
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# --- inputs/outputs ---
IN_XLSX = Path("data/Assegai_Aero_Table2.xlsx")     # adjust if needed
OUT_DIR = Path("out"); OUT_DIR.mkdir(exist_ok=True)
OUT_PNG = OUT_DIR / "cd_vs_mach.png"
OUT_CSV = OUT_DIR / "cd_vs_mach_interpolated.csv"

# --- load table (robust col detection) ---
df = pd.read_excel(IN_XLSX)  # pandas auto engine
df.columns = [str(c).strip().lower() for c in df.columns]

mach_candidates = ["mach", "m", "mach_number"]
cd_candidates   = ["c_d0", "cd0", "cd", "c_d", "c_d_0"]

mach_col = next((c for c in mach_candidates if c in df.columns), None)
cd_col   = next((c for c in cd_candidates   if c in df.columns), None)
if mach_col is None or cd_col is None:
    raise RuntimeError(f"Could not find 'mach'/'cd' columns. Found: {df.columns.tolist()}")

data = (
    df[[mach_col, cd_col]]
    .dropna()
    .rename(columns={mach_col: "mach", cd_col: "cd"})
    .sort_values("mach")
    .drop_duplicates(subset="mach")
)

# --- interpolation ---
mmin, mmax = float(data["mach"].min()), float(data["mach"].max())
mach_smooth = np.linspace(mmin, mmax, 500)
cd_smooth   = np.interp(mach_smooth, data["mach"].values, data["cd"].values,
                        left=data["cd"].iloc[0], right=data["cd"].iloc[-1])

pd.DataFrame({"mach": mach_smooth, "cd_interp": cd_smooth}).to_csv(OUT_CSV, index=False)

# --- plot ---
plt.figure(figsize=(7.5, 4.8))
plt.scatter(data["mach"], data["cd"], s=20, label="Table points")
plt.plot(mach_smooth, cd_smooth, linewidth=2, label="Linear interpolation")
plt.xlabel("Mach number"); plt.ylabel(r"$C_D$")
plt.title(r"$C_D$ vs Mach (table + linear interpolation)")
plt.grid(True, linestyle=":"); plt.legend(); plt.tight_layout()
plt.savefig(OUT_PNG, dpi=160)

print("Saved:", OUT_PNG.resolve())
print("Saved:", OUT_CSV.resolve())
