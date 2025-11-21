# --- imports & path shim ---
import math, numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# config fallback
try:
    import config as cfg
except Exception:
    class cfg:
        PROJECTILE_MASS = 43.7
        PROJECTILE_DIAMETER = 0.155
        SEA_LEVEL_GRAVITY = 9.80665
        WIND_VECTOR = None

# project imports
from main import run_single, build_initial_state
from physics.projectile6dof import integrate_6dof
from physics.attitude import q_normalize
from analytics.compare_analytical import akcay_estimate_trajectory, cut_at_impact
from analytics.akcay import k0_from_cd, trajectory_samples

st.set_page_config(page_title="6-DoF vs Akçay", layout="wide")
st.title("🚀 6-DoF vs Akçay (Analytical) — Live Demo")

# --- sidebar inputs ---
with st.sidebar:
    st.header("Shot Inputs")
    speed = st.number_input("Muzzle speed (m/s)", 100.0, 2000.0, 480.0, 10.0)
    elev  = st.slider("Elevation (deg)", 0.0, 89.0, 22.0, 0.5)
    az    = st.slider("Azimuth (deg)", -180.0, 180.0, 0.0, 1.0)
    spin  = st.number_input("Spin (rpm)", 0.0, 30000.0, 3000.0, 100.0)
    tfinal= st.number_input("Max sim time (s)", 1.0, 300.0, 120.0, 1.0)
    fast  = st.checkbox("Fast integrator", True)
    st.markdown("---")
    st.subheader("Akçay Mode")
    mode = st.radio("Choose Akçay mode", ["Estimate (Cd)", "Fit from CSV"])
    rho0 = st.number_input("ρ₀ for Akçay (kg/m³)", 0.5, 2.0, 1.225, 0.005)
    if mode == "Estimate (Cd)":
        cd    = st.number_input("Cd (estimate)", 0.05, 1.50, 0.30, 0.01)
        mass  = st.number_input("Mass m (kg)", 0.1, 200.0, float(getattr(cfg,"PROJECTILE_MASS",43.7)), 0.1)
        diam  = st.number_input("Diameter d (m)", 0.01, 1.00, float(getattr(cfg,"PROJECTILE_DIAMETER",0.155)), 0.001)
    else:
        csv_file = st.file_uploader("Calibration CSV (elev_deg,range_m,v0_mps)", type=["csv"])
        poly_deg = st.slider("Polynomial degree for K0(θ)", 1, 6, 4, 1)
        fit_strategy = st.selectbox("Fit strategy", ["fit-all (smoke test)", "leave-one-out", "holdout 30%"])

FAST_OPTS = dict(h0=1e-3, rtol=2e-6, atol=1e-8, h_max=0.05, h_min=1e-9) if fast else None

# --- run full 6-DoF and capture path ---
with st.spinner("Running 6-DoF…"):
    y0 = build_initial_state(speed, elev, az, spin, ap=None)
    ts6, ys6 = integrate_6dof(None, y0, 0.0, tfinal, integrator_opts=FAST_OPTS)
    for i in range(ys6.shape[0]):
        ys6[i, 9:13] = q_normalize(ys6[i, 9:13])
    ts6, ys6, t_imp6, p_imp6 = cut_at_impact(ts6, ys6)

rng6 = float(np.hypot(p_imp6[0], p_imp6[1]))
bearing6 = float(math.degrees(math.atan2(p_imp6[1], p_imp6[0])))
st.success(f"6-DoF impact → **range = {rng6:,.1f} m**, TOF = **{t_imp6:.2f} s**, bearing = {bearing6:.2f}°")

# --- build Akçay trajectory (estimate or fit) ---
Ya = None
if mode == "Estimate (Cd)":
    S = math.pi*(diam**2)/4.0
    K0_est = k0_from_cd(cd, S, mass, math.radians(elev))
    ta, Ya, _ = akcay_estimate_trajectory(speed, elev, rho0, cd, mass, diam, dx=5.0)
    ta, Ya, ta_imp, p_impA = cut_at_impact(ta, Ya)
    R_ak = float(np.hypot(p_impA[0], p_impA[1]))
    st.info(f"Akçay (estimate): K0={K0_est:.3e}, range={R_ak:,.1f} m (Δ={R_ak-rng6:+.1f} m)")
else:
    from analytics.compare_analytical import akcay_fit_from_csv_and_predict
    if csv_file is None:
        st.warning("Upload a calibration CSV to use fit mode.")
    else:
        # 1) Read the uploaded CSV
        try:
            df_upload = pd.read_csv(csv_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()
        # 2) Validate required columns
        required = {"elev_deg","range_m","v0_mps"}
        if not required.issubset(df_upload.columns):
            st.error(f"CSV must have columns: {sorted(required)} (found: {list(df_upload.columns)})")
            st.stop()
        # 3) Save to a real temp file
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
            df_upload.to_csv(tf.name, index=False)
            tmp_path = tf.name
        # 4) Fit/predict
        mode_key = {"fit-all (smoke test)": "fit-all",
                    "leave-one-out": "loo",
                    "holdout 30%": "holdout"}[fit_strategy]
        ta, Ya, K0_fit, coeffs, _ = akcay_fit_from_csv_and_predict(
            tmp_path, speed, elev, rho0, poly_deg=poly_deg, mode=mode_key
        )
        ta, Ya, ta_imp, p_impA = cut_at_impact(ta, Ya)
        R_ak = float(np.hypot(p_impA[0], p_impA[1]))
        st.info(f"Akçay (fit {fit_strategy}): K0={K0_fit:.3e}, range={R_ak:,.1f} m (Δ={R_ak-rng6:+.1f} m)")

# --- plotting ---
fig = plt.figure(figsize=(8,5))
plt.plot(ys6[:,3]/1000.0, ys6[:,5], label="6-DoF", linewidth=2)
if Ya is not None:
    plt.plot(Ya[:,3]/1000.0, Ya[:,5], label=("Akçay (est.)" if mode=="Estimate (Cd)" else "Akçay (fit)"))
# mark impact lines
plt.axvline(rng6/1000.0, color="C3", ls="--", lw=1.0, label="6-DoF range")
if 'R_ak' in locals():
    plt.axvline(R_ak/1000.0, color="C2", ls="--", lw=1.0, label="Akçay range")
plt.xlabel("Downrange X (km)"); plt.ylabel("Altitude Z (m)")
plt.grid(True, ls=":"); plt.legend()
st.pyplot(fig)

st.caption("Tip: For rigorous metrics and cross-validation, keep using the analytics CLI (holdout/LOO).")
