# scripts/showcase_plots.py
"""
Generate a clean set of 'hero' plots without touching the core code:
- Single-shot diagnostic bundle (trajectory, |F|, |M|, Mach, Cd, Cl, alpha, energy)
- Vacuum vs Drag-only vs Full 6-DoF comparison overlay
- Elevation sweep plot (fast, no per-angle PNGs)
- Optional hemisphere 3D surface (fast)
"""

import pathlib, math, numpy as np, datetime as dt
from importlib import import_module

OUT = pathlib.Path("out"); OUT.mkdir(exist_ok=True)

# ---- helpers: import from your project without modifying it ----
main = import_module("main")
compare_mod = None
try:
    compare_mod = import_module("compare")   # if you added compare.py earlier
except Exception:
    pass

def timestamp():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def one_diagnostic_shot(speed=480.0, elev=20.0, az=0.0, spin=3000.0, tfinal=200.0):
    """
    Run exactly one trajectory with telemetry ON to save the diagnostics bundle.
    """
    print("\n[showcase] single-shot diagnostics…")
    # Use your accurate integrator profile
    opts = dict(h0=1e-4, rtol=1e-7, atol=1e-8, h_min=1e-9, h_max=0.02)
    # record=True makes telemetry + plots; save_track=False to avoid CSV if you want
    tof, rng, bearing = main.run_single(speed, elev, az, spin, tfinal,
                                        integrator_opts=opts,
                                        record=True, save_track=False,
                                        stem=f"track_{int(speed)}ms_{int(elev)}deg")
    print(f"[showcase] done. TOF={tof:.2f}s, range={rng:.1f} m, bearing={bearing:.2f}°")
    return tof, rng, bearing

def comparison_overlay(speed=480.0, elev=20.0, az=0.0, spin=3000.0, tfinal=200.0):
    """
    Make a single comparison plot: Vacuum vs Drag-only vs Full 6-DoF.
    Requires compare.py (optional). If not present, we’ll do a minimal inline version.
    """
    print("\n[showcase] comparison overlay (vacuum/drag/full)…")
    if compare_mod is not None and hasattr(compare_mod, "run_compare"):
        compare_mod.run_compare(speed, elev, az, spin, tfinal)
        return

    # Fallback minimal version if compare.py absent:
    from physics.projectile6dof import integrate_6dof, make_state_derivative
    from physics.integrator_adaptive import rkf45_adaptive
    from physics.attitude import q_normalize
    y0 = main.build_initial_state(speed, elev, az, spin, ap=None)

    # 1) Vacuum (simple Euler for uniform sampling)
    def vacuum_trajectory(speed, elev_deg, az_deg, tfinal, g=9.80665, dt_s=0.02):
        elev = math.radians(elev_deg); az = math.radians(az_deg)
        v0 = np.array([math.cos(elev)*math.cos(az), math.cos(elev)*math.sin(az), math.sin(elev)]) * speed
        p = np.zeros(3); v = v0.copy()
        ts=[0.0]; pts=[p.copy()]
        t=0.0
        while t < tfinal and p[2] >= 0.0:
            t += dt_s
            v = v + np.array([0.0,0.0,-g])*dt_s
            p = p + v*dt_s
            ts.append(t); pts.append(p.copy())
        Y = np.zeros((len(ts),13)); Y[:,3:6] = np.array(pts)
        return np.array(ts), Y

    tv, yv = vacuum_trajectory(speed, elev, az, tfinal)

    # 2) Drag-only 3-DoF via temporary coeff patch (zero lift & all moments)
    from utils.cd_lookup import get_coeffs as real_get
    import physics.projectile6dof as p6
    def drag_only_get(mach):
        c = real_get(mach)
        c["cl_alpha"] = 0.0; c["cl_alpha3"] = 0.0
        for k in ("cm_alpha","cm_alpha3","cm_q","cm_alphadot","cspin","cmag_m"):
            c[k] = 0.0
        return c
    orig_get = p6.cd_get_coeffs
    p6.cd_get_coeffs = drag_only_get
    f_drag = p6.make_state_derivative(aero_params=None, wind_inertial=None)
    ts_d, ys_d = rkf45_adaptive(f_drag, 0.0, y0, tfinal, h0=1e-3, rtol=2e-6, atol=1e-8, h_max=0.05)
    for i in range(ys_d.shape[0]): ys_d[i,9:13] = q_normalize(ys_d[i,9:13])
    p6.cd_get_coeffs = orig_get

    # 3) Full 6-DoF
    ts_f, ys_f = integrate_6dof(None, y0, 0.0, tfinal)

    # Trim all three to impact using your helper
    tv_t, yv_t, *_ = main.cut_at_impact(tv, yv)
    ts_d_t, ys_d_t, *_ = main.cut_at_impact(ts_d, ys_d)
    ts_f_t, ys_f_t, *_ = main.cut_at_impact(ts_f, ys_f)

    # Plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(yv_t[:,3]/1000, yv_t[:,5], label="Vacuum")
    plt.plot(ys_d_t[:,3]/1000, ys_d_t[:,5], label="Drag-only")
    plt.plot(ys_f_t[:,3]/1000, ys_f_t[:,5], label="Full 6-DoF")
    plt.xlabel("Range X [km]"); plt.ylabel("Altitude Z [m]")
    plt.title(f"Trajectory comparison @ {speed} m/s, {elev}°")
    plt.grid(True, ls=":"); plt.legend()
    out = OUT / f"compare_{int(speed)}ms_{int(elev)}deg_{timestamp()}.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("Saved:", out.resolve())

def elevation_sweep_plot(speed=480.0, az=0.0, spin=3000.0, tfinal=200.0,
                         emin=5.0, emax=45.0, estep=1.0):
    """
    Run a fast elevation sweep WITHOUT generating per-shot PNGs/CSVs.
    Saves one CSV + one PNG.
    """
    print("\n[showcase] elevation sweep (fast, no per-angle plots)…")
    import pandas as pd, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt

    results=[]
    for elev in np.arange(emin, emax+1e-9, estep):
        tof, rng, bearing = main.run_single(speed, elev, az, spin, tfinal,
                                            integrator_opts=dict(h0=1e-3, rtol=2e-6, atol=1e-8, h_max=0.05),
                                            record=False, save_track=False)
        results.append((elev, tof, rng, bearing))
    df = pd.DataFrame(results, columns=["elev_deg","tof_s","range_m","bearing_deg"])
    csv = OUT / f"sweep_elev_{int(speed)}ms_{timestamp()}.csv"
    df.to_csv(csv, index=False); print("Saved:", csv.resolve())

    plt.figure()
    plt.plot(df["elev_deg"], df["range_m"])
    plt.xlabel("Elevation (deg)"); plt.ylabel("Range (m)"); plt.grid(True, ls=":")
    plt.title(f"Range vs Elevation @ {speed} m/s")
    png = OUT / f"sweep_elev_{int(speed)}ms_{timestamp()}.png"
    plt.tight_layout(); plt.savefig(png, dpi=150); plt.close()
    print("Saved:", png.resolve())

def hemisphere_surface(speed=480.0, spin=3000.0, tfinal=200.0,
                       emin=5.0, emax=45.0, estep=2.0, azstep=5.0, workers=0):
    """
    Optional 3D surface using your hemisphere sweep (fast mode, no per-angle plots).
    """
    print("\n[showcase] hemisphere sweep (fast)…")
    hemi = import_module("sweeps.hemisphere")
    # Build arrays and run
    elevs = np.arange(emin, emax + 1e-9, estep)
    azes  = np.arange(0.0, 360.0 - 1e-9, azstep)
    hemi.run(speed=speed, spin=spin, tfinal=tfinal,
             elevs=elevs, azes=azes, workers=(None if workers == 0 else workers),
             fast=True)

def main_showcase():
    # Pick your showcase defaults here
    speed = 480.0; elev = 20.0; az = 0.0; spin = 3000.0; tfinal = 200.0
    one_diagnostic_shot(speed, elev, az, spin, tfinal)
    comparison_overlay(speed, elev, az, spin, tfinal)
    elevation_sweep_plot(speed, az, spin, tfinal, emin=5.0, emax=50.0, estep=1.0)
    # Optional (comment out if you don’t want it during dev):
    # hemisphere_surface(speed, spin, tfinal, emin=5.0, emax=45.0, estep=2.5, azstep=5.0, workers=0)

if __name__ == "__main__":
    main_showcase()
