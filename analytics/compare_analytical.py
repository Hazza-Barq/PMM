# analytics/compare_analytical.py
"""
Compare full 6-DoF vs:
- Vacuum
- Drag-only point-mass
- Akçay analytical (either K0 estimated from CD,S,m OR K0(θ) fitted from CSV)

CLI examples:
  # Estimate K0 from CD,S,m and compare:
  python -m analytics.compare_analytical --mode estimate --speed 480 --elev 20 --cd 0.30 --mass 43.7 --diam 0.155

  # Fit K0(θ) polynomial from CSV (columns: elev_deg, range_m, v0_mps) and compare at a target case:
  python -m analytics.compare_analytical --mode fit --csv out/calib_demo.csv --speed 480 --elev 22

CSV schema required for --mode fit:
  elev_deg,range_m,v0_mps
"""
from __future__ import annotations
import math, numpy as np, pathlib, datetime as dt, argparse

OUT = pathlib.Path("out"); OUT.mkdir(exist_ok=True)

# ---------- helpers ----------
def _yaw_pitch_unit(elev_deg: float, az_deg: float) -> np.ndarray:
    e = math.radians(elev_deg); a = math.radians(az_deg)
    return np.array([math.cos(e)*math.cos(a), math.cos(e)*math.sin(a), math.sin(e)], dtype=float)

def vacuum_solution(speed: float, elev_deg: float, az_deg: float, g: float=9.80665, dt: float=0.01, tfinal: float=200.0):
    u0 = _yaw_pitch_unit(elev_deg, az_deg) * speed
    p  = np.zeros(3, dtype=float)
    t  = 0.0
    ts=[t]; Ps=[p.copy()]
    while t < tfinal and p[2] >= 0.0:
        t += dt
        v  = u0 + np.array([0.0,0.0,-g])*t
        p  = u0*t + 0.5*np.array([0,0,-g])*t*t
        ts.append(t); Ps.append(p.copy())
    Y = np.zeros((len(ts), 13)); Y[:, 3:6] = np.array(Ps)
    return np.array(ts), Y

def drag_only_pointmass(speed: float, elev_deg: float, az_deg: float,
                        mass: float, area: float, cd0: float,
                        rho: float=1.225, g: float=9.80665,
                        dt: float=0.002, tfinal: float=200.0):
    v = _yaw_pitch_unit(elev_deg, az_deg) * speed
    p = np.zeros(3, dtype=float)
    t = 0.0
    ts=[t]; Ps=[p.copy()]
    qdynK = 0.5*rho*area*cd0/mass
    def acc(v):
        V = np.linalg.norm(v); drag = -qdynK*V*v
        return drag + np.array([0,0,-g], dtype=float)
    while t < tfinal and p[2] >= 0.0:
        k1v = acc(v);        k1p = v
        k2v = acc(v+0.5*dt*k1v); k2p = v+0.5*dt*k1v
        k3v = acc(v+0.5*dt*k2v); k3p = v+0.5*dt*k2v
        k4v = acc(v+dt*k3v);    k4p = v+dt*k3v
        v += (dt/6.0)*(k1v+2*k2v+2*k3v+k4v)
        p += (dt/6.0)*(k1p+2*k2p+2*k3p+k4p)
        t += dt
        ts.append(t); Ps.append(p.copy())
    Y = np.zeros((len(ts), 13)); Y[:,3:6] = np.array(Ps)
    return np.array(ts), Y

def cut_at_impact(ts, ys, ground=0.0):
    z = ys[:,5]; dz = z - ground
    idx = np.where((dz[:-1] > 0.0) & (dz[1:] <= 0.0))[0]
    if idx.size == 0: return ts, ys, ts[-1], ys[-1,3:6]
    i = int(idx[0]); denom = (dz[i]-dz[i+1]); w = (dz[i]/denom) if abs(denom)>1e-15 else 0.0
    t_imp = ts[i] + w*(ts[i+1]-ts[i]); y_imp = ys[i] + w*(ys[i+1]-ys[i])
    return np.concatenate([ts[:i+1],[t_imp]]), np.vstack([ys[:i+1], y_imp[None,:]]), t_imp, y_imp[3:6]

def _poly_design(theta_rad, deg):
    # columns: [1, θ, θ^2, ...]
    return np.array([theta_rad**i for i in range(deg+1)], dtype=float)

def _fit_k0_poly_from_rows(thetas_deg, ranges_m, v0_mps, rho0, deg):
    from analytics.akcay import k0_from_measured_range
    th_rad = np.radians(thetas_deg)
    K = np.array([k0_from_measured_range(X, v0, th, rho0) for X, v0, th in zip(ranges_m, v0_mps, th_rad)], dtype=float)
    V = np.vander(th_rad, deg+1, increasing=True)
    coeffs, *_ = np.linalg.lstsq(V, K, rcond=None)
    return coeffs

def _eval_poly(coeffs, theta_deg):
    th = math.radians(theta_deg)
    V = _poly_design(th, len(coeffs)-1)
    return float(np.dot(coeffs, V))


# ---------- Akçay integration ----------
def akcay_estimate_trajectory(speed, elev_deg, rho0, cd, mass, diam, dx=5.0):
    """
    Build Akçay trajectory using K0 estimated from (cd, S, m, theta).
    """
    from analytics.akcay import k0_from_cd, trajectory_samples
    theta = math.radians(elev_deg)
    S = math.pi*(diam**2)/4.0
    K0 = k0_from_cd(cd, S, mass, theta)
    xs, ys, ts, vs = trajectory_samples(K0, speed, theta, rho0, dx=dx)
    Y = np.zeros((len(xs), 13)); Y[:,3] = xs; Y[:,5] = ys
    return ts, Y, K0

def akcay_fit_from_csv_and_predict(csv_path, speed, elev_deg, rho0, dx=5.0, poly_deg=4,
                                   mode="fit-all",  # "fit-all" | "loo" | "holdout"
                                   holdout_frac=0.3, random_seed=42):
    """
    Fit K0(θ) polynomial from calibration CSV, then predict Akçay trajectory at (speed, elev).
    CSV schema: elev_deg,range_m,v0_mps
    mode:
      - "fit-all": uses all rows (will overfit on synthetic data; good for smoke test)
      - "loo": leave-one-out at target elevation (exclude nearest row by elevation)
      - "holdout": random split; fit on (1-holdout_frac), predict on target using the fit
    """
    import pandas as pd
    from analytics.akcay import trajectory_samples

    df = pd.read_csv(csv_path)
    if not {"elev_deg","range_m","v0_mps"}.issubset(df.columns):
        raise RuntimeError("CSV must have columns: elev_deg, range_m, v0_mps")

    # pick training rows according to mode
    if mode == "fit-all":
        train = df
    elif mode == "loo":
        # exclude the row whose elevation is closest to the target
        idx = int(np.abs(df["elev_deg"].to_numpy(float) - float(elev_deg)).argmin())
        train = df.drop(df.index[idx])
    elif mode == "holdout":
        rng = np.random.default_rng(random_seed)
        mask = rng.random(len(df)) > holdout_frac
        train = df[mask]
        if len(train) < max(4, poly_deg+1):
            raise RuntimeError("Holdout left too few rows to fit. Reduce holdout_frac or degree.")
    else:
        raise RuntimeError("mode must be one of: fit-all, loo, holdout")

    coeffs = _fit_k0_poly_from_rows(
        thetas_deg=train["elev_deg"].to_numpy(float),
        ranges_m=train["range_m"].to_numpy(float),
        v0_mps=train["v0_mps"].to_numpy(float),
        rho0=rho0,
        deg=poly_deg
    )

    K0 = _eval_poly(coeffs, elev_deg)

    # build trajectory
    theta = math.radians(elev_deg)
    xs, ys, ts, vs = trajectory_samples(K0, speed, theta, rho0, dx=dx)
    Y = np.zeros((len(xs), 13)); Y[:,3] = xs; Y[:,5] = ys
    return ts, Y, K0, coeffs, mode
    """
    Fit K0(θ) polynomial from calibration CSV, then predict Akçay trajectory at (speed, elev).
    CSV schema: elev_deg,range_m,v0_mps
    """
    import pandas as pd
    from analytics.akcay import fit_k0_poly, trajectory_samples
    df = pd.read_csv(csv_path)
    if not {"elev_deg","range_m","v0_mps"}.issubset(df.columns):
        raise RuntimeError("CSV must have columns: elev_deg, range_m, v0_mps")
    thetas = df["elev_deg"].to_numpy(float)
    ranges = df["range_m"].to_numpy(float)
    v0s    = df["v0_mps"].to_numpy(float)

    coeffs = fit_k0_poly(thetas, ranges, rho0, v0_list=v0s, deg=poly_deg)
    # evaluate polynomial at target θ (in radians):
    th = math.radians(elev_deg)
    V = np.array([th**i for i in range(poly_deg+1)], dtype=float)
    K0 = float(np.dot(coeffs, V))

    xs, ys, ts, vs = trajectory_samples(K0, speed, th, rho0, dx=dx)
    Y = np.zeros((len(xs), 13)); Y[:,3] = xs; Y[:,5] = ys
    return ts, Y, K0, coeffs

# ---------- top-level compare ----------
def run_compare(mode: str,
                speed=480.0, elev=20.0, az=0.0, spin=3000.0, tfinal=200.0,
                # env/projectile defaults
                rho0=1.225, g0=9.80665, cd=0.30, mass=43.7, diam=0.155,
                # CSV for fit-mode
                csv=None, poly_deg=4):
    """
    mode: "estimate" (use CD,S,m) or "fit" (use CSV calibration)
    """
    from physics.projectile6dof import integrate_6dof
    from physics.attitude import q_normalize
    import main as main_mod

    # initial state
    y0 = main_mod.build_initial_state(speed, elev, az, spin, ap=None)

    # vacuum
    tv, yv = vacuum_solution(speed, elev, az, tfinal=tfinal)

    # drag-only
    area = math.pi*(diam**2)/4.0
    td, yd = drag_only_pointmass(speed, elev, az, mass, area, cd, rho=rho0, g=g0, tfinal=tfinal)

    # full 6-DoF
    ts, ys = integrate_6dof(None, y0, 0.0, tfinal)
    for i in range(ys.shape[0]): ys[i,9:13] = q_normalize(ys[i,9:13])

    # Akçay
        
    if mode == "estimate":
        ta, ya, K0 = akcay_estimate_trajectory(speed, elev, rho0, cd, mass, diam)
        coeffs = None; fit_mode = "estimate"
    elif mode in ("fit", "loo", "holdout"):
        if not csv:
            raise RuntimeError("--csv required for mode=fit/loo/holdout")
        ta, ya, K0, coeffs, fit_mode = akcay_fit_from_csv_and_predict(
            csv, speed, elev, rho0, poly_deg=poly_deg,
            mode=("fit-all" if mode=="fit" else mode)
        )
    else:
        raise RuntimeError("mode must be 'estimate', 'fit', 'loo', or 'holdout'")

    # trim all at impact
    tv, yv, tv_imp, pv = cut_at_impact(tv, yv)
    td, yd, td_imp, pd = cut_at_impact(td, yd)
    ts, ys, ts_imp, ps = cut_at_impact(ts, ys)
    ta, ya, ta_imp, pa = cut_at_impact(ta, ya)

    Rv = float(np.hypot(pv[0], pv[1]))
    Rd = float(np.hypot(pd[0], pd[1]))
    Rs = float(np.hypot(ps[0], ps[1]))
    Ra = float(np.hypot(pa[0], pa[1]))

    def pct(a,b): return 100.0*(a-b)/max(1e-9,b)

    print(f"[compare] Vacuum:   range={Rv:.2f} m, TOF={tv_imp:.2f} s")
    print(f"[compare] DragOnly: range={Rd:.2f} m, TOF={td_imp:.2f} s")
    print(f"[compare] 6-DoF:    range={Rs:.2f} m, TOF={ts_imp:.2f} s")
    print(f"[compare] Akçay({mode}): range={Ra:.2f} m, TOF~{ta_imp:.2f} s, K0={K0:.4e}")

    print(f"[err vs 6-DoF] Vacuum:  ΔR={pct(Rv,Rs):+.1f}%")
    print(f"[err vs 6-DoF] DragOnly:ΔR={pct(Rd,Rs):+.1f}%")
    print(f"[err vs 6-DoF] Akçay:   ΔR={pct(Ra,Rs):+.1f}%")

    # plot overlay
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tsmp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fig = plt.figure()
    plt.plot(yv[:,3]/1000.0, yv[:,5], label="Vacuum")
    plt.plot(yd[:,3]/1000.0, yd[:,5], label="Drag-only")
    plt.plot(ys[:,3]/1000.0, ys[:,5], label="Full 6-DoF")
    plt.plot(ya[:,3]/1000.0, ya[:,5], label=f"Akçay({mode})")
    plt.xlabel("X Range [km]"); plt.ylabel("Altitude Z [m]"); plt.grid(True, ls=":")
    plt.title(f"Overlay @ {speed} m/s, {elev}°  (K0={K0:.2e})")
    plt.legend()
    out = OUT / f"compare_overlay_{mode}_{int(speed)}ms_{int(elev)}deg_{tsmp}.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("Saved:", out.resolve())

    return dict(vacuum=(tv_imp,Rv), drag=(td_imp,Rd), full6dof=(ts_imp,Rs), akcay=(ta_imp,Ra), K0=K0)

# ---------- CLI ----------
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="calibration CSV for mode=fit (elev_deg,range_m,v0_mps)")
    ap.add_argument("--speed", type=float, default=480.0)
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--az", type=float, default=0.0)
    ap.add_argument("--spin", type=float, default=3000.0)
    ap.add_argument("--tfinal", type=float, default=200.0)
    ap.add_argument("--rho0", type=float, default=1.225)
    ap.add_argument("--g0", type=float, default=9.80665)
    ap.add_argument("--cd", type=float, default=0.30)
    ap.add_argument("--mass", type=float, default=43.7)
    ap.add_argument("--diam", type=float, default=0.155)
    ap.add_argument("--poly-deg", type=int, default=4)
    ap.add_argument("--mode", choices=["estimate","fit","loo","holdout"], required=True)

    args = ap.parse_args()
    run_compare(args.mode, args.speed, args.elev, args.az, args.spin, args.tfinal,
                rho0=args.rho0, g0=args.g0, cd=args.cd, mass=args.mass, diam=args.diam,
                csv=args.csv, poly_deg=args.poly_deg)

if __name__ == "__main__":
    _cli()
