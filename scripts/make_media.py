# scripts/make_media.py
# Generate missing portfolio media quickly.
# Requires your existing package layout to be importable from project root.

import math, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import datetime as dt
import sys

# --- import your project modules ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg

# Core simulation
from main import run_single, build_initial_state   # uses cut_at_impact inside
from physics.projectile6dof import integrate_6dof
from physics.attitude import q_normalize

# Telemetry
from diagnostics.telemetry import make_post_step_recorder
import physics.atmosphere as atmosphere
from utils.cd_lookup import get_coeffs as cd_get_coeffs

# Optional Akçay compare helpers (TOF vs elev overlays)
try:
    from analytics.compare_analytical import (
        akcay_fit_from_csv_and_predict,
        akcay_estimate_trajectory,
    )
    from analytics.compare_analytical import cut_at_impact as cut_at_impact_ak
    HAS_AK = True
except Exception:
    HAS_AK = False

OUT = (ROOT / "out"); OUT.mkdir(exist_ok=True)
STAMP = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------- Utilities ----------------

def cut_at_impact_local(ts: np.ndarray, ys: np.ndarray, ground: float = 0.0):
    """Local copy (safer than importing from main to avoid circulars)."""
    z = ys[:, 5]
    dz = z - ground
    idx = np.where((dz[:-1] > 0.0) & (dz[1:] <= 0.0))[0]
    if idx.size == 0:
        return ts, ys, ts[-1], ys[-1, 3:6], False
    i = int(idx[0])
    denom = (dz[i] - dz[i+1])
    w = (dz[i] / denom) if abs(denom) > 1e-15 else 0.0
    t_imp = ts[i] + w * (ts[i+1] - ts[i])
    y_imp = ys[i] + w * (ys[i+1] - ys[i])
    ts_trim = np.concatenate([ts[:i+1], np.array([t_imp])])
    ys_trim = np.vstack([ys[:i+1], y_imp[None, :]])
    return ts_trim, ys_trim, t_imp, y_imp[3:6], True

def _wind_callable():
    wv = getattr(cfg, "WIND_VECTOR", None)
    if callable(wv):
        return wv
    if hasattr(wv, "to_numpy"):
        wv = wv.to_numpy()
    w_const = np.asarray(wv if wv is not None else [0.0,0.0,0.0], dtype=float)
    return (lambda t, w=w_const: w)

def default_ap_like_config():
    class AP: pass
    ap = AP()
    ap.m  = float(getattr(cfg, "PROJECTILE_MASS", 43.7))
    ap.d  = float(getattr(cfg, "PROJECTILE_DIAMETER", 0.155))
    ap.S  = float(getattr(cfg, "REFERENCE_AREA", math.pi*(ap.d**2)/4.0))
    ap.Ix = float(getattr(cfg, "PROJECTILE_IX", 0.1444))
    ap.Iy = float(getattr(cfg, "PROJECTILE_IY", 1.7323))
    return ap

FAST_OPTS = dict(h0=1e-3, rtol=2e-6, atol=1e-8, h_max=0.05, h_min=1e-9)

# ---------------- Artifact 1: TOF vs Elevation ----------------

def plot_tof_vs_elev(speed=480.0, az=0.0, spin=3000.0, tfinal=120.0,
                     emin=5.0, emax=45.0, estep=2.0,
                     with_akcay=True, fit_mode="holdout",
                     rho0=1.225, calib_csv=None, poly_deg=4, fast=True):
    elevs = np.arange(emin, emax + 1e-9, estep)
    tof6 = []; tofA = []
    for e in elevs:
        # 6-DoF TOF via run_single (fast integrator opts)
        opts = FAST_OPTS if fast else None
        t_imp, R, brg = run_single(speed, e, az, spin, tfinal, integrator_opts=opts, save_track=False, record=False)
        tof6.append(t_imp)

        if HAS_AK and with_akcay:
            if calib_csv:
                mode_key = {"holdout":"holdout", "loo":"loo", "fit-all":"fit-all"}[fit_mode]
                ta, Ya, K0_fit, coeffs, _ = akcay_fit_from_csv_and_predict(
                    calib_csv, speed, e, rho0, poly_deg=poly_deg, mode=mode_key
                )
            else:
                # quick estimate path requires Cd, mass, diam; use a benign Cd=0.30 fallback
                mass = getattr(cfg, "PROJECTILE_MASS", 43.7)
                diam = getattr(cfg, "PROJECTILE_DIAMETER", 0.155)
                ta, Ya, _ = akcay_estimate_trajectory(speed, e, rho0, 0.30, mass, diam, dx=5.0)
            ta, Ya, tA, pA = cut_at_impact_ak(ta, Ya)
            tofA.append(tA)

    # Plot
    fig = plt.figure(figsize=(7.5,4.8))
    plt.plot(elevs, tof6, label="6-DoF", linewidth=2)
    if HAS_AK and with_akcay:
        plt.plot(elevs, tofA, label=f"Akçay ({fit_mode if calib_csv else 'est.'})", linewidth=2, linestyle="--")
    plt.xlabel("Elevation (deg)")
    plt.ylabel("Time of Flight (s)")
    plt.grid(True, ls=":")
    ttl = f"TOF vs Elevation @ {int(speed)} m/s"
    if calib_csv and with_akcay:
        ttl += f" — {fit_mode}"
    plt.title(ttl)
    plt.legend()
    fname = OUT / f"tof_vs_elev_{int(speed)}ms_{STAMP}.png"
    plt.tight_layout(); plt.savefig(fname, dpi=160)
    print("Saved:", fname.resolve())

# ---------------- Artifact 2: Quaternion norm vs time ----------------

def plot_qnorm(speed=480.0, elev=22.0, az=0.0, spin=3000.0, tfinal=120.0, fast=True):
    ap = default_ap_like_config()
    wind_fn = _wind_callable()
    y0 = build_initial_state(speed, elev, az, spin, ap=ap)

    # recorder
    rec = make_post_step_recorder(ap, atmosphere, wind_fn, cd_lookup=cd_get_coeffs)
    opts = FAST_OPTS if fast else None
    ts, ys = integrate_6dof(ap, y0, 0.0, tfinal, integrator_opts=opts, post_step_user=rec)
    for i in range(ys.shape[0]):
        ys[i, 9:13] = q_normalize(ys[i, 9:13])
    # Trim
    ts, ys, t_imp, p_imp, hit = cut_at_impact_local(ts, ys)

    qn = np.array(rec.telemetry.qnorm)
    fig = plt.figure(figsize=(7.0,4.2))
    plt.plot(rec.telemetry.t, qn, linewidth=2)
    plt.axhline(1.0, color="C2", ls="--", lw=1.0)
    plt.xlabel("Time (s)"); plt.ylabel(r"$\|\mathbf{q}\|$")
    plt.title(f"Quaternion Norm vs Time @ {int(speed)} m/s, {elev:.1f}°")
    plt.grid(True, ls=":")
    fname = OUT / f"qnorm_{int(speed)}ms_{int(round(elev))}deg_{STAMP}.png"
    plt.tight_layout(); plt.savefig(fname, dpi=160)
    print("Saved:", fname.resolve())

# ---------------- Artifact 3: Moments magnitude vs time ----------------

def plot_moments_mag(speed=480.0, elev=22.0, az=0.0, spin=3000.0, tfinal=120.0, fast=True):
    ap = default_ap_like_config()
    wind_fn = _wind_callable()
    y0 = build_initial_state(speed, elev, az, spin, ap=ap)

    rec = make_post_step_recorder(ap, atmosphere, wind_fn, cd_lookup=cd_get_coeffs)
    opts = FAST_OPTS if fast else None
    ts, ys = integrate_6dof(ap, y0, 0.0, tfinal, integrator_opts=opts, post_step_user=rec)
    for i in range(ys.shape[0]):
        ys[i, 9:13] = q_normalize(ys[i, 9:13])
    ts, ys, t_imp, p_imp, hit = cut_at_impact_local(ts, ys)

    # Use stored |dH/dt| (M_mag)
    Mmag = np.array(rec.telemetry.M_mag)
    fig = plt.figure(figsize=(7.0,4.2))
    plt.plot(rec.telemetry.t, Mmag, linewidth=2)
    plt.xlabel("Time (s)"); plt.ylabel(r"$\|\dot{\mathbf{H}}\|$ (N·m)")
    plt.title(f"Moments Magnitude vs Time @ {int(speed)} m/s, {elev:.1f}°")
    plt.grid(True, ls=":")
    fname = OUT / f"moments_mag_{int(speed)}ms_{int(round(elev))}deg_{STAMP}.png"
    plt.tight_layout(); plt.savefig(fname, dpi=160)
    print("Saved:", fname.resolve())

# ---------------- Optional: Hemisphere surface (re-make) ----------------

def hemi_surface(speed=480.0, spin=3000.0, tfinal=200.0, emin=5.0, emax=45.0, estep=2.5, azstep=5.0, fast=True):
    elevs = np.arange(emin, emax + 1e-9, estep)
    azes  = np.arange(0.0, 360.0 - 1e-9, azstep)
    R = np.zeros((len(elevs), len(azes)))
    for i, el in enumerate(elevs):
        for j, az in enumerate(azes):
            opts = FAST_OPTS if fast else None
            t_imp, rng, brg = run_single(speed, el, az, spin, tfinal, integrator_opts=opts, save_track=False, record=False)
            R[i, j] = rng

    # 3D surface
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    E, A = np.meshgrid(elevs, azes, indexing="ij")
    fig = plt.figure(figsize=(8.4,5.6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(A, E, R/1000.0, linewidth=0, antialiased=True)
    ax.set_xlabel("Azimuth (deg)"); ax.set_ylabel("Elevation (deg)"); ax.set_zlabel("Range (km)")
    ax.set_title(f"Range surface @ {int(speed)} m/s")
    fname = OUT / f"hemi_surface_{int(speed)}ms_{STAMP}.png"
    plt.tight_layout(); plt.savefig(fname, dpi=160)
    print("Saved:", fname.resolve())

# ---------------- CLI ----------------

def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Generate portfolio media quickly.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("tof", help="Time-of-flight vs elevation")
    s1.add_argument("--speed", type=float, default=480.0)
    s1.add_argument("--az", type=float, default=0.0)
    s1.add_argument("--spin", type=float, default=3000.0)
    s1.add_argument("--tfinal", type=float, default=120.0)
    s1.add_argument("--emin", type=float, default=5.0)
    s1.add_argument("--emax", type=float, default=45.0)
    s1.add_argument("--estep", type=float, default=2.0)
    s1.add_argument("--no-akcay", action="store_true")
    s1.add_argument("--fit-mode", choices=["holdout","loo","fit-all"], default="holdout")
    s1.add_argument("--rho0", type=float, default=1.225)
    s1.add_argument("--calib-csv", type=str, default="")
    s1.add_argument("--nofast", action="store_true")

    s2 = sub.add_parser("qnorm", help="Quaternion norm vs time")
    s2.add_argument("--speed", type=float, default=480.0)
    s2.add_argument("--elev", type=float, default=22.0)
    s2.add_argument("--az", type=float, default=0.0)
    s2.add_argument("--spin", type=float, default=3000.0)
    s2.add_argument("--tfinal", type=float, default=120.0)
    s2.add_argument("--nofast", action="store_true")

    s3 = sub.add_parser("moments", help="Moments magnitude vs time")
    s3.add_argument("--speed", type=float, default=480.0)
    s3.add_argument("--elev", type=float, default=22.0)
    s3.add_argument("--az", type=float, default=0.0)
    s3.add_argument("--spin", type=float, default=3000.0)
    s3.add_argument("--tfinal", type=float, default=120.0)
    s3.add_argument("--nofast", action="store_true")

    s4 = sub.add_parser("hemi", help="3D hemisphere surface (range)")
    s4.add_argument("--speed", type=float, default=480.0)
    s4.add_argument("--spin", type=float, default=3000.0)
    s4.add_argument("--tfinal", type=float, default=200.0)
    s4.add_argument("--emin", type=float, default=5.0)
    s4.add_argument("--emax", type=float, default=45.0)
    s4.add_argument("--estep", type=float, default=2.5)
    s4.add_argument("--azstep", type=float, default=5.0)
    s4.add_argument("--nofast", action="store_true")

    args = ap.parse_args()

    if args.cmd == "tof":
        calib_csv = args.calib_csv if args.calib_csv else None
        plot_tof_vs_elev(
            speed=args.speed, az=args.az, spin=args.spin, tfinal=args.tfinal,
            emin=args.emin, emax=args.emax, estep=args.estep,
            with_akcay=(not args.no_akcay),
            fit_mode=args.fit_mode, rho0=args.rho0,
            calib_csv=calib_csv, poly_deg=4,
            fast=(not args.nofast)
        )
    elif args.cmd == "qnorm":
        plot_qnorm(args.speed, args.elev, args.az, args.spin, args.tfinal, fast=(not args.nofast))
    elif args.cmd == "moments":
        plot_moments_mag(args.speed, args.elev, args.az, args.spin, args.tfinal, fast=(not args.nofast))
    elif args.cmd == "hemi":
        hemi_surface(args.speed, args.spin, args.tfinal, args.emin, args.emax, args.estep, args.azstep, fast=(not args.nofast))

if __name__ == "__main__":
    _cli()
