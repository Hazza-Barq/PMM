# main.py
import argparse, math, datetime as dt, pathlib
import numpy as np
import config as cfg
from utils.vector import Vector
from physics.projectile6dof import integrate_6dof
from physics.attitude import q_normalize

# NEW: telemetry + coeffs + atmosphere + plotting
from diagnostics.telemetry import make_post_step_recorder
from utils.cd_lookup import get_coeffs as cd_get_coeffs
import physics.atmosphere as atmosphere
from diagnostics.plots import standard_set  # saves a bundle of PNGs to out/

OUTDIR = pathlib.Path("out"); OUTDIR.mkdir(exist_ok=True)

# -------------------- helpers --------------------

def build_initial_state(muzzle_speed: float, elevation_deg: float, azimuth_deg: float,
                        spin_rpm: float, ap=None) -> np.ndarray:
    elev = math.radians(elevation_deg)
    az   = math.radians(azimuth_deg)
    u0 = np.array([
        math.cos(elev)*math.cos(az),
        math.cos(elev)*math.sin(az),
        math.sin(elev)
    ], dtype=float) * float(muzzle_speed)

    pos0 = np.array([0.0, 0.0, 0.0], dtype=float)

    # yaw (az) then pitch (elev), roll=0
    cy, sy = math.cos(az/2.0), math.sin(az/2.0)
    cp, sp = math.cos(elev/2.0), math.sin(elev/2.0)
    cr, sr = 1.0, 0.0
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    q0 = q_normalize(np.array([qw, qx, qy, qz], dtype=float))

    # Initial spin about body x → Hx = Ix * ωx
    spin_rad_s = float(spin_rpm) * 2.0 * math.pi / 60.0
    if ap is not None and hasattr(ap, "Ix"):
        Ix = float(ap.Ix)
    else:
        Ix = float(getattr(cfg, "PROJECTILE_IX", 0.1444))
    H0 = np.array([Ix * spin_rad_s, 0.0, 0.0], dtype=float)

    y0 = np.zeros(13, dtype=float)
    y0[0:3] = u0
    y0[3:6] = pos0
    y0[6:9] = H0
    y0[9:13] = q0
    return y0

def cut_at_impact(ts: np.ndarray, ys: np.ndarray, ground_level: float = 0.0):
    """Trim (ts, ys) at first downward crossing of z==ground_level."""
    z = ys[:, 5]
    dz = z - ground_level
    idx = np.where((dz[:-1] > 0.0) & (dz[1:] <= 0.0))[0]
    if idx.size == 0:
        return ts, ys, ts[-1], ys[-1, 3:6], False  # not impacted
    i = int(idx[0])
    denom = (dz[i] - dz[i+1])
    w = (dz[i] / denom) if abs(denom) > 1e-15 else 0.0
    t_imp = ts[i] + w * (ts[i+1] - ts[i])
    y_imp = ys[i] + w * (ys[i+1] - ys[i])
    ts_trim = np.concatenate([ts[:i+1], np.array([t_imp])])
    ys_trim = np.vstack([ys[:i+1], y_imp[None, :]])
    return ts_trim, ys_trim, t_imp, y_imp[3:6], True

def save_csv(ts: np.ndarray, ys: np.ndarray, stem="sim_positions"):
    import pandas as pd
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTDIR / f"{stem}_{stamp}.csv"
    pos = ys[:, 3:6]
    df = pd.DataFrame(pos, columns=["x", "y", "z"])
    df["t"] = ts
    df.to_csv(out, index=False)
    return out

def run_single(speed, elev, az, spin, tfinal, integrator_opts=None,
               save_track=False, stem="sim_positions",
               record=True):
    # optional prebuilt ap from config
    try:
        from config import ap as AP_obj
        ap = AP_obj
    except Exception:
        ap = None

    y0 = build_initial_state(speed, elev, az, spin, ap=ap)

    # Build a wind callable consistent with projectile6dof
    wv = getattr(cfg, "WIND_VECTOR", None)
    if callable(wv):
        wind_fn = wv
    else:
        if hasattr(wv, "to_numpy"): wv = wv.to_numpy()
        w_const = np.asarray(wv if wv is not None else [0.0, 0.0, 0.0], dtype=float)
        wind_fn = (lambda t, w=w_const: w)

    # Telemetry recorder (post-step hook)
    recorder = None
    if record:
        if ap is None:
            class AP: pass
            ap_tel = AP()
            ap_tel.m  = getattr(cfg, "PROJECTILE_MASS", 43.7)
            ap_tel.d  = getattr(cfg, "PROJECTILE_DIAMETER", 0.155)
            ap_tel.S  = getattr(cfg, "REFERENCE_AREA", math.pi*(ap_tel.d**2)/4.0)
            ap_tel.Ix = getattr(cfg, "PROJECTILE_IX", 0.1444)
            ap_tel.Iy = getattr(cfg, "PROJECTILE_IY", 1.7323)
        else:
            ap_tel = ap
        recorder = make_post_step_recorder(ap_tel, atmosphere, wind_fn, cd_lookup=cd_get_coeffs)

    # integrate (event in integrator will stop at impact)
    ts, ys = integrate_6dof(ap, y0, 0.0, tfinal,
                            integrator_opts=integrator_opts,
                            post_step_user=(recorder if recorder else None))

    # Trim to ground (double-safety; event should already stop there)
    ground = float(getattr(cfg, "GROUND_LEVEL_Z", 0.0))
    ts, ys, t_imp, pos_imp, hit = cut_at_impact(ts, ys, ground)
    x, y, z = pos_imp
    R = math.hypot(x, y)
    bearing = math.degrees(math.atan2(y, x))

    print(f"Impact: TOF={t_imp:.3f} s | range={R:.2f} m | bearing={bearing:.2f}° | z={z:.6f} m | hit={'yes' if hit else 'no'}")

    if save_track:
        csv_path = save_csv(ts, ys, stem=stem)
        print(f"Saved {csv_path.resolve()}")

    # Save diagnostic plots bundle
    if recorder is not None:
        label = f"{int(speed)}ms_{int(elev)}deg"
        imgs = standard_set(recorder.telemetry, speed_label=label)
        for p in imgs:
            print("Saved:", p)

    return t_imp, R, bearing

# -------------------- CLI --------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--speed", type=float, default=480.0, help="muzzle speed m/s")
    p.add_argument("--elev", type=float, default=20.0, help="elevation degrees")
    p.add_argument("--az",   type=float, default=0.0,  help="azimuth degrees")
    p.add_argument("--spin", type=float, default=3000.0, help="spin rpm")
    p.add_argument("--tfinal", type=float, default=200.0,
                   help="max sim time s (event stops at impact; large is safe)")
    # perf toggle
    p.add_argument("--fast", action="store_true",
                   help="use faster integrator settings for sweeps")
    # NEW: control I/O + diagnostics
    p.add_argument("--save-track", action="store_true",
                   help="save per-shot trimmed CSV track to out/")
    p.add_argument("--no-record", dest="record", action="store_false",
                   help="disable telemetry/plots for max speed")
    p.set_defaults(record=True)
    return p.parse_args()

def main():
    args = parse_args()

    # Integrator options
    if args.fast:
        integrator_opts = dict(h0=1e-3, rtol=2e-6, atol=1e-8, h_max=0.05, h_min=1e-9)
    else:
        integrator_opts = dict(h0=1e-4, rtol=1e-7, atol=1e-8, h_max=0.02, h_min=1e-9)

    run_single(args.speed, args.elev, args.az, args.spin, args.tfinal,
               integrator_opts=integrator_opts,
               save_track=args.save_track,
               record=args.record)

if __name__ == "__main__":
    main()
