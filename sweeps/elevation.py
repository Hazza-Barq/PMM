# sweeps/elevation.py
import numpy as np, datetime as dt, pathlib, pandas as pd
from main import run_single

FAST_OPTS = dict(h0=1e-3, rtol=2e-6, atol=1e-8, h_max=0.05, h_min=1e-9)

def run(speed=480.0, az=0.0, spin=3000.0, tfinal=120.0,
        emin=5.0, emax=45.0, estep=2.0, fast=True):
    results=[]
    for elev in np.arange(emin, emax + 1e-9, estep):
        opts = FAST_OPTS if fast else None
        # IMPORTANT: record=False stops per-angle PNGs; save_track=False stops per-angle CSVs
        tof, rng, bearing = run_single(speed, elev, az, spin, tfinal,
                                       integrator_opts=opts,
                                       save_track=False,
                                       record=False)
        results.append((elev, tof, rng, bearing))

    outdir = pathlib.Path("out"); outdir.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(results, columns=["elev_deg","tof_s","range_m","bearing_deg"])
    csv = outdir / f"sweep_elev_{int(speed)}ms_{stamp}.csv"
    df.to_csv(csv, index=False); print("Saved sweep:", csv.resolve())

    # plot only once (the summary)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(); plt.plot(df["elev_deg"], df["range_m"])
    plt.xlabel("Elevation (deg)"); plt.ylabel("Range (m)"); plt.grid(True, ls=":")
    plt.title(f"Range vs Elevation @ {speed} m/s")
    png = outdir / f"sweep_elev_{int(speed)}ms_{stamp}.png"
    plt.tight_layout(); plt.savefig(png, dpi=150); print("Saved plot:", png.resolve())

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=480.0)
    ap.add_argument("--az", type=float, default=0.0)
    ap.add_argument("--spin", type=float, default=3000.0)
    ap.add_argument("--tfinal", type=float, default=120.0)
    ap.add_argument("--emin", type=float, default=5.0)
    ap.add_argument("--emax", type=float, default=45.0)
    ap.add_argument("--estep", type=float, default=2.0)
    ap.add_argument("--nofast", action="store_true", help="disable fast integrator")
    args = ap.parse_args()
    run(args.speed, args.az, args.spin, args.tfinal,
        args.emin, args.emax, args.estep, fast=(not args.nofast))
