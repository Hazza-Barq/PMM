# sweeps/hemisphere.py
import numpy as np, pandas as pd, datetime as dt, pathlib
from multiprocessing import Pool, cpu_count
from main import run_single

FAST_OPTS = dict(h0=1e-3, rtol=2e-6, atol=1e-8, h_max=0.05, h_min=1e-9)

def _one(args):
    speed, elev, az, spin, tfinal, fast = args
    opts = FAST_OPTS if fast else None
    # No per-shot files
    tof, rng, bearing = run_single(speed, elev, az, spin, tfinal,
                                   integrator_opts=opts,
                                   save_track=False,
                                   record=False)
    return (elev, az, tof, rng, bearing)

def run(speed=480.0, spin=3000.0, tfinal=200.0,
        elevs=np.arange(5.0, 45.0+1e-9, 2.5),
        azes=np.arange(0.0, 360.0-1e-9, 5.0),
        workers=None, fast=True):
    tasks = [(speed, float(elev), float(az), spin, tfinal, fast)
             for elev in elevs for az in azes]
    workers = workers or max(1, cpu_count()-1)
    with Pool(processes=workers) as pool:
        rows = list(pool.imap_unordered(_one, tasks))

    df = pd.DataFrame(rows, columns=["elev_deg","az_deg","tof_s","range_m","bearing_deg"])
    outdir = pathlib.Path("out"); outdir.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = outdir / f"sweep_hemi_{int(speed)}ms_{stamp}.csv"
    df.to_csv(csv, index=False)
    print("Saved hemisphere sweep:", csv.resolve())

    # Mean range vs elevation
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        piv = df.pivot_table(index="elev_deg", values="range_m", aggfunc="mean")
        plt.figure(); piv.plot(kind="line", legend=False)
        plt.xlabel("Elevation (deg)"); plt.ylabel("Mean range (m)"); plt.grid(True, ls=":")
        png = outdir / f"sweep_hemi_mean_{int(speed)}ms_{stamp}.png"
        plt.tight_layout(); plt.savefig(png, dpi=150); print("Saved plot:", png.resolve())
    except Exception as e:
        print("Mean plot skipped:", e)

    # 3D surface: range(elev, az)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        piv2 = df.pivot_table(index="elev_deg", columns="az_deg", values="range_m", aggfunc="mean")
        E = piv2.index.values; A = piv2.columns.values
        R = piv2.values
        E2, A2 = np.meshgrid(E, A, indexing="ij")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(A2, E2, R/1000.0, linewidth=0, antialiased=True)
        ax.set_xlabel("Azimuth (deg)"); ax.set_ylabel("Elevation (deg)"); ax.set_zlabel("Range (km)")
        ax.set_title(f"Range surface @ {speed} m/s")
        png3d = outdir / f"sweep_hemi_surface_{int(speed)}ms_{stamp}.png"
        plt.tight_layout(); plt.savefig(png3d, dpi=150); plt.close(fig)
        print("Saved 3D surface:", png3d.resolve())
    except Exception as e:
        print("3D plot skipped:", e)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=480.0)
    ap.add_argument("--spin", type=float, default=3000.0)
    ap.add_argument("--tfinal", type=float, default=200.0)
    ap.add_argument("--emin", type=float, default=5.0)
    ap.add_argument("--emax", type=float, default=45.0)
    ap.add_argument("--estep", type=float, default=2.5)
    ap.add_argument("--azstep", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=0, help="0 = auto")
    ap.add_argument("--nofast", action="store_true")
    args = ap.parse_args()

    elevs = np.arange(args.emin, args.emax + 1e-9, args.estep)
    azes  = np.arange(0.0, 360.0 - 1e-9, args.azstep)
    run(speed=args.speed, spin=args.spin, tfinal=args.tfinal,
        elevs=elevs, azes=azes,
        workers=(None if args.workers==0 else args.workers),
        fast=(not args.nofast))
