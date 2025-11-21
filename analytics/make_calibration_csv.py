# analytics/make_calibration_csv.py
"""
Generate a calibration CSV (elev_deg, range_m, v0_mps) by running your 6-DoF
at a set of elevations. Use this to test the Akçay 'fit' pipeline.

Example:
  python -m analytics.make_calibration_csv --speed 480 --emin 10 --emax 50 --estep 5
"""
import argparse, numpy as np, pandas as pd, datetime as dt, pathlib
import main as main_mod

OUT = pathlib.Path("out"); OUT.mkdir(exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=480.0)
    ap.add_argument("--az", type=float, default=0.0)
    ap.add_argument("--spin", type=float, default=3000.0)
    ap.add_argument("--tfinal", type=float, default=200.0)
    ap.add_argument("--emin", type=float, default=10.0)
    ap.add_argument("--emax", type=float, default=50.0)
    ap.add_argument("--estep", type=float, default=5.0)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    opts = dict(h0=1e-3, rtol=2e-6, atol=1e-8, h_max=0.05, h_min=1e-9) if args.fast else None

    rows=[]
    for elev in np.arange(args.emin, args.emax+1e-9, args.estep):
        tof, rng, _ = main_mod.run_single(args.speed, elev, args.az, args.spin, args.tfinal,
                                          integrator_opts=opts, record=False, save_track=False)
        rows.append((elev, rng, args.speed))

    df = pd.DataFrame(rows, columns=["elev_deg","range_m","v0_mps"])
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = OUT / f"calib_demo_{int(args.speed)}ms_{stamp}.csv"
    df.to_csv(csv, index=False)
    print("Saved calibration CSV:", csv.resolve())

if __name__ == "__main__":
    main()
