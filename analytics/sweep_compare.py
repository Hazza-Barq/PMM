# analytics/sweep_compare.py
import argparse, numpy as np, pandas as pd, pathlib, datetime as dt
from analytics.compare_analytical import run_compare

OUT = pathlib.Path("out"); OUT.mkdir(exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="calibration CSV used for Akçay fit (elev_deg,range_m,v0_mps)")
    ap.add_argument("--speed", type=float, default=480.0)
    ap.add_argument("--az", type=float, default=0.0)
    ap.add_argument("--spin", type=float, default=3000.0)
    ap.add_argument("--tfinal", type=float, default=200.0)
    ap.add_argument("--rho0", type=float, default=1.225)
    ap.add_argument("--g0", type=float, default=9.80665)
    ap.add_argument("--cd", type=float, default=0.30)   # used only for vacuum/drag baseline
    ap.add_argument("--mass", type=float, default=43.7)
    ap.add_argument("--diam", type=float, default=0.155)
    ap.add_argument("--emin", type=float, default=10.0)
    ap.add_argument("--emax", type=float, default=50.0)
    ap.add_argument("--estep", type=float, default=2.0)
    ap.add_argument("--poly-deg", type=int, default=4)
    args = ap.parse_args()

    rows=[]
    for elev in np.arange(args.emin, args.emax + 1e-9, args.estep):
        res = run_compare(
            mode="holdout",
            speed=args.speed, elev=float(elev), az=args.az, spin=args.spin, tfinal=args.tfinal,
            rho0=args.rho0, g0=args.g0, cd=args.cd, mass=args.mass, diam=args.diam,
            csv=args.csv, poly_deg=args.poly_deg
        )
        (tv, Rv) = res["vacuum"]; (td, Rd) = res["drag"]; (ts, Rs) = res["full6dof"]; (ta, Ra) = res["akcay"]
        rows.append(dict(
            elev_deg=float(elev),
            range_6dof=Rs, range_akcay=Ra, range_err_m=Ra-Rs, range_err_pct=100.0*(Ra-Rs)/max(1e-9,Rs),
            tof_6dof=ts, tof_akcay=ta, tof_err_s=ta-ts, tof_err_pct=100.0*(ta-ts)/max(1e-9,ts)
        ))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    df = pd.DataFrame(rows)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_out = OUT / f"loo_errors_{int(args.speed)}ms_{stamp}.csv"
    df.to_csv(csv_out, index=False); print("Saved:", csv_out.resolve())

    # Summary metrics
    mae_m = df["range_err_m"].abs().mean()
    mape = df["range_err_pct"].abs().mean()
    mae_t = df["tof_err_s"].abs().mean()
    print(f"MAE(range) = {mae_m:.2f} m | MAPE(range) = {mape:.2f}% | MAE(TOF) = {mae_t:.3f} s")

    # Plots
    fig1 = plt.figure()
    plt.plot(df["elev_deg"], df["range_err_pct"], marker="o")
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("Elevation (deg)"); plt.ylabel("Akçay − 6DoF  (range error, %)")
    plt.grid(True, ls=":"); plt.title(f"Akçay LOO error vs elevation @ {args.speed} m/s")
    png1 = OUT / f"loo_range_errpct_{int(args.speed)}ms_{stamp}.png"
    fig1.tight_layout(); fig1.savefig(png1, dpi=150); plt.close(fig1); print("Saved:", png1.resolve())

    fig2 = plt.figure()
    plt.plot(df["elev_deg"], df["tof_err_s"], marker="o")
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("Elevation (deg)"); plt.ylabel("Akçay − 6DoF  (TOF error, s)")
    plt.grid(True, ls=":"); plt.title(f"Akçay LOO TOF error vs elevation @ {args.speed} m/s")
    png2 = OUT / f"loo_tof_err_{int(args.speed)}ms_{stamp}.png"
    fig2.tight_layout(); fig2.savefig(png2, dpi=150); plt.close(fig2); print("Saved:", png2.resolve())

if __name__ == "__main__":
    main()
