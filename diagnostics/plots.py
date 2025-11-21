# diagnostics/plots.py
import pathlib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = pathlib.Path("out"); OUTDIR.mkdir(exist_ok=True)

def _save(fig, name):
    p = OUTDIR / name
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p.resolve()

def plot_trajectory(tel, name="traj.png"):
    import numpy as np
    fig = plt.figure()
    plt.plot([x/1000 for x in tel.x], tel.z)
    plt.xlabel("Range X [km]"); plt.ylabel("Altitude Z [m]")
    plt.title("Trajectory (x–z)")
    plt.grid(True, ls=":")
    return _save(fig, name)

def plot_timeseries(t, y, xlabel, ylabel, title, name):
    fig = plt.figure()
    plt.plot(t, y)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title); plt.grid(True, ls=":")
    return _save(fig, name)

def standard_set(tel, speed_label=""):
    paths = []
    paths.append(plot_trajectory(tel, f"traj_{speed_label}.png"))
    paths.append(plot_timeseries(tel.t, tel.F_mag, "t [s]", "|F_aero| [N]", "Aero Force", f"force_{speed_label}.png"))
    paths.append(plot_timeseries(tel.t, tel.M_mag, "t [s]", "|M_aero| [N·m]", "Aero Moment", f"moment_{speed_label}.png"))
    paths.append(plot_timeseries(tel.t, tel.mach,  "t [s]", "Mach", "Mach vs time", f"mach_{speed_label}.png"))
    paths.append(plot_timeseries(tel.t, tel.cd,    "t [s]", "Cd", "Cd vs time", f"cd_{speed_label}.png"))
    paths.append(plot_timeseries(tel.t, tel.cl,    "t [s]", "Cl", "Cl vs time", f"cl_{speed_label}.png"))
    paths.append(plot_timeseries(tel.t, tel.alpha_deg, "t [s]", "alpha [deg]", "AoA vs time", f"alpha_{speed_label}.png"))
    # energy drift
    e0 = tel.E_mech[0] if tel.E_mech else 1.0
    e_rel = [(e-e0)/abs(e0) for e in tel.E_mech]
    paths.append(plot_timeseries(tel.t, e_rel, "t [s]", "ΔE/E0", "Energy drift (translational+potential)", f"energy_{speed_label}.png"))
    return paths
