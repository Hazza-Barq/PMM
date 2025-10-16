# sweep.py
import numpy as np
import pandas as pd
import json
import config
from projectile import initialize_projectile, acceleration_func
from physics.integrator import rk4_step

def simulate_until_ground(muzzle_velocity, elevation_deg, azimuth_deg, dt=0.02):
    """Run one trajectory until projectile hits ground (y <= 0)."""
    pos, vel = initialize_projectile(muzzle_velocity, elevation_deg, azimuth_deg)
    state = (pos, vel)

    t = 0.0
    traj = []

    while pos.y >= 0:
        traj.append({"t": t, "x": pos.x, "y": pos.y, "z": pos.z})
        pos, vel = rk4_step(state, dt, acceleration_func)
        state = (pos, vel)
        t += dt

    return traj


def sweep_angles(angle_list, muzzle_velocity, azimuth_deg=0.0, dt=0.02,
                 out_csv="sweep.csv", out_json="sweep.json"):
    """Sweep through angles and store results for Akçay fitting."""
    results = []

    for theta in angle_list:
        traj = simulate_until_ground(muzzle_velocity, theta, azimuth_deg, dt)
        final = traj[-1]

        # Range in ground plane (sqrt(x²+z²))
        range_m = np.sqrt(final["x"]**2 + final["z"]**2)
        tof_s = final["t"]

        # Downsample trajectory for fitting (~50 points)
        N = max(1, len(traj)//50)
        sample = traj[::N]
        xs = [p["x"] for p in sample]
        ys = [p["y"] for p in sample]
        zs = [p["z"] for p in sample]

        results.append({
            "theta_deg": theta,
            "range_m": range_m,
            "tof_s": tof_s,
            "sample_x": xs,
            "sample_y": ys,
            "sample_z": zs
        })

    # Save CSV (just the essentials)
    df = pd.DataFrame([{
        "theta_deg": r["theta_deg"],
        "range_m": r["range_m"],
        "tof_s": r["tof_s"]
    } for r in results])
    df.to_csv(out_csv, index=False)

    # Save JSON (full trajectories)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} cases to {out_csv} and {out_json}")
    return results


if __name__ == "__main__":
    muzzle_velocity = 950.0   # m/s, set what you want
    config.LATITUDE = 0.0     # ignore Coriolis for training
    config.AZIMUTH = 0.0

    # Sweep angles 20°–80° in 2° steps
    angle_list = np.arange(20, 81, 2)

    sweep_angles(angle_list, muzzle_velocity)
