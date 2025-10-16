
from projectile import initialize_projectile, acceleration_func
from physics.integrator import rk4_step
from utils.vector import Vector
from ui.plotter import plot_trajectory_3d
import pandas as pd
import config

# ---- User inputs ----
muzzle_velocity = float(input("Enter muzzle velocity (m/s): "))
quadrant_elevation = float(input("Enter quadrant elevation (deg): "))
config.LATITUDE = float(input("Enter launch latitude (degrees): "))
config.AZIMUTH = float(input("Enter launch azimuth (degrees): "))

# Initial state
pos, vel = initialize_projectile(muzzle_velocity, quadrant_elevation, config.AZIMUTH)
state = (pos, vel)
dt = 0.0002  # timestep

# ---- Simulation loop ----
results = []  # store all rows in memory
time = 0.0

while state[0].y > 0:
    pos, vel = state
    speed = vel.norm()
    results.append((time, pos.x, pos.y, pos.z, speed))  # tuple is faster than list
    state = rk4_step(state, dt, acceleration_func)
    time += dt

# ---- Save once at the end ----
df = pd.DataFrame(results, columns=["time", "north", "up", "east", "speed"])

# For backward compatibility with any code expecting x,y,z, also add aliases:
# x -> east, y -> north, z -> up
df["x"] = df["east"]
df["y"] = df["north"]
df["z"] = df["up"]
# Uncomment if you want to save:
#df.to_excel("trajectories.xlsx", index=False)

# ---- Plot results ----
plot_trajectory_3d(df)

print(f"Final displacement: {df['x'].iloc[-1]:.3f} m")
print(f"Max altitude: {df['y'].max():.3f} m")
