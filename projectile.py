# projectile.py
from utils.vector import Vector
import math
from physics.forces import compute_drag_force, compute_gravity_force, compute_coriolis_force
from physics.integrator import rk4_step, euler_step
from config import PROJECTILE_MASS, REFERENCE_AREA, WIND_VECTOR, LATITUDE, AZIMUTH, SPEED_OF_SOUND  # example constants
from physics.atmosphere import isa_properties
from utils.cd_lookup import get_cd

def deg_to_rad(deg):
    return math.radians(deg)

def initialize_projectile(muzzle_velocity, quadrant_elevation_deg, azimuth_deg):
    # Convert degrees to radians
    qe = deg_to_rad(quadrant_elevation_deg)   # Quadrant Elevation (vertical angle from horizontal)
    az = deg_to_rad(azimuth_deg)              # Azimuth from North, clockwise to East

    # Direction cosines for our coordinate system: 
    # 1 = North, 2 = Up, 3 = East
    north_comp = math.cos(qe) * math.cos(az)
    up_comp    = math.sin(qe)
    east_comp  = math.cos(qe) * math.sin(az)

    # Initial position unit vector (direction only)
    x0 = Vector(north_comp, up_comp, east_comp)

    # Initial velocity vector
    v0 = Vector(
        muzzle_velocity * north_comp,
        muzzle_velocity * up_comp,
        muzzle_velocity * east_comp
    )

    return (x0, v0)

    # Convert degrees to radians
    qe = deg_to_rad(quadrant_elevation_deg)
    az = deg_to_rad(azimuth_deg)

    # Initial position unit vector in launch direction
    x0 = Vector(
        math.cos(qe) * math.cos(az),
        math.sin(qe),
        math.cos(qe) * math.sin(az) 
    )

    # Initial velocity vector
    v0 = Vector(
        muzzle_velocity * math.cos(qe) * math.cos(az),
        muzzle_velocity * math.sin(qe),
        muzzle_velocity * math.cos(qe) * math.sin(az)
    )

    # State vector: (position, velocity)
    return (x0, v0)

def acceleration_func(position: Vector, velocity: Vector) -> Vector:
    """
    Acceleration from total forces at current state, including atmospheric updates.
    """
    # Get altitude from position (y component)
    altitude = position.y  # <-- changed from position[1]

    temp, pressure, RHO = isa_properties(altitude)

    speed = velocity.norm()  # <-- changed from magnitude()
    mach = speed / SPEED_OF_SOUND
    Cd = get_cd(mach)

    drag = compute_drag_force(velocity, WIND_VECTOR, RHO, Cd, REFERENCE_AREA)
    gravity = compute_gravity_force(position, LATITUDE)
    coriolis = compute_coriolis_force(velocity, LATITUDE, AZIMUTH)

    total_force = drag + gravity + coriolis 
    #print(f"Altitude: {altitude:.2f} m, Speed: {speed:.2f} m/s, Mach: {mach:.2f}, Cd: {Cd:.3f}")
    #print(f"Drag: {drag}, Gravity: {gravity}, Coriolis: {coriolis}")

    
     # you can use + since Vector implements __add__
    return total_force / PROJECTILE_MASS  # Vector supports __truediv__
       

    

def simulate_projectile(initial_state, dt, steps, method="rk4"):
    """
    Simulates projectile motion using chosen integration method.
    method: "rk4" or "euler"
    """
    state = initial_state
    trajectory = [state]

    step_func = rk4_step if method.lower() == "rk4" else euler_step

    for _ in range(steps):
        state = step_func(state, dt, acceleration_func)
        trajectory.append(state)

    return trajectory
