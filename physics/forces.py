from utils.vector import Vector
import config
from config import R_EARTH, OMEGA_EARTH, PROJECTILE_MASS
import math
import numpy as np

def compute_drag_force(velocity: Vector, wind: Vector, rho: float, Cd: float, area: float) -> Vector:
    # Convert to NumPy arrays for fast math
    v_rel = np.array([velocity.x - wind.x,
                      velocity.y - wind.y,
                      velocity.z - wind.z])
    speed = np.linalg.norm(v_rel)
    drag = -0.5 * rho * Cd * area * speed * v_rel
    return Vector(*drag)

def compute_gravity_force(position: Vector, lat_deg: float) -> Vector:
    lat_rad = math.radians(config.LATITUDE)
    g0 = 9.80665 * (1 - 0.0026 * math.cos(2 * lat_rad))  # adjusted for latitude

    pos_arr = np.array([position.x, position.y, position.z])
    grav = np.array([
        -g0 * (pos_arr[0] / R_EARTH) * 0,
        -g0 * (1 - (2 * pos_arr[1] / R_EARTH)),
        -g0 * (pos_arr[2] / R_EARTH) * 0
    ]) * PROJECTILE_MASS
    return Vector(*grav)

def compute_coriolis_force(velocity: Vector, lat_deg: float, azimuth_deg: float) -> Vector:
    lat_rad = math.radians(config.LATITUDE)
    az_rad = math.radians(config.AZIMUTH)

    omega = np.array([
        OMEGA_EARTH * math.cos(lat_rad) * math.cos(az_rad),
        OMEGA_EARTH * math.sin(lat_rad),
        OMEGA_EARTH * math.cos(lat_rad) * math.sin(az_rad)
    ])

    vel_arr = np.array([velocity.x, velocity.y, velocity.z])
    coriolis = -2 * np.cross(omega, vel_arr)
    return Vector(*coriolis)
