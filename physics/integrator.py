
# integrators.py
from utils.vector import Vector
from physics.atmosphere import isa_properties
from utils.cd_lookup import get_cd
from config import SPEED_OF_SOUND

def euler_step(state, dt, acceleration_func):
    """
    One step of Euler integration.
    state: tuple (position: Vector, velocity: Vector)
    dt: timestep
    acceleration_func: function taking (position, velocity) and returning acceleration Vector
    """
    pos, vel = state
    acc = acceleration_func(pos, vel)

    # Update velocity and position
    vel_new = vel  + (acc * (dt))
    pos_new = pos  + (vel * (dt))

    return pos_new, vel_new


def rk4_step(state, dt, acceleration_func):
    pos, vel = state

    # k1
    acc1 = acceleration_func(pos, vel)
    k1_v = acc1 * dt
    k1_x = vel * dt

    # k2
    acc2 = acceleration_func(
        pos + k1_x * 0.5,
        vel + k1_v * 0.5
    )
    k2_v = acc2 * dt
    k2_x = (vel + k1_v * 0.5) * dt

    # k3
    acc3 = acceleration_func(
        pos + k2_x * 0.5,
        vel + k2_v * 0.5
    )
    k3_v = acc3 * dt
    k3_x = (vel + k2_v * 0.5) * dt

    # k4
    acc4 = acceleration_func(
        pos + k3_x,
        vel + k3_v
    )
    k4_v = acc4 * dt
    k4_x = (vel + k3_v) * dt

    # Combine increments
    vel_new = vel + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
    pos_new = pos + (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6

    return pos_new, vel_new
    """
    One step of Runge-Kutta 4 integration.
    state: tuple (position: Vector, velocity: Vector)
    dt: timestep
    acceleration_func: function taking (position, velocity) and returning acceleration Vector
    """
    pos, vel = state

    # k1
    acc1 = acceleration_func(pos, vel)
    k1_v = acc1 * (dt)
    k1_x = vel * (dt)

    # k2
    acc2 = acceleration_func(
        pos  + (k1_x * (0.5)),
        vel  + (k1_v * (0.5))
    )
    k2_v = acc2 * (dt)
    k2_x = vel  + (k1_v * (0.5)) * (dt)

    # k3
    acc3 = acceleration_func(
        pos  + (k2_x * (0.5)),
        vel  + (k2_v * (0.5))
    )
    k3_v = acc3 * (dt)
    k3_x = vel  + (k2_v * (0.5)) * (dt)

    # k4
    acc4 = acceleration_func(
        pos  + (k3_x),
        vel  + (k3_v)
    )
    k4_v = acc4 * (dt)
    k4_x = vel  + (k3_v) * (dt)

    # Combine increments
    vel_new = vel  + (
        k1_v * (1/6)  + (
        k2_v * (1/3))  + (
        k3_v * (1/3))  + (
        k4_v * (1/6))
    )

    pos_new = pos  + (
        k1_x * (1/6)  + (
        k2_x * (1/3))  + (
        k3_x * (1/3))  + (
        k4_x * (1/6))
    )

    return pos_new, vel_new
