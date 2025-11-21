# utils/atmosphere.py

import math

# Constants
g = 9.80665  # m/s^2
R = 287.00   # J/(kg·K)

# Layer data
lapse_rates = [-0.0065, 0, 0.001, 0.0028]      # K/m
layer_bounds = [11000, 20000, 32000, 47000]    # meters

# Base values for ISA
base_pressure = 101325  # Pa
base_temperature = 288.15  # K

def _calculate_layer(p0, t0, a, h0, h1):
    """Calculate temperature and pressure at h1 given base at h0."""
    if a != 0:
        t1 = t0 + a * (h1 - h0)
        p1 = p0 * (t1 / t0) ** (-g / (a * R))
    else:
        t1 = t0
        p1 = p0 * math.exp(-g * (h1 - h0) / (R * t0))
    return t1, p1

def isa_properties(altitude: float):
    """
    Returns (temperature [K], pressure [Pa], density [kg/m^3])
    for a given altitude [m] from 0 to 47000 m using ISA model.
    Altitudes outside this range are clamped.
    """

    # Clamp altitude to [0, 47000]
    if altitude < 0:
        altitude = 0
    elif altitude > 47000:
        altitude = 47000

    p0 = base_pressure
    t0 = base_temperature
    h_prev = 0

    for i in range(len(layer_bounds)):
        h_layer = layer_bounds[i]
        a = lapse_rates[i]

        if altitude <= h_layer:
            temperature, pressure = _calculate_layer(p0, t0, a, h_prev, altitude)
            break
        else:
            t0, p0 = _calculate_layer(p0, t0, a, h_prev, h_layer)
            h_prev = h_layer

    RHO = pressure / (R * temperature)
    return temperature, pressure, RHO
