import math
from utils.vector import Vector


# === Projectile Constants ===
PROJECTILE_MASS = 43.7 # kg
PROJECTILE_DIAMETER = 0.155  # m
REFERENCE_AREA = math.pi * (PROJECTILE_DIAMETER / 2) ** 2  # m²

# === Physical Constants ===
SEA_LEVEL_GRAVITY = 9.80665
R_EARTH = 6356766     # m
R_AIR = 287.0 
OMEGA_EARTH = 0.00007292115 #spin of earth
# === Configuration Switches ===
USE_VARIABLE_GRAVITY = False  
DEBUG = False                 


WIND_VECTOR = Vector(0.0, 0.0, 0.0)

LATITUDE = None  # deg
AZIMUTH = None  # deg
SPEED_OF_SOUND = 343.0 

# config.py
PROJECTILE_IX = 0.1444   # kg·m²  (about body x / spin axis)
PROJECTILE_IY = 1.7323   # kg·m²  (about body y,z — axisymmetric Iy = Iz)
