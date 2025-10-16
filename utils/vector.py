# utils/vector.py

import math
import numpy as np

class Vector:
    __slots__ = ("x", "y", "z")  # prevents dict overhead, faster & lighter

    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # --- Basic Arithmetic ---
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float):
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float):
        if scalar == 0:
            raise ZeroDivisionError("Division by zero in Vector")
        return Vector(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def __eq__(self, other):
        return (
            math.isclose(self.x, other.x, abs_tol=1e-9) and
            math.isclose(self.y, other.y, abs_tol=1e-9) and
            math.isclose(self.z, other.z, abs_tol=1e-9)
        )

    def __hash__(self):
        return hash((round(self.x, 9), round(self.y, 9), round(self.z, 9)))

    # --- Dot and Cross Products ---
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    # --- Magnitude and Normalization ---
    def norm(self):
        return math.sqrt(self.norm_squared())

    def norm_squared(self):
        return self.dot(self)

    def unit(self):
        n = self.norm()
        if n == 0:
            return Vector(0.0, 0.0, 0.0)
        return self / n

    # --- Angle and Projection ---
    def angle_with(self, other):
        dot = self.dot(other)
        norms = self.norm() * other.norm()
        if norms == 0:
            return 0.0
        return math.acos(max(-1.0, min(1.0, dot / norms)))  # clamp for stability

    def project_onto(self, other):
        unit_other = other.unit()
        return unit_other * self.dot(unit_other)
    

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_numpy(self):
        """Convert to NumPy array for SciPy integration."""
        return np.array([self.x, self.y, self.z], dtype=float)

    @staticmethod
    def from_numpy(arr):
        """Convert NumPy array back to Vector."""
        return Vector(arr[0], arr[1], arr[2])


    # --- Rotation (Rodrigues' rotation formula) ---
    def rotate_about_axis(self, axis, angle_rad):
        """
        Rotate vector about given axis by angle (radians).
        """
        k = axis.unit()
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)
        return (
            self * cos_theta +
            k.cross(self) * sin_theta +
            k * (k.dot(self) * (1 - cos_theta))
        )

    # --- Utilities ---
    def to_list(self):
        return [self.x, self.y, self.z]

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __repr__(self):
        return f"Vector3D({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"
