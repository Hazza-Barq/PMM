"""
Vector arithmetic utilities.

This module defines a lightweight, immutable-like ``Vector`` class for
three‑dimensional Cartesian coordinates. Instances of ``Vector`` support
common arithmetic operations (addition, subtraction, scalar multiplication,
division), geometric operations (dot and cross products, norm, normalisation),
projection and rotation. Internally the coordinates are always stored as
``float`` values and the class uses ``__slots__`` to minimise per‑instance
memory overhead. The provided methods return new ``Vector`` instances rather
than mutating the existing object.

Examples
--------
>>> from utils.vector import Vector
>>> v = Vector(1, 2, 3)
>>> w = Vector(4, -1, 0.5)
>>> v + w
Vector(5.0000, 1.0000, 3.5000)
>>> v.dot(w)
2.0
>>> v.cross(w)
Vector(1.5000, 11.5000, -9.0000)
"""

from __future__ import annotations

import math
from typing import Iterator, List

import numpy as np


class Vector:
    """A simple three‑dimensional vector class.

    Parameters
    ----------
    x, y, z : float
        The Cartesian components of the vector. Values are coerced to
        ``float`` on initialisation.

    Notes
    -----
    * ``Vector`` instances are lightweight thanks to the use of ``__slots__``;
      they do not support arbitrary new attributes.
    * Instances are effectively immutable; arithmetic operations return new
      ``Vector`` objects and do not modify the originals.
    """

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # ------------------------------------------------------------------
    # Basic arithmetic operations
    # ------------------------------------------------------------------
    def __add__(self, other: "Vector") -> "Vector":
        """Vector addition (elementwise)."""
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector") -> "Vector":
        """Vector subtraction (elementwise)."""
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector":
        """Scalar multiplication from the right."""
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vector":
        """Scalar multiplication from the left."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector":
        """Scalar division."""
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide a Vector by zero.")
        return Vector(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> "Vector":
        """Additive inverse of the vector."""
        return Vector(-self.x, -self.y, -self.z)

    # ------------------------------------------------------------------
    # Comparison and hashing
    # ------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return (
            math.isclose(self.x, other.x, abs_tol=1e-9)
            and math.isclose(self.y, other.y, abs_tol=1e-9)
            and math.isclose(self.z, other.z, abs_tol=1e-9)
        )

    def __hash__(self) -> int:
        return hash((round(self.x, 9), round(self.y, 9), round(self.z, 9)))

    # ------------------------------------------------------------------
    # Geometric operations
    # ------------------------------------------------------------------
    def dot(self, other: "Vector") -> float:
        """Dot product with another vector."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector") -> "Vector":
        """Cross product with another vector."""
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm(self) -> float:
        """Compute the Euclidean norm (magnitude) of the vector."""
        return math.sqrt(self.norm_squared())

    def norm_squared(self) -> float:
        """Compute the squared Euclidean norm (more efficient than ``norm``)."""
        return self.dot(self)

    def unit(self) -> "Vector":
        """Return a unit (normalised) vector in the same direction as this one."""
        n = self.norm()
        if n == 0:
            return Vector(0.0, 0.0, 0.0)
        return self / n

    def angle_with(self, other: "Vector") -> float:
        """Compute the angle in radians between this vector and another."""
        norms = self.norm() * other.norm()
        if norms == 0:
            return 0.0
        cos_theta = self.dot(other) / norms
        # Guard against slight numerical errors that might push cos_theta out of [-1,1]
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.acos(cos_theta)

    def project_onto(self, other: "Vector") -> "Vector":
        """Project this vector onto another vector and return the result."""
        u = other.unit()
        return u * self.dot(u)

    def rotate_about_axis(self, axis: "Vector", angle_rad: float) -> "Vector":
        """Rotate this vector about a given axis by ``angle_rad`` radians.

        This uses Rodrigues' rotation formula. The axis need not be normalised;
        it will be internally converted to a unit vector.
        """
        k = axis.unit()
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)
        return (
            self * cos_theta
            + k.cross(self) * sin_theta
            + k * (k.dot(self) * (1.0 - cos_theta))
        )

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def to_numpy(self) -> np.ndarray:
        """Return a ``numpy.ndarray`` representation of this vector."""
        return np.array([self.x, self.y, self.z], dtype=float)

    @staticmethod
    def from_numpy(arr: np.ndarray) -> "Vector":
        """Construct a ``Vector`` from a 3‑element array or sequence."""
        return Vector(float(arr[0]), float(arr[1]), float(arr[2]))

    def to_list(self) -> List[float]:
        """Return the components as a list ``[x, y, z]``."""
        return [self.x, self.y, self.z]

    def __iter__(self) -> Iterator[float]:
        """Yield the components of the vector in order x, y, z."""
        yield self.x
        yield self.y
        yield self.z

    def __repr__(self) -> str:
        return f"Vector({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"
