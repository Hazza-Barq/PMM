# analytics/akcay.py
"""
Akçay approximate trajectory model helpers.

Functions:
 - k0_from_measured_range(X, v0, theta, rho0, g0) -> K0
 - k0_from_cd(cd, S, m, theta) -> K0_est (simple geometry-based estimate)
 - range_from_k0(K0, v0, theta, rho0, g0) -> X (closed-form positive root)
 - trajectory_samples(K0, v0, theta, rho0, g0, dx=1.0) -> arrays x,y,t,v
 - xs_ymax_from_K0(...) -> xs, ymax
 - fit_k0_poly(theta_list, X_list, rho0, v0_list=None, deg=4) -> poly coeffs
Notes: angles are radians internally; inputs may be provided in degrees using helper wrappers.
"""
from __future__ import annotations
import math, numpy as np
from typing import Sequence, Tuple, Optional

# ---- utilities ----
def deg2rad(d):
    return math.radians(d)

def rad2deg(r):
    return math.degrees(r)

# ---- core formula helpers ----
def k0_from_measured_range(X: float, v0: float, theta_rad: float,
                           rho0: float, g0: float = 9.80665) -> float:
    """
    Eq (26): K0 = 1/(rho0*X) * ( v0^2 sin(2θ) / (g0 X) - 1 )
    Returns K0 (scalar).
    """
    X = float(X)
    if X <= 0:
        raise ValueError("X must be positive")
    s2 = math.sin(2.0 * theta_rad)
    term = (v0 * v0 * s2) / (g0 * X)
    K0 = (term - 1.0) / (rho0 * X)
    return float(K0)

def k0_from_cd(cd: float, S: float, m: float, theta_rad: float) -> float:
    """
    Estimate K0 from geometry using Akcay eq (23): K0 = CD * S / (3 m cos(theta))
    If cos(theta) is near zero, this formula blows up (vertical shots) — use measured fit instead.
    """
    c = float(cd)
    denom = 3.0 * float(m) * max(1e-12, math.cos(theta_rad))
    return float(c * float(S) / denom)

def range_from_k0(K0: float, v0: float, theta_rad: float,
                  rho0: float, g0: float = 9.80665) -> float:
    """
    Solve for nonzero range X from Eq (24) → leads to quadratic:
      B*K0*rho0 * X^2 + B * X - A = 0
    where A = tan(theta), B = g0 / (2 v0^2 cos^2 theta)

    Return positive root (float). Raises if no positive root.
    """
    K0 = float(K0)
    A = math.tan(theta_rad)
    cos_t = math.cos(theta_rad)
    cos2 = cos_t * cos_t
    if cos2 <= 0:
        raise ValueError("cos(theta)^2 <= 0; theta too near ±90deg")
    B = g0 / (2.0 * (v0 * v0) * cos2)

    a = B * K0 * rho0
    b = B
    c = -A

    # Solve a X^2 + b X + c = 0
    if abs(a) < 1e-16:
        # degenerate -> linear: b X + c = 0 -> X = -c/b
        if abs(b) < 1e-16:
            raise RuntimeError("degenerate coefficients in range solve")
        X = -c / b
        if X <= 0:
            raise RuntimeError("No positive root for X (degenerate).")
        return float(X)

    disc = b * b - 4.0 * a * c
    if disc < 0:
        raise RuntimeError(f"No real root for X (disc={disc})")
    sqrt_d = math.sqrt(disc)

    # two roots:
    x1 = (-b + sqrt_d) / (2.0 * a)
    x2 = (-b - sqrt_d) / (2.0 * a)
    # choose positive
    Xs = [x for x in (x1, x2) if x > 0 and np.isfinite(x)]
    if not Xs:
        raise RuntimeError(f"No positive real root for X (roots {x1}, {x2})")
    X = max(Xs)  # choose larger positive root (physical)
    return float(X)

def t_at_x(x: float, v0: float, theta_rad: float, rho0: float, K0: float) -> float:
    """
    Akcay Eq (29):
      t = (2 / (9 v0 cosθ)) * (1 / (ρ0 K0)) * [ (1 + 3 ρ0 K0 x)^{3/2} - 1 ]
    """
    cos_t = math.cos(theta_rad)
    denom = 9.0 * v0 * cos_t
    if denom == 0:
        raise RuntimeError("degenerate denom in t_at_x")
    inner = 1.0 + 3.0 * rho0 * K0 * x
    if inner < 0:
        # mathematically possible if K0 negative and x large -> treat as NaN
        return float("nan")
    return (2.0 / denom) * (1.0 / (rho0 * K0)) * (inner ** 1.5 - 1.0)

def v_at_x(x: float, v0: float, theta_rad: float, rho0: float, K0: float) -> float:
    """
    Akcay Eq (30): v = v0 * (cos theta / cos phi) * (1 + 3ρ0K0 x)^(-1/2)
    But phi depends on x; we can compute cos(phi) from geometry: cosφ = ?
    Simpler: using energy-like, use v = v0 * (cosθ / cosφ) * factor.
    The paper gives only that relation; for plotting we use the factor form:
      v(x) = (v0 * cosθ) / cosφ * (1 + 3ρ0 K0 x)^(-1/2)
    We'll return the nominal speed magnitude factor approximation:
      v(x) ≈ v0 * (1 + 3ρ0 K0 x)^(-1/2)
    This follows the paper's dependence on (1+3ρ0K0x)^-1/2 ignoring small cosθ/cosφ correction.
    """
    inner = 1.0 + 3.0 * rho0 * K0 * x
    if inner <= 0:
        return float("nan")
    return float(v0 * (inner ** -0.5))

def trajectory_samples(K0: float, v0: float, theta_rad: float,
                       rho0: float, g0: float = 9.80665,
                       dx: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sampled arrays (x, y, t, v) from x=0 up to impact X using Akcay approximate model.
    dx: sample spacing in meters (small positive).
    """
    X = range_from_k0(K0, v0, theta_rad, rho0, g0)
    if X <= 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]))
    xs = np.arange(0.0, X + dx, dx)
    tan_t = math.tan(theta_rad)
    cos_t = math.cos(theta_rad)
    cos2 = cos_t * cos_t
    B = g0 / (2.0 * v0 * v0 * cos2)
    ys = tan_t * xs - B * xs * xs * (1.0 + K0 * rho0 * xs)
    ts = np.array([t_at_x(float(x), v0, theta_rad, rho0, K0) for x in xs], dtype=float)
    vs = np.array([v_at_x(float(x), v0, theta_rad, rho0, K0) for x in xs], dtype=float)
    return xs, ys, ts, vs

def xs_ymax_from_K0(K0: float, v0: float, theta_rad: float, rho0: float, g0: float=9.80665) -> Tuple[float,float]:
    """
    Compute x_s (x where ymax occurs) and ymax using Akcay eq (31)-(32).
    Returns (xs, ymax).
    """
    X = range_from_k0(K0, v0, theta_rad, rho0, g0)
    # use eq (32) for xs; eq (31) for ymax
    denom = 3.0 * rho0 * K0
    if abs(denom) < 1e-16:
        # fallback: no K0 -> classical xs = X/2 (approx) and ymax from parabola
        xs = X / 2.0
    else:
        term = 1.0 + 3.0 * rho0 * K0 * X * (1.0 + K0 * rho0 * X)
        inner = math.sqrt(term)
        xs = (inner - 1.0) / denom
    # ymax eq (31): ymax = xs tanθ / (1 + 2ρ0K0 xs^2 + 3K0 xs)
    tan_t = math.tan(theta_rad)
    denom2 = 1.0 + 2.0 * rho0 * K0 * xs * xs + 3.0 * K0 * xs
    ymax = xs * tan_t / denom2 if denom2 != 0 else float("nan")
    return float(xs), float(ymax)

# ---- polynomial fit for K0(theta) from multiple measurements ----
def fit_k0_poly(theta_list_deg: Sequence[float], X_list: Sequence[float],
                rho0: float, v0_list: Optional[Sequence[float]] = None,
                deg: int = 4) -> np.ndarray:
    """
    Fit K0(theta) polynomial coefficients (deg degree) from measurements.
    Inputs:
      theta_list_deg: list of elevation angles in degrees (θ_i)
      X_list: list of measured ranges for those θ_i
      rho0: air density at test location
      v0_list: either single v0 (if same) or list of muzzle velocities for each measurement
    Returns:
      numpy array of polynomial coeffs [a0, a1, ..., adeg] such that K0(θ) = a0 + a1 θ + a2 θ^2 + ...
      (angles θ used in radians in the polynomial fit)
    """
    thetas_rad = np.array([math.radians(t) for t in theta_list_deg], dtype=float)
    Xs = np.array(X_list, dtype=float)
    if v0_list is None:
        v0 = None
        v0s = None
    else:
        if np.isscalar(v0_list):
            v0s = np.full_like(Xs, float(v0_list))
        else:
            v0s = np.array(v0_list, dtype=float)

    K_vals = np.zeros_like(Xs)
    for i, X in enumerate(Xs):
        v0i = float(v0s[i]) if v0s is not None else None
        # We require v0 to compute K0 via Eq (26):
        if v0i is None:
            raise ValueError("v0_list must be provided when fitting K0 from measured ranges")
        K_vals[i] = k0_from_measured_range(X, v0i, thetas_rad[i], rho0)

    # Fit polynomial in θ (radians)
    V = np.vander(thetas_rad, deg + 1, increasing=True)  # columns: 1, θ, θ^2, ...
    coeffs, *_ = np.linalg.lstsq(V, K_vals, rcond=None)
    return coeffs  # array length deg+1

# ---- small example helpers ----
def example_usage():
    """
    Minimal example:
      v0=480 m/s, elev=20 deg, rho0=1.225, assume CD≈0.3, S, m given,
      compute K0 estimate from CD and compute analytic range.
    """
    v0 = 480.0
    elev_deg = 20.0
    theta = math.radians(elev_deg)
    rho0 = 1.225
    CD = 0.3; S = 0.019  # example area (m^2)
    m = 43.7
    K0_est = k0_from_cd(CD, S, m, theta)
    X = range_from_k0(K0_est, v0, theta, rho0)
    xs, ys, ts, vs = trajectory_samples(K0_est, v0, theta, rho0, dx=10.0)
    return {"K0_est": K0_est, "range": X, "xs_len": len(xs)}
