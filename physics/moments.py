# physics/moments.py
import numpy as np, math
from utils.vector import Vector

def _as_np(v):
    if isinstance(v, Vector):
        return v.to_numpy()
    return np.asarray(v, dtype=float)

def _safe_norm(a, eps=1e-12):
    a = np.asarray(a, dtype=float)
    s = float(np.dot(a, a))
    return math.sqrt(s) if s > eps else 0.0

def _clamp_array(a, lo=-1e12, hi=1e12):
    A = np.asarray(a, dtype=float)
    A = np.where(np.isfinite(A), A, 0.0)
    return np.clip(A, lo, hi)

def aerodynamic_moments_body(v_body, x_body, H_body, aero_params, rho, coeffs=None):
    """
    Returns (dHdt_body, x_dot_body) in BODY frame.
    v_body, x_body, H_body as BODY-frame vectors.
    """
    v_body = _as_np(v_body); x_body = _as_np(x_body); H_body = _as_np(H_body)

    if isinstance(aero_params, dict):
        d = aero_params["d"]; Ix = aero_params["Ix"]; Iy = aero_params["Iy"]
    else:
        d = getattr(aero_params, "d"); Ix = getattr(aero_params, "Ix"); Iy = getattr(aero_params, "Iy")

    V = _safe_norm(v_body)
    if V <= 1e-12:
        return np.zeros(3), np.zeros(3)

    u, v, w = v_body
    alpha = math.atan2(-w, u)
    dot_vx = float(np.dot(v_body, x_body))

    # coefficients (dict preferred)
    CM_alpha  = (coeffs.get("cm_alpha") if coeffs else getattr(aero_params, "CM_alpha", 0.0))
    CM_alpha3 = (coeffs.get("cm_alpha3") if coeffs else getattr(aero_params, "CM_alpha3", 0.0))
    CM_q      = (coeffs.get("cm_q") if coeffs else getattr(aero_params, "CM_q", 0.0))
    CM_adot   = (coeffs.get("cm_alphadot") if coeffs else getattr(aero_params, "CM_alphadot", 0.0))
    Cspin     = (coeffs.get("cspin") if coeffs else getattr(aero_params, "Cspin", 0.0))
    Cmag_m    = (coeffs.get("cmag_m") if coeffs else getattr(aero_params, "Cmag_m", 0.0))

    # Dynamic pressure scaling
    qdyn = 0.5 * rho * (V**2)

    # Terms (vector forms mirroring your earlier equations, in BODY)
    OM  = (math.pi * rho * (d**3) * V / 2.0) * (CM_alpha + CM_alpha3 * (math.sin(alpha)**2)) * np.cross(v_body, x_body)
    SDM = (math.pi * rho * (d**4) * V / (8.0 * Ix)) * Cspin  * float(np.dot(H_body, x_body)) * x_body
    PDM = (math.pi * rho * (d**4) * V / (8.0 * Iy)) * (CM_q + CM_adot) * (H_body - float(np.dot(H_body, x_body)) * x_body)
    MM  = (math.pi * rho * (d**4) / (8.0 * Ix)) * Cmag_m * float(np.dot(H_body, x_body)) * (v_body - (dot_vx * x_body / max(1e-9, V)))

    dHdt_body = OM + PDM + MM + SDM

    # Geometric kinematics for body x-axis (if you still need ẋ):
    # x_dot = ω × x ;  with ω computed in projectile6dof from H/I.  Here return 0 and let caller handle ω.
    x_dot_body = np.zeros(3)

    return _clamp_array(dHdt_body), _clamp_array(x_dot_body)
