# physics/forces.py
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

def aerodynamic_forces_body(v_body, x_body, H_body, aero_params, rho, coeffs=None):
    """
    BODY-FRAME aerodynamic forces (+ gravity returned separately by caller if needed).

    v_body : air-relative velocity in BODY frame (3,)
    x_body : body x-axis (unit) in BODY frame = [1,0,0]
    H_body : angular momentum in BODY frame (3,)
    aero_params : object/dict with m, d, S, etc.
    rho : air density (kg/m^3)
    coeffs : dict from utils.cd_lookup.get_coeffs(mach) (optional)

    Returns F_body (3,)
    """
    v_body = _as_np(v_body); x_body = _as_np(x_body); H_body = _as_np(H_body)

    # geometry/mass
    if isinstance(aero_params, dict):
        d = aero_params["d"]; S = aero_params["S"]; m = aero_params["m"]
    else:
        d = getattr(aero_params, "d"); S = getattr(aero_params, "S"); m = getattr(aero_params, "m")

    V = _safe_norm(v_body)
    if V <= 1e-12:
        return np.zeros(3)

    # angle-of-attack (body z sign convention: if z-up in inertial, in body we used alpha = atan2(-w,u))
    u, v, w = v_body
    alpha = math.atan2(-w, u)  # adjust sign if your body z is opposite
    # beta = math.asin(np.clip(v / V, -1.0, 1.0))  # available if you need it

    # coefficients (prefer dict from cd_lookup; fallback to static ap values)
    CD0 = (coeffs.get("cd0") if coeffs else getattr(aero_params, "CD0", 0.0))
    CD_a2 = (coeffs.get("cd_alpha2") if coeffs else getattr(aero_params, "CD_alpha2", 0.0))
    CL_a  = (coeffs.get("cl_alpha") if coeffs else getattr(aero_params, "CL_alpha", 0.0))
    CL_a3 = (coeffs.get("cl_alpha3") if coeffs else getattr(aero_params, "CL_alpha3", 0.0))
    Cmag_f = (coeffs.get("cmag_f") if coeffs else getattr(aero_params, "Cmag_f", 0.0))

    # Drag (opposite to velocity)
    CD = CD0 + CD_a2 * (math.sin(alpha)**2)
    qdyn = 0.5 * rho * (V**2)
    DF_body = -qdyn * S * CD * (v_body / V)

    # Lift — vector form (in body): ∝ ( V * x_hat - (u·x) * v / V )
    dot_vx = float(np.dot(v_body, x_body))
    CL = CL_a + CL_a3 * (math.sin(alpha)**2)
    LF_body = qdyn * S * CL * ((V * x_body) - (dot_vx * v_body / V))

    # Magnus force — proportional to spin and cross(v, x)
    MF_pref = qdyn * S * d
    MF_body = MF_pref * Cmag_f * float(np.dot(H_body, x_body)) * np.cross(v_body, x_body)

    F_body = DF_body + LF_body + MF_body
    return _clamp_array(F_body)
