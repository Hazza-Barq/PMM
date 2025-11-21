# physics/projectile6dof.py
import numpy as np, math
import config as cfg
from utils.vector import Vector
from utils.cd_lookup import get_coeffs as cd_get_coeffs
from physics.attitude import q_normalize, rotate_body_to_inertial, rotate_inertial_to_body, omega_to_qdot, q_to_dcm
from physics.forces import aerodynamic_forces_body
from physics.moments import aerodynamic_moments_body
from physics.integrator_adaptive import rkf45_adaptive
from physics import atmosphere as atm  # your ISA module

# State layout (len=13):
# y = [ u_i(3), pos_i(3), H_b(3),  q(4) ]   (inertial i / body b)

def _as_np(v):
    if isinstance(v, Vector):
        return v.to_numpy()
    return np.asarray(v, dtype=float)

def _gravity_inertial():
    g = getattr(cfg, "SEA_LEVEL_GRAVITY", 9.80665)
    return np.array([0.0, 0.0, -g], dtype=float)

def _air_density(alt_m):
    # Prefer your atmosphere module
    try:
        return float(atm.density(alt_m))
    except Exception:
        return 1.225

def _speed_of_sound(alt_m):
    try:
        return float(atm.speed_of_sound(alt_m))
    except Exception:
        return float(getattr(cfg, "SPEED_OF_SOUND", 343.0))

def compute_alpha_beta_from_body(v_body):
    u, v, w = v_body
    V = np.linalg.norm(v_body)
    if V < 1e-9:
        return 0.0, 0.0, V
    alpha = math.atan2(-w, u)  # adjust if body-z sign differs
    beta  = math.asin(np.clip(v / V, -1.0, 1.0))
    return alpha, beta, V

def make_state_derivative(aero_params=None, wind_inertial=None):
    """
    Build f(t,y). Uses cfg + atmosphere + cd_lookup every step.
    - aero_params: object/dict with m,d,S,Ix,Iy; if None, build from config.py
    - wind_inertial: None, constant np.array(3,), or callable t->np.array(3,)
    """
    # make default AP from config if needed
    if aero_params is None:
        class AP: pass
        ap = AP()
        ap.m  = getattr(cfg, "PROJECTILE_MASS", 43.7)
        ap.d  = getattr(cfg, "PROJECTILE_DIAMETER", 0.155)
        ap.S  = getattr(cfg, "REFERENCE_AREA", math.pi*ap.d**2/4.0)
        ap.Ix = getattr(cfg, "PROJECTILE_IX", 0.1444)
        ap.Iy = getattr(cfg, "PROJECTILE_IY", 1.7323)
        aero_params = ap

    if wind_inertial is None:
        # from config (Vector)
        wv = getattr(cfg, "WIND_VECTOR", Vector(0.0,0.0,0.0))
        const_wind = _as_np(wv)
        wind_inertial = lambda t: const_wind
    elif not callable(wind_inertial):
        w_const = _as_np(wind_inertial)
        wind_inertial = lambda t: w_const

    g_i = _gravity_inertial()
    x_body = np.array([1.0, 0.0, 0.0], dtype=float)

    def f(t, y):
        y = np.asarray(y, dtype=float)
        u_i  = y[0:3]
        pos  = y[3:6]
        H_b  = y[6:9]
        q    = q_normalize(y[9:13])

        alt = pos[2]
        rho = _air_density(alt)
        a_snd = _speed_of_sound(alt)

        v_air_i = u_i - wind_inertial(t)
        # BODY velocity
        v_b = rotate_inertial_to_body(q, v_air_i)
        alpha, beta, V = compute_alpha_beta_from_body(v_b)

        # Mach number from V
        mach = V / max(1e-9, a_snd)
        coeffs = cd_get_coeffs(mach)  # dict of cd0, cl_alpha, etc.

        # BODY forces
        F_b = aerodynamic_forces_body(v_b, x_body, H_b, aero_params, rho, coeffs=coeffs)

        # rotate to INERTIAL and add gravity
        D = q_to_dcm(q)
        F_i = D @ F_b + aero_params.m * g_i

        # linear acceleration (INERTIAL)
        u_dot = F_i / max(1e-12, aero_params.m)

        # BODY moments → dH/dt
        dHdt_b, _x_dot_b_unused = aerodynamic_moments_body(v_b, x_body, H_b, aero_params, rho, coeffs=coeffs)

        # BODY angular rate from H = I ω (axisymmetric)
        Ix = aero_params.Ix; Iy = aero_params.Iy
        omega_b = np.array([H_b[0]/max(1e-12, Ix),
                            H_b[1]/max(1e-12, Iy),
                            H_b[2]/max(1e-12, Iy)], dtype=float)

        # Quaternion derivative
        q_dot = omega_to_qdot(q, omega_b)

        # Kinematics
        pos_dot = u_i

        dy = np.zeros_like(y)
        dy[0:3] = u_dot
        dy[3:6] = pos_dot
        dy[6:9] = dHdt_b
        dy[9:13] = q_dot
        return dy

    return f

def integrate_6dof(aero_params, y0, t0, t_final, wind=None, integrator_opts=None, post_step_user=None):
    """
    Integrate with adaptive RKF45 (normalizes q each accepted step).
    Now stops exactly at ground impact (z == GROUND_LEVEL_Z) via event.
    """
    f = make_state_derivative(aero_params=aero_params, wind_inertial=wind)

    # defaults (tunable)
    opts = dict(h0=1e-4, atol=1e-8, rtol=1e-7, h_min=1e-9, h_max=0.02)
    if integrator_opts:
        opts.update(integrator_opts)

    # post-step wrapper: q normalize + optional user hook
    def post_step_check(t, y):
        y = np.asarray(y, dtype=float)
        if y.shape[0] >= 13:
            y[9:13] = q_normalize(y[9:13])
        # Clamp extremes (tweak as needed)
        y[0:3] = np.clip(y[0:3], -5000.0, 5000.0)
        y[6:9] = np.clip(y[6:9], -1e8, 1e8)
        if post_step_user is not None:
            y = post_step_user(t, y)
        if not np.all(np.isfinite(y)):
            raise RuntimeError(f"Non-finite state after post_step at t={t}")
        return y

    # Event: ground impact at z = ground_level
    ground = float(getattr(cfg, "GROUND_LEVEL_Z", 0.0))
    def event_fn(t, y):
        # y[5] is position z in our state layout
        return float(y[5] - ground)  # >0 above ground, <=0 at/under ground

    ts, ys = rkf45_adaptive(
        f, t0, y0, t_final,
        post_step=post_step_check,
        event_fn=event_fn, event_dir=-1, event_terminate=True, event_refine=True,
        **opts
    )

    # final safety normalization
    for i in range(ys.shape[0]):
        ys[i, 9:13] = q_normalize(ys[i, 9:13])
    return ts, ys
    """
    Integrate with adaptive RKF45 (already normalizes q each accepted step).
    """
    f = make_state_derivative(aero_params=aero_params, wind_inertial=wind)

    # defaults (tunable)
    h0 = 1e-4; atol = 1e-8; rtol = 1e-7; h_min=1e-9; h_max=0.02
    if integrator_opts:
        h0   = integrator_opts.get("h0", h0)
        atol = integrator_opts.get("atol", atol)
        rtol = integrator_opts.get("rtol", rtol)
        h_min= integrator_opts.get("h_min", h_min)
        h_max= integrator_opts.get("h_max", h_max)

    # post-step wrapper: q normalize + optional user hook (integrator does normalize too; this is extra safety)
    def post_step_check(t, y):
        y = np.asarray(y, dtype=float)
        if y.shape[0] >= 13:
            y[9:13] = q_normalize(y[9:13])
        # Clamp extremes (tweak as needed)
        y[0:3] = np.clip(y[0:3], -5000.0, 5000.0)
        y[6:9] = np.clip(y[6:9], -1e8, 1e8)
        if post_step_user is not None:
            y = post_step_user(t, y)
        if not np.all(np.isfinite(y)):
            raise RuntimeError(f"Non-finite state after post_step at t={t}")
        return y

    ts, ys = rkf45_adaptive(f, t0, y0, t_final,
                            h0=h0, atol=atol, rtol=rtol,
                            h_min=h_min, h_max=h_max,
                            post_step=post_step_check)
    # final safety normalization
    for i in range(ys.shape[0]):
        ys[i, 9:13] = q_normalize(ys[i, 9:13])
    return ts, ys
