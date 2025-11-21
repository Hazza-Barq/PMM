# diagnostics/telemetry.py
import math, numpy as np
from dataclasses import dataclass, field

@dataclass
class Telemetry:
    t: list = field(default_factory=list)
    x: list = field(default_factory=list)
    y: list = field(default_factory=list)
    z: list = field(default_factory=list)
    speed: list = field(default_factory=list)
    mach: list = field(default_factory=list)
    alpha_deg: list = field(default_factory=list)
    cd: list = field(default_factory=list)
    cl: list = field(default_factory=list)
    F_mag: list = field(default_factory=list)   # |aero force|
    M_mag: list = field(default_factory=list)   # |aero moment|
    qnorm: list = field(default_factory=list)   # quaternion norm (should stay ~1)
    E_mech: list = field(default_factory=list)  # mgh + 0.5 m v^2 (monitor drift)

def make_post_step_recorder(aero_params, atmosphere, wind_fn, cd_lookup, ground_level=0.0):
    """
    Returns a function (t,y) -> y that appends diagnostics into a Telemetry object.
    Compute from accepted states (no extra RHS calls).
    """
    tel = Telemetry()

    def post_step(t, y):
        # unpack
        u = y[0:3]; pos = y[3:6]; H = y[6:9]; q = y[9:13]
        # environment
        z = pos[2]
        rho = atmosphere.density(z) if hasattr(atmosphere, "density") else 1.225
        a_snd = atmosphere.speed_of_sound(z) if hasattr(atmosphere, "speed_of_sound") else 343.0
        w_i = wind_fn(t) if callable(wind_fn) else np.zeros(3)
        v_air_i = u - w_i

        # body/angles
        from physics.attitude import rotate_inertial_to_body, q_normalize
        qn = q_normalize(q)
        v_b = rotate_inertial_to_body(qn, v_air_i)
        V = float(np.linalg.norm(v_b))
        alpha = math.atan2(-v_b[2], v_b[0]) if V > 1e-12 else 0.0
        M = float(V / max(1e-9, a_snd))

        coeffs_raw = cd_lookup(M) or {}
        coeffs = {str(k).lower(): float(v) for k, v in coeffs_raw.items()
          if isinstance(v, (int, float)) and np.isfinite(v)}       
        CD0 = coeffs.get("cd0", 0.0)
        CDa2= coeffs.get("cd_alpha2", 0.0)
        CL  = coeffs.get("cl_alpha", 0.0) + coeffs.get("cl_alpha3", 0.0)*(math.sin(alpha)**2)
        CD  = CD0 + CDa2*(math.sin(alpha)**2)

        # force & moment magnitudes (recompute with current state, cheap)
        from physics.forces import aerodynamic_forces_body
        from physics.moments import aerodynamic_moments_body
        x_body = np.array([1.0,0.0,0.0])
        F_b = aerodynamic_forces_body(v_b, x_body, H, aero_params, rho, coeffs)
        dHdt_b, _ = aerodynamic_moments_body(v_b, x_body, H, aero_params, rho, coeffs)

        # energies (mechanical only: translational + potential)
        m = getattr(aero_params, "m")
        g = getattr(__import__("config"), "SEA_LEVEL_GRAVITY", 9.80665)
        E = m* g * z + 0.5 * m * float(np.dot(u,u))

        # store
        tel.t.append(float(t))
        tel.x.append(float(pos[0])); tel.y.append(float(pos[1])); tel.z.append(float(pos[2]))
        tel.speed.append(float(np.linalg.norm(u))); tel.mach.append(M)
        tel.alpha_deg.append(math.degrees(alpha)); tel.cd.append(CD); tel.cl.append(CL)
        tel.F_mag.append(float(np.linalg.norm(F_b))); tel.M_mag.append(float(np.linalg.norm(dHdt_b)))
        tel.qnorm.append(float(np.linalg.norm(qn)))
        tel.E_mech.append(E)
        return y  # must return the (possibly normalized) state

    # attach the Telemetry instance so caller can read it after integrate
    post_step.telemetry = tel
    return post_step
