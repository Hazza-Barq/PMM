# physics/integrator_adaptive.py
"""
Adaptive RKF45 (Cash-Karp-like) integrator with post-step hook and quaternion normalization.
Now supports an optional event function to terminate early (e.g., ground impact).
"""
import numpy as np

# Try to import quaternion normalizer from attitude module (optional)
try:
    from physics.attitude import q_normalize
except Exception:
    def q_normalize(q):
        q = np.asarray(q, dtype=float)
        n = np.linalg.norm(q)
        if n == 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return q / n

def rkf45_adaptive(
    f, t0, y0, t_end,
    h0=1e-4, atol=1e-7, rtol=1e-6,
    h_min=1e-9, h_max=0.02, max_steps=200000,
    post_step=None,
    event_fn=None,         # NEW: callable (t,y)-> float (e.g., y[5]-ground_z)
    event_dir=-1,          # NEW: 0:any, +1: neg->pos, -1: pos->neg
    event_terminate=True,  # NEW: stop when event triggers
    event_refine=True      # NEW: linearly refine root between prev/curr accepted steps
):
    """
    Returns: (ts, ys) numpy arrays (ys shape (N, len(y0)))
    """
    t = float(t0)
    y = np.array(y0, dtype=float)
    h = float(h0)
    ts = [t]
    ys = [y.copy()]
    steps = 0
    safety = 0.9

    # Cash-Karp coefficients
    a2 = 1/5; a3 = 3/10; a4 = 3/5; a5 = 1; a6 = 7/8
    b21=1/5
    b31=3/40; b32=9/40
    b41=3/10; b42=-9/10; b43=6/5
    b51=-11/54; b52=5/2; b53=-70/27; b54=35/27
    b61=1631/55296; b62=175/512; b63=575/13824; b64=44275/110592; b65=253/4096
    c1=37/378; c3=250/621; c4=125/594; c6=512/1771
    c1s=2825/27648; c3s=18575/48384; c4s=13525/55296; c5s=277/14336; c6s=1/4

    if h <= 0:
        raise ValueError("h0 must be positive")

    # Event book-keeping
    prev_t = t
    prev_y = y.copy()
    prev_event_val = event_fn(t, y) if event_fn is not None else None

    while t < t_end and steps < max_steps:
        if not np.all(np.isfinite(y)):
            raise RuntimeError(f"Non-finite state at t={t}")

        if t + h > t_end:
            h = t_end - t

        # compute stages
        k1 = f(t, y)
        k2 = f(t + a2*h, y + h*(b21*k1))
        k3 = f(t + a3*h, y + h*(b31*k1 + b32*k2))
        k4 = f(t + a4*h, y + h*(b41*k1 + b42*k2 + b43*k3))
        k5 = f(t + a5*h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
        k6 = f(t + a6*h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))

        y_high = y + h*(c1*k1 + c3*k3 + c4*k4 + c6*k6)
        y_low  = y + h*(c1s*k1 + c3s*k3 + c4s*k4 + c5s*k5 + c6s*k6)

        err = np.abs(y_high - y_low)
        tol = atol + rtol * np.maximum(np.abs(y), np.abs(y_high))
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.max(err / tol)

        if not np.isfinite(ratio):
            h *= 0.2
            if h < h_min:
                raise RuntimeError("Step underflow in adaptive integrator (non-finite ratio)")
            steps += 1
            continue

        if ratio <= 1.0:
            # Accept step
            t_new = t + h
            y_new = y_high

            # normalize quaternion slice if present (assume q at indices 9..12)
            try:
                if y_new.shape[0] >= 13:
                    y_new[9:13] = q_normalize(y_new[9:13])
            except Exception:
                pass

            # user post-step hook
            if post_step is not None:
                y_new = post_step(t_new, y_new)
                if not isinstance(y_new, np.ndarray):
                    y_new = np.array(y_new, dtype=float)
                if not np.all(np.isfinite(y_new)):
                    raise RuntimeError(f"post_step produced non-finite state at t={t_new}")

            # ---- NEW: event handling on accepted step ----
            if event_fn is not None:
                v_prev = prev_event_val
                v_curr = event_fn(t_new, y_new)
                triggered = False
                if v_prev is not None and np.isfinite(v_prev) and np.isfinite(v_curr):
                    if event_dir == 0 and v_prev * v_curr <= 0: triggered = True
                    elif event_dir == -1 and (v_prev > 0 and v_curr <= 0): triggered = True
                    elif event_dir == +1 and (v_prev < 0 and v_curr >= 0): triggered = True

                if triggered and event_terminate:
                    if event_refine:
                        denom = (v_prev - v_curr)
                        w = (v_prev / denom) if abs(denom) > 1e-15 else 0.0
                        t_evt = prev_t + w * (t_new - prev_t)
                        y_evt = prev_y + w * (y_new - prev_y)
                        ts.append(t_evt)
                        ys.append(y_evt.copy())
                    else:
                        ts.append(t_new)
                        ys.append(y_new.copy())
                    return np.array(ts), np.array(ys)

                prev_event_val = v_curr
            # ----------------------------------------------

            # accept & adapt
            t = t_new
            y = y_new
            ts.append(t)
            ys.append(y.copy())
            prev_t = t
            prev_y = y.copy()
            h = min(h_max, max(h_min, safety * h * (1.0 / max(ratio, 1e-12))**0.2))
        else:
            # Reject step
            h = max(h_min, safety * h * (1.0 / ratio)**0.25)
            if h <= h_min:
                raise RuntimeError("Required step smaller than h_min; integration unstable")
        steps += 1

    if steps >= max_steps:
        raise RuntimeError("Max steps reached in rkf45_adaptive")

    return np.array(ts), np.array(ys)
