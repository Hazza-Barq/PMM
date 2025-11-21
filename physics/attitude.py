# physics/attitude.py
import numpy as np

def q_normalize(q):
    """Normalize quaternion [qw,qx,qy,qz]."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n

def q_mul(q1, q2):
    """Hamilton product q = q1 * q2. Both as [qw,qx,qy,qz]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z], dtype=float)

def q_conj(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

def q_to_dcm(q):
    """
    Convert quaternion -> direction cosine matrix (DCM).
    D maps body -> inertial: v_inertial = D @ v_body
    """
    qw, qx, qy, qz = q_normalize(q)
    qx2 = qx*qx; qy2 = qy*qy; qz2 = qz*qz
    qwqx = qw*qx; qwqy = qw*qy; qwqz = qw*qz
    qxqy = qx*qy; qxqz = qx*qz; qyqz = qy*qz
    D = np.array([
        [qw*qw + qx2 - qy2 - qz2, 2*(qxqy - qwqz),         2*(qxqz + qwqy)],
        [2*(qxqy + qwqz),         qw*qw - qx2 + qy2 - qz2, 2*(qyqz - qwqx)],
        [2*(qxqz - qwqy),         2*(qyqz + qwqx),         qw*qw - qx2 - qy2 + qz2]
    ], dtype=float)
    return D

def dcm_to_q(D):
    """Convert a 3x3 DCM to quaternion [qw,qx,qy,qz] (numerically stable)."""
    D = np.asarray(D, dtype=float)
    tr = D[0,0] + D[1,1] + D[2,2]
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        qw = 0.25 / s
        qx = (D[2,1] - D[1,2]) * s
        qy = (D[0,2] - D[2,0]) * s
        qz = (D[1,0] - D[0,1]) * s
    else:
        if (D[0,0] > D[1,1]) and (D[0,0] > D[2,2]):
            s = 2.0 * np.sqrt(1.0 + D[0,0] - D[1,1] - D[2,2])
            qw = (D[2,1] - D[1,2]) / s
            qx = 0.25 * s
            qy = (D[0,1] + D[1,0]) / s
            qz = (D[0,2] + D[2,0]) / s
        elif D[1,1] > D[2,2]:
            s = 2.0 * np.sqrt(1.0 + D[1,1] - D[0,0] - D[2,2])
            qw = (D[0,2] - D[2,0]) / s
            qx = (D[0,1] + D[1,0]) / s
            qy = 0.25 * s
            qz = (D[1,2] + D[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + D[2,2] - D[0,0] - D[1,1])
            qw = (D[1,0] - D[0,1]) / s
            qx = (D[0,2] + D[2,0]) / s
            qy = (D[1,2] + D[2,1]) / s
            qz = 0.25 * s
    return q_normalize(np.array([qw, qx, qy, qz], dtype=float))

def rotate_body_to_inertial(q, v_body):
    D = q_to_dcm(q)
    return D.dot(v_body)

def rotate_inertial_to_body(q, v_inertial):
    D = q_to_dcm(q)
    return D.T.dot(v_inertial)

def omega_to_qdot(q, omega_body):
    """
    Convert angular velocity vector in body frame (p,q,r) to quaternion derivative.
    Returns q_dot (4,)
    """
    p, q_, r = omega_body
    Omega = np.array([0.0, p, q_, r], dtype=float)
    # we compute q_dot = 0.5 * Omega_quat * q (this ordering is consistent with earlier files)
    return 0.5 * q_mul(Omega, q)
