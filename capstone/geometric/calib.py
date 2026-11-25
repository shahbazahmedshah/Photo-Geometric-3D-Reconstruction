"""Camera calibration: normalised DLT -> RQ decomposition -> nonlinear refinement."""
import numpy as np
from scipy.linalg import rq
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .camera import projection_matrix, project


def calibration_object(n=7, spacing=1.0):
    """Points on three mutually orthogonal planes (a 'corner' calibration target)."""
    g = spacing * np.arange(1, n + 1)
    A, B = np.meshgrid(g, g)
    a, b, o = A.ravel(), B.ravel(), np.zeros(A.size)
    return np.vstack([np.column_stack([o, a, b]),    # plane X = 0
                      np.column_stack([a, o, b]),    # plane Y = 0
                      np.column_stack([a, b, o])])   # plane Z = 0


def _normaliser(pts):
    d = pts.shape[1]
    c = pts.mean(0)
    s = np.sqrt(d) / np.mean(np.linalg.norm(pts - c, axis=1))
    T = np.eye(d + 1)
    T[:d, :d] *= s
    T[:d, d] = -s * c
    return T


def dlt(Xw, x):
    """Normalised Direct Linear Transform. Xw: (n, 3), x: (n, 2), n >= 6."""
    T, U = _normaliser(x), _normaliser(Xw)
    xn = (T @ np.hstack([x, np.ones((len(x), 1))]).T).T
    Xn = (U @ np.hstack([Xw, np.ones((len(Xw), 1))]).T).T
    rows = []
    for (u, v, _), X in zip(xn, Xn):
        rows.append(np.concatenate([X, np.zeros(4), -u * X]))
        rows.append(np.concatenate([np.zeros(4), X, -v * X]))
    _, _, Vt = np.linalg.svd(np.array(rows))
    Pn = Vt[-1].reshape(3, 4)
    return np.linalg.inv(T) @ Pn @ U


def decompose(P):
    """P -> K (K[2,2] = 1, positive diagonal), R (det +1), t."""
    if np.linalg.det(P[:, :3]) < 0:
        P = -P
    K, R = rq(P[:, :3])
    D = np.diag(np.sign(np.diag(K)))
    K, R = K @ D, D @ R
    t = np.linalg.solve(K, P[:, 3])
    return K / K[2, 2], R, t


def refine(K, R, t, Xw, x):
    """Minimise reprojection error over (fx, fy, cx, cy, skew, rvec, t)."""
    def unpack(v):
        K_ = np.array([[v[0], v[4], v[2]], [0, v[1], v[3]], [0, 0, 1]])
        return K_, Rotation.from_rotvec(v[5:8]).as_matrix(), v[8:11]

    def resid(v):
        K_, R_, t_ = unpack(v)
        return (project(projection_matrix(K_, R_, t_), Xw) - x).ravel()

    v0 = np.concatenate([[K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1]],
                         Rotation.from_matrix(R).as_rotvec(), t])
    sol = least_squares(resid, v0, method="lm")
    return unpack(sol.x)


def calibrate(Xw, x, do_refine=True):
    P = dlt(Xw, x)
    K, R, t = decompose(P)
    if do_refine:
        K, R, t = refine(K, R, t, Xw, x)
    return K, R, t
