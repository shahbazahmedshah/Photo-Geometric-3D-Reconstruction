"""Uncalibrated stereo: normalised 8-point F, essential matrix, (R, t), triangulation."""
import numpy as np

from .camera import skew


def _norm2d(x):
    c = x.mean(0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(x - c, axis=1))
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])


def eight_point(x1, x2):
    """Hartley-normalised 8-point algorithm: x2^T F x1 = 0, rank-2 enforced."""
    T1, T2 = _norm2d(x1), _norm2d(x2)
    a = (T1 @ np.column_stack([x1, np.ones(len(x1))]).T).T
    b = (T2 @ np.column_stack([x2, np.ones(len(x2))]).T).T
    A = np.column_stack([b[:, [0]] * a, b[:, [1]] * a, a])
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F)
    F = U @ np.diag([S[0], S[1], 0]) @ Vt
    F = T2.T @ F @ T1
    return F / np.linalg.norm(F)


def essential_from_F(F, K1, K2):
    E = K2.T @ F @ K1
    U, S, Vt = np.linalg.svd(E)
    return U @ np.diag([1, 1, 0]) @ Vt


def decompose_essential(E):
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    Wm = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1, R2 = U @ Wm @ Vt, U @ Wm.T @ Vt
    t = U[:, 2]
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def triangulate(P1, P2, x1, x2):
    """Linear (DLT) triangulation, one 4x4 SVD per correspondence."""
    X = np.empty((len(x1), 3))
    for i, (a, b) in enumerate(zip(x1, x2)):
        A = np.stack([a[0] * P1[2] - P1[0], a[1] * P1[2] - P1[1],
                      b[0] * P2[2] - P2[0], b[1] * P2[2] - P2[1]])
        _, _, Vt = np.linalg.svd(A)
        X[i] = Vt[-1, :3] / Vt[-1, 3]
    return X


def relative_pose(x1, x2, K1, K2):
    """Returns F, E, R, t (unit norm) and triangulated points in camera-1 frame."""
    F = eight_point(x1, x2)
    E = essential_from_F(F, K1, K2)
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    best = None
    for R, t in decompose_essential(E):
        P2 = K2 @ np.hstack([R, t.reshape(3, 1)])
        X = triangulate(P1, P2, x1, x2)
        front = np.sum((X[:, 2] > 0) & ((R @ X.T).T[:, 2] + t[2] > 0))
        if best is None or front > best[0]:
            best = (front, R, t, X)
    _, R, t, X = best
    return F, E, R, t, X


def epipolar_residual(F, x1, x2):
    """Mean symmetric point-to-epipolar-line distance in pixels."""
    a = np.column_stack([x1, np.ones(len(x1))])
    b = np.column_stack([x2, np.ones(len(x2))])
    l2 = (F @ a.T).T
    l1 = (F.T @ b.T).T
    num = np.abs(np.sum(b * l2, 1))
    d = num / np.hypot(l2[:, 0], l2[:, 1]) + num / np.hypot(l1[:, 0], l1[:, 1])
    return float(np.mean(d / 2))


__all__ = ["eight_point", "essential_from_F", "decompose_essential", "triangulate",
           "relative_pose", "epipolar_residual", "skew"]
