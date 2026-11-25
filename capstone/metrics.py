"""Error metrics against analytic ground truth."""
import numpy as np


def angular_error_deg(n_est, n_true, mask=None):
    c = np.clip(np.sum(n_est * n_true, axis=-1), -1, 1)
    e = np.degrees(np.arccos(c))
    return float(np.mean(e if mask is None else e[mask]))


def depth_rms(z_est, z_true, mask=None, align="offset"):
    """RMS depth error. align: 'none', 'offset' (best constant), 'scale_offset'."""
    m = np.ones(z_true.shape, bool) if mask is None else mask
    a, b = z_est[m], z_true[m]
    if align == "offset":
        a = a - np.mean(a - b)
    elif align == "scale_offset":
        A = np.stack([a, np.ones_like(a)], 1)
        s, c = np.linalg.lstsq(A, b, rcond=None)[0]
        a = s * a + c
    return float(np.sqrt(np.mean((a - b) ** 2)))


def reprojection_rms(P, Xw, x_obs):
    Xh = np.hstack([Xw, np.ones((len(Xw), 1))])
    x = (P @ Xh.T).T
    x = x[:, :2] / x[:, 2:]
    return float(np.sqrt(np.mean(np.sum((x - x_obs) ** 2, 1))))


def rotation_error_deg(R_est, R_true):
    c = (np.trace(R_est.T @ R_true) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def endpoint_error(flow_est, flow_true, mask=None):
    e = np.linalg.norm(flow_est - flow_true, axis=-1)
    return float(np.mean(e if mask is None else e[mask]))


def similarity_align(A, B, allow_reflection=False):
    """Umeyama: find s, R, t minimising ||s R A + t - B|| (A, B: (n, 3)).
    Returns aligned A and (s, R, t)."""
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(B0.T @ A0)
    D = np.eye(3)
    if not allow_reflection and np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / np.sum(A0 ** 2)
    t = mb - s * R @ ma
    return (s * (R @ A.T)).T + t, (s, R, t)


def interior(shape, border):
    m = np.zeros(shape, bool)
    m[border:-border, border:-border] = True
    return m
