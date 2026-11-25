"""Structure from motion by factorisation under orthography (Tomasi & Kanade 1992)."""
import numpy as np

from .camera import rot


def orbit_cameras(F=20, sweep_deg=40.0, tilt_deg=25.0):
    """Rotations of an orthographic camera orbiting the object (rows = i, j, k axes)."""
    Rs = []
    for a in np.linspace(-sweep_deg / 2, sweep_deg / 2, F):
        Rs.append(rot((1, 0, 0), tilt_deg) @ rot((0, 0, 1), a))
    return Rs


def measurement_matrix(S, Rs, rng=None, sigma=0.0, t_scale=0.5):
    """W (2F x P): u = i.S + tx, v = j.S + ty (orthographic) plus image noise."""
    rows = []
    rng = np.random.default_rng(0) if rng is None else rng
    for R in Rs:
        t = rng.uniform(-t_scale, t_scale, 2)
        rows.append(R[0] @ S + t[0])
        rows.append(R[1] @ S + t[1])
    W = np.array(rows)
    if sigma > 0:
        W = W + rng.normal(0, sigma, W.shape)
    return W


def _g(a, b):
    return np.array([a[0] * b[0], a[0] * b[1] + a[1] * b[0], a[0] * b[2] + a[2] * b[0],
                     a[1] * b[1], a[1] * b[2] + a[2] * b[1], a[2] * b[2]])


def factorize(W):
    """Returns motion M (2F x 3), shape S (3 x P) after metric upgrade, and the
    singular values (rank-3 check)."""
    Wc = W - W.mean(1, keepdims=True)
    U, s, Vt = np.linalg.svd(Wc, full_matrices=False)
    M = U[:, :3] * np.sqrt(s[:3])
    S = np.sqrt(s[:3])[:, None] * Vt[:3]
    # metric upgrade: find L = Q Q^T with i^T L i = j^T L j = 1, i^T L j = 0
    F = M.shape[0] // 2
    G, c = [], []
    for f in range(F):
        i, j = M[2 * f], M[2 * f + 1]
        G += [_g(i, i), _g(j, j), _g(i, j)]
        c += [1, 1, 0]
    l = np.linalg.lstsq(np.array(G), np.array(c, float), rcond=None)[0]
    L = np.array([[l[0], l[1], l[2]], [l[1], l[3], l[4]], [l[2], l[4], l[5]]])
    w, V = np.linalg.eigh(L)
    Q = V @ np.diag(np.sqrt(np.clip(w, 1e-12, None)))
    return M @ Q, np.linalg.inv(Q) @ S, s


def recovered_rotations(M):
    Rs = []
    for f in range(M.shape[0] // 2):
        i, j = M[2 * f], M[2 * f + 1]
        k = np.cross(i, j)
        R = np.stack([i, j, k])
        U, _, Vt = np.linalg.svd(R)
        Rs.append(U @ Vt)
    return Rs
