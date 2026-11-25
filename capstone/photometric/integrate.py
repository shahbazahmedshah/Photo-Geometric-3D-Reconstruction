"""Normal / gradient-field integration: gradients (p, q) -> depth z.

* Frankot-Chellappa (1988): projection onto integrable Fourier basis.
* Poisson least squares: min ||Dx z - p||^2 + ||Dy z - q||^2 with sparse
  finite-difference operators (the same operators are reused by fusion.py).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def frankot_chellappa(p, q, dx=1.0, mirror=True):
    H, W = p.shape
    if mirror:   # even/odd reflection removes the periodic-boundary artefact
        p = np.block([[p, -p[:, ::-1]], [p[::-1, :], -p[::-1, ::-1]]])
        q = np.block([[q, q[:, ::-1]], [-q[::-1, :], -q[::-1, ::-1]]])
    h, w = p.shape
    wx = 2 * np.pi * np.fft.fftfreq(w, d=dx)[None, :]
    wy = 2 * np.pi * np.fft.fftfreq(h, d=dx)[:, None]
    P, Q = np.fft.fft2(p), np.fft.fft2(q)
    den = wx ** 2 + wy ** 2
    den[0, 0] = 1.0
    Z = (-1j * wx * P - 1j * wy * Q) / den
    Z[0, 0] = 0.0
    z = np.real(np.fft.ifft2(Z))
    return z[:H, :W]


def grad_ops(H, W, dx=1.0):
    """Forward-difference operators on a row-major H x W grid.
    Dx: (H*(W-1), H*W), Dy: ((H-1)*W, H*W)."""
    def d1(n):
        return sp.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n)) / dx
    Dx = sp.kron(sp.eye(H), d1(W), format="csr")
    Dy = sp.kron(d1(H), sp.eye(W), format="csr")
    return Dx, Dy


def midpoint_gradients(p, q):
    """Gradients at the edge mid-points used by the forward differences."""
    return 0.5 * (p[:, 1:] + p[:, :-1]), 0.5 * (q[1:, :] + q[:-1, :])


def solve_normal_equations(A, b, pin_weight=1e-6):
    """Sparse least squares via normal equations + tiny ridge on one pixel (fixes the
    constant null space). Exact and fast for grids up to a few hundred pixels."""
    n = A.shape[1]
    AtA = (A.T @ A).tocsc()
    AtA = AtA + sp.csc_matrix(([pin_weight], ([0], [0])), shape=(n, n))
    return spsolve(AtA, A.T @ b)


def poisson_integrate(p, q, dx=1.0, mask=None):
    H, W = p.shape
    Dx, Dy = grad_ops(H, W, dx)
    pm, qm = midpoint_gradients(p, q)
    A = sp.vstack([Dx, Dy]).tocsr()
    b = np.concatenate([pm.ravel(), qm.ravel()])
    if mask is not None:   # drop equations touching invalid pixels
        mx = (mask[:, 1:] & mask[:, :-1]).ravel()
        my = (mask[1:, :] & mask[:-1, :]).ravel()
        keep = np.concatenate([mx, my])
        A, b = A[keep], b[keep]
    z = solve_normal_equations(A, b).reshape(H, W)
    return z - z.mean()
