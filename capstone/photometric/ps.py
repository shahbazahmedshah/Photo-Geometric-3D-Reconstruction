"""Lambertian photometric stereo (Woodham 1980).

Per pixel: I_k = s_k . g with g = rho * n. With shadow masking each pixel uses
only lights where I_k > thr, so the solve is a batched weighted least squares.
"""
import numpy as np


def photometric_stereo(I, S, thr=1e-3):
    """I: (k, H, W) images, S: (k, 3) light directions.
    Returns normals (H, W, 3), albedo (H, W), valid mask (>= 3 lit images)."""
    k, H, W = I.shape
    w = (I > thr).astype(float)                                  # (k, H, W)
    A = np.einsum("khw,ki,kj->hwij", w, S, S)                    # (H, W, 3, 3)
    b = np.einsum("khw,khw,ki->hwi", w, I, S)                    # (H, W, 3)
    valid = w.sum(0) >= 3
    A[~valid] = np.eye(3)
    g = np.linalg.solve(A, b[..., None])[..., 0]
    rho = np.linalg.norm(g, axis=-1)
    n = g / np.maximum(rho, 1e-12)[..., None]
    n[~valid] = (0, 0, 1)
    return n, rho, valid


def analytic_g_covariance(S, sigma):
    """Covariance of the least-squares estimate of g under iid noise sigma."""
    return sigma ** 2 * np.linalg.inv(S.T @ S)
