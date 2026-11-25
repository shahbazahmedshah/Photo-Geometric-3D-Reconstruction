"""Optical flow from brightness constancy  Ix u + Iy v + It = 0.

* Lucas-Kanade: local least squares over a Gaussian window (structure tensor).
* Horn-Schunck: global smoothness regularisation (Jacobi iterations).
Both run coarse-to-fine with image warping so motions of a few pixels work.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, zoom


def _grads(I1, I2w):
    A = 0.5 * (I1 + I2w)
    Iy, Ix = np.gradient(A)
    return Ix, Iy, I2w - I1


def warp(I, flow):
    H, W = I.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    return map_coordinates(I, [yy + flow[..., 1], xx + flow[..., 0]], order=3, mode="nearest")


def structure_tensor_min_eig(I, sigma=2.0):
    Iy, Ix = np.gradient(I)
    a = gaussian_filter(Ix * Ix, sigma)
    b = gaussian_filter(Ix * Iy, sigma)
    c = gaussian_filter(Iy * Iy, sigma)
    return 0.5 * (a + c - np.sqrt((a - c) ** 2 + 4 * b ** 2))


def lk_step(I1, I2w, flow, sigma=2.0, eps=1e-9):
    Ix, Iy, It = _grads(I1, I2w)
    a = gaussian_filter(Ix * Ix, sigma)
    b = gaussian_filter(Ix * Iy, sigma)
    c = gaussian_filter(Iy * Iy, sigma)
    d = gaussian_filter(Ix * It, sigma)
    e = gaussian_filter(Iy * It, sigma)
    det = a * c - b * b
    det = np.where(np.abs(det) < eps, np.inf, det)
    du = (-c * d + b * e) / det
    dv = (b * d - a * e) / det
    return flow + np.stack([du, dv], -1)


def _avg(u):
    k = np.pad(u, 1, mode="edge")
    return (k[:-2, 1:-1] + k[2:, 1:-1] + k[1:-1, :-2] + k[1:-1, 2:]) / 6 + \
           (k[:-2, :-2] + k[:-2, 2:] + k[2:, :-2] + k[2:, 2:]) / 12


def hs_step(I1, I2w, flow, alpha=0.05, iters=200):
    Ix, Iy, It = _grads(I1, I2w)
    u0, v0 = flow[..., 0], flow[..., 1]
    u, v = u0.copy(), v0.copy()
    den = alpha ** 2 + Ix ** 2 + Iy ** 2
    for _ in range(iters):
        ub, vb = _avg(u), _avg(v)
        r = (Ix * (ub - u0) + Iy * (vb - v0) + It) / den
        u, v = ub - Ix * r, vb - Iy * r
    return np.stack([u, v], -1)


def coarse_to_fine(I1, I2, step, levels=3, warps=5, **kw):
    pyr = [(I1, I2)]
    for _ in range(levels - 1):
        a, b = pyr[-1]
        pyr.append((gaussian_filter(a, 1.0)[::2, ::2], gaussian_filter(b, 1.0)[::2, ::2]))
    flow = np.zeros(pyr[-1][0].shape + (2,))
    for lvl in range(levels - 1, -1, -1):
        a, b = pyr[lvl]
        if flow.shape[:2] != a.shape:
            f = np.stack([zoom(flow[..., i], (a.shape[0] / flow.shape[0], a.shape[1] / flow.shape[1]),
                               order=1) for i in range(2)], -1)
            flow = 2.0 * f
        for _ in range(warps):
            flow = step(a, warp(b, flow), flow, **kw)
    return flow


def lucas_kanade(I1, I2, levels=3, warps=5, sigma=2.0):
    return coarse_to_fine(I1, I2, lk_step, levels, warps, sigma=sigma)


def horn_schunck(I1, I2, levels=3, warps=5, alpha=0.05, iters=200):
    return coarse_to_fine(I1, I2, hs_step, levels, warps, alpha=alpha, iters=iters)
