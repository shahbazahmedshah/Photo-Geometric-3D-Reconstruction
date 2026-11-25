"""Variational shape from shading (Ikeuchi & Horn 1981 style, gradient space).

Minimise  sum (E - R(p,q))^2 + lam * (|grad p|^2 + |grad q|^2)
with a Lambertian reflectance map of known source s and albedo rho. Boundary
gradients are given (here: from the analytic surface, as occluding-boundary /
known-boundary constraints would be in practice). Jacobi iteration:
    p <- pbar + (1/lam') (E - R(pbar,qbar)) dR/dp
"""
import numpy as np


def reflectance_map(p, q, s, rho=1.0):
    D = np.sqrt(1 + p ** 2 + q ** 2)
    num = -p * s[0] - q * s[1] + s[2]
    R = rho * num / D
    Rp = rho * (-s[0] / D - num * p / D ** 3)
    Rq = rho * (-s[1] / D - num * q / D ** 3)
    return R, Rp, Rq


def _avg4(a):
    b = a.copy()
    b[1:-1, 1:-1] = 0.25 * (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:])
    return b


def shape_from_shading(E, s, rho, p_bnd, q_bnd, lam=0.05, iters=6000, tol=1e-7):
    """E: image; s: light; p_bnd/q_bnd: arrays whose outer ring holds boundary values.
    Returns p, q and the number of iterations used."""
    p = np.zeros_like(E)
    q = np.zeros_like(E)
    ring = np.ones(E.shape, bool)
    ring[1:-1, 1:-1] = False
    p[ring], q[ring] = p_bnd[ring], q_bnd[ring]
    for it in range(iters):
        pb, qb = _avg4(p), _avg4(q)
        R, Rp, Rq = reflectance_map(pb, qb, s, rho)
        r = E - R
        pn = pb + r * Rp / (lam + Rp ** 2 + Rq ** 2)
        qn = qb + r * Rq / (lam + Rp ** 2 + Rq ** 2)
        pn[ring], qn[ring] = p_bnd[ring], q_bnd[ring]
        change = max(np.abs(pn - p).max(), np.abs(qn - q).max())
        p, q = pn, qn
        if change < tol:
            break
    return p, q, it + 1
