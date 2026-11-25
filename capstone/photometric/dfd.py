"""Two-image depth from defocus, spatial domain (Subbarao & Surya 1994).

For a Gaussian PSF and a locally cubic focused image f, the S-transform gives
    f = g_i - (sigma_i^2 / 2) lap(g_i)        (i = 1, 2)
so  sigma_1^2 - sigma_2^2 = 2 (g_1 - g_2) / lap(g),   g = (g_1 + g_2)/2.
(sigma = std. dev. of the Gaussian PSF; Subbarao's spread parameter is sqrt(2) sigma,
which is why the original paper writes the factor as 1/4.)
The ratio is estimated by local least squares over a window, then mapped to
depth through the monotone thin-lens relation Delta(Z) = sigma_1^2 - sigma_2^2.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, laplace, uniform_filter


def relative_blur(g1, g2, win=15, presmooth=0.5, min_energy=1e-6):
    a = gaussian_filter(g1, presmooth)
    b = gaussian_filter(g2, presmooth)
    L = laplace(0.5 * (a + b))
    num = uniform_filter(2 * (a - b) * L, win)
    den = uniform_filter(L * L, win)
    conf = den
    delta = num / np.maximum(den, min_energy)
    return delta, conf


def depth_lookup(delta, lens, v1, v2, Zmin, Zmax, n=4000):
    Zs = np.linspace(Zmin, Zmax, n)
    D = lens.sigma(Zs, v1) ** 2 - lens.sigma(Zs, v2) ** 2
    order = np.argsort(D)
    if not np.all(np.diff(D) > 0) and not np.all(np.diff(D) < 0):
        raise ValueError("Delta(Z) is not monotone over the depth range")
    return np.interp(delta, D[order], Zs[order])


def depth_from_defocus(g1, g2, lens, v1, v2, Zmin, Zmax, win=15):
    """Note: presmoothing adds the same variance to both images, so the
    difference sigma_1^2 - sigma_2^2 is unchanged."""
    delta, conf = relative_blur(g1, g2, win)
    return depth_lookup(delta, lens, v1, v2, Zmin, Zmax), conf
