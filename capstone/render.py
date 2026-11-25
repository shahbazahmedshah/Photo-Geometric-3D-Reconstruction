"""Forward image-formation models (the 'Phi' operator of the plan).

All images are computed from closed-form expressions; nothing is captured.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from .scenes import texture


# ---------------------------------------------------------------- lighting
def light_dir(theta_deg, phi_deg):
    """Unit light direction: theta = angle from viewing axis (+z), phi = azimuth."""
    t, p = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])


def light_ring(k=4, theta_deg=35.0, phi0_deg=10.0):
    return np.stack([light_dir(theta_deg, phi0_deg + 360.0 * i / k) for i in range(k)])


def perturb_lights(S, deg, rng):
    """Rotate every light direction by `deg` about a random axis (calibration error)."""
    out = []
    for s in S:
        a = rng.normal(size=3)
        a -= a.dot(s) * s
        a /= np.linalg.norm(a)
        th = np.deg2rad(deg)
        out.append(s * np.cos(th) + np.cross(a, s) * np.sin(th))
    return np.array(out)


# ---------------------------------------------------------------- shading
def render_lambertian(n, albedo, s):
    """I = rho * max(0, n . s) for a distant point source s (unit vector)."""
    return albedo * np.clip(n @ s, 0.0, None)


def render_blinn_phong(n, albedo, s, ks=0.3, shininess=30.0, v=(0, 0, 1)):
    v = np.asarray(v, float)
    h = (s + v) / np.linalg.norm(s + v)
    spec = ks * np.clip(n @ h, 0, None) ** shininess * (n @ s > 0)
    return render_lambertian(n, albedo, s) + spec


def render_stack(n, albedo, S, model="lambertian"):
    fn = render_lambertian if model == "lambertian" else render_blinn_phong
    return np.stack([fn(n, albedo, s) for s in S])


def add_noise(I, sigma, rng, quantize=False):
    J = I + rng.normal(0.0, sigma, I.shape) if sigma > 0 else I.copy()
    if quantize:
        J = np.round(np.clip(J, 0, 1) * 255) / 255
    return J


# ---------------------------------------------------------------- defocus
class ThinLens:
    """Thin-lens camera; blur standard deviation in pixels for scene depth Z.

    b(Z) = (A v / 2) |1/F - 1/v - 1/Z| is the blur-circle radius on the sensor
    (aperture diameter A, sensor distance v, focal length F); sigma = kappa*b/pix.
    """

    def __init__(self, F=25.0, A=3.0, pix=0.01, kappa=0.5):
        self.F, self.A, self.pix, self.kappa = F, A, pix, kappa

    def v_for_focus(self, Zf):
        return 1.0 / (1.0 / self.F - 1.0 / Zf)

    def sigma(self, Z, v):
        b = 0.5 * self.A * v * np.abs(1.0 / self.F - 1.0 / v - 1.0 / Z)
        return self.kappa * b / self.pix


def spatially_varying_blur(T, sig, levels=48):
    """Blur image T with per-pixel Gaussian sigma by interpolating a blur stack."""
    smin, smax = float(sig.min()), float(sig.max())
    grid = np.linspace(smin, smax + 1e-9, levels)
    stack = np.stack([gaussian_filter(T, g, mode="reflect") if g > 1e-3 else T for g in grid])
    t = (sig - smin) / (grid[1] - grid[0])
    i0 = np.clip(np.floor(t).astype(int), 0, levels - 2)
    w = t - i0
    r, c = np.indices(T.shape)
    return (1 - w) * stack[i0, r, c] + w * stack[i0 + 1, r, c]


def render_defocus_pair(Z, lens, Zf1, Zf2, tex_scale=8.0):
    """Two images of a textured (pinhole-aligned) scene focused at Zf1 and Zf2."""
    H, W = Z.shape
    u = np.linspace(-1, 1, W)
    v = np.linspace(-1, 1, H)
    U, V = np.meshgrid(u, v)
    T = texture(U * tex_scale / 8, V * tex_scale / 8, seed=3, fmin=2, fmax=6)
    v1, v2 = lens.v_for_focus(Zf1), lens.v_for_focus(Zf2)
    s1, s2 = lens.sigma(Z, v1), lens.sigma(Z, v2)
    return T, spatially_varying_blur(T, s1), spatially_varying_blur(T, s2), (v1, v2), (s1, s2)


# ---------------------------------------------------------------- stereo / flow
def raycast_heightfield(cam, H, W, Zfun, iters=40):
    """Depth seen by each pixel of a pinhole camera looking along +Z at a surface
    Z = Zfun(X, Y). Solved by fixed-point iteration (converges for moderate slopes).
    cam: dict(K, C) with camera centre C, axes aligned with world (rectified rig).
    """
    K, C = cam["K"], cam["C"]
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    rx = (uu - K[0, 2]) / K[0, 0]
    ry = (vv - K[1, 2]) / K[1, 1]
    Z = np.full((H, W), float(np.mean(Zfun(np.array(C[0]), np.array(C[1])))))
    for _ in range(iters):
        X = C[0] + rx * (Z - C[2])
        Y = C[1] + ry * (Z - C[2])
        Z = Zfun(X, Y)
    X = C[0] + rx * (Z - C[2])
    Y = C[1] + ry * (Z - C[2])
    return X, Y, Z


def render_textured_view(cam, H, W, Zfun, tex_seed=11):
    X, Y, Z = raycast_heightfield(cam, H, W, Zfun)
    I = texture(X, Y, seed=tex_seed, fmin=3, fmax=12)
    return I, Z


def warp_pair(T, w, H, W, scale=1 / 40.0, iters=30):
    """I1(x) = T(x); I2(x + w(x)) = T(x) exactly (forward flow w in pixels).

    I2 at pixel y is T(x*) where x* solves x* + w(x*) = y (fixed point).
    Returns I1, I2 and the true flow on the I1 grid.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    I1 = T(xx * scale, yy * scale)
    px, py = xx.copy(), yy.copy()
    for _ in range(iters):
        u, v = w(px, py)
        px, py = xx - u, yy - v
    I2 = T(px * scale, py * scale)
    return I1, I2, np.stack(w(xx, yy), axis=-1)
