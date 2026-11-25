"""Analytic test scenes.

Every surface is a height field z = f(x, y) on the square [-1, 1]^2 with
closed-form gradients p = dz/dx and q = dz/dy, so depth and normals are known
exactly. Array convention: rows index y, columns index x (meshgrid 'xy').
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Surface:
    name: str
    f: Callable          # z(x, y)
    fx: Callable         # dz/dx
    fy: Callable         # dz/dy


# ---------------------------------------------------------------- surfaces
def _sphere_cap(R=1.6):
    c = np.sqrt(R ** 2 - 2.0)   # makes z = 0 at the corners

    def f(x, y):
        return np.sqrt(R ** 2 - x ** 2 - y ** 2) - c

    def fx(x, y):
        return -x / np.sqrt(R ** 2 - x ** 2 - y ** 2)

    def fy(x, y):
        return -y / np.sqrt(R ** 2 - x ** 2 - y ** 2)
    return Surface("sphere", f, fx, fy)


def _gaussian(a=0.6, s=0.35):
    def f(x, y):
        return a * np.exp(-(x ** 2 + y ** 2) / (2 * s ** 2))

    def fx(x, y):
        return -x / s ** 2 * f(x, y)

    def fy(x, y):
        return -y / s ** 2 * f(x, y)
    return Surface("gaussian", f, fx, fy)


def _peaks(k=3.0, a=0.04):
    """MATLAB 'peaks' mapped from [-3,3]^2 to [-1,1]^2 and scaled by a."""
    def P(X, Y):
        return (3 * (1 - X) ** 2 * np.exp(-X ** 2 - (Y + 1) ** 2)
                - 10 * (X / 5 - X ** 3 - Y ** 5) * np.exp(-X ** 2 - Y ** 2)
                - 1 / 3 * np.exp(-(X + 1) ** 2 - Y ** 2))

    def PX(X, Y):
        e1 = np.exp(-X ** 2 - (Y + 1) ** 2)
        e2 = np.exp(-X ** 2 - Y ** 2)
        e3 = np.exp(-(X + 1) ** 2 - Y ** 2)
        g = X / 5 - X ** 3 - Y ** 5
        return (-6 * (1 - X) * e1 + 3 * (1 - X) ** 2 * e1 * (-2 * X)
                - 10 * (1 / 5 - 3 * X ** 2) * e2 - 10 * g * e2 * (-2 * X)
                - 1 / 3 * e3 * (-2 * (X + 1)))

    def PY(X, Y):
        e1 = np.exp(-X ** 2 - (Y + 1) ** 2)
        e2 = np.exp(-X ** 2 - Y ** 2)
        e3 = np.exp(-(X + 1) ** 2 - Y ** 2)
        g = X / 5 - X ** 3 - Y ** 5
        return (3 * (1 - X) ** 2 * e1 * (-2 * (Y + 1))
                - 10 * (-5 * Y ** 4) * e2 - 10 * g * e2 * (-2 * Y)
                - 1 / 3 * e3 * (-2 * Y))

    return Surface("peaks",
                   lambda x, y: a * P(k * x, k * y),
                   lambda x, y: a * k * PX(k * x, k * y),
                   lambda x, y: a * k * PY(k * x, k * y))


def _ripple(a=0.1):
    def f(x, y):
        return a * np.sin(2 * np.pi * x) * np.cos(np.pi * y)

    def fx(x, y):
        return a * 2 * np.pi * np.cos(2 * np.pi * x) * np.cos(np.pi * y)

    def fy(x, y):
        return -a * np.pi * np.sin(2 * np.pi * x) * np.sin(np.pi * y)
    return Surface("ripple", f, fx, fy)


SURFACES = {s.name: s for s in (_sphere_cap(), _gaussian(), _peaks(), _ripple())}


# ---------------------------------------------------------------- grid data
@dataclass
class Scene:
    x: np.ndarray        # 1-D coordinates (columns)
    y: np.ndarray        # 1-D coordinates (rows)
    X: np.ndarray
    Y: np.ndarray
    z: np.ndarray
    p: np.ndarray
    q: np.ndarray
    n: np.ndarray        # (H, W, 3) unit normals
    albedo: np.ndarray
    dx: float
    surface: Surface


def normals_from_pq(p, q):
    n = np.stack([-p, -q, np.ones_like(p)], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def pq_from_normals(n, eps=1e-6):
    nz = np.clip(n[..., 2], eps, None)
    return -n[..., 0] / nz, -n[..., 1] / nz


def texture(x, y, seed=7, n_waves=12, fmin=1.0, fmax=8.0):
    """Smooth procedural texture in [0.2, 1]: a deterministic sum of sinusoids.

    It is a closed-form function of continuous (x, y), so it can be evaluated
    exactly at any warped or projected coordinate (used by stereo, flow, DfD).
    """
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(fmin, fmax, n_waves)
    angles = rng.uniform(0, np.pi, n_waves)
    phases = rng.uniform(0, 2 * np.pi, n_waves)
    t = np.zeros(np.broadcast(x, y).shape)
    for f, a, ph in zip(freqs, angles, phases):
        t += np.sin(2 * np.pi * f * (np.cos(a) * x + np.sin(a) * y) + ph)
    t /= n_waves ** 0.5
    return 0.6 + 0.4 * np.tanh(t)


def make_scene(name="sphere", N=128, textured_albedo=False):
    s = SURFACES[name]
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    z, p, q = s.f(X, Y), s.fx(X, Y), s.fy(X, Y)
    alb = texture(X, Y) if textured_albedo else np.full_like(z, 0.8)
    return Scene(x, y, X, Y, z, p, q, normals_from_pq(p, q), alb, x[1] - x[0], s)
