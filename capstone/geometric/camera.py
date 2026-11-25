"""Pinhole camera model: x ~ K [R | t] X (world -> camera: Xc = R X + t)."""
import numpy as np


def intrinsics(f=800.0, cx=320.0, cy=240.0, fy=None, skew=0.0):
    return np.array([[f, skew, cx], [0, f if fy is None else fy, cy], [0, 0, 1.0]])


def look_at(C, target, up=(0, 0, 1.0)):
    C, target, up = (np.asarray(a, float) for a in (C, target, up))
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross(z, (0, 1.0, 0))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ C


def projection_matrix(K, R, t):
    return K @ np.hstack([R, t.reshape(3, 1)])


def project(P, X):
    Xh = np.hstack([X, np.ones((len(X), 1))])
    x = (P @ Xh.T).T
    return x[:, :2] / x[:, 2:]


def camera_center(P):
    _, _, Vt = np.linalg.svd(P)
    C = Vt[-1]
    return C[:3] / C[3]


def rot(axis, deg):
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    th = np.deg2rad(deg)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def sample_surface_points(surface, n, rng, extent=1.0, height_scale=1.0):
    xy = rng.uniform(-extent, extent, (n, 2))
    z = height_scale * surface.f(xy[:, 0], xy[:, 1])
    return np.column_stack([xy, z])
