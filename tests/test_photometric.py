"""Noise-free unit tests against analytic ground truth (single viewpoint)."""
import numpy as np
import pytest

from capstone.scenes import SURFACES, make_scene, pq_from_normals
from capstone.render import (light_dir, light_ring, render_lambertian, render_stack, ThinLens,
                             render_defocus_pair)
from capstone.photometric.ps import photometric_stereo, analytic_g_covariance
from capstone.photometric.integrate import poisson_integrate, frankot_chellappa
from capstone.photometric.sfs import shape_from_shading
from capstone.photometric.dfd import depth_from_defocus, relative_blur
from capstone.metrics import angular_error_deg, depth_rms


@pytest.mark.parametrize("name", list(SURFACES))
def test_analytic_gradients(name):
    s = SURFACES[name]
    x, y = np.random.default_rng(0).uniform(-0.95, 0.95, (2, 200))
    h = 1e-6
    assert np.allclose((s.f(x + h, y) - s.f(x - h, y)) / (2 * h), s.fx(x, y), atol=1e-6)
    assert np.allclose((s.f(x, y + h) - s.f(x, y - h)) / (2 * h), s.fy(x, y), atol=1e-6)


def test_renderer_closed_form():
    sc = make_scene("peaks")
    s = light_dir(30, 40)
    D = np.sqrt(1 + sc.p ** 2 + sc.q ** 2)
    closed = sc.albedo * np.clip((-sc.p * s[0] - sc.q * s[1] + s[2]) / D, 0, None)
    assert np.abs(render_lambertian(sc.n, sc.albedo, s) - closed).max() < 1e-12


@pytest.mark.parametrize("name", list(SURFACES))
def test_photometric_stereo_exact(name):
    sc = make_scene(name, textured_albedo=True)
    S = light_ring(8)
    n, rho, valid = photometric_stereo(render_stack(sc.n, sc.albedo, S), S)
    assert angular_error_deg(n, sc.n, valid) < 1e-4
    assert np.abs(rho - sc.albedo)[valid].max() < 1e-8


def test_ps_covariance_matches_theory():
    sc = make_scene("sphere", N=48)
    S = light_ring(8)
    clean = render_stack(sc.n, sc.albedo, S)
    rng = np.random.default_rng(0)
    gs = []
    for _ in range(200):
        n, r, _ = photometric_stereo(clean + rng.normal(0, 0.01, clean.shape), S)
        gs.append(n * r[..., None])
    emp = np.std(np.array(gs), 0).reshape(-1, 3).mean(0)
    pred = np.sqrt(np.diag(analytic_g_covariance(S, 0.01)))
    assert np.allclose(emp, pred, rtol=0.05)


@pytest.mark.parametrize("name", list(SURFACES))
def test_integration(name):
    sc = make_scene(name)
    assert depth_rms(poisson_integrate(sc.p, sc.q, sc.dx), sc.z) < 2e-4
    assert depth_rms(frankot_chellappa(sc.p, sc.q, sc.dx), sc.z) < 1e-3


def test_sfs_sphere():
    sc = make_scene("sphere", N=48)
    s = light_dir(20, 45)
    E = render_lambertian(sc.n, 1.0, s)
    p, q, _ = shape_from_shading(E, s, 1.0, sc.p, sc.q, lam=0.01, iters=20000)
    assert depth_rms(poisson_integrate(p, q, sc.dx), sc.z) < 0.03


def test_dfd_planes():
    lens = ThinLens()
    for Zc in (600, 750):
        Z = np.full((96, 96), float(Zc))
        _, g1, g2, (v1, v2), (s1, s2) = render_defocus_pair(Z, lens, 450, 1050)
        d, _ = relative_blur(g1, g2)
        assert abs(np.median(d[20:-20, 20:-20]) - (s1[0, 0] ** 2 - s2[0, 0] ** 2)) < 0.15
        Zh, _ = depth_from_defocus(g1, g2, lens, v1, v2, 500, 1000)
        assert abs(np.median(Zh[20:-20, 20:-20]) - Zc) < 10
