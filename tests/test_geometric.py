"""Noise-free unit tests against analytic ground truth (multiple viewpoints + fusion)."""
import numpy as np
import pytest

from capstone.scenes import SURFACES, make_scene, texture
from capstone.render import render_textured_view, warp_pair, light_ring, render_stack
from capstone.geometric.camera import (intrinsics, look_at, projection_matrix, project,
                                       sample_surface_points)
from capstone.geometric.calib import calibration_object, calibrate
from capstone.geometric.stereo import block_match, disparity_to_depth
from capstone.geometric.epipolar import relative_pose, epipolar_residual
from capstone.geometric.flow import lucas_kanade, horn_schunck
from capstone.geometric.sfm import orbit_cameras, measurement_matrix, factorize
from capstone.photometric.ps import photometric_stereo
from capstone.fusion import fuse, sparse_points_from_stereo
from capstone.metrics import (rotation_error_deg, endpoint_error, interior, similarity_align,
                              depth_rms)


def test_calibration_exact():
    K = intrinsics(800, 320, 240)
    R, t = look_at((22, 16, 14), (3.5, 3.5, 3.5))
    Xw = calibration_object()
    Kh, Rh, th = calibrate(Xw, project(projection_matrix(K, R, t), Xw))
    assert np.allclose(Kh, K, atol=1e-6)
    assert rotation_error_deg(Rh, R) < 1e-6
    assert np.allclose(th, t, atol=1e-8)


def test_simple_stereo():
    H, W, f, b = 120, 160, 200.0, 1.0
    K = intrinsics(f, W / 2, H / 2)
    s = SURFACES["gaussian"]
    Zfun = lambda X, Y: 10 - 1.5 * s.f(np.clip(X / 4, -1, 1), np.clip(Y / 4, -1, 1))  # noqa: E731
    IL, ZL = render_textured_view(dict(K=K, C=np.zeros(3)), H, W, Zfun)
    IR, _ = render_textured_view(dict(K=K, C=np.array([b, 0, 0.0])), H, W, Zfun)
    d = f * b / ZL
    disp, valid = block_match(IL, IR, int(d.min()) - 2, int(d.max()) + 2, win=9)
    assert np.median(np.abs(disp - d)[valid]) < 0.15          # sub-pixel disparity
    assert np.median(np.abs(disparity_to_depth(disp, f, b) - ZL)[valid]) < 0.1


def test_uncalibrated_exact():
    rng = np.random.default_rng(0)
    K = intrinsics(800, 320, 240)
    Xw = sample_surface_points(SURFACES["peaks"], 100, rng, height_scale=3)
    R1, t1 = look_at((-1.5, -3.5, 3.5), (0, 0, 0))
    R2, t2 = look_at((1.5, -3.5, 4.0), (0, 0, 0))
    x1 = project(projection_matrix(K, R1, t1), Xw)
    x2 = project(projection_matrix(K, R2, t2), Xw)
    F, E, R, t, X = relative_pose(x1, x2, K, K)
    assert epipolar_residual(F, x1, x2) < 1e-8
    Rrel = R2 @ R1.T
    assert rotation_error_deg(R, Rrel) < 1e-6
    Xc1 = (R1 @ Xw.T).T + t1
    Xa, _ = similarity_align(X, Xc1)
    assert np.abs(Xa - Xc1).max() < 1e-8


@pytest.mark.parametrize("method", [lucas_kanade, horn_schunck])
def test_flow_translation(method):
    T = lambda x, y: texture(x, y, seed=5, fmin=0.5, fmax=3)  # noqa: E731
    w = lambda x, y: (np.full_like(x, 1.2), np.full_like(x, -0.7))  # noqa: E731
    I1, I2, ft = warp_pair(T, w, 96, 96)
    assert endpoint_error(method(I1, I2), ft, interior((96, 96), 12)) < 0.02


def test_sfm_exact():
    rng = np.random.default_rng(0)
    S = sample_surface_points(SURFACES["peaks"], 200, rng, height_scale=3).T
    W = measurement_matrix(S, orbit_cameras(12), rng, 0.0)
    M, Sh, sv = factorize(W)
    assert sv[3] / sv[0] < 1e-10
    St = (S - S.mean(1, keepdims=True)).T
    Sa, _ = similarity_align(Sh.T, St, allow_reflection=True)
    assert np.abs(Sa - St).max() < 1e-8


def test_fusion_exact_inputs():
    sc = make_scene("peaks", N=64)
    rng = np.random.default_rng(0)
    S = light_ring(4)
    n, _, _ = photometric_stereo(render_stack(sc.n, sc.albedo, S), S)
    idx, zh = sparse_points_from_stereo(sc, 0.02, 0.0, rng)
    z = fuse(n, sc.dx, idx, zh, lam=10)
    assert depth_rms(z, sc.z, align="none") < 2e-3
