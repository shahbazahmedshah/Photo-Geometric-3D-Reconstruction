"""Week 2 experiments: calibration, simple stereo, uncalibrated stereo, optical flow, SfM.

Usage: python -m experiments.exp_multi_view [--trials 100]
"""
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

from capstone.scenes import SURFACES, texture
from capstone.render import render_textured_view, add_noise, warp_pair
from capstone.geometric.camera import (intrinsics, look_at, projection_matrix, project,
                                       sample_surface_points)
from capstone.geometric.calib import calibration_object, calibrate
from capstone.geometric.stereo import block_match, disparity_to_depth
from capstone.geometric.epipolar import relative_pose, epipolar_residual
from capstone.geometric.flow import lucas_kanade, horn_schunck, structure_tensor_min_eig
from capstone.geometric.sfm import orbit_cameras, measurement_matrix, factorize
from capstone.metrics import (reprojection_rms, rotation_error_deg, endpoint_error, interior,
                              similarity_align)
from experiments.common import savefig, mean_std, dump, hide_axes, COLORS

NAMES = list(SURFACES)


# ------------------------------------------------------------------ calibration
def exp_calibration(trials, rng):
    K = intrinsics(800, 320, 240)
    R, t = look_at((22, 16, 14), (3.5, 3.5, 3.5))
    P = projection_matrix(K, R, t)
    Xw = calibration_object()
    x = project(P, Xw)
    sigmas = [0.0, 0.25, 0.5, 1.0, 2.0]
    keys = ["reproj_px", "f_rel_err_pct", "c_err_px", "rot_err_deg"]
    res = {k: [] for k in keys}
    for s in sigmas:
        acc = {k: [] for k in keys}
        for _ in range(trials if s > 0 else 1):
            xn = x + rng.normal(0, s, x.shape)
            Kh, Rh, th = calibrate(Xw, xn)
            acc["reproj_px"].append(reprojection_rms(projection_matrix(Kh, Rh, th), Xw, xn))
            acc["f_rel_err_pct"].append(100 * abs(Kh[0, 0] - K[0, 0]) / K[0, 0])
            acc["c_err_px"].append(float(np.hypot(Kh[0, 2] - K[0, 2], Kh[1, 2] - K[1, 2])))
            acc["rot_err_deg"].append(rotation_error_deg(Rh, R))
        for k in keys:
            res[k].append(mean_std(acc[k]))
    n = len(Xw)
    res["noise_floor_px"] = [float(s * np.sqrt(2 * (1 - 11 / (2 * n)))) for s in sigmas]
    res["sigmas"] = sigmas

    xn = x + rng.normal(0, 0.5, x.shape)
    Kh, Rh, th = calibrate(Xw, xn)
    xr = project(projection_matrix(Kh, Rh, th), Xw)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    ax[0].scatter(xn[:, 0], xn[:, 1], s=8, c=COLORS[0], label="observed (sigma 0.5 px)")
    ax[0].scatter(xr[:, 0], xr[:, 1], s=14, facecolors="none", edgecolors=COLORS[1], label="reprojected")
    ax[0].set_xlim(0, 640); ax[0].set_ylim(480, 0); ax[0].set_aspect("equal")
    ax[0].set_title("Virtual 3-plane target in the image")
    ax[0].legend(fontsize=7, loc="lower right")
    for c, k in zip(COLORS, ["reproj_px", "f_rel_err_pct"]):
        m = np.array(res[k])
        a = ax[1] if k == "reproj_px" else ax[2]
        a.errorbar(sigmas, m[:, 0], yerr=m[:, 1], marker="o", ms=3, capsize=2, color=c, label=k)
    ax[1].plot(sigmas, res["noise_floor_px"], "k--", lw=1, label="noise floor")
    ax[1].set_xlabel("pixel noise sigma"); ax[1].set_ylabel("reprojection RMS (px)"); ax[1].legend(fontsize=7)
    ax[2].set_xlabel("pixel noise sigma"); ax[2].set_ylabel("|f error| (%)")
    ax[1].set_title("Reprojection error"); ax[2].set_title("Focal-length error")
    savefig(fig, "fig09_calibration.png")
    return res


# ------------------------------------------------------------------ simple stereo
def _zfun(nm, Z0=10.0, A=1.5, L=4.0):
    s = SURFACES[nm]
    return lambda X, Y: Z0 - A * s.f(np.clip(X / L, -1, 1), np.clip(Y / L, -1, 1))


def exp_simple_stereo(trials, rng):
    H, W, f = 240, 320, 400.0
    K = intrinsics(f, W / 2, H / 2)
    out = {"baselines": [0.25, 0.5, 1.0, 1.5], "rms": {}, "theory": {}}
    for nm in NAMES:
        Zfun = _zfun(nm)
        IL, ZL = render_textured_view(dict(K=K, C=np.zeros(3)), H, W, Zfun)
        out["rms"][nm] = []
        out["theory"][nm] = []
        for b in out["baselines"]:
            IR, _ = render_textured_view(dict(K=K, C=np.array([b, 0, 0.0])), H, W, Zfun)
            d = f * b / ZL
            e = []
            for _ in range(max(1, trials // 25)):
                disp, valid = block_match(add_noise(IL, 0.01, rng), add_noise(IR, 0.01, rng),
                                          int(d.min()) - 3, int(d.max()) + 3, win=9)
                e.append(float(np.sqrt(np.mean((disparity_to_depth(disp, f, b) - ZL)[valid] ** 2))))
            out["rms"][nm].append(mean_std(e))
        # theory curve scaled from the b = 1 measurement: dZ = Z^2/(f b) * dd
        Zm = float(np.mean(ZL))
        dd = out["rms"][nm][2][0] * f * 1.0 / Zm ** 2
        out["theory"][nm] = [Zm ** 2 / (f * b) * dd for b in out["baselines"]]
        out.setdefault("disparity_err_px", {})[nm] = dd
    # figure for 'peaks'
    Zfun = _zfun("peaks")
    b = 1.0
    IL, ZL = render_textured_view(dict(K=K, C=np.zeros(3)), H, W, Zfun)
    IR, _ = render_textured_view(dict(K=K, C=np.array([b, 0, 0.0])), H, W, Zfun)
    d = f * b / ZL
    disp, valid = block_match(add_noise(IL, 0.01, rng), add_noise(IR, 0.01, rng),
                              int(d.min()) - 3, int(d.max()) + 3, win=9)
    Zh = np.where(valid, disparity_to_depth(disp, f, b), np.nan)
    fig, ax = plt.subplots(1, 5, figsize=(14, 2.9))
    ax[0].imshow(IL, cmap="gray"); ax[0].set_title("left view")
    ax[1].imshow(IR, cmap="gray"); ax[1].set_title("right view (b = 1)")
    ax[2].imshow(np.where(valid, disp, np.nan)); ax[2].set_title("disparity (px)")
    im = ax[3].imshow(Zh - ZL, cmap="coolwarm", vmin=-0.05, vmax=0.05); ax[3].set_title("depth error")
    fig.colorbar(im, ax=ax[3], shrink=0.8)
    for a in ax[:4]:
        hide_axes(a)
    for c, nm in zip(COLORS, NAMES):
        m = np.array(out["rms"][nm])
        ax[4].errorbar(out["baselines"], m[:, 0], yerr=m[:, 1], marker="o", ms=3, color=c, capsize=2, label=nm)
        ax[4].plot(out["baselines"], out["theory"][nm], ":", color=c, lw=1)
    ax[4].set_xlabel("baseline b"); ax[4].set_ylabel("depth RMS"); ax[4].set_yscale("log")
    ax[4].set_title("Error vs baseline (dotted: Z^2/(f b))"); ax[4].legend(fontsize=6)
    savefig(fig, "fig10_simple_stereo.png")
    return out


# ------------------------------------------------------------------ uncalibrated stereo
def _two_view(s, relief, n, rng):
    K = intrinsics(800, 320, 240)
    Xw = sample_surface_points(s, n, rng, height_scale=relief)
    R1, t1 = look_at((-1.5, -3.5, 3.5), (0, 0, 0))
    R2, t2 = look_at((1.5, -3.5, 4.0), (0, 0, 0))
    return K, Xw, (R1, t1), (R2, t2)


def exp_uncalibrated(trials, rng):
    sigmas = [0.0, 0.25, 0.5, 1.0]
    reliefs = [0.5, 1.0, 2.0, 3.0]
    keys = ["epi_px", "rot_deg", "tdir_deg", "pts_rel_rms"]
    out = {"sigmas": sigmas, "reliefs": reliefs, "by_relief": {}}
    s = SURFACES["peaks"]
    for relief in reliefs:
        rows = {k: [] for k in keys}
        for sg in sigmas:
            acc = {k: [] for k in keys}
            for _ in range(trials if sg > 0 else 1):
                K, Xw, (R1, t1), (R2, t2) = _two_view(s, relief, 200, rng)
                P1, P2 = projection_matrix(K, R1, t1), projection_matrix(K, R2, t2)
                x1 = project(P1, Xw) + rng.normal(0, sg, (len(Xw), 2))
                x2 = project(P2, Xw) + rng.normal(0, sg, (len(Xw), 2))
                F, E, R, t, X = relative_pose(x1, x2, K, K)
                Rrel = R2 @ R1.T
                trel = t2 - Rrel @ t1
                Xc1 = (R1 @ Xw.T).T + t1
                Xa, _ = similarity_align(X, Xc1)
                acc["epi_px"].append(epipolar_residual(F, x1, x2))
                acc["rot_deg"].append(rotation_error_deg(R, Rrel))
                acc["tdir_deg"].append(float(np.degrees(np.arccos(np.clip(abs(t @ trel) / np.linalg.norm(trel), -1, 1)))))
                acc["pts_rel_rms"].append(float(np.sqrt(np.mean(np.sum((Xa - Xc1) ** 2, 1))) / np.ptp(Xc1[:, 2])))
            for k in keys:
                rows[k].append(mean_std(acc[k]))
        out["by_relief"][str(relief)] = rows
    # figure: epipolar lines + error vs relief
    K, Xw, (R1, t1), (R2, t2) = _two_view(s, 2.0, 150, rng)
    P1, P2 = projection_matrix(K, R1, t1), projection_matrix(K, R2, t2)
    x1 = project(P1, Xw) + rng.normal(0, 0.5, (150, 2))
    x2 = project(P2, Xw) + rng.normal(0, 0.5, (150, 2))
    F, *_ = relative_pose(x1, x2, K, K)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    for a, pts, title in [(ax[0], x1, "view 1"), (ax[1], x2, "view 2 + epipolar lines")]:
        a.scatter(pts[:, 0], pts[:, 1], s=6, c=COLORS[0])
        a.set_xlim(0, 640); a.set_ylim(480, 0); a.set_aspect("equal"); a.set_title(title)
    for p in x1[::15]:
        l = F @ np.array([p[0], p[1], 1.0])
        xs = np.array([0, 640])
        ax[1].plot(xs, -(l[0] * xs + l[2]) / l[1], color=COLORS[1], lw=0.7)
    for c, sg_i in zip(COLORS, range(1, len(sigmas))):
        r = [out["by_relief"][str(rl)]["rot_deg"][sg_i][0] for rl in reliefs]
        ax[2].plot(reliefs, r, marker="o", ms=3, color=c, label=f"sigma = {sigmas[sg_i]} px")
    ax[2].set_xlabel("relief (height scale)"); ax[2].set_ylabel("rotation error (deg)")
    ax[2].set_yscale("log"); ax[2].set_title("Near-planar scenes are ill-conditioned"); ax[2].legend(fontsize=7)
    savefig(fig, "fig11_uncalibrated_stereo.png")
    return out


# ------------------------------------------------------------------ optical flow
def exp_flow():
    H = W = 128
    c = W / 2
    T = lambda x, y: texture(x, y, seed=5, fmin=0.5, fmax=3)  # noqa: E731
    fields = {
        "translation": lambda x, y: (np.full_like(x, 1.6), np.full_like(x, -0.9)),
        "rotation": lambda x, y: (-(y - c) * np.radians(2), (x - c) * np.radians(2)),
        "divergence": lambda x, y: (0.03 * (x - c), 0.03 * (y - c)),
        "sinusoidal": lambda x, y: (1.5 * np.sin(2 * np.pi * y / W), 1.0 * np.cos(2 * np.pi * x / W)),
    }
    out = {}
    m = interior((H, W), 12)
    fig, ax = plt.subplots(2, 4, figsize=(11, 5.6), layout="constrained")
    for j, (nm, w) in enumerate(fields.items()):
        I1, I2, ft = warp_pair(T, w, H, W)
        fl = lucas_kanade(I1, I2)
        fh = horn_schunck(I1, I2)
        out[nm] = {"max_motion_px": float(np.abs(ft).max()), "lk_epe": endpoint_error(fl, ft, m),
                   "hs_epe": endpoint_error(fh, ft, m)}
        ax[0, j].imshow(I1, cmap="gray")
        st = 10
        yy, xx = np.mgrid[0:H:st, 0:W:st]
        ax[0, j].quiver(xx, yy, ft[::st, ::st, 0], ft[::st, ::st, 1], color="#ffcc00", angles="xy",
                        scale_units="xy", scale=0.25, width=0.006)
        ax[0, j].set_title(f"{nm}: true flow")
        e = np.linalg.norm(fl - ft, axis=-1)
        im = ax[1, j].imshow(np.where(m, e, np.nan), cmap="magma", vmin=0, vmax=0.05)
        ax[1, j].set_title(f"LK EPE {out[nm]['lk_epe']:.4f} | HS {out[nm]['hs_epe']:.4f}")
        hide_axes(ax[0, j]); hide_axes(ax[1, j])
    fig.colorbar(im, ax=ax[1, :], shrink=0.7, label="LK endpoint error (px)")
    savefig(fig, "fig12_optical_flow.png")
    # aperture problem: 1-D grating
    G = lambda x, y: 0.5 + 0.4 * np.sin(2 * np.pi * 1.5 * x)  # noqa: E731
    I1, I2, ft = warp_pair(G, fields["translation"], H, W)
    lam_min = structure_tensor_min_eig(I1)
    fl = lucas_kanade(I1, I2)
    out["aperture"] = {"min_eig_grating": float(np.median(lam_min[m])),
                       "min_eig_texture": float(np.median(structure_tensor_min_eig(warp_pair(T, fields["translation"], H, W)[0])[m])),
                       "lk_v_error_px": float(np.nanmedian(np.abs(fl[..., 1] - ft[..., 1])[m])),
                       "lk_u_error_px": float(np.nanmedian(np.abs(fl[..., 0] - ft[..., 0])[m])),
                       "normal_flow_u_error_px": _normal_flow_error(I1, I2, ft, m)}
    return out


def _normal_flow_error(I1, I2, ft, m):
    """Only the flow component along the image gradient is observable (aperture problem).
    Normal flow u_n = -It grad(I) / |grad(I)|^2, refined coarse-to-fine like LK."""
    from capstone.geometric.flow import warp
    flow = np.zeros(I1.shape + (2,))
    for _ in range(10):
        I2w = warp(I2, flow)
        Iy, Ix = np.gradient(0.5 * (I1 + I2w))
        It = I2w - I1
        g2 = Ix ** 2 + Iy ** 2 + 1e-9
        flow = flow + np.stack([-It * Ix / g2, -It * Iy / g2], -1)
    ok = m & (np.abs(np.gradient(I1)[1]) > 1e-3)
    return float(np.median(np.abs(flow[..., 0] - ft[..., 0])[ok]))


# ------------------------------------------------------------------ SfM
def exp_sfm(trials, rng):
    s = SURFACES["peaks"]
    sigmas = [0.0, 0.002, 0.005, 0.01, 0.02]
    frames = [5, 10, 20, 40]
    out = {"sigmas": sigmas, "frames": frames, "shape_rel_rms": {}, "axis_err_deg": {}}
    for F in frames:
        sr, rr = [], []
        for sg in sigmas:
            a, b = [], []
            for _ in range(max(1, trials // 4) if sg > 0 else 1):
                S = sample_surface_points(s, 300, rng, height_scale=3.0).T
                Rs = orbit_cameras(F)
                W = measurement_matrix(S, Rs, rng, sg)
                M, Sh, sv = factorize(W)
                St = (S - S.mean(1, keepdims=True)).T
                Sa, (sc_, Ra, _) = similarity_align(Sh.T, St, allow_reflection=True)
                a.append(float(np.sqrt(np.mean(np.sum((Sa - St) ** 2, 1))) / np.ptp(St[:, 2])))
                # cameras in the aligned frame: M_true = M_hat Ra^T / s  (reflection allowed)
                Mt = M @ Ra.T / sc_
                ang = []
                for f_, R_ in enumerate(Rs):
                    for r_ in range(2):
                        v = Mt[2 * f_ + r_] / np.linalg.norm(Mt[2 * f_ + r_])
                        ang.append(np.degrees(np.arccos(np.clip(v @ R_[r_], -1, 1))))
                b.append(float(np.mean(ang)))
            sr.append(mean_std(a))
            rr.append(mean_std(b))
        out["shape_rel_rms"][str(F)] = sr
        out["axis_err_deg"][str(F)] = rr
    S = sample_surface_points(s, 300, rng, height_scale=3.0).T
    W = measurement_matrix(S, orbit_cameras(20), rng, 0.005)
    M, Sh, sv = factorize(W)
    St = (S - S.mean(1, keepdims=True)).T
    Sa, _ = similarity_align(Sh.T, St, allow_reflection=True)
    out["singular_values_norm"] = (sv[:8] / sv[0]).tolist()
    fig = plt.figure(figsize=(12, 3.6))
    a0 = fig.add_subplot(1, 3, 1)
    a0.semilogy(np.arange(1, 9), sv[:8] / sv[0], "o-", color=COLORS[0])
    a0.axvline(3.5, color="k", lw=0.8, ls="--")
    a0.set_title("Singular values of W (rank-3 gap)"); a0.set_xlabel("index")
    a1 = fig.add_subplot(1, 3, 2, projection="3d")
    a1.scatter(*St.T, s=4, c=COLORS[0], label="true")
    a1.scatter(*Sa.T, s=8, facecolors="none", edgecolors=COLORS[1], label="recovered")
    a1.set_title("Shape (20 frames, sigma = 0.005)"); a1.legend(fontsize=7)
    a2 = fig.add_subplot(1, 3, 3)
    for c, F in zip(COLORS, frames):
        mm = np.array(out["shape_rel_rms"][str(F)])
        a2.errorbar(sigmas, 100 * mm[:, 0], yerr=100 * mm[:, 1], marker="o", ms=3, capsize=2, color=c, label=f"F = {F}")
    a2.set_xlabel("track noise sigma"); a2.set_ylabel("shape RMS (% of depth range)")
    a2.set_title("Factorisation accuracy"); a2.legend(fontsize=7)
    savefig(fig, "fig13_sfm.png")
    return out


def run(trials=100, seed=1):
    rng = np.random.default_rng(seed)
    t = time.time()
    res = {"calibration": exp_calibration(trials, rng)}
    res["simple_stereo"] = exp_simple_stereo(trials, rng)
    res["uncalibrated"] = exp_uncalibrated(trials, rng)
    res["flow"] = exp_flow()
    res["sfm"] = exp_sfm(trials, rng)
    res["seconds"] = time.time() - t
    dump("multi_view.json", res)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    r = run(a.trials, a.seed)
    print(f"multi-view experiments done in {r['seconds']:.1f} s")
