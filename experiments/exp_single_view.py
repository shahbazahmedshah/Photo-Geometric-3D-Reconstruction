"""Week 1 experiments: forward model, photometric stereo, integration, SfS, DfD.

Usage: python -m experiments.exp_single_view [--trials 100]
"""
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

from capstone.scenes import SURFACES, make_scene, normals_from_pq, pq_from_normals
from capstone.render import (light_dir, light_ring, render_stack, render_lambertian, add_noise,
                             ThinLens, render_defocus_pair)
from capstone.photometric.ps import photometric_stereo, analytic_g_covariance
from capstone.photometric.integrate import frankot_chellappa, poisson_integrate
from capstone.photometric.sfs import shape_from_shading, reflectance_map
from capstone.photometric.dfd import depth_from_defocus
from capstone.metrics import angular_error_deg, depth_rms, interior
from experiments.common import savefig, normal_rgb, mean_std, dump, hide_axes, COLORS

NAMES = list(SURFACES)


def exp_scenes():
    fig, ax = plt.subplots(2, 4, figsize=(10, 5))
    worst = 0.0
    for j, nm in enumerate(NAMES):
        sc = make_scene(nm)
        im = ax[0, j].imshow(sc.z, extent=(-1, 1, -1, 1), origin="lower")
        ax[0, j].set_title(f"{nm}: z(x,y)")
        fig.colorbar(im, ax=ax[0, j], shrink=0.75)
        ax[1, j].imshow(normal_rgb(sc.n), extent=(-1, 1, -1, 1), origin="lower")
        ax[1, j].set_title("normal map (RGB = n)")
        for a in ax[:, j]:
            hide_axes(a)
        # O1 check: renderer equals closed form
        s = light_dir(30, 40)
        I = render_lambertian(sc.n, sc.albedo, s)
        D = np.sqrt(1 + sc.p ** 2 + sc.q ** 2)
        closed = sc.albedo * np.clip((-sc.p * s[0] - sc.q * s[1] + s[2]) / D, 0, None)
        worst = max(worst, float(np.abs(I - closed).max()))
    savefig(fig, "fig01_scenes.png")

    # reflectance map contours
    pp, qq = np.meshgrid(np.linspace(-2, 2, 400), np.linspace(-2, 2, 400))
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.3))
    for a, (th, ph) in zip(ax, [(0, 0), (30, 45), (60, 45)]):
        s = light_dir(th, ph)
        R, _, _ = reflectance_map(pp, qq, s)
        cs = a.contour(pp, qq, np.clip(R, 0, None), levels=np.linspace(0.05, 0.95, 10), cmap="viridis")
        a.clabel(cs, fontsize=6, fmt="%.1f")
        a.set_title(f"R(p,q), source theta={th} deg")
        a.set_xlabel("p")
        a.set_ylabel("q")
        a.set_aspect("equal")
    savefig(fig, "fig02_reflectance_map.png")
    return {"renderer_max_abs_diff": worst}


def exp_ps(trials, rng):
    out = {}
    # qualitative figure
    sc = make_scene("peaks", textured_albedo=True)
    S = light_ring(4)
    I = add_noise(render_stack(sc.n, sc.albedo, S), 0.01, rng)
    n, rho, valid = photometric_stereo(I, S)
    err = np.degrees(np.arccos(np.clip(np.sum(n * sc.n, -1), -1, 1)))
    fig, ax = plt.subplots(2, 4, figsize=(10, 5))
    for k in range(4):
        ax[0, k].imshow(I[k], cmap="gray", origin="lower")
        ax[0, k].set_title(f"I{k + 1}, s = ({S[k, 0]:.2f},{S[k, 1]:.2f},{S[k, 2]:.2f})")
    ax[1, 0].imshow(normal_rgb(sc.n), origin="lower"); ax[1, 0].set_title("true normals")
    ax[1, 1].imshow(normal_rgb(n), origin="lower"); ax[1, 1].set_title("PS normals")
    ax[1, 2].imshow(rho, cmap="gray", origin="lower"); ax[1, 2].set_title("PS albedo")
    im = ax[1, 3].imshow(err, origin="lower", cmap="magma"); ax[1, 3].set_title("angular error (deg)")
    fig.colorbar(im, ax=ax[1, 3], shrink=0.75)
    for a in ax.ravel():
        hide_axes(a)
    savefig(fig, "fig03_ps_example.png")

    # noise-free accuracy on all surfaces (O2)
    out["noise_free_deg"] = {}
    for nm in NAMES:
        sc = make_scene(nm)
        S = light_ring(8)
        n, _, v = photometric_stereo(render_stack(sc.n, sc.albedo, S), S)
        out["noise_free_deg"][nm] = angular_error_deg(n, sc.n, v)

    # Monte Carlo: error vs noise for k lights (N = 64 grid, per-pixel independent)
    sigmas = [0.0, 0.005, 0.01, 0.02, 0.05]
    ks = [3, 4, 8, 12]
    sc = make_scene("sphere", N=64)
    curves = {}
    for k in ks:
        S = light_ring(k)
        clean = render_stack(sc.n, sc.albedo, S)
        curves[k] = []
        for s in sigmas:
            e = [angular_error_deg(photometric_stereo(add_noise(clean, s, rng), S)[0], sc.n)
                 for _ in range(trials if s > 0 else 1)]
            curves[k].append(mean_std(e))
    out["noise_curves_deg"] = {str(k): v for k, v in curves.items()}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for c, k in zip(COLORS, ks):
        m = np.array(curves[k])
        ax[0].errorbar(sigmas, m[:, 0], yerr=m[:, 1], marker="o", ms=3, capsize=2, color=c, label=f"k = {k}")
    ax[0].set_xlabel("image noise sigma")
    ax[0].set_ylabel("mean angular error (deg)")
    ax[0].set_title(f"PS error vs noise ({trials} trials/point)")
    ax[0].legend()

    # conditioning: light elevation sweep at fixed noise
    thetas = [5, 10, 15, 25, 35, 50, 65]
    cond, errs = [], []
    for th in thetas:
        S = light_ring(4, theta_deg=th)
        clean = render_stack(sc.n, sc.albedo, S)
        cond.append(float(np.linalg.cond(S)))
        errs.append(mean_std([angular_error_deg(photometric_stereo(add_noise(clean, 0.01, rng), S)[0], sc.n)
                              for _ in range(trials)]))
    errs = np.array(errs)
    ax[1].errorbar(cond, errs[:, 0], yerr=errs[:, 1], marker="o", ms=3, capsize=2, color=COLORS[0])
    for c_, th, e in zip(cond, thetas, errs[:, 0]):
        ax[1].annotate(f"{th} deg", (c_, e), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("cond(S)  (labels: light elevation)")
    ax[1].set_ylabel("mean angular error (deg)")
    ax[1].set_title("Conditioning of the light matrix (sigma = 0.01, k = 4)")
    savefig(fig, "fig04_ps_noise_conditioning.png")
    out["conditioning"] = {"theta_deg": thetas, "cond": cond, "err_deg": errs.tolist()}

    # analytic vs empirical covariance of g (no shadows on the sphere for k = 8)
    S = light_ring(8)
    clean = render_stack(sc.n, sc.albedo, S)
    sig = 0.01
    gs = []
    for _ in range(max(trials, 50)):
        n_, r_, _ = photometric_stereo(add_noise(clean, sig, rng), S)
        gs.append(n_ * r_[..., None])
    emp = np.std(np.array(gs), axis=0).reshape(-1, 3).mean(0)
    pred = np.sqrt(np.diag(analytic_g_covariance(S, sig)))
    out["g_std_empirical"] = emp
    out["g_std_analytic"] = pred
    return out


def exp_integration(trials, rng):
    out = {"noise_free_rms": {}}
    for nm in NAMES:
        sc = make_scene(nm)
        out["noise_free_rms"][nm] = {
            "poisson": depth_rms(poisson_integrate(sc.p, sc.q, sc.dx), sc.z),
            "frankot_chellappa": depth_rms(frankot_chellappa(sc.p, sc.q, sc.dx), sc.z)}
    sigmas = [0.0, 0.005, 0.01, 0.02, 0.05]
    sc = make_scene("gaussian", N=96)
    S = light_ring(4)
    clean = render_stack(sc.n, sc.albedo, S)
    res = {"poisson": [], "frankot_chellappa": []}
    for s in sigmas:
        ep, ef = [], []
        for _ in range(max(1, trials // 4) if s > 0 else 1):
            n, _, _ = photometric_stereo(add_noise(clean, s, rng), S)
            p, q = pq_from_normals(n)
            ep.append(depth_rms(poisson_integrate(p, q, sc.dx), sc.z))
            ef.append(depth_rms(frankot_chellappa(p, q, sc.dx), sc.z))
        res["poisson"].append(mean_std(ep))
        res["frankot_chellappa"].append(mean_std(ef))
    out["rms_vs_noise"] = res
    # visual: integrated depth & error for one noisy trial
    n, _, _ = photometric_stereo(add_noise(clean, 0.02, rng), S)
    p, q = pq_from_normals(n)
    zp = poisson_integrate(p, q, sc.dx)
    zp += np.mean(sc.z - zp)
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.9))
    ax[0].imshow(sc.z, origin="lower"); ax[0].set_title("true z")
    ax[1].imshow(zp, origin="lower"); ax[1].set_title("Poisson z (sigma = 0.02)")
    im = ax[2].imshow(zp - sc.z, origin="lower", cmap="coolwarm"); ax[2].set_title("error: smooth drift")
    fig.colorbar(im, ax=ax[2], shrink=0.8)
    for a in ax[:3]:
        hide_axes(a)
    for c, (k, v) in zip(COLORS, res.items()):
        m = np.array(v)
        ax[3].errorbar(sigmas, m[:, 0], yerr=m[:, 1], marker="o", ms=3, capsize=2, color=c, label=k)
    ax[3].set_xlabel("image noise sigma")
    ax[3].set_ylabel("depth RMS")
    ax[3].legend(fontsize=7)
    ax[3].set_title("Integration error vs noise")
    savefig(fig, "fig05_integration.png")
    return out


def exp_sfs():
    out = {}
    s = light_dir(20, 45)
    fig, ax = plt.subplots(3, 4, figsize=(10, 7.2), layout="constrained")
    for j, nm in enumerate(NAMES):
        sc = make_scene(nm, N=64)
        E = render_lambertian(sc.n, 1.0, s)
        t0 = time.time()
        p, q, it = shape_from_shading(E, s, 1.0, sc.p, sc.q, lam=0.01, iters=20000)
        z_sfs = poisson_integrate(p, q, sc.dx)
        S = light_ring(4)
        n_ps, _, _ = photometric_stereo(render_stack(sc.n, 1.0, S), S)
        z_ps = poisson_integrate(*pq_from_normals(n_ps), sc.dx)
        n_sfs = normals_from_pq(p, q)
        out[nm] = {"sfs_ang_deg": angular_error_deg(n_sfs, sc.n), "sfs_rms": depth_rms(z_sfs, sc.z),
                   "ps_ang_deg": angular_error_deg(n_ps, sc.n), "ps_rms": depth_rms(z_ps, sc.z),
                   "iterations": it, "seconds": time.time() - t0}
        ax[0, j].imshow(E, cmap="gray", origin="lower"); ax[0, j].set_title(f"{nm}: single image")
        ax[1, j].imshow(z_sfs, origin="lower"); ax[1, j].set_title(f"SfS z, rms {out[nm]['sfs_rms']:.3f}")
        e = np.degrees(np.arccos(np.clip(np.sum(n_sfs * sc.n, -1), -1, 1)))
        im = ax[2, j].imshow(e, origin="lower", cmap="magma", vmin=0, vmax=30)
        ax[2, j].set_title(f"normal error, mean {out[nm]['sfs_ang_deg']:.1f} deg")
        for a in ax[:, j]:
            hide_axes(a)
    fig.colorbar(im, ax=ax[2, :], shrink=0.7, label="deg")
    savefig(fig, "fig06_sfs.png")
    return out


def exp_dfd(trials, rng):
    lens = ThinLens()
    out = {}
    sigmas = [0.0, 0.002, 0.005, 0.01]
    fig, ax = plt.subplots(2, 4, figsize=(10, 5))
    for j, nm in enumerate(NAMES):
        sc = make_scene(nm)
        zn = (sc.z - sc.z.min()) / np.ptp(sc.z)
        Z = 1000 - 500 * zn                      # mm: near = 500, far = 1000
        T, g1, g2, (v1, v2), (s1, s2) = render_defocus_pair(Z, lens, 450, 1050)
        m = interior(Z.shape, 10)
        out[nm] = {"sigma1_px": [float(s1.min()), float(s1.max())],
                   "sigma2_px": [float(s2.min()), float(s2.max())], "rms_mm": []}
        for s in sigmas:
            e = []
            for _ in range(max(1, trials // 10) if s > 0 else 1):
                Zh, _ = depth_from_defocus(add_noise(g1, s, rng), add_noise(g2, s, rng), lens, v1, v2, 500, 1000)
                e.append(float(np.sqrt(np.mean((Zh - Z)[m] ** 2))))
            out[nm]["rms_mm"].append(mean_std(e))
        if nm in ("sphere", "peaks"):
            Zh, _ = depth_from_defocus(g1, g2, lens, v1, v2, 500, 1000)
            c0 = 0 if nm == "sphere" else 2
            ax[0, c0].imshow(g1, cmap="gray", origin="lower"); ax[0, c0].set_title(f"{nm}: near focus")
            ax[0, c0 + 1].imshow(g2, cmap="gray", origin="lower"); ax[0, c0 + 1].set_title("far focus")
            ax[1, c0].imshow(Z, origin="lower", vmin=500, vmax=1000); ax[1, c0].set_title("true Z (mm)")
            ax[1, c0 + 1].imshow(np.where(m, Zh, np.nan), origin="lower", vmin=500, vmax=1000)
            ax[1, c0 + 1].set_title("DfD Z (mm)")
    for a in ax.ravel():
        hide_axes(a)
    savefig(fig, "fig07_dfd.png")
    fig, a = plt.subplots(figsize=(5, 3.3))
    for c, nm in zip(COLORS, NAMES):
        mm = np.array(out[nm]["rms_mm"])
        a.errorbar(sigmas, mm[:, 0], yerr=mm[:, 1], marker="o", ms=3, capsize=2, color=c, label=nm)
    a.set_xlabel("image noise sigma")
    a.set_ylabel("depth RMS (mm), range 500-1000")
    a.set_title("Depth from defocus: noise sensitivity")
    a.legend(fontsize=7)
    savefig(fig, "fig08_dfd_noise.png")
    out["noise_sigmas"] = sigmas
    return out


def run(trials=100, seed=0):
    rng = np.random.default_rng(seed)
    t = time.time()
    res = {"scenes": exp_scenes()}
    res["ps"] = exp_ps(trials, rng)
    res["integration"] = exp_integration(trials, rng)
    res["sfs"] = exp_sfs()
    res["dfd"] = exp_dfd(trials, rng)
    res["seconds"] = time.time() - t
    dump("single_view.json", res)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    r = run(a.trials, a.seed)
    print(f"single-view experiments done in {r['seconds']:.1f} s")
