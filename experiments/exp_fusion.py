"""Capstone experiment: fusion of dense PS normals with sparse triangulated points.

Inputs per trial:
  * PS: 4 images (noise sigma_I) solved with slightly mis-calibrated lights
    (light_err_deg) -> dense normals with a smooth, low-frequency bias.
  * Stereo: a fraction of grid points imaged by a calibrated rig with pixel noise
    sigma_px and triangulated -> sparse metric heights with white noise.
Methods compared (no ground-truth alignment is used for any of them):
  PS-only  : Poisson integration + offset fitted to the sparse points
  Stereo   : piecewise-linear interpolation of the sparse points
  Fused    : joint sparse least squares, lambda chosen by hold-out validation

Usage: python -m experiments.exp_fusion [--trials 100]
"""
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

from capstone.scenes import SURFACES, make_scene, pq_from_normals
from capstone.render import light_ring, render_stack, add_noise, perturb_lights
from capstone.photometric.ps import photometric_stereo
from capstone.photometric.integrate import poisson_integrate
from capstone.fusion import fuse, densify_sparse, sparse_points_from_stereo, select_lambda
from capstone.metrics import depth_rms
from experiments.common import savefig, mean_std, dump, hide_axes, COLORS

NAMES = list(SURFACES)
CFG = dict(N=128, k=4, sigma_I=0.01, light_err_deg=2.0, frac=0.02, sigma_px=0.5)


def one_trial(sc, rng, cfg, lam=None):
    S = light_ring(cfg["k"])
    I = add_noise(render_stack(sc.n, sc.albedo, S), cfg["sigma_I"], rng)
    n, _, _ = photometric_stereo(I, perturb_lights(S, cfg["light_err_deg"], rng))
    idx, zh = sparse_points_from_stereo(sc, cfg["frac"], cfg["sigma_px"], rng)
    z_ps = poisson_integrate(*pq_from_normals(n), sc.dx)
    z_ps += np.mean(zh - z_ps.ravel()[idx])
    z_st = densify_sparse(idx, zh, sc.z.shape)
    cv = None
    if lam is None:
        lam, lams, scores = select_lambda(n, sc.dx, idx, zh, rng=rng)
        cv = (lams, scores)
    z_f = fuse(n, sc.dx, idx, zh, lam)
    r = {"ps_only": depth_rms(z_ps, sc.z, align="none"),
         "stereo_only": depth_rms(z_st, sc.z, align="none"),
         "fused": depth_rms(z_f, sc.z, align="none"), "lam": lam}
    return r, (z_ps, z_st, z_f, idx), cv


def run(trials=100, seed=2):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    out = {"config": CFG, "per_surface": {}, "lambda_cv": {}}
    for nm in NAMES:
        sc = make_scene(nm, N=CFG["N"])
        # choose lambda once per surface by hold-out validation (no ground truth)
        r0, maps, cv = one_trial(sc, rng, CFG)
        lam = r0["lam"]
        out["lambda_cv"][nm] = {"lams": cv[0], "holdout_rms": cv[1], "chosen": lam}
        acc = {"ps_only": [], "stereo_only": [], "fused": []}
        for _ in range(trials):
            r, _, _ = one_trial(sc, rng, CFG, lam)
            for k in acc:
                acc[k].append(r[k])
        wins = int(np.sum(np.array(acc["fused"]) < np.minimum(acc["ps_only"], acc["stereo_only"])))
        out["per_surface"][nm] = {k: mean_std(v) for k, v in acc.items()}
        out["per_surface"][nm]["fused_best_fraction"] = wins / trials
        if nm == "peaks":
            fig_maps = (sc, maps)

    # density sweep on 'peaks' (fixed lambda chosen above)
    sc = make_scene("peaks", N=CFG["N"])
    fracs = [0.005, 0.01, 0.02, 0.05, 0.1]
    dens = {"ps_only": [], "stereo_only": [], "fused": []}
    for fr in fracs:
        cfg = dict(CFG, frac=fr)
        acc = {k: [] for k in dens}
        for _ in range(max(5, trials // 5)):
            r, _, _ = one_trial(sc, rng, cfg, out["lambda_cv"]["peaks"]["chosen"])
            for k in acc:
                acc[k].append(r[k])
        for k in dens:
            dens[k].append(mean_std(acc[k]))
    out["density_sweep"] = {"fracs": fracs, **dens}

    # ---- figures
    sc, (z_ps, z_st, z_f, idx) = fig_maps
    fig, ax = plt.subplots(2, 4, figsize=(11, 5.4), layout="constrained")
    vmin, vmax = sc.z.min(), sc.z.max()
    r_, c_ = np.unravel_index(idx, sc.z.shape)
    for a, z, t in zip(ax[0], [sc.z, z_ps, z_st, z_f], ["ground truth", "PS only", "stereo only", "fused"]):
        a.imshow(z, origin="lower", vmin=vmin, vmax=vmax)
        a.set_title(t)
        hide_axes(a)
    ax[0, 2].scatter(c_, r_, s=1.5, c="w")
    ax[1, 0].axis("off")
    ax[1, 0].text(0.0, 0.5, f"peaks surface\n{len(idx)} sparse points\nlight error {CFG['light_err_deg']} deg\n"
                            f"image noise {CFG['sigma_I']}\npixel noise {CFG['sigma_px']} px",
                  fontsize=8.5, va="center")
    e_lim = 0.04
    for a, z, t in zip(ax[1, 1:], [z_ps, z_st, z_f], ["PS: smooth drift", "stereo: noise + interpolation",
                                                      "fused error"]):
        im = a.imshow(z - sc.z, origin="lower", cmap="coolwarm", vmin=-e_lim, vmax=e_lim)
        a.set_title(f"{t}\nRMS {np.sqrt(np.mean((z - sc.z) ** 2)):.4f}")
        hide_axes(a)
    fig.colorbar(im, ax=ax[1, 1:], shrink=0.75, label="depth error")
    savefig(fig, "fig14_fusion_maps.png")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    w = 0.26
    xs = np.arange(len(NAMES))
    for i, (k, c) in enumerate(zip(["ps_only", "stereo_only", "fused"], COLORS)):
        m = np.array([out["per_surface"][nm][k] for nm in NAMES])
        ax[0].bar(xs + (i - 1) * w, m[:, 0], w, yerr=m[:, 1], color=c, capsize=2, label=k)
    ax[0].set_xticks(xs, NAMES)
    ax[0].set_ylabel("depth RMS (no GT alignment)")
    ax[0].set_title(f"Monte Carlo, {trials} trials per surface")
    ax[0].legend(fontsize=7)
    for c, nm in zip(COLORS, NAMES):
        cvd = out["lambda_cv"][nm]
        ax[1].semilogx(cvd["lams"], cvd["holdout_rms"], "o-", ms=3, color=c, label=nm)
    ax[1].set_xlabel("lambda"); ax[1].set_ylabel("hold-out RMS at sparse points")
    ax[1].set_title("Choosing lambda without ground truth"); ax[1].legend(fontsize=7)
    for k, c in zip(["ps_only", "stereo_only", "fused"], COLORS):
        m = np.array(dens[k])
        ax[2].errorbar(100 * np.array(fracs), m[:, 0], yerr=m[:, 1], marker="o", ms=3, capsize=2, color=c, label=k)
    ax[2].set_xscale("log"); ax[2].set_xlabel("sparse points (% of pixels)"); ax[2].set_ylabel("depth RMS")
    ax[2].set_title("Effect of sparse-point density (peaks)"); ax[2].legend(fontsize=7)
    savefig(fig, "fig15_fusion_monte_carlo.png")

    out["seconds"] = time.time() - t0
    dump("fusion.json", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()
    r = run(a.trials, a.seed)
    print(f"fusion experiments done in {r['seconds']:.1f} s")
