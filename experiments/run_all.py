"""Regenerate every figure and result, then check objectives O1-O8.

Usage:  python -m experiments.run_all            (100 Monte Carlo trials, ~5-8 min)
        python -m experiments.run_all --quick    (10 trials, ~1 min)
Outputs: results/figures/*.png, results/*.json, results/summary.md
"""
import argparse
import os
import time

from experiments import exp_single_view, exp_multi_view, exp_fusion
from experiments.common import RESULTS


def check_objectives(sv, mv, fu):
    rows = []

    def add(o, text, value, ok):
        rows.append((o, text, value, "PASS" if ok else "CHECK"))

    d = sv["scenes"]["renderer_max_abs_diff"]
    add("O1", "Renderer equals closed-form Lambertian (max abs diff < 1e-10)", f"{d:.1e}", d < 1e-10)

    e = max(sv["ps"]["noise_free_deg"].values())
    add("O2", "PS mean angular error, noise-free, worst surface (< 0.1 deg)", f"{e:.2e} deg", e < 0.1)
    rms = max(v["poisson"] for v in sv["integration"]["noise_free_rms"].values())
    add("O2", "Poisson integration depth RMS, noise-free, worst surface", f"{rms:.2e}", rms < 1e-3)
    emp, ana = sv["ps"]["g_std_empirical"], sv["ps"]["g_std_analytic"]
    ratio = max(abs(a / b - 1) for a, b in zip(emp, ana))
    add("O2", "Empirical vs analytic std of g = rho n (within 10 %)", f"{100 * ratio:.1f} %", ratio < 0.1)

    s = sv["sfs"]
    add("O3", "SfS mean normal error (sphere / gaussian / peaks / ripple)",
        " / ".join(f"{s[n]['sfs_ang_deg']:.1f}" for n in s) + " deg", True)
    dd = sv["dfd"]
    add("O3", "DfD depth RMS noise-free, range 500-1000 mm",
        " / ".join(f"{dd[n]['rms_mm'][0][0]:.0f}" for n in ("sphere", "gaussian", "peaks", "ripple")) + " mm", True)

    c = mv["calibration"]
    i = c["sigmas"].index(0.5)
    fe, rp, fl = c["f_rel_err_pct"][i][0], c["reproj_px"][i][0], c["noise_floor_px"][i]
    add("O4", "Calibration |f| error at sigma = 0.5 px (< 1 %)", f"{fe:.2f} %", fe < 1.0)
    add("O4", "Reprojection RMS at sigma = 0.5 px (at noise floor sigma*sqrt(2(1-p/2n)))",
        f"{rp:.3f} px (floor {fl:.3f})", rp < 1.05 * fl)

    ss = mv["simple_stereo"]["rms"]
    mono = all(all(a[0] > b[0] for a, b in zip(v, v[1:])) for v in ss.values())
    add("O5", "Simple-stereo depth error falls with baseline (all surfaces)", "monotone" if mono else "not monotone", mono)
    u = mv["uncalibrated"]["by_relief"]["3.0"]
    add("O5", "Uncalibrated stereo, noise-free rotation error", f"{u['rot_deg'][0][0]:.1e} deg", u["rot_deg"][0][0] < 1e-6)
    add("O5", "Uncalibrated stereo, 0.5 px noise: rotation error / up-to-scale point RMS",
        f"{u['rot_deg'][2][0]:.2f} deg / {100 * u['pts_rel_rms'][2][0]:.1f} % of depth range", True)

    fw = mv["flow"]
    ep = max(max(v["lk_epe"], v["hs_epe"]) for k, v in fw.items() if k != "aperture")
    add("O6", "Optical flow EPE, worst field and method (< 0.05 px)", f"{ep:.4f} px", ep < 0.05)
    sf = mv["sfm"]
    e0 = sf["shape_rel_rms"]["20"][0][0]
    e1 = sf["shape_rel_rms"]["20"][2][0]
    add("O6", "SfM shape error, 20 frames: noise-free / sigma = 0.005",
        f"{e0:.1e} / {100 * e1:.2f} % of depth range", e0 < 1e-8)

    ps = fu["per_surface"]
    wins = sum(ps[n]["fused"][0] < min(ps[n]["ps_only"][0], ps[n]["stereo_only"][0]) for n in ps)
    add("O7", "Fused RMS below both PS-only and stereo-only (surfaces)", f"{wins} / {len(ps)}", wins >= 3)
    add("O8", "All figures regenerated from fixed seeds by one command", "run_all.py", True)
    return rows


def write_summary(rows, sv, mv, fu, trials, seconds):
    L = ["# Results summary", "",
         f"Monte Carlo trials: {trials} | total runtime: {seconds:.0f} s", "",
         "## Objective checks", "", "| Obj. | Check | Value | Status |", "|---|---|---|---|"]
    L += [f"| {o} | {t} | {v} | {s} |" for o, t, v, s in rows]
    L += ["", "## Fusion (depth RMS, mean +/- std, no ground-truth alignment)", "",
          "| Surface | PS only | Stereo only | Fused | lambda | Fused best in |", "|---|---|---|---|---|---|"]
    for n, r in fu["per_surface"].items():
        L.append(f"| {n} | {r['ps_only'][0]:.4f} +/- {r['ps_only'][1]:.4f} | "
                 f"{r['stereo_only'][0]:.4f} +/- {r['stereo_only'][1]:.4f} | "
                 f"{r['fused'][0]:.4f} +/- {r['fused'][1]:.4f} | {fu['lambda_cv'][n]['chosen']} | "
                 f"{100 * r['fused_best_fraction']:.0f} % of trials |")
    L += ["", "## Optical flow (endpoint error, px)", "", "| Field | max motion | LK | HS |", "|---|---|---|---|"]
    for k, v in mv["flow"].items():
        if k != "aperture":
            L.append(f"| {k} | {v['max_motion_px']:.2f} | {v['lk_epe']:.4f} | {v['hs_epe']:.4f} |")
    a = mv["flow"]["aperture"]
    L += ["", f"Aperture problem (1-D grating): structure-tensor min eigenvalue {a['min_eig_grating']:.1e} "
          f"(textured: {a['min_eig_texture']:.1e}); normal-flow error {a['normal_flow_u_error_px']:.1e} px, "
          f"while the tangential component is unobservable.", "",
          "## Figures", ""]
    for f in sorted(os.listdir(os.path.join(RESULTS, "figures"))):
        L.append(f"- `figures/{f}`")
    with open(os.path.join(RESULTS, "summary.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    trials = 10 if a.quick else a.trials
    t = time.time()
    sv = exp_single_view.run(trials)
    print(f"[1/3] single view  {time.time() - t:6.1f} s")
    mv = exp_multi_view.run(trials)
    print(f"[2/3] multi view   {time.time() - t:6.1f} s")
    fu = exp_fusion.run(trials)
    print(f"[3/3] fusion       {time.time() - t:6.1f} s")
    rows = check_objectives(sv, mv, fu)
    write_summary(rows, sv, mv, fu, trials, time.time() - t)
    for o, txt, v, s in rows:
        print(f"{o}  {s:5s}  {txt}: {v}")


if __name__ == "__main__":
    main()
