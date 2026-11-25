# Results summary

Monte Carlo trials: 100 | total runtime: 127 s

## Objective checks

| Obj. | Check | Value | Status |
|---|---|---|---|
| O1 | Renderer equals closed-form Lambertian (max abs diff < 1e-10) | 4.4e-16 | PASS |
| O2 | PS mean angular error, noise-free, worst surface (< 0.1 deg) | 2.44e-07 deg | PASS |
| O2 | Poisson integration depth RMS, noise-free, worst surface | 6.90e-05 | PASS |
| O2 | Empirical vs analytic std of g = rho n (within 10 %) | 0.4 % | PASS |
| O3 | SfS mean normal error (sphere / gaussian / peaks / ripple) | 1.5 / 15.0 / 12.0 / 9.2 deg | PASS |
| O3 | DfD depth RMS noise-free, range 500-1000 mm | 5 / 25 / 13 / 24 mm | PASS |
| O4 | Calibration |f| error at sigma = 0.5 px (< 1 %) | 0.47 % | PASS |
| O4 | Reprojection RMS at sigma = 0.5 px (at noise floor sigma*sqrt(2(1-p/2n))) | 0.692 px (floor 0.694) | PASS |
| O5 | Simple-stereo depth error falls with baseline (all surfaces) | monotone | PASS |
| O5 | Uncalibrated stereo, noise-free rotation error | 0.0e+00 deg | PASS |
| O5 | Uncalibrated stereo, 0.5 px noise: rotation error / up-to-scale point RMS | 0.39 deg / 0.6 % of depth range | PASS |
| O6 | Optical flow EPE, worst field and method (< 0.05 px) | 0.0237 px | PASS |
| O6 | SfM shape error, 20 frames: noise-free / sigma = 0.005 | 3.1e-15 / 0.79 % of depth range | PASS |
| O7 | Fused RMS below both PS-only and stereo-only (surfaces) | 4 / 4 | PASS |
| O8 | All figures regenerated from fixed seeds by one command | run_all.py | PASS |

## Fusion (depth RMS, mean +/- std, no ground-truth alignment)

| Surface | PS only | Stereo only | Fused | lambda | Fused best in |
|---|---|---|---|---|---|
| sphere | 0.0171 +/- 0.0052 | 0.0232 +/- 0.0033 | 0.0043 +/- 0.0008 | 30 | 100 % of trials |
| gaussian | 0.0153 +/- 0.0052 | 0.0158 +/- 0.0009 | 0.0042 +/- 0.0008 | 30 | 100 % of trials |
| peaks | 0.0144 +/- 0.0067 | 0.0185 +/- 0.0013 | 0.0042 +/- 0.0008 | 30 | 99 % of trials |
| ripple | 0.0133 +/- 0.0064 | 0.0201 +/- 0.0015 | 0.0042 +/- 0.0007 | 30 | 94 % of trials |

## Optical flow (endpoint error, px)

| Field | max motion | LK | HS |
|---|---|---|---|
| translation | 1.60 | 0.0059 | 0.0018 |
| rotation | 2.23 | 0.0079 | 0.0018 |
| divergence | 1.92 | 0.0143 | 0.0084 |
| sinusoidal | 1.50 | 0.0118 | 0.0237 |

Aperture problem (1-D grating): structure-tensor min eigenvalue 0.0e+00 (textured: 3.4e-04); normal-flow error 3.0e-05 px, while the tangential component is unobservable.

## Figures

- `figures/fig01_scenes.png`
- `figures/fig02_reflectance_map.png`
- `figures/fig03_ps_example.png`
- `figures/fig04_ps_noise_conditioning.png`
- `figures/fig05_integration.png`
- `figures/fig06_sfs.png`
- `figures/fig07_dfd.png`
- `figures/fig08_dfd_noise.png`
- `figures/fig09_calibration.png`
- `figures/fig10_simple_stereo.png`
- `figures/fig11_uncalibrated_stereo.png`
- `figures/fig12_optical_flow.png`
- `figures/fig13_sfm.png`
- `figures/fig14_fusion_maps.png`
- `figures/fig15_fusion_monte_carlo.png`
