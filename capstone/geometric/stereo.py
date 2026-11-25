"""Simple (rectified) binocular stereo: block matching -> disparity -> Z = f b / d."""
import numpy as np
from scipy.ndimage import uniform_filter


def block_match(I_left, I_right, d_min, d_max, win=9):
    """SSD block matching along rows with parabolic sub-pixel refinement.
    Left pixel u matches right pixel u - d. Returns disparity and valid mask."""
    H, W = I_left.shape
    ds = np.arange(d_min, d_max + 1)
    cost = np.full((len(ds), H, W), np.inf)
    for k, d in enumerate(ds):
        shifted = np.full_like(I_right, np.nan)
        shifted[:, d:] = I_right[:, :W - d]
        c = uniform_filter(np.nan_to_num((I_left - shifted) ** 2, nan=1.0), win)
        c[:, :d + win // 2] = np.inf
        cost[k] = c
    k = np.argmin(cost, axis=0)
    disp = ds[k].astype(float)
    inner = (k > 0) & (k < len(ds) - 1)
    r, c = np.indices((H, W))
    km, kp = np.clip(k - 1, 0, len(ds) - 1), np.clip(k + 1, 0, len(ds) - 1)
    c0, cm, cp = cost[k, r, c], cost[km, r, c], cost[kp, r, c]
    with np.errstate(invalid="ignore"):
        den = cm - 2 * c0 + cp
    ok = inner & np.isfinite(den) & (den > 1e-12)
    disp[ok] += 0.5 * (cm[ok] - cp[ok]) / den[ok]
    valid = np.isfinite(c0) & inner
    valid[:, :d_max + win] = False
    valid[:, -win:] = False
    valid[:win, :] = False
    valid[-win:, :] = False
    return disp, valid


def disparity_to_depth(disp, f, b):
    return f * b / np.maximum(disp, 1e-9)


def depth_resolution(Z, f, b, delta_d=1.0):
    """First-order depth uncertainty for a disparity error delta_d: Z^2/(f b) * dd."""
    return Z ** 2 / (f * b) * delta_d
