"""Photo-geometric fusion (after Nehab et al. 2005, linear form).

Unknown: depth z on the photometric-stereo grid. Two families of linear rows:
  normal rows   : n_z (z[i,j+1] - z[i,j]) / dx + n_x = 0   (and the y analogue)
  position rows : lam * z[i,j] = lam * z_hat[i,j]         (sparse metric points)
Positions fix scale/offset and low frequencies; normals give dense detail.
"""
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import griddata
from scipy.sparse.linalg import lsqr, spsolve

from .geometric.camera import intrinsics, look_at, projection_matrix, project
from .geometric.epipolar import triangulate
from .photometric.integrate import grad_ops


def normal_rows(n, dx):
    H, W, _ = n.shape
    Dx, Dy = grad_ops(H, W, dx)
    nxm = 0.5 * (n[:, 1:, 0] + n[:, :-1, 0])
    nzx = 0.5 * (n[:, 1:, 2] + n[:, :-1, 2])
    nym = 0.5 * (n[1:, :, 1] + n[:-1, :, 1])
    nzy = 0.5 * (n[1:, :, 2] + n[:-1, :, 2])
    A = sp.vstack([sp.diags(nzx.ravel()) @ Dx, sp.diags(nzy.ravel()) @ Dy])
    b = -np.concatenate([nxm.ravel(), nym.ravel()])
    return A.tocsr(), b


def position_rows(idx, zhat, N_pix):
    m = len(idx)
    A = sp.csr_matrix((np.ones(m), (np.arange(m), idx)), shape=(m, N_pix))
    return A, zhat


def fuse(n, dx, idx, zhat, lam=1.0, solver="direct"):
    H, W, _ = n.shape
    An, bn = normal_rows(n, dx)
    Ap, bp = position_rows(idx, zhat, H * W)
    A = sp.vstack([An, lam * Ap]).tocsr()
    b = np.concatenate([bn, lam * bp])
    if solver == "lsqr":
        z = lsqr(A, b, atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    else:
        z = spsolve((A.T @ A).tocsc(), A.T @ b)
    return z.reshape(H, W)


def select_lambda(n, dx, idx, zhat, lams=(1, 3, 10, 30, 100, 300), holdout=0.2, rng=None):
    """Choose lam without ground truth: hold out a fraction of the sparse points,
    fuse with the rest, and score the prediction error at the held-out points."""
    rng = np.random.default_rng(0) if rng is None else rng
    m = len(idx)
    perm = rng.permutation(m)
    k = max(4, int(holdout * m))
    test, train = perm[:k], perm[k:]
    scores = []
    for lam in lams:
        z = fuse(n, dx, idx[train], zhat[train], lam).ravel()
        scores.append(float(np.sqrt(np.mean((z[idx[test]] - zhat[test]) ** 2))))
    return lams[int(np.argmin(scores))], list(lams), scores


def densify_sparse(idx, zhat, shape):
    """Stereo-only baseline: piecewise-linear interpolation of the sparse points."""
    H, W = shape
    r, c = np.unravel_index(idx, shape)
    gr, gc = np.mgrid[0:H, 0:W]
    z = griddata((r, c), zhat, (gr, gc), method="linear")
    zn = griddata((r, c), zhat, (gr, gc), method="nearest")
    return np.where(np.isnan(z), zn, z)


def stereo_rig(height=6.0, baseline=1.5, f=800.0):
    K = intrinsics(f, 320, 240)
    R1, t1 = look_at((-baseline / 2, 0, height), (0, 0, 0), up=(0, 1, 0))
    R2, t2 = look_at((baseline / 2, 0, height), (0, 0, 0), up=(0, 1, 0))
    return projection_matrix(K, R1, t1), projection_matrix(K, R2, t2)


def sparse_points_from_stereo(scene, frac, sigma_px, rng, rig=None):
    """Sample a fraction of PS-grid pixels, image them with a calibrated stereo rig
    (pixel noise sigma_px), triangulate, and snap (X, Y) back to the grid.
    Returns flat indices and triangulated heights."""
    H, W = scene.z.shape
    P1, P2 = stereo_rig() if rig is None else rig
    m = max(8, int(frac * H * W))
    idx = rng.choice(H * W, m, replace=False)
    Xw = np.column_stack([scene.X.ravel()[idx], scene.Y.ravel()[idx], scene.z.ravel()[idx]])
    x1 = project(P1, Xw) + rng.normal(0, sigma_px, (m, 2))
    x2 = project(P2, Xw) + rng.normal(0, sigma_px, (m, 2))
    Xt = triangulate(P1, P2, x1, x2)
    col = np.clip(np.rint((Xt[:, 0] - scene.x[0]) / scene.dx), 0, W - 1).astype(int)
    row = np.clip(np.rint((Xt[:, 1] - scene.y[0]) / scene.dx), 0, H - 1).astype(int)
    flat = row * W + col
    # merge duplicates after snapping (average)
    uniq, inv = np.unique(flat, return_inverse=True)
    zsum = np.bincount(inv, Xt[:, 2])
    cnt = np.bincount(inv)
    return uniq, zsum / cnt
