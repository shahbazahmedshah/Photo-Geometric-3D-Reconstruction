"""Shared helpers for experiment scripts."""
import json
import warnings
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")
os.makedirs(FIGS, exist_ok=True)
warnings.filterwarnings("ignore", message=".*tight_layout.*")

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9,
    "axes.titlesize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "image.cmap": "viridis",
})
COLORS = ["#2a6f97", "#e07a1f", "#3a9d5d", "#c0392b", "#7d5ba6"]


def savefig(fig, name):
    path = os.path.join(FIGS, name)
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def normal_rgb(n):
    return np.clip(0.5 * (n + 1), 0, 1)


def mean_std(values):
    a = np.asarray(values, float)
    return float(a.mean()), float(a.std())


def dump(name, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=2, default=conv)


def hide_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
