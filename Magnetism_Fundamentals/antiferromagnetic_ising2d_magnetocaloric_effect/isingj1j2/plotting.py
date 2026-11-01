"""
Estilo gráfico común y exportación de figuras.

Fuente serif, 300 dpi al guardar y una paleta apta para daltonismo (Wong, 2011),
usada de forma consistente: un color fijo por tamaño de red y por condición de frontera.
"""

import matplotlib.pyplot as plt
import numpy as np

from . import config

STYLE = {
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 12,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "errorbar.capsize": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}

# Paleta de Wong (2011).
WONG = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000")
COLOR_PBC = "#0072B2"
COLOR_FBC = "#D55E00"
COLOR_GREY = "#7F7F7F"
L_COLORS = {8: "#56B4E9", 12: "#E69F00", 16: "#009E73", 20: "#56B4E9", 24: "#CC79A7", 32: "#0072B2",
            48: "#D55E00", 64: "#000000"}


def apply_style():
    """Aplica el estilo común a matplotlib."""
    plt.rcParams.update(STYLE)


def color_for_L(L):
    return L_COLORS.get(int(L), COLOR_GREY)


def field_colors(H, cmap="viridis"):
    """Un color por campo, de una escala continua (los campos son una variable ordenada)."""
    H = np.asarray(H, dtype=float)
    span = H.max() - H.min() if H.max() > H.min() else 1.0
    return {float(h): plt.get_cmap(cmap)(0.05 + 0.85 * (h - H.min()) / span) for h in H}


def savefig(fig, name, formats=("png", "pdf")):
    """Guarda ``fig`` en la carpeta de figuras como name.png y name.pdf (300 dpi)."""
    folder = config.figures_dir()
    for ext in formats:
        fig.savefig(folder / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    return folder / f"{name}.png"


def errorband(ax, x, y, err, color, label=None, marker="o", ms=3.5, lw=1.4, alpha=0.25):
    """Curva con marcadores y banda de +-1 error (más legible que barras cuando hay muchos puntos)."""
    ax.plot(x, y, marker=marker, ms=ms, lw=lw, color=color, label=label)
    if err is not None:
        ax.fill_between(x, y - err, y + err, color=color, alpha=alpha, lw=0)
