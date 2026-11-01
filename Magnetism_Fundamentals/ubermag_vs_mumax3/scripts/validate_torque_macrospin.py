#!/usr/bin/env python3
"""Validación del barrido de torque (MuMax3) frente a un modelo analítico y a Ubermag.

Comprueba, con los datos de torque_vs_angle.txt (salida de torque_magnetometry_mumax3.mx3):

  1. propiedades de la curva: amplitud, periodo pi, simetría impar, componentes tau_y y tau_z;
  2. factores desmagnetizantes de la caja de 300 x 300 x 100 nm con la fórmula analítica de
     Aharoni (J. Appl. Phys. 83, 3432, 1998), comprobada con el cubo (1/3) y la regla de suma;
  3. el modelo de macroespín de Stoner-Wohlfarth sin y con anisotropía de forma
     (rama continua de equilibrios, igual que el barrido del notebook);
  4. la comparación con Ubermag, de dos maneras:
       a) digitalizando los 71 marcadores de la figura que el notebook de Ubermag guarda en
          su celda del barrido angular (resolución de la figura: unos 5.6e3 N/m2, 1 píxel);
       b) si existe, con los datos numéricos exactos que guarda benchmark_ubermag_vs_mumax3.py
          con la opción --dump-arrays (ubermag_arrays_torque_magnetometry_uniaxial_box.npz).

Uso (por defecto lee mumax3/torque_magnetometry_mumax3.out/ y results/, y escribe
figures/fig_torque_comparison.png):
    python scripts/validate_torque_macrospin.py
    python scripts/validate_torque_macrospin.py --notebook torque_magnetometry_uniaxial_box.ipynb

--notebook añade la digitalización de la figura guardada en el notebook (opción a); con los datos
numéricos de Ubermag en results/ la figura usa esos datos y no la digitalización.

Requiere numpy y matplotlib; Pillow solo para la digitalización (opción a).
"""
import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent   # carpeta del proyecto
MU0 = 4e-7 * np.pi
# Parámetros del notebook (Ubermag) y del script de MuMax3
MS, A_EX, K, H = 9.65e5, 8.78e-13, 1.18e6, 2.38e6
DIMS = (300e-9, 300e-9, 100e-9)  # caja; el eje fácil z es la dirección corta


def load_mumax3(path):
    d = np.loadtxt(path, comments="#")
    return d[:, 0], d[:, 1], d[:, 2], d[:, 3]


def _n_aharoni(a, b, c):
    """Factor desmagnetizante a lo largo de c para un prisma de semiejes a, b, c (Aharoni, 1998)."""
    r = np.sqrt(a * a + b * b + c * c)
    ab, bc, ac = np.sqrt(a * a + b * b), np.sqrt(b * b + c * c), np.sqrt(a * a + c * c)
    return (1 / np.pi) * (
        (b * b - c * c) / (2 * b * c) * np.log((r - a) / (r + a))
        + (a * a - c * c) / (2 * a * c) * np.log((r - b) / (r + b))
        + b / (2 * c) * np.log((ab + a) / (ab - a))
        + a / (2 * c) * np.log((ab + b) / (ab - b))
        + c / (2 * a) * np.log((bc - b) / (bc + b))
        + c / (2 * b) * np.log((ac - a) / (ac + a))
        + 2 * np.arctan(a * b / (c * r))
        + (a ** 3 + b ** 3 - 2 * c ** 3) / (3 * a * b * c)
        + (a * a + b * b - 2 * c * c) / (3 * a * b * c) * r
        + c / (a * b) * (ac + bc)
        - ((a * a + b * b) ** 1.5 + (b * b + c * c) ** 1.5 + (c * c + a * a) ** 1.5) / (3 * a * b * c)
    )


def demag_factors(lx, ly, lz):
    """(Nx, Ny, Nz) de un prisma de aristas lx, ly, lz (magnetización uniforme)."""
    a, b, c = lx / 2, ly / 2, lz / 2
    return _n_aharoni(b, c, a), _n_aharoni(c, a, b), _n_aharoni(a, b, c)


def macrospin_torque(phis, Nz=None, Ny=None):
    """tau_x(phi) del macroespín: e = K sin^2(th) [+ forma] - mu0 Ms H cos(th - phi).

    Con Nz y Ny se añade la energía magnetostática (1/2) mu0 Ms^2 (Nz cos^2 th + Ny sin^2 th).
    Se sigue la rama continua de equilibrios (Newton desde el estado del ángulo anterior).
    """
    shape = Nz is not None
    dd = 0.5 * MU0 * MS ** 2 * (Ny - Nz) if shape else 0.0
    th, out = 0.0, []
    for ph in phis:
        for _ in range(200):
            g = K * np.sin(2 * th) + MU0 * MS * H * np.sin(th - ph) + dd * np.sin(2 * th)
            h = 2 * K * np.cos(2 * th) + MU0 * MS * H * np.cos(th - ph) + 2 * dd * np.cos(2 * th)
            th -= g / h
            if abs(g / h) < 1e-14:
                break
        out.append(MU0 * MS * H * np.sin(th - ph))
    return np.array(out)


def digitize_stored_figure(notebook_path):
    """Lee los 71 marcadores de la figura guardada en la celda del barrido angular del notebook.

    Como la posición horizontal de cada marcador es conocida (phi_k = 2 pi k / 70), se mide solo
    la altura del centro del marcador en una banda de +-1 píxel alrededor de esa columna. La
    escala vertical sale de las dos líneas rojas en +-K, y la horizontal de los bordes del eje
    (xlim = [0, 2 pi]).
    """
    from PIL import Image

    nb = json.load(open(notebook_path, encoding="utf-8"))
    png = None
    for cell in nb["cells"]:
        src = "".join(cell["source"]) if cell["cell_type"] == "code" else ""
        if "torque_val_x" in src and "scatter" in src:
            for o in cell.get("outputs", []):
                if "image/png" in o.get("data", {}):
                    png = base64.b64decode(o["data"]["image/png"])
    if png is None:
        raise RuntimeError("no se encontró la figura del barrido en el notebook")
    im = np.array(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)
    vals, counts = np.unique(im.reshape(-1, 3), axis=0, return_counts=True)
    order = np.argsort(-counts)
    bg = next(v for v in vals[order] if tuple(v) != (255, 255, 255))      # fondo del eje
    blue = np.abs(im - np.array([76, 114, 176])).sum(axis=2) < 60         # marcadores
    red = (im[:, :, 0] > 200) & (im[:, :, 1] < 80) & (im[:, :, 2] < 80)   # líneas +-K
    isbg = np.abs(im - bg).sum(axis=2) < 6
    cols = np.where(isbg.sum(axis=0) > 100)[0]
    x0, x1 = cols.min(), cols.max()
    rr = red.sum(axis=1)
    rl = np.where(rr > 0.3 * (x1 - x0))[0]
    groups = np.split(rl, np.where(np.diff(rl) > 3)[0] + 1)
    cy = [(g * rr[g]).sum() / rr[g].sum() for g in groups]
    y_top, y_bot = min(cy), max(cy)
    slope = (K - (-K)) / (y_top - y_bot)                                   # N/m2 por fila
    phi = 2 * np.pi * np.arange(71) / 70
    xp = x0 + phi / (2 * np.pi) * (x1 - x0 + 1) - 0.5
    tau = np.full(71, np.nan)
    for k, x in enumerate(xp):
        xi = int(round(x))
        sub = blue[:, [c for c in range(xi - 1, xi + 2) if x0 <= c <= x1]]
        rows = np.where(sub.any(axis=1))[0]
        if len(rows) == 0:
            continue
        g = max(np.split(rows, np.where(np.diff(rows) > 1)[0] + 1), key=len)
        w = sub[g, :].sum(axis=1).astype(float)
        tau[k] = K + ((g * w).sum() / w.sum() - y_top) * slope
    return phi, tau, abs(slope)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "mumax3" / "torque_magnetometry_mumax3.out" / "torque_vs_angle.txt"))
    ap.add_argument("--notebook", default=None, help="notebook de Ubermag (para digitalizar su figura)")
    ap.add_argument("--ubermag-npz", default=str(ROOT / "results" / "ubermag_arrays_torque_magnetometry_uniaxial_box.npz"))
    ap.add_argument("--fig", default=str(ROOT / "figures" / "fig_torque_comparison.png"), help="ruta de la figura de comparación (PNG)")
    args = ap.parse_args()
    if not Path(args.data).exists():
        raise SystemExit(f"No se encontró {args.data}. Ejecuta antes mumax3/torque_magnetometry_mumax3.mx3 (ver README).")

    phi, tx, ty, tz = load_mumax3(args.data)
    amp = 0.5 * (tx.max() - tx.min())
    print("== Curva de MuMax3 ==")
    print(f"  tau_x máx = {tx.max():.2f} (phi = {phi[tx.argmax()]:.4f}), mín = {tx.min():.2f} (phi = {phi[tx.argmin()]:.4f}) N/m2")
    print(f"  semirrango (amplitud) = {amp:.1f} N/m2 = {amp / K:.3f} K")
    print(f"  max|tau(phi) - tau(phi+pi)| = {np.abs(tx[:35] - tx[35:70]).max():.3f} N/m2 ({np.abs(tx[:35] - tx[35:70]).max() / amp:.1e} de la amplitud)")
    print(f"  max|tau(phi) + tau(2pi-phi)| = {np.abs(tx + tx[::-1]).max():.3f} N/m2")
    print(f"  max|tau_y| = {np.abs(ty).max():.2e}, max|tau_z| = {np.abs(tz).max():.2e} N/m2")

    nx, ny, nz = demag_factors(*DIMS)
    print("\n== Factores desmagnetizantes (Aharoni, 1998) ==")
    cube = demag_factors(1, 1, 1)[2]
    print(f"  comprobaciones: cubo N = {cube:.6f} (1/3), Nx+Ny+Nz = {nx + ny + nz:.8f}")
    print(f"  Nx = Ny = {ny:.4f}, Nz = {nz:.4f}")
    kd = 0.5 * MU0 * MS ** 2
    keff = K - kd * (nz - ny)
    print(f"  Kd = mu0 Ms^2/2 = {kd:.4e} J/m3; anisotropía de forma Kd (Nz-Ny) = {kd * (nz - ny):.4e}; K_eff = {keff:.4e} J/m3")
    print(f"  H/H_K = {H / (2 * K / (MU0 * MS)):.3f}; H/H_K,eff = {H / (2 * keff / (MU0 * MS)):.3f}")

    fine = np.linspace(0, 2 * np.pi, 2001)
    t_k, t_ks = macrospin_torque(phi), macrospin_torque(phi, nz, ny)
    f_k, f_ks = macrospin_torque(fine), macrospin_torque(fine, nz, ny)
    print("\n== Macroespín frente a MuMax3 (diferencia máxima, % de la amplitud de MuMax3) ==")
    print(f"  solo K          : amplitud {np.abs(f_k).max():.4e} N/m2; diferencia {100 * np.abs(t_k - tx).max() / amp:.1f} %")
    print(f"  K + forma       : amplitud {np.abs(f_ks).max():.4e} N/m2; diferencia {100 * np.abs(t_ks - tx).max() / amp:.2f} %")
    print(f"  extremos (K solo)  : phi = {fine[f_k.argmin()] % np.pi:.3f} y {fine[f_k.argmax()] % np.pi:.3f} rad (módulo pi)")
    print(f"  extremos (K+forma) : phi = {fine[f_ks.argmin()] % np.pi:.3f} y {fine[f_ks.argmax()] % np.pi:.3f} rad (módulo pi)")

    dig = None
    if args.notebook:
        try:
            _, tau, pxres = digitize_stored_figure(args.notebook)
            dig = tau
            d = tau - tx
            print("\n== Ubermag (figura guardada en el notebook, digitalizada) - MuMax3 ==")
            print(f"  marcadores leídos: {np.isfinite(tau).sum()} de 71; resolución de la figura: 1 px = {pxres:.0f} N/m2 ({100 * pxres / amp:.2f} % de la amplitud)")
            print(f"  diferencia media = {d.mean():.0f} N/m2 ({100 * d.mean() / amp:.2f} %); rms = {np.sqrt(np.mean(d ** 2)):.0f} ({100 * np.sqrt(np.mean(d ** 2)) / amp:.2f} %);"
                  f" máx. |dif.| = {np.abs(d).max():.0f} ({100 * np.abs(d).max() / amp:.2f} %, {np.abs(d).max() / pxres:.1f} px)")
        except Exception as e:  # noqa: BLE001
            print("\n(No se pudo digitalizar la figura del notebook:", e, ")")
            pxres = None
    exact = Path(args.ubermag_npz)
    ub_exact = None
    if exact.exists():
        z = np.load(exact)
        ub_exact = z["tx"]
        d = z["tx"] - tx
        print("\n== Ubermag (datos numéricos exactos) - MuMax3 ==")
        print(f"  máx. |dif.| = {np.abs(d).max():.3g} N/m2 = {np.abs(d).max() / amp:.2e} de la amplitud; rms = {np.sqrt(np.mean(d ** 2)):.3g}")

    if args.fig:
        fig, (ax, ar) = plt.subplots(2, 1, figsize=(7.6, 7.2), dpi=130, sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1.35]})
        ax.axhline(K, color="r", ls="--", lw=1, label=r"$\pm K$")
        ax.axhline(-K, color="r", ls="--", lw=1)
        ax.plot(fine, f_k, color="gray", ls=":", lw=1.6, label="Macroespín, solo $K$")
        ax.plot(fine, f_ks, color="k", lw=1.3, label="Macroespín, $K$ + anisotropía de forma")
        ax.scatter(phi, tx, s=22, color="#DD8452", zorder=3, label="MuMax3")
        # Con los datos numéricos de Ubermag se dibujan esos; la digitalización de la figura solo se usa si faltan.
        if ub_exact is not None:
            ax.scatter(phi, ub_exact, s=16, marker="x", color="#4C72B0", zorder=4, label="Ubermag (datos numéricos)")
        elif dig is not None:
            ax.scatter(phi, dig, s=16, marker="x", color="#4C72B0", zorder=4, label="Ubermag (digitalizado de la figura)")
        ax.set_ylabel(r"$\tau_x$ (N m$^{-2}$)")
        ax.set_ylim(-1.32e6, 1.95e6)
        ax.legend(fontsize=8, loc="upper center", ncol=2)
        ar.axhline(0, color="k", lw=0.6)
        ar.plot(phi, 100 * (t_ks - tx) / amp, "k.-", lw=0.8, ms=3, label="macroespín + forma − MuMax3")
        if ub_exact is not None:
            ar.plot(phi, 100 * (ub_exact - tx) / amp, "x", color="#4C72B0", ms=4, label="Ubermag − MuMax3")
        elif dig is not None:
            ar.axhspan(-100 * pxres / amp, 100 * pxres / amp, color="#4C72B0", alpha=0.12, lw=0, label="±1 píxel de la figura")
            ar.plot(phi, 100 * (dig - tx) / amp, "x", color="#4C72B0", ms=4, label="Ubermag (digitalizado) − MuMax3")
        ar.set_xlabel(r"Ángulo del campo $\varphi$ (rad)")
        ar.set_ylabel("Diferencia (% de la amplitud)")
        ar.set_xlim(0, 2 * np.pi)
        ar.set_ylim(-1.5, 2.4)
        ar.legend(fontsize=7.5, loc="upper center", ncol=2)
        fig.tight_layout()
        fig.savefig(args.fig, bbox_inches="tight")
        print("\nFigura guardada en", args.fig)


if __name__ == "__main__":
    main()
