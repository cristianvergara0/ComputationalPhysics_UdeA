#!/usr/bin/env python3
"""Análisis de la curva de torque simulada como si fuera una medición de laboratorio.

En un magnetómetro de torque se conoce el ángulo del campo, phi, y se mide tau(phi). La constante
de anisotropía se obtiene de la amplitud de la curva, pero con un campo finito la magnetización se
retrasa respecto del campo y la curva frente a phi no es un seno. El procedimiento habitual
(Cullity y Graham, 2009, cap. 7) es:

  1. ajustar directamente tau(phi) con una serie de Fourier (lo que se obtiene sin corregir nada);
  2. corregir el desfase: como tau = mu0 Ms H sin(theta - phi), el ángulo de la magnetización es
     theta = phi + arcsin(tau / (mu0 Ms H)), y frente a theta la curva de un material uniaxial es
     tau = -K sin(2 theta);
  3. restar la anisotropía de forma de la muestra para pasar de la anisotropía efectiva medida a
     la del material: K = K_ef + (1/2) mu0 Ms^2 (N_z - N_y).

Aquí la "medición" es la curva de torque_magnetometry_uniaxial_box.ipynb, cuya K de entrada se
conoce (1.18e6 J/m^3); así se comprueba cuánto se equivoca cada paso. Escribe
figures/fig_torque_curve_analysis.png.

Uso:
    python scripts/torque_curve_analysis.py
    python scripts/torque_curve_analysis.py --source mumax3

Requiere numpy y matplotlib.
"""
import argparse
from pathlib import Path

import numpy as np

import validate_torque_macrospin as vt

ROOT = Path(__file__).resolve().parent.parent   # carpeta del proyecto
MU0 = 4e-7 * np.pi
MS, H_MAG, K_IN = 9.65e5, 2.38e6, 1.18e6        # parámetros del notebook de torque
UBERMAG = ROOT / "results" / "ubermag_arrays_torque_magnetometry_uniaxial_box.npz"
MUMAX3 = ROOT / "mumax3" / "torque_magnetometry_mumax3.out" / "torque_vs_angle.txt"


def load_curve(source):
    """Ángulos del campo y tau_x de la simulación elegida ('ubermag' o 'mumax3')."""
    if source == "ubermag":
        if not UBERMAG.exists():
            raise SystemExit(f"No se encontró {UBERMAG}. Ver README (pasada con --dump-arrays).")
        z = np.load(UBERMAG)
        return np.asarray(z["angles"], float), np.asarray(z["tx"], float)
    if not MUMAX3.exists():
        raise SystemExit(f"No se encontró {MUMAX3}. Ejecuta antes mumax3/torque_magnetometry_mumax3.mx3.")
    d = np.loadtxt(MUMAX3, comments="#")
    return d[:, 0], d[:, 1]


def fourier_sin(phi, tau, n):
    """Coeficiente de sin(n phi) sobre una vuelta muestreada uniformemente (sin repetir 2 pi)."""
    if np.isclose(phi[-1] - phi[0], 2 * np.pi):
        phi, tau = phi[:-1], tau[:-1]
    return 2.0 / len(phi) * np.sum(tau * np.sin(n * phi))


def magnetization_angle(phi, tau, ms=MS, h=H_MAG):
    """theta = phi + arcsin(tau / (mu0 Ms H)): el ángulo de M a partir del torque medido."""
    return phi + np.arcsin(tau / (MU0 * ms * h))


def fit_k(theta, tau):
    """K de tau = -K sin(2 theta) por mínimos cuadrados, y residuo rms y máximo relativos a K."""
    s = np.sin(2 * theta)
    k = -np.sum(tau * s) / np.sum(s * s)
    res = tau + k * s
    return k, float(np.sqrt(np.mean(res ** 2)) / k), float(np.abs(res).max() / k)


def analyze(phi, tau):
    b2, b4 = fourier_sin(phi, tau, 2), fourier_sin(phi, tau, 4)
    theta = magnetization_angle(phi, tau)
    k_ef, rms, worst = fit_k(theta, tau)
    nx, ny, nz = vt.demag_factors(*vt.DIMS)
    shape = 0.5 * MU0 * MS ** 2 * (nz - ny)
    return {"b2": b2, "b4": b4, "theta": theta, "k_ef": k_ef, "rms": rms, "worst": worst,
            "lag_deg": float(np.degrees(np.abs(phi - theta)).max()), "shape": shape,
            "k_rec": k_ef + shape, "h_over_hk": H_MAG / (2 * K_IN / (MU0 * MS)),
            "amplitude": 0.5 * (tau.max() - tau.min())}


def make_figure(phi, tau, r, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c_data, c_fit, c_soft = "#4C72B0", "#C4622D", "0.45"
    fine = np.linspace(0, 2 * np.pi, 500)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    ax = axes[0]
    ax.plot(fine, -(-r["b2"]) * np.sin(2 * fine) / 1e5, color=c_soft, lw=2, ls="--",
            label=rf"ajuste directo, amplitud {abs(r['b2']) / 1e5:.2f}")
    ax.plot(phi, tau / 1e5, "o", color=c_data, ms=6, label="simulación")
    ax.set_xlabel(r"ángulo del campo $\varphi$ (rad)")
    ax.set_ylabel(r"$\tau_x$ ($10^5$ N m$^{-2}$)")
    ax.set_title("(a) Frente al ángulo del campo, como sale del instrumento", fontsize=11)
    ax = axes[1]
    ax.plot(fine, -r["k_ef"] * np.sin(2 * fine) / 1e5, color=c_fit, lw=2.5,
            label=rf"$-K_\mathrm{{ef}}\sin 2\theta$, $K_\mathrm{{ef}}$ = {r['k_ef'] / 1e5:.2f}")
    ax.plot(r["theta"], tau / 1e5, "o", color=c_data, ms=6, label="simulación, corregida")
    ax.set_xlabel(r"ángulo de la magnetización $\theta$ (rad)")
    ax.set_title("(b) Frente al ángulo de la magnetización", fontsize=11)
    for a in axes:
        a.axhline(0, color="0.8", lw=1)
        a.grid(True, color="0.92", lw=0.8)
        a.set_axisbelow(True)
        a.spines[["top", "right"]].set_visible(False)
        a.legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("ubermag", "mumax3"), default="ubermag")
    ap.add_argument("--fig", default=str(ROOT / "figures" / "fig_torque_curve_analysis.png"))
    args = ap.parse_args()
    phi, tau = load_curve(args.source)
    r = analyze(phi, tau)
    rel = lambda v: f"{100 * (v / K_IN - 1):+.1f} %"   # noqa: E731
    print(f"Curva de {args.source}: {len(phi)} ángulos; H / H_K = {r['h_over_hk']:.2f}")
    print(f"1. Ajuste directo frente a phi: |b2| = {abs(r['b2']):.4e} J/m3 ({rel(abs(r['b2']))} respecto de K de entrada)")
    print(f"   armónico sin(4 phi): b4 = {r['b4']:.3e} J/m3 (un material uniaxial con H infinito no lo tendría)")
    print(f"2. Corrección del desfase (máximo {r['lag_deg']:.1f}°): K_ef = {r['k_ef']:.4e} J/m3;"
          f" residuo rms {100 * r['rms']:.2f} %, máximo {100 * r['worst']:.2f} % de K_ef")
    print(f"   amplitud de la curva = {r['amplitude']:.4e} J/m3")
    print(f"3. Anisotropía de forma (Aharoni): {r['shape']:.4e} J/m3 -> K = {r['k_rec']:.4e} J/m3 ({rel(r['k_rec'])})")
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    make_figure(phi, tau, r, args.fig)
    print(f"Figura: {args.fig}")


if __name__ == "__main__":
    main()
