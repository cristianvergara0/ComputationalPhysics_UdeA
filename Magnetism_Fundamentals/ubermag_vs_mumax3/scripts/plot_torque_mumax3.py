#!/usr/bin/env python3
"""Grafica el torque por unidad de volumen frente al ángulo del campo (salida de
mumax3/torque_magnetometry_mumax3.mx3), con las mismas líneas en ±K que la gráfica del notebook.

Uso (desde cualquier carpeta):
    python scripts/plot_torque_mumax3.py
    python scripts/plot_torque_mumax3.py ruta/a/torque_vs_angle.txt

La figura se guarda como torque_mumax3.png junto al archivo de datos.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent   # carpeta del proyecto
K = 1.18e6  # J/m3, anisotropía uniaxial del notebook


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="?", default=str(ROOT / "mumax3" / "torque_magnetometry_mumax3.out" / "torque_vs_angle.txt"),
                    help="archivo torque_vs_angle.txt de MuMax3")
    path = Path(ap.parse_args().data)
    if not path.exists():
        raise SystemExit(f"No se encontró {path}. Ejecuta antes mumax3/torque_magnetometry_mumax3.mx3 (ver README).")
    data = np.loadtxt(path, comments="#")
    angle, tau_x = data[:, 0], data[:, 1]

    fig, ax = plt.subplots(dpi=100)
    ax.hlines([K, -K], xmin=0, xmax=2 * np.pi, color="r", linestyle="--")
    ax.scatter(angle, tau_x, color="#DD8452")
    ax.set_xlabel("Ángulo del campo (rad)")
    ax.set_ylabel(r"Torque (N m$^{-2}$)")
    ax.set_xlim([0, 2 * np.pi])
    out = path.parent / "torque_mumax3.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Guardado {out}; |tau_x| máximo = {np.abs(tau_x).max():.6g} N/m2")


if __name__ == "__main__":
    main()
