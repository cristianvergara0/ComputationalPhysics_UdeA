#!/usr/bin/env python3
"""Figuras de tiempos de ejecución (Ubermag+OOMMF frente a MuMax3) a partir de la salida
de benchmark_ubermag_vs_mumax3.py.

Lee benchmark_results.csv y benchmark_cells.csv y escribe, para cada caso,
fig_benchmark_<caso>.png con dos paneles:
  (a) tiempo de pared total (mediana, barras de error = mín.-máx.), escala logarítmica;
  (b) tiempo por celda del notebook de Ubermag y tiempo no atribuible a ninguna celda
      (arranque del kernel, cierre).
También imprime la tabla de cifras que aparece en la sección 8 de los notebooks.

Uso (desde cualquier carpeta; por defecto lee results/ y escribe en figures/):
    python scripts/make_benchmark_figure.py
    python scripts/make_benchmark_figure.py --dir ruta/a/los/csv --out ruta/de/salida
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent   # carpeta del proyecto
UBERMAG = "Ubermag+OOMMF (notebook completo)"
MUMAX = "MuMax3 (script completo)"
C_UB, C_MX, C_OTHER, C_PY = "#4C72B0", "#DD8452", "#8C8C8C", "#A9BCDA"

# Etiqueta de cada celda de código (índice de celda -> texto, ¿lanza OOMMF?, nº de llamadas).
CELLS = {
    "micromagnetic_energy_terms_skyrmion": {
        "titulo": "Energías del skyrmion",
        4: ("Importaciones, sistema y relajación", True, 1),
        7: ("Magnetización reducida (solo Python)", False, 0),
        9: ("Zeeman", True, 3),
        11: ("Anisotropía uniaxial", True, 3),
        13: ("Intercambio", True, 3),
        15: ("DMI (clase T)", True, 3),
        18: ("Euler explícito (solo Python)", False, 0),
    },
    "torque_magnetometry_uniaxial_box": {
        "titulo": "Magnetometría de torque",
        4: ("Importaciones y sistema", False, 0),
        6: ("Relajación a campo cero", True, 1),
        8: ("Torque a campo cero (mag2exp)", False, 0),
        10: ("Barrido angular", True, 71),
    },
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(ROOT / "results"), help="carpeta con los CSV")
    ap.add_argument("--out", default=str(ROOT / "figures"), help="carpeta de salida de las figuras")
    args = ap.parse_args()
    base = Path(args.dir)
    out = Path(args.out) if args.out else base
    out.mkdir(parents=True, exist_ok=True)
    if not (base / "benchmark_results.csv").exists():
        raise SystemExit(f"No se encontró {base / 'benchmark_results.csv'}. Ejecuta antes scripts/benchmark_ubermag_vs_mumax3.py.")
    res = read_csv(base / "benchmark_results.csv")
    cel = read_csv(base / "benchmark_cells.csv")

    for case, spec in CELLS.items():
        rows = {r["enfoque"]: r for r in res if r["caso"] == case}
        if UBERMAG not in rows or MUMAX not in rows:
            print(f"[{case}] faltan filas de Ubermag o de MuMax3 en benchmark_results.csv; se omite.")
            continue
        u, m = rows[UBERMAG], rows[MUMAX]
        um, ulo, uhi = float(u["mediana_s"]), float(u["min_s"]), float(u["max_s"])
        mm_, mlo, mhi = float(m["mediana_s"]), float(m["min_s"]), float(m["max_s"])
        cell_t = {int(r["celda"]): float(r["mediana_s"]) for r in cel if r["caso"] == case}

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(10.6, 3.9), dpi=150, gridspec_kw={"width_ratios": [1, 1.5]}
        )
        # (a) totales
        for x, (med, lo, hi, color) in enumerate([(um, ulo, uhi, C_UB), (mm_, mlo, mhi, C_MX)]):
            ax1.bar(x, med, color=color, width=0.62)
            ax1.errorbar(x, med, yerr=[[med - lo], [hi - med]], color="k", capsize=4, lw=1.2)
            ax1.text(x, hi * 1.18, f"{med:.2f} s" if med < 10 else f"{med:.1f} s", ha="center", fontsize=9)
        ax1.set_yscale("log")
        ax1.set_xticks([0, 1], ["Ubermag + OOMMF\n(notebook completo)", "MuMax3\n(script completo)"], fontsize=8.5)
        ax1.set_ylabel("Tiempo de pared (s)")
        ax1.set_ylim(min(mlo, mm_) / 3, um * 6)
        ax1.set_title(f"{spec['titulo']}: mediana de {u['repeticiones']}\n(cociente de medianas {um / mm_:.1f})", fontsize=9.5)

        # (b) desglose por celdas del notebook de Ubermag
        labels, vals, colors = [], [], []
        for idx in sorted(k for k in spec if isinstance(k, int)):
            name, uses_oommf, ncalls = spec[idx]
            t = cell_t.get(idx)
            if t is None:
                continue
            extra = f" ({ncalls} llamada{'s' if ncalls != 1 else ''} a OOMMF)" if uses_oommf else ""
            labels.append(f"{name}{extra}")
            vals.append(t)
            colors.append(C_UB if uses_oommf else C_PY)
        resto = um - sum(vals)
        labels.append("Arranque del kernel y otros")
        vals.append(resto)
        colors.append(C_OTHER)
        ypos = range(len(vals))[::-1]
        ax2.barh(list(ypos), vals, color=colors, height=0.66)
        ax2.set_yticks(list(ypos), labels, fontsize=7.6)
        vmax = max(vals)
        for y, v in zip(ypos, vals):
            ax2.text(v + vmax * 0.012, y, f"{v:.1f} s" if v >= 1 else f"{v:.2f} s", va="center", fontsize=8)
        ax2.set_xlim(0, vmax * 1.16)
        ax2.set_xlabel("Tiempo por celda en Ubermag (s, mediana)")
        ax2.set_title("Dónde se va el tiempo en el notebook de Ubermag", fontsize=9.5)
        for a in (ax1, ax2):
            a.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fp = out / f"fig_benchmark_{case}.png"
        fig.savefig(fp, bbox_inches="tight")
        plt.close(fig)

        # cifras para el texto de los notebooks
        n_oommf = sum(spec[i][2] for i in spec if isinstance(i, int) and i in cell_t)
        t_oommf_cells = sum(cell_t[i] for i in spec if isinstance(i, int) and i in cell_t and spec[i][1])
        print(f"\n[{case}] -> {fp.name}")
        print(f"  Ubermag: mediana {um:.2f} s (mín. {ulo:.2f}, máx. {uhi:.2f}) | MuMax3: {mm_:.3f} s (mín. {mlo:.3f}, máx. {mhi:.3f}) | cociente {um / mm_:.1f}")
        print(f"  llamadas a OOMMF: {n_oommf}; celdas que las contienen: {t_oommf_cells:.1f} s = {100 * t_oommf_cells / um:.1f} % del notebook")
        for idx in sorted(k for k in spec if isinstance(k, int)):
            if idx in cell_t and spec[idx][2] > 1:
                print(f"  celda {idx}: {cell_t[idx]:.2f} s / {spec[idx][2]} llamadas = {cell_t[idx] / spec[idx][2]:.2f} s por llamada")
        print(f"  no atribuible a celdas (kernel, cierre): {resto:.2f} s")


if __name__ == "__main__":
    main()
