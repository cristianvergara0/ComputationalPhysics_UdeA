#!/usr/bin/env python3
"""Tiempo de una minimización en función del número de celdas, en Ubermag/OOMMF y en MuMax3.

Separa el costo fijo de cada llamada del costo que crece con el tamaño del problema. Usa el sistema
del notebook de torque (intercambio, anisotropía uniaxial, campo desmagnetizante y Zeeman, con los
mismos parámetros y celdas de 5 nm) en cajas de n x n x n/3 celdas, con un campo de 2.38e6 A/m a 45°
en el plano yz aplicado sobre el estado uniforme +z. Para cada tamaño:

  Ubermag/OOMMF: md.drive(system, verbose=0) con oc.MinDriver(), como en el notebook. Cada llamada
      escribe los archivos de entrada, lanza un proceso de OOMMF y lee el resultado; se mide el
      tiempo de pared de la llamada completa. Antes de medir se hace una llamada de calentamiento
      con el tamaño más pequeño, que no se cuenta.
  MuMax3: dos scripts con la misma configuración. Uno prepara el sistema y evalúa la energía total
      (lo que obliga a calcular el núcleo desmagnetizante); el otro, además, llama a Minimize(). Se
      mide el tiempo de pared de cada proceso (`mumax3 -f`), tras una ejecución de calentamiento de
      cada script. El tiempo de la minimización es la diferencia de las medianas de los dos; si la
      minimización dura menos que la dispersión entre procesos, esa diferencia no la resuelve.

En cada caso se ajusta t = t0 + c N por mínimos cuadrados: t0 es el costo fijo por llamada y c el
costo por celda. El lenguaje de MuMax3 no tiene funciones de reloj, por eso su minimización se mide
por diferencia de dos procesos.

Uso (en el entorno con Ubermag y con mumax3 en el PATH):
    python scripts/benchmark_scaling.py
    python scripts/benchmark_scaling.py --sizes 12 24 36 48 60 --repeats 3
    python scripts/benchmark_scaling.py --skip-ubermag        # solo MuMax3
    python scripts/benchmark_scaling.py --plot-only           # rehace tabla y figura con el CSV medido

Con --skip-ubermag o --skip-mumax3 se conservan las filas ya medidas del otro código.
Salidas: results/benchmark_scaling.csv, results/benchmark_scaling.md y figures/fig_benchmark_scaling.png.
Requiere numpy y matplotlib; la parte de Ubermag, discretisedfield, micromagneticmodel y oommfc.
"""
import argparse
import csv
import importlib.metadata as imd
import os
import platform
import shutil
import statistics
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent   # carpeta del proyecto
CELL = 5e-9
MS, A_EX, K_U, H_MAG = 9.65e5, 8.78e-13, 1.18e6, 2.38e6   # parámetros del notebook de torque
ANGLE_DEG = 45.0
DEFAULT_SIZES = [6, 12, 24, 36, 48, 60]                    # 60 x 60 x 20 es la malla del notebook

UB = "Ubermag+OOMMF: md.drive"
MX_FULL = "MuMax3: proceso con Minimize()"
MX_SETUP = "MuMax3: proceso sin Minimize()"
MX_MIN = "MuMax3: Minimize() (diferencia)"
FIELDS = ["codigo", "n", "celdas", "repeticiones", "mediana_s", "min_s", "max_s", "mz_final", "sesion"]


def grid(n):
    """Malla de n x n x n/3 celdas (la proporción 3:3:1 de la caja del notebook)."""
    return n, n, max(1, n // 3)


def n_cells(n):
    nx, ny, nz = grid(n)
    return nx * ny * nz


# ---------------------------------------------------------------------------------------------
# MuMax3
# ---------------------------------------------------------------------------------------------
def mx3_script(n, minimize):
    """Script de MuMax3 para la caja de tamaño n; con minimize=False solo prepara el sistema."""
    nx, ny, nz = grid(n)
    lines = [
        f"// benchmark_scaling.py: caja de {nx} x {ny} x {nz} celdas de 5 nm, sistema del notebook de torque",
        f"SetGridSize({nx}, {ny}, {nz})",
        f"SetCellSize({CELL:g}, {CELL:g}, {CELL:g})",
        f"Msat = {MS:g}",
        f"Aex = {A_EX:g}",
        f"Ku1 = {K_U:g}",
        "anisU = vector(0, 0, 1)",
        f"phi := {ANGLE_DEG:g} * pi / 180",
        f"B_ext = vector(0, mu0*{H_MAG:g}*sin(phi), mu0*{H_MAG:g}*cos(phi))",
        "m = Uniform(0, 0, 1)",
        'print("E_total inicial (J):", E_total.Get())',
    ]
    if minimize:
        lines += [
            "Minimize()",
            'print("<m> tras Minimize:", m.Comp(0).Average(), m.Comp(1).Average(), m.Comp(2).Average())',
            'print("MaxTorque (T):", MaxTorque.Get())',
        ]
    return "\n".join(lines) + "\n"


def parse_mz(text):
    """m_z de la línea "<m> tras Minimize:" de la salida de MuMax3 (None si no aparece)."""
    for line in text.splitlines():
        if "<m> tras Minimize:" in line and not line.lstrip().startswith("print"):
            try:
                return float(line.split(":", 1)[1].split()[2])
            except (IndexError, ValueError):
                return None
    return None


def run_mumax(script_text, name, mumax_cmd, gpu, workdir):
    """Ejecuta un script en un directorio temporal; devuelve (tiempo_s, salida)."""
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        path = Path(tmp) / name
        path.write_text(script_text, encoding="utf-8")
        cmd = [mumax_cmd, "-f"] + (["-gpu", str(gpu)] if gpu is not None else []) + [name]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        elapsed = time.perf_counter() - t0
        out = (proc.stdout or "") + (proc.stderr or "")
        log = Path(tmp) / (path.stem + ".out") / "log.txt"
        if log.exists():
            out += "\n" + log.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"mumax3 falló con {name}:\n{out[-2000:]}")
    return elapsed, out


def measure_mumax(sizes, repeats, mumax_cmd, gpu, workdir, session):
    rows = []
    for n in sizes:
        scripts = {False: mx3_script(n, False), True: mx3_script(n, True)}
        times, mz = {False: [], True: []}, None
        for minimize, text in scripts.items():
            name = f"scaling_{n}_{'min' if minimize else 'setup'}.mx3"
            print(f"[MuMax3] n = {n} ({n_cells(n)} celdas), {'con' if minimize else 'sin'} Minimize: calentamiento ...", flush=True)
            run_mumax(text, name, mumax_cmd, gpu, workdir)
            for r in range(repeats):
                s, out = run_mumax(text, name, mumax_cmd, gpu, workdir)
                times[minimize].append(s)
                if minimize:
                    mz = parse_mz(out)
                print(f"         repetición {r + 1}/{repeats}: {s:.2f} s", flush=True)
        for label, vals in ((MX_FULL, times[True]), (MX_SETUP, times[False])):
            rows.append(row(label, n, vals, mz if label == MX_FULL else None, session))
        diff = statistics.median(times[True]) - statistics.median(times[False])
        rows.append({"codigo": MX_MIN, "n": n, "celdas": n_cells(n), "repeticiones": repeats,
                     "mediana_s": diff, "min_s": "", "max_s": "", "mz_final": "", "sesion": session})
    return rows


# ---------------------------------------------------------------------------------------------
# Ubermag / OOMMF
# ---------------------------------------------------------------------------------------------
def ubermag_system(n):
    """El sistema del notebook de torque en una caja de n x n x n/3 celdas, con el campo a 45°."""
    import discretisedfield as df
    import micromagneticmodel as mm

    nx, ny, nz = grid(n)
    region = df.Region(p1=(-nx * CELL / 2, -ny * CELL / 2, 0), p2=(nx * CELL / 2, ny * CELL / 2, nz * CELL))
    mesh = df.Mesh(region=region, cell=(CELL, CELL, CELL))
    phi = np.deg2rad(ANGLE_DEG)
    system = mm.System(name=f"scaling_{n}")
    system.energy = (
        mm.Exchange(A=A_EX)
        + mm.UniaxialAnisotropy(K=K_U, u=(0, 0, 1))
        + mm.Demag()
        + mm.Zeeman(H=(0, H_MAG * np.sin(phi), H_MAG * np.cos(phi)))
    )
    return system, mesh


def measure_ubermag(sizes, repeats, workdir, session):
    import discretisedfield as df
    import oommfc as oc

    rows = []
    md = oc.MinDriver()
    old_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(dir=workdir)
    try:
        os.chdir(tmp)   # OOMMF escribe en <system.name>/drive-*/ dentro de la carpeta actual
        system, mesh = ubermag_system(min(sizes))
        system.m = df.Field(mesh, nvdim=3, value=(0, 0, 1), norm=MS)
        print("[Ubermag] calentamiento ...", flush=True)
        md.drive(system, verbose=0)
        print(f"[Ubermag] runner: {oc.runner.runner}", flush=True)
        for n in sizes:
            system, mesh = ubermag_system(n)
            times = []
            for r in range(repeats):
                system.m = df.Field(mesh, nvdim=3, value=(0, 0, 1), norm=MS)
                t0 = time.perf_counter()
                md.drive(system, verbose=0)
                times.append(time.perf_counter() - t0)
                print(f"[Ubermag] n = {n} ({n_cells(n)} celdas), repetición {r + 1}/{repeats}: {times[-1]:.2f} s", flush=True)
            mz = float(system.m.orientation.array[..., 2].mean())
            rows.append(row(UB, n, times, mz, session))
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    return rows


# ---------------------------------------------------------------------------------------------
# Tablas, ajuste y figura
# ---------------------------------------------------------------------------------------------
def row(label, n, vals, mz, session):
    return {"codigo": label, "n": n, "celdas": n_cells(n), "repeticiones": len(vals),
            "mediana_s": statistics.median(vals), "min_s": min(vals), "max_s": max(vals),
            "mz_final": "" if mz is None else mz, "sesion": session}


def fit_linear(cells, t):
    """Ajuste t = t0 + c N por mínimos cuadrados; devuelve (t0, c, error_t0, error_c)."""
    x, y = np.asarray(cells, float), np.asarray(t, float)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    dof = len(x) - 2
    if dof > 0:
        s2 = float(((y - X @ coef) ** 2).sum()) / dof
        cov = s2 * np.linalg.inv(X.T @ X)
        err = np.sqrt(np.diag(cov))
    else:
        err = np.array([np.nan, np.nan])
    return float(coef[0]), float(coef[1]), float(err[0]), float(err[1])


def read_rows(path):
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["n"], r["celdas"], r["repeticiones"] = int(r["n"]), int(r["celdas"]), int(r["repeticiones"])
        for k in ("mediana_s", "min_s", "max_s"):
            r[k] = float(r[k]) if r[k] not in ("", None) else ""
    return rows


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def series(rows, label):
    sel = sorted((r for r in rows if r["codigo"] == label), key=lambda r: r["celdas"])
    return sel


def summary_markdown(rows, info):
    lines = ["| Código | Celdas | Mediana (s) | Mín. (s) | Máx. (s) |", "|---|---:|---:|---:|---:|"]
    for label in (UB, MX_FULL, MX_SETUP, MX_MIN):
        for r in series(rows, label):
            lo = f"{r['min_s']:.2f}" if r["min_s"] != "" else ""
            hi = f"{r['max_s']:.2f}" if r["max_s"] != "" else ""
            lines.append(f"| {label} | {r['celdas']} | {r['mediana_s']:.3f} | {lo} | {hi} |")
    lines.append("\nAjuste t = t0 + c N (mínimos cuadrados sobre las medianas):\n")
    lines.append("| Código | t0 (s) | c (µs por celda) |")
    lines.append("|---|---:|---:|")
    for label in (UB, MX_FULL, MX_MIN):
        s = series(rows, label)
        if len(s) >= 2:
            t0, c, e0, ec = fit_linear([r["celdas"] for r in s], [r["mediana_s"] for r in s])
            lines.append(f"| {label} | {t0:.3f} ± {e0:.3f} | {c * 1e6:.3f} ± {ec * 1e6:.3f} |")
    sessions = sorted({r["sesion"] for r in rows})
    if len(sessions) > 1:
        lines.append("\nAviso: las filas proceden de sesiones distintas (" + ", ".join(sessions) + ").")
    if info:
        lines.append("\nEntorno: " + "; ".join(f"{k}={v}" for k, v in info.items()))
    return "\n".join(lines)


def make_figure(rows, path):
    """(a) tiempo por llamada frente a N; (b) la parte que crece con N, t - t0, con el ajuste lineal.

    La diferencia de procesos de MuMax3 (solo Minimize) no se dibuja: queda por debajo de la
    dispersión entre procesos y está en la tabla.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c_ub, c_mx = "#4C72B0", "#DD8452"   # los mismos colores que make_benchmark_figure.py
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6), layout="constrained")
    styles = [(UB, c_ub, "o", "Ubermag + OOMMF (una llamada a md.drive)"),
              (MX_FULL, c_mx, "s", "MuMax3 (un proceso con Minimize())")]
    handles = []
    for label, color, marker, text in styles:
        s = series(rows, label)
        if not s:
            continue
        x = np.array([r["celdas"] for r in s], float)
        y = np.array([r["mediana_s"] for r in s], float)
        lo = y - np.array([r["min_s"] for r in s], float)
        hi = np.array([r["max_s"] for r in s], float) - y
        h = ax1.errorbar(x, y, yerr=[lo, hi], color=color, lw=2, marker=marker, ms=7, capsize=3,
                         elinewidth=1, label=text)
        handles.append(h)
        t0, c, e0, ec = fit_linear(x, y)
        ax1.annotate(f"$t_0$ = {t0:.2f} s", (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                     va="center", fontsize=8.5, color="0.2")
        ax2.errorbar(x / 1e3, y - t0, yerr=[lo, hi], color=color, ls="none", marker=marker, ms=7,
                     capsize=3, elinewidth=1)
        xf = np.linspace(0, x.max(), 50)
        ax2.plot(xf / 1e3, c * xf, color=color, lw=1.5, ls="--")
        ax2.annotate(f"$c$ = {c * 1e6:.1f} µs por celda", (x[-1] / 1e3, c * x[-1]), xytext=(-6, 10),
                     textcoords="offset points", ha="right", fontsize=8.5, color="0.2")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Número de celdas $N$")
    ax1.set_ylabel("Tiempo de pared por llamada (s)")
    ax1.set_title("(a) Tiempo total (mediana; barras: mín.-máx.)", fontsize=10)
    xmax = max((r["celdas"] for r in rows), default=1)
    ax1.set_xlim(right=xmax * 5)   # espacio para las etiquetas directas
    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.set_xlabel("Número de celdas $N$ (miles)")
    ax2.set_ylabel("$t - t_0$ (s)")
    ax2.set_title("(b) Parte que crece con el tamaño y ajuste $t = t_0 + cN$", fontsize=10)
    for a in (ax1, ax2):
        a.grid(True, which="major", color="0.88", lw=0.8)
        a.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Una minimización del sistema de torque en mallas de 72 a 72 000 celdas", fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def environment_info(mumax_cmd):
    info = {"python": platform.python_version(), "so": platform.platform()}
    for pkg in ("discretisedfield", "micromagneticmodel", "oommfc"):
        try:
            info[pkg] = imd.version(pkg)
        except imd.PackageNotFoundError:
            info[pkg] = "no instalado"
    try:
        info["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        info["gpu"] = "nvidia-smi no disponible"
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
                    help="lados n de las cajas de n x n x n/3 celdas (por defecto: %(default)s)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--mumax3", default="mumax3", help="ejecutable de MuMax3")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--workdir", default=None, help="carpeta para los directorios temporales")
    ap.add_argument("--skip-ubermag", action="store_true")
    ap.add_argument("--skip-mumax3", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="solo rehace la tabla y la figura")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--figures", default=str(ROOT / "figures"))
    args = ap.parse_args()
    res, figs = Path(args.results), Path(args.figures)
    res.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    csv_path = res / "benchmark_scaling.csv"
    rows = read_rows(csv_path)
    info = {}

    if not args.plot_only:
        if not rows and (args.skip_ubermag or args.skip_mumax3):
            print("Aviso: no hay mediciones previas del otro código en", csv_path)
        session = datetime.now().strftime("%Y-%m-%d %H:%M")
        info = environment_info(args.mumax3)
        print("Entorno:", info)
        new = []
        if not args.skip_ubermag:
            new += measure_ubermag(args.sizes, args.repeats, args.workdir, session)
        if not args.skip_mumax3:
            new += measure_mumax(args.sizes, args.repeats, args.mumax3, args.gpu, args.workdir, session)
        replaced = {r["codigo"] for r in new}
        rows = [r for r in rows if r["codigo"] not in replaced] + new
        write_rows(csv_path, rows)
    elif not rows:
        raise SystemExit(f"No se encontró {csv_path}. Ejecuta antes el benchmark sin --plot-only.")

    md_path = res / "benchmark_scaling.md"
    text = summary_markdown(rows, info)
    if not info and md_path.exists():   # --plot-only: se conserva la línea de entorno de la medición
        env = [line for line in md_path.read_text(encoding="utf-8").splitlines() if line.startswith("Entorno: ")]
        if env:
            text += "\n\n" + env[0]
    md_path.write_text(text, encoding="utf-8")
    make_figure(rows, figs / "fig_benchmark_scaling.png")
    print("\n" + text)
    print(f"\nFigura: {figs / 'fig_benchmark_scaling.png'}")


if __name__ == "__main__":
    main()
