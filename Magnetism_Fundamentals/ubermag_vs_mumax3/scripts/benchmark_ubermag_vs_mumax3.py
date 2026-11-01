#!/usr/bin/env python3
"""Mide el tiempo de ejecucion de los notebooks de Ubermag/OOMMF y de los scripts de MuMax3.

Los notebooks se ejecutan COPIADOS en un directorio temporal (los originales no se tocan)
con nbclient, registrando el tiempo de cada celda. Los scripts .mx3 se ejecutan con
`mumax3 -f` y se mide el tiempo de pared del proceso completo (incluye arranque del proceso,
inicializacion de CUDA, la simulacion y la escritura de salidas).

Que mide y que no: es el tiempo de pared del FLUJO DE TRABAJO completo en cada herramienta
(notebook con sus llamadas a OOMMF frente a un script de MuMax3 en un solo proceso). No aisla
la velocidad de los solvers: cada llamada de Ubermag lanza un proceso de OOMMF nuevo, y en el
equipo donde se midio ese coste fijo (unos 6 s por llamada) domina el total. Ver la seccion 8
de los notebooks.

Ejecutar desde el entorno de Anaconda que tiene Ubermag y mumax3 en el PATH:

    python scripts/benchmark_ubermag_vs_mumax3.py --repeats 3
    python scripts/benchmark_ubermag_vs_mumax3.py --skip-ubermag          # solo MuMax3
    python scripts/benchmark_ubermag_vs_mumax3.py --skip-mumax3 --repeats 1 --dump-arrays
    python scripts/benchmark_ubermag_vs_mumax3.py --workdir D:\\tmp       # copias fuera de %TEMP%

Los resultados se escriben en results/ (benchmark_results.csv, benchmark_cells.csv,
benchmark_results.md y, con --dump-arrays, ubermag_arrays_<caso>.npz).

Con --skip-ubermag o --skip-mumax3 se CONSERVAN las filas ya medidas del otro enfoque en
benchmark_results.csv y benchmark_cells.csv (se sustituyen solo las que se vuelven a medir); la
columna "sesion" indica cuando se midio cada fila y la tabla avisa si se mezclan sesiones.

--dump-arrays añade, solo en la COPIA del notebook, una ultima celda que guarda con numpy las
magnitudes de Ubermag (angulos y torque; campos de magnetizacion) en
ubermag_arrays_<caso>.npz. Los notebooks originales no cambian. Esos archivos los usan
validate_torque_macrospin.py y validate_skyrmion_numpy.py para comparar con datos exactos.
Esa pasada no modifica los CSV de tiempos: ejecuta una celda extra y no es una medicion del benchmark.

Requiere: nbformat y nbclient (vienen con Jupyter). Salidas: benchmark_results.csv,
benchmark_cells.csv y benchmark_results.md (tabla lista para pegar en markdown).
"""
import argparse
import csv
import importlib.metadata as md
import platform
import shutil
import statistics
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # carpeta del proyecto (los notebooks están en la raíz)
PAIRS = [
    ("micromagnetic_energy_terms_skyrmion.ipynb", "mumax3/skyrmion_energy_terms_mumax3.mx3"),
    ("torque_magnetometry_uniaxial_box.ipynb", "mumax3/torque_magnetometry_mumax3.mx3"),
]
UBERMAG_LABEL = "Ubermag+OOMMF (notebook completo)"
MUMAX_LABEL = "MuMax3 (script completo)"

# Celda que se anade a la COPIA del notebook con --dump-arrays (cada caso usa sus variables).
DUMP_CELLS = {
    "micromagnetic_energy_terms_skyrmion": (
        "try:\n"
        "    import numpy as _np\n"
        "    _np.savez('ubermag_arrays.npz', m_relaxed=system.m.array, M_euler=M.array)\n"
        "except Exception as _e:\n"
        "    print('dump-arrays: no se pudo guardar:', _e)\n"
    ),
    "torque_magnetometry_uniaxial_box": (
        "try:\n"
        "    import numpy as _np\n"
        "    _np.savez('ubermag_arrays.npz', angles=angles, tx=torque_val_x, ty=torque_val_y, tz=torque_val_z)\n"
        "except Exception as _e:\n"
        "    print('dump-arrays: no se pudo guardar:', _e)\n"
    ),
}


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def run_notebook(nb_path, kernel, timeout, workdir=None, dump_to=None):
    """Ejecuta una copia del notebook; devuelve (tiempo_total_s, [(indice, primera_linea, s)]).

    Si dump_to no es None, se anade una celda final (solo en la copia) que guarda los arrays de
    Ubermag y el archivo resultante se copia a dump_to. El tiempo de esa celda (milisegundos)
    queda dentro del total, pero no se incluye en la lista por celdas.
    """
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(str(nb_path), as_version=4)
    n_original = len(nb.cells)
    case = Path(nb_path).stem
    if dump_to is not None and case in DUMP_CELLS:
        nb.cells.append(nbformat.v4.new_code_cell(DUMP_CELLS[case]))
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name=kernel or nb.metadata.get("kernelspec", {}).get("name", "python3"),
            resources={"metadata": {"path": tmp}},
            record_timing=True,
        )
        t0 = time.perf_counter()
        client.execute()
        total = time.perf_counter() - t0
        produced = Path(tmp) / "ubermag_arrays.npz"
        if dump_to is not None and produced.exists():
            shutil.copy(produced, dump_to)
    cells = []
    for i, c in enumerate(nb.cells[:n_original]):
        if c.cell_type != "code":
            continue
        ex = c.metadata.get("execution", {})
        try:
            dt = (_ts(ex["shell.execute_reply"]) - _ts(ex["iopub.execute_input"])).total_seconds()
        except KeyError:
            dt = float("nan")
        first = (c.source.strip().splitlines() or [""])[0][:60]
        cells.append((i, first, dt))
    return total, cells


def run_mumax(script, mumax_cmd, gpu, workdir=None):
    """Ejecuta `mumax3 -f script` en un directorio temporal; devuelve (s, primera_linea_log)."""
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        dst = Path(tmp) / script.name
        shutil.copy(script, dst)
        cmd = [mumax_cmd, "-f"] + (["-gpu", str(gpu)] if gpu is not None else []) + [dst.name]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        elapsed = time.perf_counter() - t0
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            raise RuntimeError(f"mumax3 fallo con {script.name}:\n{out[-2000:]}")
        first = out.strip().splitlines()[0] if out.strip() else ""
    return elapsed, first


def cpu_name():
    """Nombre comercial de la CPU (mejor esfuerzo).

    En Windows platform.processor() solo devuelve la familia ("Intel64 Family 6 Model 183 ...") y
    `wmic` ya no existe en las versiones recientes, asi que se consulta PowerShell.
    """
    try:
        system = platform.system()
        if system == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=30).stdout.strip()
            if out:
                return out.splitlines()[0].strip()
        elif system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            if out:
                return out
    except Exception:
        pass
    return platform.processor() or platform.machine()


def environment_info(mumax_cmd):
    info = {"python": platform.python_version(), "so": platform.platform(),
            "cpu": cpu_name(), "cpu_id": platform.processor() or platform.machine()}
    for pkg in ("ubermag", "discretisedfield", "micromagneticmodel", "oommfc", "mag2exp"):
        try:
            info[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            info[pkg] = "no instalado"
    try:
        info["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        info["gpu"] = "nvidia-smi no disponible"
    return info


def summarize(values):
    return statistics.median(values), min(values), max(values)


def read_previous(base):
    """Filas ya medidas: results = {(caso, enfoque): [..]}, cells = lista de filas del CSV de celdas."""
    results, cells = {}, []
    p = base / "benchmark_results.csv"
    if p.exists():
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                results[(r["caso"], r["enfoque"])] = [
                    r["caso"], r["enfoque"], int(r["repeticiones"]), float(r["mediana_s"]),
                    float(r["min_s"]), float(r["max_s"]), r.get("sesion") or ""]
    p = base / "benchmark_cells.csv"
    if p.exists():
        with open(p, newline="", encoding="utf-8") as f:
            cells = [[r["caso"], int(r["celda"]), r["primera_linea"], float(r["mediana_s"])] for r in csv.DictReader(f)]
    return results, cells


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=3, help="repeticiones medidas por caso")
    ap.add_argument("--warmup-mumax", type=int, default=1,
                    help="ejecuciones de calentamiento de MuMax3 (cache de kernels CUDA), no se cuentan")
    ap.add_argument("--kernel", default=None, help="nombre del kernel de Jupyter (por defecto el del notebook)")
    ap.add_argument("--timeout", type=int, default=7200, help="limite por celda (s)")
    ap.add_argument("--mumax3", default="mumax3", help="ejecutable de MuMax3")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--dir", default=str(ROOT), help="carpeta del proyecto (notebooks en la raíz y scripts de MuMax3 en mumax3/)")
    ap.add_argument("--workdir", default=None,
                    help="carpeta donde se crean los directorios temporales de ejecucion (por defecto, el temporal del sistema)")
    ap.add_argument("--dump-arrays", action="store_true",
                    help="guarda ubermag_arrays_<caso>.npz con los datos de Ubermag (celda extra solo en la copia)")
    ap.add_argument("--skip-ubermag", action="store_true")
    ap.add_argument("--skip-mumax3", action="store_true")
    args = ap.parse_args()
    base = Path(args.dir)
    out = base / "results"          # tablas de tiempos y datos de Ubermag
    out.mkdir(exist_ok=True)
    session = datetime.now().strftime("%Y-%m-%d %H:%M")

    info = environment_info(args.mumax3)
    print("Entorno:", info)
    prev_results, prev_cells = read_previous(out)
    new_results, new_cells, version_line = {}, {}, ""

    for nb_name, mx_name in PAIRS:
        nb_path, mx_path = base / nb_name, base / mx_name
        case = nb_name.replace(".ipynb", "")
        if not args.skip_ubermag:
            totals, per_cell = [], {}
            dump_to = out / f"ubermag_arrays_{case}.npz" if args.dump_arrays else None
            for r in range(args.repeats):
                print(f"[Ubermag] {nb_name}  repeticion {r + 1}/{args.repeats} ...", flush=True)
                total, cells = run_notebook(nb_path, args.kernel, args.timeout, args.workdir, dump_to)
                totals.append(total)
                for i, first, dt in cells:
                    per_cell.setdefault((i, first), []).append(dt)
                print(f"          {total:.1f} s", flush=True)
            med, lo, hi = summarize(totals)
            new_results[(case, UBERMAG_LABEL)] = [case, UBERMAG_LABEL, args.repeats, med, lo, hi, session]
            new_cells[case] = [[case, i, first, statistics.median(v)] for (i, first), v in per_cell.items()]
        if not args.skip_mumax3:
            for _ in range(args.warmup_mumax):
                print(f"[MuMax3] {mx_name}  calentamiento ...", flush=True)
                run_mumax(mx_path, args.mumax3, args.gpu, args.workdir)
            totals = []
            for r in range(args.repeats):
                print(f"[MuMax3] {mx_name}  repeticion {r + 1}/{args.repeats} ...", flush=True)
                s, first = run_mumax(mx_path, args.mumax3, args.gpu, args.workdir)
                version_line = version_line or first
                totals.append(s)
                print(f"         {s:.2f} s", flush=True)
            med, lo, hi = summarize(totals)
            new_results[(case, MUMAX_LABEL)] = [case, MUMAX_LABEL, args.repeats, med, lo, hi, session]

    if args.dump_arrays:
        # Esta pasada ejecuta una celda extra y suele hacerse con una sola repeticion: sus tiempos no
        # sustituyen a los del benchmark, que quedan como estaban en benchmark_results.csv.
        print("\n--dump-arrays: datos de Ubermag guardados; los tiempos de esta pasada no se escriben en los CSV.")
        return

    # Fusion con lo ya medido: solo se sustituye lo que se volvio a medir en esta ejecucion.
    merged = dict(prev_results)
    merged.update(new_results)
    order = {nb.replace(".ipynb", ""): i for i, (nb, _) in enumerate(PAIRS)}
    rows = sorted(merged.values(), key=lambda r: (order.get(r[0], 99), 0 if r[1].startswith("Ubermag") else 1))
    cell_rows = [r for r in prev_cells if r[0] not in new_cells]
    for case_rows in new_cells.values():
        cell_rows.extend(case_rows)
    cell_rows.sort(key=lambda r: (order.get(r[0], 99), r[1]))

    with open(out / "benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["caso", "enfoque", "repeticiones", "mediana_s", "min_s", "max_s", "sesion"])
        w.writerows(rows)
    with open(out / "benchmark_cells.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["caso", "celda", "primera_linea", "mediana_s"])
        w.writerows(cell_rows)

    lines = ["| Caso | Enfoque | Mediana (s) | Min (s) | Max (s) |", "|---|---|---:|---:|---:|"]
    for case, appr, _, med, lo, hi, _s in rows:
        lines.append(f"| {case} | {appr} | {med:.2f} | {lo:.2f} | {hi:.2f} |")
    for case in dict.fromkeys(r[0] for r in rows):
        u = [r for r in rows if r[0] == case and r[1].startswith("Ubermag")]
        m = [r for r in rows if r[0] == case and r[1].startswith("MuMax3")]
        if u and m:
            lines.append(f"\nFactor Ubermag/MuMax3 en {case}: {u[0][3] / m[0][3]:.1f}x")
    if len({r[6] for r in rows}) > 1:
        lines.append("\nAviso: las filas proceden de sesiones distintas (" + ", ".join(s or "sin fecha" for s in
                     sorted({r[6] for r in rows})) + "). El tiempo por llamada de OOMMF cambia entre sesiones en "
                     "algunos equipos; para comparar conviene medir ambos programas en la misma sesion.")
    lines.append("\nEntorno: " + "; ".join(f"{k}={v}" for k, v in info.items()))
    if version_line:
        lines.append("MuMax3: " + version_line)
    text = "\n".join(lines)
    (out / "benchmark_results.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
