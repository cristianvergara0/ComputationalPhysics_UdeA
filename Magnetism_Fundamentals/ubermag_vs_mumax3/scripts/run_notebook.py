#!/usr/bin/env python3
"""Ejecuta un notebook completo y guarda sus salidas en el mismo archivo.

El kernel se lanza desde este proceso de Python, de modo que hereda el entorno de la terminal: con
el entorno de conda de Ubermag activado, el kernel encuentra OOMMF. Es la forma recomendada de
ejecutar ubermag_numba_calculator.ipynb, cuya comparación con OOMMF necesita el ejecutable oommf en
el PATH. Las celdas se ejecutan con la carpeta del proyecto como directorio de trabajo.

Uso, desde la carpeta del proyecto:
    python scripts/run_notebook.py ubermag_numba_calculator.ipynb
    python scripts/run_notebook.py ubermag_numba_calculator.ipynb --output ejecutado.ipynb

Requiere nbformat, nbclient e ipykernel.
"""
import argparse
import asyncio
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # carpeta del proyecto


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook", help="ruta del notebook")
    ap.add_argument("--output", default=None, help="archivo de salida (por defecto, el mismo notebook)")
    ap.add_argument("--kernel", default="python3", help="nombre del kernel de Jupyter (por defecto: %(default)s)")
    args = ap.parse_args()

    import nbformat
    from nbclient import NotebookClient

    if sys.platform == "win32":   # el bucle por defecto de Windows no admite todo lo que usa zmq (Jupyter)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    path = Path(args.notebook).resolve()
    if not path.exists():
        raise SystemExit(f"No se encontró {path}.")
    nb = nbformat.read(path, as_version=4)
    print(f"Ejecutando {path.name} con el kernel '{args.kernel}' (las salidas aparecen al terminar) ...", flush=True)
    start = time.perf_counter()
    client = NotebookClient(nb, timeout=None, kernel_name=args.kernel, resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    finally:
        out = Path(args.output).resolve() if args.output else path
        nbformat.write(nb, out)
        print(f"Guardado {out.name} ({time.perf_counter() - start:.0f} s)")


if __name__ == "__main__":
    main()
