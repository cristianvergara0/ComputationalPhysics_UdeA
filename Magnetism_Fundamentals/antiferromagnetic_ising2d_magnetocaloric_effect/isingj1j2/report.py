"""
Presentación de resultados: formato valor ± error, tablas numeradas y exportación a CSV.
"""

import math
import re

import numpy as np
import pandas as pd

from . import config


def fmt(value, err=None, digits=None):
    """
    'valor ± error' con el número de decimales fijado por dos cifras significativas
    del error (o ``digits`` decimales si se indica). Sin error, el valor con
    ``digits`` decimales (4 por defecto).
    """
    if value is None or not np.isfinite(value):
        return "n/d"
    if err is None or not np.isfinite(err) or err <= 0:
        d = 4 if digits is None else digits
        return f"{value:.{d}f}"
    d = digits if digits is not None else max(0, 1 - int(math.floor(math.log10(err))))
    return f"{value:.{d}f} ± {err:.{d}f}"


def fmt_pair(pair, digits=None):
    """fmt aplicado a una tupla (valor, error)."""
    return fmt(pair[0], pair[1], digits)


def _slug(text):
    text = text.lower()
    for a, b in zip("áéíóúñü", "aeiounu"):
        text = text.replace(a, b)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")[:60]


def show_table(df, number, caption, name=None, index=False):
    """
    Muestra una tabla con su número y leyenda y la guarda en results/tablaNN_<nombre>.csv.

    En Jupyter la leyenda se muestra en Markdown y la tabla como HTML; fuera de
    Jupyter, como texto. No devuelve nada, para que Jupyter no imprima la ruta del
    archivo cuando la llamada es la última línea de una celda.
    """
    path = config.results_dir() / f"tabla{number:02d}_{name or _slug(caption)}.csv"
    df.to_csv(path, index=index)
    try:
        from IPython.display import Markdown, display
        display(Markdown(f"**Tabla {number}.** {caption}"))
        display(df if index else df.style.hide(axis="index"))
    except Exception:
        print(f"Tabla {number}. {caption}")
        print(df.to_string(index=index))


def compatibility(value, err, references, n_sigma=2.0):
    """Nombres de las referencias compatibles con value ± n_sigma err."""
    if not (np.isfinite(value) and np.isfinite(err)):
        return "n/d"
    ok = [name for name, ref in references.items() if abs(value - ref) <= n_sigma * err]
    return ", ".join(ok) if ok else "ninguna"


def deviation_sigma(a, b):
    """(a - b) / sqrt(err_a^2 + err_b^2) para pares (valor, error)."""
    s = math.hypot(a[1], b[1])
    return (a[0] - b[0]) / s if s > 0 else np.nan


def as_frame(rows):
    return pd.DataFrame(rows)
