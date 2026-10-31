"""Entorno de ejecución, rutas del proyecto y guardado de resultados.

Las rutas se calculan a partir de la ubicación del paquete, no del directorio de trabajo,
de modo que los cuadernos y las pruebas guardan sus resultados en el mismo lugar con
independencia de desde dónde se ejecuten:

* ``resultados/tablas``: tablas en CSV.
* ``resultados/figuras``: figuras en PNG.
* ``resultados/entorno_<cuaderno>.json``: versiones del entorno de cada ejecución.

La carpeta ``resultados`` puede cambiarse con la variable de entorno ``TFMC_RESULTADOS``.
"""

import json
import os
import platform
from datetime import datetime
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
# La variable de entorno TFMC_RESULTADOS permite redirigir las salidas (por ejemplo, en las pruebas).
DIR_RESULTADOS = Path(os.environ.get("TFMC_RESULTADOS", RAIZ_PROYECTO / "resultados"))


def _directorio(*partes):
    """Devuelve ``resultados/<partes>`` y lo crea si no existe."""
    ruta = DIR_RESULTADOS.joinpath(*partes)
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def ruta_figura(nombre):
    """Ruta (``str``) de una figura dentro de ``resultados/figuras``."""
    return str(_directorio("figuras") / nombre)


def ruta_tabla(nombre):
    """Ruta (``str``) de una tabla dentro de ``resultados/tablas``."""
    return str(_directorio("tablas") / nombre)


def guardar_tabla(tabla, nombre, indice=False):
    """Guarda un DataFrame como CSV en ``resultados/tablas``.

    No devuelve nada, de modo que, como última línea de una celda, no imprime la ruta local del
    archivo; ``ruta_tabla(nombre)`` la da cuando hace falta.

    Parámetros
    ----------
    tabla : pandas.DataFrame
        Tabla que se guarda.
    nombre : str
        Nombre del archivo, con o sin la extensión ``.csv``.
    indice : bool
        Si se escribe el índice del DataFrame (por defecto no).
    """
    if not nombre.endswith(".csv"):
        nombre = nombre + ".csv"
    tabla.to_csv(ruta_tabla(nombre), index=indice)


def versiones():
    """Versiones de Python, del sistema y de las bibliotecas numéricas."""
    import matplotlib
    import numpy
    import pandas
    import scipy

    try:
        import numba
        version_numba = numba.__version__
    except ImportError:
        version_numba = None
    return {
        "python": platform.python_version(),
        "sistema": f"{platform.system()} {platform.release()}",
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "matplotlib": matplotlib.__version__,
        "numba": version_numba,
    }


def registrar_entorno(cuaderno=None, guardar=True):
    """Imprime las versiones del entorno y, si se indica el cuaderno, las guarda en JSON.

    La salida impresa queda en el propio cuaderno, de modo que cada resultado guardado
    indica con qué versiones se obtuvo. Devuelve el diccionario de versiones.
    """
    info = versiones()
    info["fecha"] = datetime.now().isoformat(timespec="seconds")
    print("Entorno de ejecución")
    print(f"  Python     : {info['python']} ({info['sistema']})")
    for nombre in ("numpy", "scipy", "pandas", "matplotlib"):
        print(f"  {nombre.capitalize():<11}: {info[nombre]}")
    if info["numba"] is None:
        print("  Numba      : no disponible (el cálculo se ejecuta en Python puro, más lento)")
    else:
        print(f"  Numba      : {info['numba']}")
    if guardar and cuaderno:
        ruta = _directorio() / f"entorno_{cuaderno}.json"
        with open(ruta, "w", encoding="utf-8") as archivo:
            json.dump(info, archivo, ensure_ascii=False, indent=2)
    return info


def crear_fila(ancho):
    """Devuelve una función ``fila(etiqueta, valor)`` que imprime una línea con la etiqueta alineada.

    ``ancho`` es el número de caracteres reservados a la etiqueta; los cuadernos lo fijan
    una vez y todas sus salidas impresas comparten el mismo formato.
    """
    def fila(etiqueta, valor):
        print(f"{etiqueta:<{ancho}}: {valor}")

    return fila
